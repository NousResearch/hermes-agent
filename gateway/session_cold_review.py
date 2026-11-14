"""``/review`` and messaging ``/refine`` for a session whose cached agent is gone.

The gateway drops a session's cached agent after an idle hour (``agent.agent_cache.idle_ttl_secs``),
under LRU or memory pressure, and around ``/moa``. The conversation outlives it in the session's
stored transcript, but ``/review`` read only the cached agent, so a session left alone for an hour
answered "Nothing to review yet" until the next message rebuilt one. A cold ``/review`` reviews the
stored transcript and dispatches the reviewer from a fresh agent on the route, toolsets and session
id the session's next turn would use, so its completion comes back exactly as a resident agent's.
"""
import asyncio
import concurrent.futures
import contextlib
import logging

logger = logging.getLogger(__name__)


async def cold_review_history(runner, source):
    """``(session_id, transcript)`` of the session ``source`` belongs to: a local session's current
    physical transcript (``gateway/session_refine.py`` reads the same one), else the messaging entry's."""
    from gateway.session_refine import _local_session
    authority, live = _local_session(runner, source)
    if live is not None:
        from gateway.session_contract import SessionRef
        from gateway.session_local_recovery import transcript_target

        def load():
            target = transcript_target(authority, SessionRef(authority.profile_id, source.chat_id))
            return target, authority.db.get_messages_as_conversation(target, include_ancestors=True)
        return await asyncio.to_thread(load)
    entry = await runner.async_session_store.get_or_create_session(source)
    return entry.session_id, await runner.async_session_store.load_transcript(entry.session_id)


async def stored_transcript(runner, session_key):
    """``(session_id, transcript)`` of a messaging session; the transcript is empty when the chat has no
    session or no user/assistant turn. Read-only: ``/refine`` in a new chat must not create a session."""
    session_id = await runner.async_session_store.peek_session_id(session_key)
    history = await runner.async_session_store.load_transcript(session_id) if session_id else []
    if not any(m.get("role") in {"user", "assistant"} for m in history):
        return session_id, []
    return session_id, list(history)


@contextlib.contextmanager
def cold_review_parent(runner, source, session_key, session_id):
    """A fresh agent on the session's next-turn route (built as ``session_execution._build_fresh_agent``
    builds one, minus the turn-only prompt pieces), inside the session's policy scope. On exit only its
    own LLM clients are released: ``close()`` would tear down the session's terminal and browser, and the
    reviewer it dispatched is detached and builds its own clients."""
    from gateway.run import _load_gateway_config, _platform_config_key
    from gateway.session_api_turn import api_runtime_locked
    from gateway.session_authorities import active_authority
    from gateway.session_policy import policy_for_source, policy_scope, policy_turn_toolsets
    from gateway.session_worker_construct import session_prefill, session_routing
    from run_agent import AIAgent
    policy = policy_for_source(runner, source)
    authority = active_authority(runner) if policy else None
    user_config = policy.config(authority, secrets=False) if policy else _load_gateway_config()
    platform_key = policy.platform if policy else _platform_config_key(source.platform)
    enabled, disabled = runner._resolve_turn_toolsets(user_config, source, platform_key)
    if policy:
        enabled = policy_turn_toolsets(authority, policy, user_config, source.chat_id)
    # Each input below is the one the turn runner reads for this session (``run_turn_runner``).
    pr = session_routing(authority, policy) if policy else runner._provider_routing
    prefill = session_prefill(authority, policy, source.chat_id) if policy else runner._prefill_messages
    tier = (policy.current_service_tier(runner, session_key) if policy else
            runner._resolve_session_service_tier(source=source, session_key=session_key))
    with policy_scope(policy, authority=authority):
        model, runtime = runner._resolve_session_agent_runtime(
            source=source, session_key=session_key, user_config=user_config)
        route = runner._resolve_turn_agent_config("", model, runtime, tier=tier)  # /fast request options
        agent = AIAgent(
            model=route["model"], **route["runtime"], quiet_mode=True, verbose_logging=False,
            enabled_toolsets=enabled, disabled_toolsets=disabled,
            # Launch-preloaded skills (``hermes -s``): the reviewer is told to load them too.
            ephemeral_system_prompt=(policy.skills_prompt if policy else None) or None,
            prefill_messages=prefill or None,
            reasoning_config=runner._resolve_session_reasoning_config(source=source, session_key=session_key,
                                                                      model=model),
            service_tier=tier, request_overrides=route.get("request_overrides"),
            providers_allowed=pr.get("only"), providers_ignored=pr.get("ignore"), providers_order=pr.get("order"),
            provider_sort=pr.get("sort"), provider_require_parameters=pr.get("require_parameters", False),
            provider_data_collection=pr.get("data_collection"),
            session_id=session_id, platform=platform_key, gateway_session_key=session_key,
            session_db=getattr(runner._session_db, "_db", runner._session_db),
            # The session's backup providers, as its next turn would carry them: the reviewer inherits the chain.
            fallback_model=None if api_runtime_locked() else runner._refresh_fallback_model(),
            # The reviewer reads the transcript excerpt; this agent never runs a turn of its own.
            skip_memory=True)
        agent._end_session_on_close = False
        try:
            yield agent
        finally:
            with contextlib.suppress(Exception):
                agent.release_clients()


def start_cold_refine(runner, source, session_key, session_id, history, focus):
    """``/refine`` for a messaging session whose cached agent is gone: the resident path's own
    ``_spawn_background_review`` call, made from the cold parent. A resident agent sends its review
    summary into the chat through its last turn's callback (``run_turn_runner._make_bg_review_callbacks``);
    this one sends it to the same chat and thread directly. Returns a future that resolves once the review
    is spawned (or raises its start error); the parent's clients are released when the review's request ends,
    on the thread that built it."""
    from agent.async_utils import safe_schedule_threadsafe
    from agent.memory_provider import spawn_context_thread
    from gateway.run import _interim_metadata, _non_conversational_metadata
    loop = asyncio.get_running_loop()
    adapter = runner._delivery_adapter_for(source)
    metadata = _interim_metadata(_non_conversational_metadata(
        runner._thread_metadata_for_source(source), platform=source.platform))
    started = concurrent.futures.Future()

    def deliver(message):
        if adapter is not None:
            safe_schedule_threadsafe(adapter.send(source.chat_id, str(message), metadata=metadata), loop,
                                     logger=logger, log_message="cold /refine summary delivery scheduling failed",
                                     log_level=logging.WARNING)

    def body():
        try:
            with cold_review_parent(runner, source, session_key, session_id) as parent:
                parent.background_review_callback = deliver
                parent._spawn_background_review(
                    messages_snapshot=history, review_memory=True,
                    review_skills="skill_manage" in getattr(parent, "valid_tool_names", set()), focus=focus or None,
                    explicit=True)
                started.set_result(None)
                run = getattr(parent, "_background_review_run", None)
                if run is not None:
                    run.request_done.wait()
        except Exception as exc:  # health: allow BLE001 -- the start error becomes the /refine reply (start_failed)
            if started.done():
                logger.warning("cold /refine failed after it started", exc_info=True)
            else:
                started.set_exception(exc)

    spawn_context_thread(body, name="cold-refine").start()
    return asyncio.wrap_future(started)

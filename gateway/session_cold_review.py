"""``/review`` for a session whose cached agent is gone.

The gateway drops a session's cached agent after an idle hour (``agent.agent_cache.idle_ttl_secs``),
under LRU or memory pressure, and around ``/moa``. The conversation outlives it in the session's
stored transcript, but ``/review`` read only the cached agent, so a session left alone for an hour
answered "Nothing to review yet" until the next message rebuilt one. A cold ``/review`` reviews the
stored transcript and dispatches the reviewer from a fresh agent on the route, toolsets and session
id the session's next turn would use, so its completion comes back exactly as a resident agent's.
"""
import asyncio
import contextlib


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

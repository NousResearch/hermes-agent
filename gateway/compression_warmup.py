"""Post-compression idle prefix-cache warmup for sessions on a local model server.

Compression rebuilds the transcript, so the next turn's request is a brand-new token
prefix for the local model server (a paged prefix cache: oMLX, llama.cpp, vLLM, Ollama
all keep one): the first turn after a manual /compress or a deferred (watermark-fenced)
hygiene adoption pays a full cold prefill — measured 2026-10-05 on oMLX with a 35B:
~23K tokens ≈ 41s TTFT, versus 1-4s for a cache-hit turn. When a compression commit
lands while the session is idle, replay the rebuilt context once with ``max_tokens=1``
so the server stores the boundary snapshots before the user's next message arrives;
that turn then hits the prefix cache.

Byte-alignment is the whole game: the chat template renders the tool list into the
front of the token stream, so the warmup request is built through the SAME machinery the
next real turn uses — ``_resolve_session_agent_runtime`` / ``_resolve_turn_agent_config``
/ ``TurnRunner._build_fresh_agent`` (agent construction incl. toolset + ephemeral prompt
pins), ``session_store.load_transcript`` + ``_build_gateway_agent_history`` (transcript
replay), ``conversation_loop._restore_or_build_system_prompt`` (stored system prompt +
pinned tools), ``turn_request_assembly.assemble_api_request`` +
``agent._build_api_kwargs`` (request assembly) — with ``max_tokens=1`` overlaid at the
end via the agent's own ``_max_tokens_param``.

Gates (all fail-open, failure only logs):

* the route must resolve to a LOCAL model server (loopback / LAN / container name — see
  ``agent.model_metadata.is_local_endpoint``). Cloud providers bill per token and manage
  their own prompt caches, so warming them is pure cost;
* the session must be idle at fire time — no running turn, no adapter-pending or
  overflow-queued messages, no compression still in flight. A local server's serial FCFS
  queue would double the prefill instead of hiding it, so a busy session is re-polled
  briefly and then abandoned;
* the response is discarded: nothing is appended to the transcript, state.db, or the
  cached agent — this is cache filling, not conversation;
* the whole feature is opt-in (``gateway.prefix_cache_warmup.enabled``), default off:
  a local server is the only route it can help, and not every local server keeps a
  prefix cache the user cares about.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from contextlib import suppress
from typing import Any, Optional

# Log-record parity with gateway/run.py so warmup lines land in gateway.log.
logger = logging.getLogger("gateway.run")

# Wait before the first idle check: a commit that lands mid-turn (the common case for a
# deferred hygiene adoption) sees the turn still running; a /compact between messages is
# already idle and this delay is the whole warmup latency budget.
_INITIAL_DELAY_SECONDS = 4.0
# Re-poll a busy session for this long before abandoning the warmup — the turn that
# triggered hygiene typically runs for minutes after the summary lands.
_RECHECK_INTERVAL_SECONDS = 20.0
_MAX_WAIT_SECONDS = 900.0

# One scheduled warmup per session; a newer compression commit supersedes the older one
# (its rebuilt prefix is already stale).
_generations: dict[str, int] = {}
_generation_counter = itertools.count(1)
_generation_lock = threading.Lock()
_tasks: set = set()


def _next_generation(session_key: str) -> int:
    with _generation_lock:
        gen = next(_generation_counter)
        _generations[session_key] = gen
        return gen


def _is_current(session_key: str, generation: int) -> bool:
    with _generation_lock:
        return _generations.get(session_key) == generation


def _profile_scope(gateway: Any, source: Any):
    """The routed profile's runtime scope (config + credentials), or a no-op scope.

    Kept as its own function so the scope can be exercised without importing the
    gateway runtime (which reads the checkout's git directory at import time).
    """
    from contextlib import nullcontext

    try:
        from gateway.run import _profile_runtime_scope

        return _profile_runtime_scope(gateway._resolve_profile_home_for_source(source))
    except Exception:
        return nullcontext()


def _is_local_runtime(runtime_kwargs: dict) -> bool:
    """True only for a chat-completions route that resolves to a LOCAL model server.

    Cloud providers bill per token and manage their own prompt caches, so warming them
    is pure cost. A local server (loopback, LAN, container name) keeps a paged prefix
    cache, and a compression commit invalidates the prefix the next turn would hit.
    """
    api_mode = str(runtime_kwargs.get("api_mode") or "chat_completions").strip().lower()
    if api_mode != "chat_completions":
        return False
    base_url = str(runtime_kwargs.get("base_url") or "").strip()
    if not base_url:
        return False
    try:
        from agent.model_metadata import is_local_endpoint

        return bool(is_local_endpoint(base_url))
    except Exception:
        return False


def _warmup_enabled() -> bool:
    """``gateway.prefix_cache_warmup.enabled`` — the whole feature is opt-in.

    Must be called INSIDE the routed profile's scope (``_profile_runtime_scope``): a
    multiplexed gateway serves several profiles, and each owns its own config.yaml, so
    reading the flag outside the scope would consult the wrong profile's file.
    """
    try:
        from gateway.run import _load_gateway_config

        section = (_load_gateway_config() or {}).get("gateway")
        block = section.get("prefix_cache_warmup") if isinstance(section, dict) else None
        return bool(block.get("enabled", False)) if isinstance(block, dict) else False
    except Exception:
        return False


def _session_idle(gateway: Any, session_key: str, source: Any) -> bool:
    """No running turn (incl. pending sentinel) and nothing queued for the session."""
    with suppress(Exception):
        if session_key in dict(gateway._running_agent_items()):
            return False
    adapter = None
    with suppress(Exception):
        adapter = gateway._delivery_adapter_for(source)
    pending = getattr(adapter, "_pending_messages", None)
    if isinstance(pending, dict) and pending.get(session_key) is not None:
        return False
    with suppress(Exception):
        if gateway._overflow_queue(session_key):
            return False
    return True


def schedule_compression_warmup(
    gateway: Any, *, session_key: Optional[str], session_entry: Any, source: Any, reason: str,
) -> None:
    """Schedule an idle-gated prefix warmup after a compression commit. Never raises.

    Called from the gateway points where a committed compression can leave the session
    idle: the watermark-fenced hygiene adoption done-callback and manual /compress.
    In-turn commits (preflight / overflow recovery) are not hooked — the continuing turn
    re-prefills itself, and warming beside it would double the serial-queue prefill.
    """
    try:
        if not session_key or session_entry is None or source is None:
            return
        loop = asyncio.get_running_loop()
        generation = _next_generation(session_key)
        task = loop.create_task(
            _warmup_task(
                gateway, session_key=session_key, session_entry=session_entry,
                source=source, reason=reason, generation=generation,
            ),
            name=f"compression-warmup-{str(session_key)[:40]}",
        )
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except Exception:
        logger.debug("Compression warmup scheduling failed for %s", session_key, exc_info=True)


async def _warmup_task(
    gateway: Any, *, session_key: str, session_entry: Any, source: Any, reason: str, generation: int,
) -> None:
    try:
        # Multiplex: re-enter the owning profile's scope (config + secret contextvars) the
        # same way slash dispatch does — the scheduled task must not read the launch
        # profile's config/credentials for a routed session.
        scope = _profile_scope(gateway, source)
        with scope:
            # The opt-in flag is read INSIDE the routed profile's scope: a multiplexed
            # gateway serves several profiles, each with its own config.yaml.
            if not _warmup_enabled():
                return
            await asyncio.sleep(_INITIAL_DELAY_SECONDS)
            deadline = time.monotonic() + _MAX_WAIT_SECONDS
            while True:
                if not _is_current(session_key, generation):
                    logger.info(
                        "Compression warmup skipped for session %s: superseded by a newer commit",
                        getattr(session_entry, "session_id", None),
                    )
                    return
                if _session_idle(gateway, session_key, source) and not (
                    await gateway._session_has_compression_in_flight(session_key)
                ):
                    break
                if time.monotonic() >= deadline:
                    logger.info(
                        "Compression warmup skipped for session %s: session stayed busy "
                        "(running turn / queued messages) for over %.0fs",
                        getattr(session_entry, "session_id", None), _MAX_WAIT_SECONDS,
                    )
                    return
                await asyncio.sleep(_RECHECK_INTERVAL_SECONDS)
            # Pins survive restarts in the routing store; an internal-event read reuses the
            # pinned bytes verbatim, which is exactly what the next human turn replays.
            with suppress(Exception):
                await gateway._rehydrate_prompt_pins(session_key, getattr(session_entry, "session_id", None))
            loop = asyncio.get_running_loop()
            from contextvars import copy_context

            outcome = await loop.run_in_executor(
                None, copy_context().run,
                lambda: _fire_warmup_sync(gateway, source=source, session_key=session_key,
                                          session_entry=session_entry),
            )
            if outcome and outcome.startswith("fired"):
                logger.info(
                    "Compression warmup %s for session %s (reason=%s)",
                    outcome, getattr(session_entry, "session_id", None), reason,
                )
            elif outcome:
                logger.info(
                    "Compression warmup skipped for session %s: %s (reason=%s)",
                    getattr(session_entry, "session_id", None), outcome, reason,
                )
    except asyncio.CancelledError:
        logger.debug("Compression warmup cancelled for session %s", session_key)
        raise
    except Exception:
        logger.warning(
            "Compression warmup failed for session %s (ignored; the next turn pays the cold prefill)",
            session_key, exc_info=True,
        )


def _fire_warmup_sync(gateway: Any, *, source: Any, session_key: str, session_entry: Any) -> str:
    """Build the next turn's exact request through the turn's own machinery and replay it
    with max_tokens=1. Runs on a worker thread; returns an outcome string for the log."""
    from gateway.config import Platform
    from gateway.run import (
        _build_gateway_agent_history, _current_max_iterations, _load_gateway_config,
        _message_timestamps_enabled, _platform_config_key,
    )
    from gateway.session import build_session_context
    from gateway.turn_context import TurnContext
    from gateway.run_turn_runner import TurnRunner
    from run_agent import AIAgent

    session_id = session_entry.session_id
    user_config = _load_gateway_config()
    model, runtime_kwargs = gateway._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=user_config,
    )
    if not _is_local_runtime(runtime_kwargs):
        return "not_local"

    history = gateway.session_store.load_transcript(session_id)
    if not history:
        return "empty_history"
    if not _session_idle(gateway, session_key, source):
        return "session_busy_at_fire"

    redact_pii = False
    with suppress(Exception):
        redact_pii = bool((user_config.get("privacy") or {}).get("redact_pii", False))
    context = build_session_context(source, gateway.config, session_entry)
    context_prompt = gateway._pinned_session_context_prompt(context, redact_pii, session_key, internal=True)
    channel_prompt, pinned_source = gateway._pinned_channel_inputs(session_key, None, source, internal=True)

    platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
    enabled_toolsets, disabled_toolsets = gateway._resolve_turn_toolsets(user_config, source, platform_key)
    turn_ctx = TurnContext(
        source=pinned_source, session_key=session_key, session_id=session_id,
        user_config=user_config, history=history, context_prompt=context_prompt,
        channel_prompt=channel_prompt, enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets, AIAgent=AIAgent,
    )
    turn_runner = TurnRunner(gateway, turn_ctx)
    combined_ephemeral = turn_runner._combined_ephemeral_prompt()
    skip_context_files = turn_runner._skip_context_files(platform_key)
    reasoning_config = gateway._resolve_session_reasoning_config(
        source=source, session_key=session_key, model=model,
    )
    turn_route = gateway._resolve_turn_agent_config("", model, runtime_kwargs)
    # _build_fresh_agent reads the tier off the runner (run_sync assigns it per turn);
    # pin this session's resolution only for the build.
    prev_tier = getattr(gateway, "_service_tier", None)
    gateway._service_tier = gateway._resolve_session_service_tier(source=source, session_key=session_key)
    try:
        agent = turn_runner._build_fresh_agent(
            turn_route, platform_key, combined_ephemeral, _current_max_iterations(),
            reasoning_config, gateway._provider_routing, skip_context_files,
        )
    finally:
        gateway._service_tier = prev_tier
    try:
        agent_history, _observed = _build_gateway_agent_history(
            history, channel_prompt=channel_prompt,
            inject_timestamps=_message_timestamps_enabled(user_config),
        )
        if not agent_history:
            return "empty_replay"
        from agent.conversation_loop import _restore_or_build_system_prompt

        _restore_or_build_system_prompt(agent, None, agent_history)

        from agent.turn_request_assembly import assemble_api_request

        # build_api_messages reads the turn's frozen expiry clock; outside a turn it must
        # be stamped here (every real turn re-stamps it in _reset_per_turn_agent_state).
        agent._current_turn_timestamp = time.time()
        # There is no current turn: current_turn_user_idx marks the freshly appended user
        # message in a real turn (live injection + context-engine incoming_message), and the
        # next real turn will treat every transcript row as historical replay. One past the
        # end selects "no current turn" in both consumers (their range guards), so all rows
        # replay their api_content sidecars byte-exactly like the real next turn's prefix.
        assembled = assemble_api_request(
            agent, messages=agent_history, current_turn_user_idx=len(agent_history),
            _ext_prefetch_cache=None, _plugin_user_context=None, moa_config=None,
            active_system_prompt=agent._cached_system_prompt, original_user_message=None,
            pending_moa_prepared_request=None, request_logger=logger,
        )
        api_kwargs = agent._build_api_kwargs(assembled.api_messages, tools_for_api=assembled.tools_for_api)
        from agent.message_sanitization import sanitize_outbound_kwargs

        sanitize_outbound_kwargs(agent, api_kwargs)
        # hermes normally sends no max_tokens; the warmup must cap generation at one token
        # (the prefill is the point). Route through the provider's own param-name mapping.
        max_tokens_param = getattr(agent, "_max_tokens_param", None)
        if callable(max_tokens_param):
            api_kwargs.update(max_tokens_param(1))
        else:
            api_kwargs["max_tokens"] = 1
        api_kwargs.pop("stream", None)
        api_kwargs.pop("stream_options", None)

        if not _session_idle(gateway, session_key, source):
            return "session_busy_before_send"
        client = agent._create_request_openai_client(reason="compression-warmup", api_kwargs=api_kwargs)
        try:
            from agent.sdk_transform_bypass import bypass_chat_sdk_request_transform

            payload = bypass_chat_sdk_request_transform(api_kwargs, client)
            started = time.monotonic()
            client.chat.completions.create(**payload)
            elapsed = time.monotonic() - started
        finally:
            agent._close_request_openai_client(client, reason="compression-warmup")
        return f"fired in {elapsed:.1f}s messages={len(api_kwargs.get('messages') or [])} model={model}"
    finally:
        # Never cached: release the throwaway agent like an evicted one (soft — no sandbox,
        # browser or background processes were ever started on it).
        gateway._spawn_release_thread(
            gateway._release_evicted_agent_soft, (agent,),
            f"compression-warmup-release-{str(session_key)[:24]}", inline_fallback=True,
        )

"""Production post-run finish for gateway foreground turns.

Promoted from ``GatewayRunner._handle_message_with_agent`` success path after
``_run_agent`` returns: typing stop, stale discard, response normalize,
transcript persistence, compression-exhaust reset, voice/media delivery.

Does not own ``_run_agent`` itself or the handler except/finally cleanup.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from gateway.config import Platform


def _is_gateway_hidden_reasoning_incomplete_turn(agent_result: dict) -> bool:
    from gateway.run import _is_gateway_hidden_reasoning_incomplete_turn as _fn
    return _fn(agent_result)


def _empty_response_fallback(*args, **kwargs):
    from gateway.run import _empty_response_fallback as _fn
    return _fn(*args, **kwargs)


def _normalize_empty_agent_response(*args, **kwargs):
    from gateway.run import _normalize_empty_agent_response as _fn
    return _fn(*args, **kwargs)


def _sanitize_gateway_final_response(*args, **kwargs):
    from gateway.run import _sanitize_gateway_final_response as _fn
    return _fn(*args, **kwargs)


def _should_clear_resume_pending_after_turn(agent_result: dict) -> bool:
    from gateway.run import _should_clear_resume_pending_after_turn as _fn
    return _fn(agent_result)


def _resolve_gateway_display_bool(*args, **kwargs):
    from gateway.run import _resolve_gateway_display_bool as _fn
    return _fn(*args, **kwargs)


def _platform_config_key(platform) -> str:
    from gateway.run import _platform_config_key as _fn
    return _fn(platform)


def _load_gateway_config() -> dict:
    from gateway.run import _load_gateway_config as _fn
    return _fn()


def _resolve_gateway_model():
    from gateway.run import _resolve_gateway_model as _fn
    return _fn()


def _drain_gateway_watch_events(queue):
    from gateway.run import _drain_gateway_watch_events as _fn
    return _fn(queue)


def _format_gateway_process_notification(evt):
    from gateway.run import _format_gateway_process_notification as _fn
    return _fn(evt)


@dataclass(slots=True)
class GatewayFinishedAgentTurn:
    """Success-path outcome after ``_run_agent``.

    ``response`` is ``None`` when discarded (stale generation) or streaming
    already delivered the body. Empty string means intentional silence.
    """

    response: str | None = None
    session_entry: Any = None


async def finish_gateway_agent_turn(
    *,
    runner: Any,
    event: Any,
    source: Any,
    session_entry: Any,
    session_key: str,
    history: list,
    agent_result: dict,
    run_start_session_id: str,
    message_text: str,
    persist_user_message: Any,
    persist_user_timestamp: Any,
    quick_key: str,
    run_generation: int,
    msg_start_time: float,
    platform_name: str,
    hook_ctx: dict,
    logger: Any,
) -> GatewayFinishedAgentTurn:
    """Finish a successful agent turn (post-run orchestration).

    Side effects on ``runner`` / session store match the former inline success
    path in ``_handle_message_with_agent``.
    """
    _quick_key = quick_key
    _run_start_session_id = run_start_session_id
    _msg_start_time = msg_start_time
    _platform_name = platform_name
    history = list(history) if history is not None else []

    # Stop persistent typing indicator now that the agent is done.
    # Slack AI status is scoped to a thread/workspace, so preserve the
    # same routing metadata used by the response delivery path.
    try:
        _typing_adapter = runner._adapter_for_source(source)
        _stop_with_metadata = getattr(
            type(_typing_adapter), "_stop_typing_with_metadata", None
        )
        _stop_typing = getattr(type(_typing_adapter), "stop_typing", None)
        if _typing_adapter and callable(_stop_with_metadata):
            await _typing_adapter._stop_typing_with_metadata(
                source.chat_id,
                runner._thread_metadata_for_source(
                    source, runner._reply_anchor_for_event(event)
                ),
            )
        elif _typing_adapter and callable(_stop_typing):
            await _typing_adapter.stop_typing(source.chat_id)
    except Exception:
        pass

    if not runner._is_session_run_current(_quick_key, run_generation):
        logger.info(
            "Discarding stale agent result for %s — generation %d is no longer current",
            _quick_key or "?",
            run_generation,
        )
        _stale_adapter = runner._adapter_for_source(source)
        if getattr(type(_stale_adapter), "pop_post_delivery_callback", None) is not None:
            _stale_adapter.pop_post_delivery_callback(
                _quick_key,
                generation=run_generation,
            )
        elif _stale_adapter and hasattr(_stale_adapter, "_post_delivery_callbacks"):
            _stale_adapter._post_delivery_callbacks.pop(_quick_key, None)
        return GatewayFinishedAgentTurn(response=None, session_entry=session_entry)

    response = agent_result.get("final_response") or ""
    # Hidden-reasoning-only retry exhaustion: the loop's sentinel text
    # ("Codex response remained incomplete after 3 continuation
    # attempts") doubles as final_response, so it would be delivered
    # verbatim into the channel — where peer agents can ingest it as a
    # completed assistant turn (#51628). Blank it here so the normal
    # empty-response handling (and the suppression below) applies.
    if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
        response = ""
    try:
        from gateway.response_filters import is_intentional_silence_agent_result
        _intentional_silence = is_intentional_silence_agent_result(
            agent_result, response,
        )
    except Exception:
        _intentional_silence = False

    # Convert the agent's internal "(empty)" / "[[NO_REPLY]]" sentinels
    # into either a platform-aware visible fallback (QQ explicit address)
    # or intentional silence. Generic English empty text is a last resort.
    _raw_response_marker = str(response or "").strip()
    if (
        _raw_response_marker in {"(empty)", "[[NO_REPLY]]"}
        and not _intentional_silence
    ):
        empty_kind = (
            "no_reply" if _raw_response_marker == "[[NO_REPLY]]" else "empty"
        )
        try:
            _is_admin = bool(runner._is_admin_user(source))
        except Exception:
            _is_admin = False
        fallback = _empty_response_fallback(
            source,
            event.text or "",
            empty_kind=empty_kind,
            is_admin_user=_is_admin,
            raw_message=getattr(event, "raw_message", None),
            event=event,
        )
        if fallback:
            response = fallback
            agent_result = dict(agent_result)
            agent_result["suppress_reply"] = False
            agent_result["final_response"] = fallback
            # Rewrite any assistant message that still carries the
            # empty/NO_REPLY sentinel so transcript persistence matches
            # the user-visible reply.
            _msgs = list(agent_result.get("messages") or [])
            for _idx in range(len(_msgs) - 1, -1, -1):
                _msg = _msgs[_idx]
                if not isinstance(_msg, dict):
                    continue
                if _msg.get("role") != "assistant":
                    continue
                if str(_msg.get("content") or "").strip() in {
                    "(empty)",
                    "[[NO_REPLY]]",
                    "NO_REPLY",
                }:
                    _msgs[_idx] = {**_msg, "content": fallback}
                    break
            agent_result["messages"] = _msgs
            _intentional_silence = False
        elif empty_kind == "no_reply":
            # Keep NO_REPLY silent when no platform fallback applies.
            # Treat [[NO_REPLY]] as intentional silence even though the
            # upstream filter only matches bare NO_REPLY.
            response = ""
            _intentional_silence = True
        else:
            # Empty response with no platform-specific fallback:
            # suppress delivery rather than dumping a generic English
            # error into QQ group ambient turns.
            response = ""
            _intentional_silence = True
    agent_messages = agent_result.get("messages", [])
    _response_time = time.time() - _msg_start_time
    _api_calls = agent_result.get("api_calls", 0)
    _resp_len = len(response)
    logger.info(
        "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
        _platform_name, source.chat_id or "unknown",
        _response_time, _api_calls, _resp_len,
    )

    # NOTE: the cross-process cache-coherence re-baseline
    # (_refresh_agent_cache_message_count) is intentionally deferred
    # until AFTER this turn's transcript persistence block below — it
    # must include the first-turn `session_meta` marker row and the
    # compression session_id swap, both of which happen later.  See
    # the call site after the `update_session(...)` write.

    # Successful turn — clear any stuck-loop counter for this session.
    # This ensures the counter only accumulates across CONSECUTIVE
    # restarts where the session was active (never completed).
    #
    # Also clear the resume_pending flag (set by drain-timeout
    # shutdown) — the turn ran to completion, so recovery
    # succeeded and subsequent messages should no longer receive
    # the restart-interruption system note.
    if session_key and _should_clear_resume_pending_after_turn(agent_result):
        runner._clear_restart_failure_count(session_key)
        try:
            await runner.async_session_store.clear_resume_pending(session_key)
        except Exception as _e:
            logger.debug(
                "clear_resume_pending failed for %s: %s",
                session_key, _e,
            )

    # Normalize empty responses: surface errors, partial failures, and
    # the case where agent did work but returned no text. Fix for #18765.
    if not _intentional_silence:
        response = _normalize_empty_agent_response(
            agent_result, response, history_len=len(history),
        )
        response = _sanitize_gateway_final_response(source.platform, response)

    # Ordering contract: the agent thread already updated the contextvar
    # in conversation_compression.py; propagate to SessionEntry + _save().
    # If the agent's session_id changed during compression, update
    # session_entry so transcript writes below go to the right session.
    if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
        if session_entry.session_id == _run_start_session_id:
            session_entry.session_id = agent_result["session_id"]
            # The held turn lease follows the rotation: the transcript
            # persistence below writes to the NEW id, so the
            # serialization boundary must move with it or an alias
            # key resolving the fresh child could interleave (#64934).
            runner._rebind_turn_lease(
                _quick_key, run_generation, session_entry.session_id
            )
            await runner.async_session_store._save()
            await runner.async_session_store._record_gateway_session_peer(
                session_entry.session_id,
                session_key,
                source,
            )
            await asyncio.to_thread(
                runner._sync_telegram_topic_binding,
                source, session_entry, reason="agent-result-compression",
            )
        else:
            logger.info(
                "Skipping agent-result session split sync for %s because "
                "the session binding moved from %s to %s before "
                "compression finished",
                session_key or "?",
                _run_start_session_id,
                session_entry.session_id,
            )

    # Prepend reasoning/thinking if display is enabled (per-platform).
    # Mattermost requires explicit per-platform opt-in because this is
    # scratch text, not ordinary final-answer content.
    try:
        _show_reasoning_effective = _resolve_gateway_display_bool(
            _load_gateway_config(),
            _platform_config_key(source.platform),
            "show_reasoning",
            default=bool(getattr(runner, "_show_reasoning", False)),
            platform=source.platform,
            require_platform_override_for={Platform.MATTERMOST},
        )
    except Exception:
        _show_reasoning_effective = (
            False
            if source.platform == Platform.MATTERMOST
            else getattr(runner, "_show_reasoning", False)
        )
    if _show_reasoning_effective and response and not _intentional_silence:
        last_reasoning = agent_result.get("last_reasoning")
        if last_reasoning:
            # Collapse long reasoning to keep messages readable
            lines = last_reasoning.strip().splitlines()
            if len(lines) > 15:
                display_reasoning = "\n".join(lines[:15])
                display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
            else:
                display_reasoning = last_reasoning.strip()
            # Render style is per-platform: Discord defaults to "-# "
            # subtext (native small grey metadata text); other
            # platforms keep the fenced code block.
            try:
                from gateway.display_config import resolve_display_setting
                _reasoning_style = resolve_display_setting(
                    _load_gateway_config(),
                    _platform_config_key(source.platform),
                    "reasoning_style",
                    "code",
                )
            except Exception:
                _reasoning_style = "code"
            if _reasoning_style == "subtext":
                _quoted = "\n".join(
                    f"-# {ln}" if ln else "-#" for ln in display_reasoning.splitlines()
                )
                response = f"-# 💭 Reasoning\n{_quoted}\n\n{response}"
            elif _reasoning_style == "blockquote":
                _quoted = "\n".join(
                    f"> {ln}" if ln else ">" for ln in display_reasoning.splitlines()
                )
                response = f"> 💭 **Reasoning:**\n{_quoted}\n\n{response}"
            else:
                response = f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

    # Runtime-metadata footer — only on the FINAL message of the turn.
    # Off by default (display.runtime_footer.enabled=false).  When
    # streaming already delivered the body, we can't mutate the sent
    # text, so we fire a separate trailing send below.
    _footer_line = ""
    try:
        from gateway.runtime_footer import build_footer_line as _bfl
        _footer_line = _bfl(
            user_config=_load_gateway_config(),
            platform_key=_platform_config_key(source.platform),
            model=agent_result.get("model"),
            context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
            context_length=agent_result.get("context_length") or None,
            cwd=os.environ.get("TERMINAL_CWD", ""),
        )
    except Exception as _footer_err:
        logger.debug("runtime_footer build failed: %s", _footer_err)
        _footer_line = ""
    if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
        response = f"{response}\n\n{_footer_line}"

    # Emit agent:end hook
    await runner.hooks.emit("agent:end", {
        **hook_ctx,
        "response": (response or "")[:500],
    })

    # Check for pending process watchers (check_interval on background processes)
    try:
        from tools.process_registry import process_registry
        # Detach the current batch atomically (see crash-recovery drain
        # above): reassign to a fresh list so a watcher appended by a
        # concurrent session during the yield isn't dropped by clear().
        watchers = process_registry.pending_watchers
        process_registry.pending_watchers = []
        for i, watcher in enumerate(watchers):
            asyncio.create_task(runner._run_process_watcher(watcher))
            if i % 100 == 99:
                await asyncio.sleep(0)
    except Exception as e:
        logger.error("Process watcher setup error: %s", e)

    # Drain watch pattern notifications that arrived during the agent run.
    # Watch events and completions share the same queue; process
    # completions are already handled by the per-process watcher task
    # above, so we only inject watch-type events here.
    #
    # Async-delegation completions ALSO ride this shared queue but are
    # owned by the dedicated _async_delegation_watcher (started at
    # boot), which covers both the idle and post-turn cases with a
    # single consumer — so we leave them on the queue here.
    try:
        from tools.process_registry import process_registry as _pr
        _watch_events = _drain_gateway_watch_events(_pr.completion_queue)
        for evt in _watch_events:
            synth_text = _format_gateway_process_notification(evt)
            if synth_text:
                try:
                    await runner._inject_watch_notification(synth_text, evt)
                except Exception as e2:
                    logger.error("Watch notification injection error: %s", e2)
    except Exception as e:
        logger.debug("Watch queue drain error: %s", e)

    # NOTE: Dangerous command approvals are now handled inline by the
    # blocking gateway approval mechanism in tools/approval.py.  The agent
    # thread blocks until the user responds with /approve or /deny, so by
    # the time we reach here the approval has already been resolved.  The
    # old post-loop pop_pending + approval_hint code was removed in favour
    # of the blocking approach that mirrors CLI's synchronous input().

    # Save the full conversation to the transcript, including tool calls.
    # This preserves the complete agent loop (tool_calls, tool results,
    # intermediate reasoning) so sessions can be resumed with full context
    # and transcripts are useful for debugging and training data.
    #
    # IMPORTANT: For context-overflow failures (compression exhausted,
    # generic 400 on large sessions) we must NOT persist the user's
    # message — doing so would grow the session further and cause the
    # same failure on the next attempt, an infinite loop. (#1630, #9893)
    #
    # Transient failures (429, timeout, connection error, provider 5xx)
    # are different: the session is not oversized, and silently dropping
    # the user message causes severe context loss on retry — the agent
    # forgets what was just asked.  Persist the user turn so the
    # conversation is preserved. (#7100)
    agent_failed_early = bool(agent_result.get("failed"))
    hidden_reasoning_incomplete = _is_gateway_hidden_reasoning_incomplete_turn(
        agent_result
    )
    _err_str_for_classify = str(agent_result.get("error", "")).lower()
    # Use specific multi-word phrases (not bare "exceed" or "token")
    # to avoid false positives on transient errors like "rate limit
    # exceeded" or "invalid auth token". Matches run_agent.py's
    # own context-length classifier.
    is_context_overflow_failure = agent_failed_early and (
        bool(agent_result.get("compression_exhausted"))
        or any(p in _err_str_for_classify for p in (
            "context length", "context size", "context window",
            "maximum context", "token limit", "too many tokens",
            "reduce the length", "exceeds the limit",
            "request entity too large", "prompt is too long",
            "payload too large", "input is too long",
        ))
        or ("400" in _err_str_for_classify and len(history) > 50)
    )
    if is_context_overflow_failure:
        logger.info(
            "Skipping transcript persistence for context-overflow "
            "failure in session %s to prevent session growth loop.",
            session_entry.session_id,
        )
    elif agent_failed_early:
        logger.info(
            "Transient agent failure in session %s — persisting user "
            "message so conversation context is preserved on retry.",
            session_entry.session_id,
        )
    elif hidden_reasoning_incomplete:
        logger.warning(
            "Suppressing hidden-reasoning-only incomplete gateway turn "
            "for session %s: %s",
            session_entry.session_id,
            agent_result.get("error", "processing incomplete"),
        )

    # When compression is exhausted, the session is permanently too
    # large to process.  Auto-reset it so the next message starts
    # fresh instead of replaying the same oversized context in an
    # infinite fail loop.  (#9893)
    if agent_result.get("compression_exhausted") and session_entry and session_key:
        logger.info(
            "Auto-resetting session %s after compression exhaustion.",
            session_entry.session_id,
        )
        new_entry = await runner.async_session_store.reset_session(session_key)
        runner._evict_cached_agent(session_key)
        # Conversation boundary: one funnel call clears every
        # conversation-scoped per-session dict (#58403 and siblings).
        # See _CONVERSATION_SCOPED_STATE.
        runner._clear_conversation_scope(
            session_key, reason="compression_exhausted_reset"
        )
        if new_entry is not None:
            # Drop the stale reference to the bloated compressed child and
            # re-point the Telegram topic binding at the fresh session.
            # Compression rotated session_entry.session_id to the oversized
            # compressed child earlier this turn (the agent-result sync
            # above), and that _sync also rewrote the (chat_id, thread_id)
            # -> bloated-child binding. reset_session swaps in a clean,
            # parentless session, but without re-syncing the binding the
            # next inbound message in this topic gets switch_session'd back
            # onto the bloated child by the binding-heal walk, reloads the
            # oversized transcript, and re-triggers compression exhaustion
            # forever (#35809 — regression of the #9893/#10063 auto-reset).
            # No-op on non-topic lanes.
            session_entry = new_entry
            await asyncio.to_thread(
                runner._sync_telegram_topic_binding,
                source, session_entry, reason="compression-exhausted-reset",
            )
        response = (response or "") + (
            "\n\n🔄 Session auto-reset — the conversation exceeded the "
            "maximum context size and could not be compressed further. "
            "Your next message will start a fresh session."
        )

    ts = time.time()  # Unix epoch float — consistent with DB storage

    # If this is a fresh session (no history), write the full tool
    # definitions as the first entry so the transcript is self-describing
    # -- the same list of dicts sent as tools=[...] in the API request.
    if is_context_overflow_failure:
        pass  # Skip all transcript writes — don't grow a broken session
    elif not history:
        tool_defs = agent_result.get("tools", [])
        await runner.async_session_store.append_to_transcript(
            session_entry.session_id,
            {
                "role": "session_meta",
                "tools": tool_defs or [],
                "model": _resolve_gateway_model(),
                "platform": source.platform.value if source.platform else "",
                "timestamp": ts,
            }
        )

    # The agent already persisted these messages to SQLite via
    # _flush_messages_to_session_db(), so skip the DB write here
    # to prevent the duplicate-write bug (#860 / #42039). This holds
    # for the codex app-server runtime too: although it early-returns
    # and bypasses conversation_loop's per-step flushes, it flushes its
    # own projected assistant/tool messages before returning and
    # reports agent_persisted=True (see agent/codex_runtime.py). Reading
    # the flag (default = runner._session_db is not None) keeps the
    # persistence contract explicit and lets any future non-persisting
    # runtime opt into a gateway-side write by returning False.
    agent_persisted = agent_result.get("agent_persisted", runner._session_db is not None)

    # Find only the NEW messages from this turn (skip history we loaded).
    # Use the filtered history length (history_offset) that was actually
    # passed to the agent, not len(history) which includes session_meta
    # entries that were stripped before the agent saw them.
    if is_context_overflow_failure:
        pass  # handled above — skip all transcript writes
    elif agent_failed_early or hidden_reasoning_incomplete:
        # Transient failure (429/timeout/5xx): persist only the user
        # message so the next message can load a transcript that
        # reflects what was said.  Skip the assistant error text since
        # it's a gateway-generated hint, not model output. Hidden-
        # reasoning-only incomplete turns follow the same persistence
        # rule so peer-agent channels don't ingest them as completed
        # assistant turns. (#7100, #51628)
        _user_entry = {
            "role": "user",
            "content": (
                persist_user_message
                if persist_user_message is not None
                else message_text
            ),
            "timestamp": (
                persist_user_timestamp
                if persist_user_timestamp is not None
                else ts
            ),
        }
        if event.message_id:
            _user_entry["message_id"] = str(event.message_id)
        # Dedupe: skip if this platform message_id is already in the
        # transcript (prevents duplicate user turns on Telegram retries
        # after transient failures). #47237
        _skip_persist = (
            event.message_id
            and await runner.async_session_store.has_platform_message_id(
                session_entry.session_id, str(event.message_id)
            )
        )
        if _skip_persist:
            logger.info(
                "Skipping duplicate user turn "
                "(message_id=%s) in session %s",
                event.message_id, session_entry.session_id,
            )
        else:
            await runner.async_session_store.append_to_transcript(
                session_entry.session_id,
                _user_entry,
                skip_db=agent_persisted,
            )
    else:
        history_len = agent_result.get("history_offset", len(history))
        new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []

        # If no new messages found (edge case), fall back to simple user/assistant
        if not new_messages:
            _user_entry = {
                "role": "user",
                "content": (
                    persist_user_message
                    if persist_user_message is not None
                    else message_text
                ),
                "timestamp": (
                    persist_user_timestamp
                    if persist_user_timestamp is not None
                    else ts
                ),
            }
            if event.message_id:
                _user_entry["message_id"] = str(event.message_id)
            await runner.async_session_store.append_to_transcript(
                session_entry.session_id,
                _user_entry,
                skip_db=agent_persisted,
            )
            if response:
                await runner.async_session_store.append_to_transcript(
                    session_entry.session_id,
                    {"role": "assistant", "content": response, "timestamp": ts},
                    skip_db=agent_persisted,
                )
        else:
            # Attach the inbound platform message_id to the first user
            # entry written this turn so platform-level quote-resolution
            # (e.g. Yuanbao QuoteContextMiddleware's transcript fallback)
            # can find earlier @bot messages by their original message_id.
            _user_msg_id_attached = False
            for msg in new_messages:
                # Skip system messages (they're rebuilt each run)
                if msg.get("role") == "system":
                    continue
                # Add timestamp to each message for debugging
                entry = {**msg, "timestamp": ts}
                if (
                    not _user_msg_id_attached
                    and msg.get("role") == "user"
                    and event.message_id
                    and "message_id" not in entry
                ):
                    entry["message_id"] = str(event.message_id)
                    _user_msg_id_attached = True
                await runner.async_session_store.append_to_transcript(
                    session_entry.session_id, entry,
                    skip_db=agent_persisted,
                )

    # Token counts and model are now persisted by the agent directly.
    # Keep only last_prompt_tokens here for context-window tracking and
    # compression decisions.
    await runner.async_session_store.update_session(
        session_entry.session_key,
        last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
    )

    # Re-baseline the cached agent's message_count snapshot now that
    # ALL of this turn's transcript writes are done — the agent's
    # flushed user/assistant/tool rows AND the first-turn `session_meta`
    # marker appended above.  The cross-process coherence guard (#45966)
    # snapshots the count at agent-BUILD time (before this turn's own
    # writes) and never refreshes it on reuse, so without this the
    # process's own turn grows message_count and the next turn sees a
    # mismatch and rebuilds the agent — destroying prompt caching.
    #
    # This MUST run after the `session_meta` append: that row also
    # increments message_count, so re-baselining before it (the old
    # position) left the snapshot one short and the guard mis-fired on
    # turn 2 of EVERY fresh gateway conversation, rebuilding the cached
    # agent and busting the prompt cache.  Running here also uses the
    # compaction-updated session_id (the agent_result session_id swap
    # above), matching this function's documented contract.  Refreshing
    # here makes the guard fire only on a DIFFERENT process's writes.
    # Fail-safe inside the helper.
    await runner._refresh_agent_cache_message_count(
        session_key, session_entry.session_id
    )

    # Intentional silence is a delivery decision, not a transcript
    # mutation.  The agent's [SILENT]/NO_REPLY assistant turn above is
    # still persisted in session history so later turns keep normal
    # user/assistant alternation; only the outbound chat delivery is
    # suppressed.
    if _intentional_silence:
        logger.info(
            "Suppressing intentional silence marker for session %s",
            session_entry.session_id,
        )
        # Empty string (not None) is the stable "no delivery" signal
        # expected by adapters and gateway silence tests.
        response = ""

    # Auto voice reply: send TTS audio before the text response
    _already_sent = bool(agent_result.get("already_sent"))
    if runner._should_send_voice_reply(event, response, agent_messages, already_sent=_already_sent):
        await runner._send_voice_reply(event, response)

    # If streaming already delivered the response, extract and
    # deliver any MEDIA: files before returning None.  Streaming
    # sends raw text chunks that include MEDIA: tags — the normal
    # post-processing in _process_message_background is skipped
    # when already_sent is True, so media files would never be
    # delivered without this.
    #
    # Never skip when the agent failed — the error message is new
    # content the user hasn't seen (streaming only sent earlier
    # partial output before the failure).  Without this guard,
    # users see the agent "stop responding without explanation."
    if agent_result.get("already_sent") and not agent_result.get("failed"):
        if response:
            _media_adapter = runner._adapter_for_source(source)
            if _media_adapter:
                await runner._deliver_media_from_response(
                    response, event, _media_adapter,
                )
        # Streaming already delivered the body text, but the footer was
        # intentionally held back (see the `not already_sent` gate above).
        # Send it now as a small trailing message so Telegram/Discord/etc.
        # still surface the runtime metadata on the final reply.
        if _footer_line:
            try:
                _foot_adapter = runner._adapter_for_source(source)
                if _foot_adapter:
                    await _foot_adapter.send(
                        source.chat_id,
                        _footer_line,
                        metadata=runner._thread_metadata_for_source(source, runner._reply_anchor_for_event(event)),
                    )
            except Exception as _e:
                logger.debug("trailing footer send failed: %s", _e)
        return GatewayFinishedAgentTurn(response=None, session_entry=session_entry)

    return GatewayFinishedAgentTurn(response=response, session_entry=session_entry)

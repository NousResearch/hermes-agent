"""Gateway turn persistence phases; inherited by GatewayTurnMixin."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import dataclasses
import inspect
import json
import os
import queue
import threading
import time
from agent.i18n import t
from agent.session_activity import format_iteration_progress
from contextlib import nullcontext, suppress
from contextvars import copy_context
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import (
    SessionSource, _session_key_namespace, build_channel_continuity_note,
    build_session_context,
)
from gateway.session_transcript import TranscriptReadError
from gateway.turn_context import TurnContext
from gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from hermes_constants import get_hermes_home_override
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import base_url_hostname

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTurnPersistenceMixin:
    async def _hmwa_shape_agent_response(
        self, agent_result, source, history, session_entry, session_key,
        _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
    ):
        """Turn the raw agent result into the outbound text: sentinel/silence handling, response
        logging, resume-pending clear, empty-response normalization, and identity-guarded
        post-compression session_id propagation. Returns
        ``(response, _intentional_silence, agent_messages)``."""
        from gateway.run import (
            _is_gateway_hidden_reasoning_incomplete_turn, _normalize_empty_agent_response,
            _sanitize_gateway_final_response, _should_clear_resume_pending_after_turn,
        )
        response = agent_result.get("final_response") or ""
        # Hidden-reasoning-only retry exhaustion: the loop's sentinel text doubles as final_response
        # and would be delivered verbatim (peer agents would ingest it as a completed turn).
        if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
            response = ""
        _intentional_silence = self._is_intentional_silence(agent_result, response)

        # "(empty)" = the model produced no visible content after exhausting all retries.
        if response == "(empty)" and not _intentional_silence:
            response = (
                "⚠️ The model returned no response after processing tool results. This can happen "
                "with some models — try again or rephrase your question."
            )
        agent_messages = agent_result.get("messages", [])
        logger.info(
            "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
            _platform_name, source.chat_id or "unknown",
            time.time() - _msg_start_time, agent_result.get("api_calls", 0), len(response),
        )

        # Successful turn: clear the consecutive-restart stuck-loop counter and resume_pending (set
        # by drain-timeout shutdown) so later messages don't get the restart-interruption note.
        if session_key and _should_clear_resume_pending_after_turn(agent_result):
            await self._clear_restart_failure_count(session_key)
            try:
                await self.async_session_store.clear_resume_pending(session_key)
            except Exception as _e:
                logger.debug("clear_resume_pending failed for %s: %s", session_key, _e)

        # Normalize empty responses: surface errors, partial failures, and work-without-text.
        # Fix for #18765.
        if not _intentional_silence:
            response = _normalize_empty_agent_response(agent_result, response, history_len=len(history))
            response = _sanitize_gateway_final_response(source.platform, response)

        # The agent thread already updated the contextvar; propagate to SessionEntry + _save() only
        # if the binding still points at the session this run was launched against.
        if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
            if session_entry.session_id == _run_start_session_id:
                session_entry.session_id = agent_result["session_id"]
                # The held turn lease follows the rotation (persistence writes to the NEW id).
                self._rebind_turn_lease(_quick_key, run_generation, session_entry.session_id)
                await self.async_session_store._save()
                await self.async_session_store._record_gateway_session_peer(
                    session_entry.session_id, session_key, source,
                )
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="agent-result-compression",
                )
            else:
                logger.info(
                    "Skipping agent-result session split sync for %s because the session binding "
                    "moved from %s to %s before compression finished",
                    session_key or "?", _run_start_session_id, session_entry.session_id,
                )
        return response, _intentional_silence, agent_messages

    # reasoning_style → (header line, per-line quote prefix for blank / non-blank lines)
    _REASONING_QUOTE_STYLES = {
        "subtext": ("-# 💭 Reasoning", "-# ", "-#"), "blockquote": ("> 💭 **Reasoning:**", "> ", ">")
    }

    def _hmwa_prepend_reasoning(self, agent_result, response, source, _intentional_silence):
        """Prepend the last reasoning block when show_reasoning is on for this platform. Mattermost
        requires an explicit per-platform opt-in (scratch text, not final-answer content)."""
        from gateway.run import _load_gateway_config, _platform_config_key, _resolve_gateway_display_bool
        try:
            _show_reasoning_effective = _resolve_gateway_display_bool(
                _load_gateway_config(), _platform_config_key(source.platform), "show_reasoning",
                default=bool(getattr(self, "_show_reasoning", False)), platform=source.platform,
                require_platform_override_for={Platform.MATTERMOST},
            )
        except Exception:
            _show_reasoning_effective = (
                False if source.platform == Platform.MATTERMOST else getattr(self, "_show_reasoning", False)
            )
        last_reasoning = agent_result.get("last_reasoning")
        if not (_show_reasoning_effective and response and not _intentional_silence and last_reasoning):
            return response
        from gateway.stream_consumer_fences import escape_code_fences_for_display
        # Collapse long reasoning to keep messages readable
        lines = last_reasoning.strip().splitlines()
        if len(lines) > 15:
            display_reasoning = "\n".join(lines[:15]) + f"\n_... ({len(lines) - 15} more lines)_"
        else:
            display_reasoning = last_reasoning.strip()
        # Per-platform render style: Discord defaults to "-# " subtext, others keep the code block.
        try:
            from gateway.display_config import resolve_display_setting
            _reasoning_style = resolve_display_setting(
                _load_gateway_config(), _platform_config_key(source.platform), "reasoning_style", "code",
            )
        except Exception:
            _reasoning_style = "code"
        _quote = self._REASONING_QUOTE_STYLES.get(_reasoning_style)
        if _quote:
            header, prefix, empty = _quote
            _quoted = "\n".join(f"{prefix}{ln}" if ln else empty for ln in display_reasoning.splitlines())
            return f"{header}\n{_quoted}\n\n{response}"
        # Escape ``` inside reasoning so inner fences don't break the outer code block.
        display_reasoning = escape_code_fences_for_display(display_reasoning)
        return f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

    def _hmwa_runtime_footer_line(self, agent_result, source, _turn_seconds):
        """Runtime-metadata footer for the FINAL message of the turn; off by default
        (display.runtime_footer.enabled=false)."""
        from gateway.run import _load_gateway_config, _platform_config_key, _terminal_scope_cwd
        try:
            from gateway.runtime_footer import build_footer_line as _bfl
            return _bfl(
                user_config=_load_gateway_config(),
                platform_key=_platform_config_key(source.platform), model=agent_result.get("model"),
                context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                context_length=agent_result.get("context_length") or None,
                cwd=_terminal_scope_cwd(""), turn_seconds=_turn_seconds,
            )
        except Exception as _footer_err:
            logger.debug("runtime_footer build failed: %s", _footer_err)
            return ""

    async def _hmwa_post_turn_hooks(self, hook_ctx, agent_result, response):
        """agent:end hook, process-watcher scheduling, and watch-notification drain."""
        await self.hooks.emit("agent:end", {
            **hook_ctx, "response": (response or "")[:500], "model": agent_result.get("model", ""),
            "provider": agent_result.get("provider", ""),
        })

        # Pending process watchers (check_interval on background processes)
        try:
            from tools.process_registry import process_registry
            # Detach the batch atomically (reassign, not clear()) so concurrent appends aren't dropped.
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            for i, watcher in enumerate(watchers):
                asyncio.create_task(self._run_process_watcher(watcher))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Process watcher setup error: %s", e)

        # Drain watch notifications that arrived during the run; the queue also carries process /
        # async-delegation completions owned elsewhere — inject only watch-type events.
        try:
            from tools.process_registry import process_registry as _pr
            await self._drain_watch_notifications(_pr.completion_queue)
        except Exception as e:
            logger.debug("Watch queue drain error: %s", e)

    _FAILED_TURN_NOTICE = (
        "Your request was not processed. Send it again if you still want me to carry it out."
    )
    _PARTIAL_FAILED_TURN_NOTICE = (
        "This turn did not complete. Some actions may already have run; verify their effects "
        "before resending."
    )

    def _hmwa_add_failed_turn_notice(self, response, notice):
        """Make failed-turn delivery explicit without replacing the provider-specific guidance."""
        response = str(response or "").strip()
        return f"{response}\n\n{notice}" if response else notice

    def _hmwa_failed_turn_notice(self, agent_result):
        """Choose retry guidance without assuming completed tool effects can be repeated safely."""
        from gateway.media_repair import _current_turn_messages
        # Compression during the failed turn can move the slice boundary; the shared helper falls
        # back to the last user row so tool evidence is not silently dropped.
        turn_messages = _current_turn_messages(
            agent_result.get("messages", []) or [], agent_result.get("history_offset", 0),
        )
        if any(
            message.get("role") == "tool"
            or (message.get("role") == "assistant" and message.get("tool_calls"))
            for message in turn_messages
        ):
            return self._PARTIAL_FAILED_TURN_NOTICE
        return self._FAILED_TURN_NOTICE

    async def _hmwa_close_failed_turn(self, session_id, notice):
        """Append the gateway-owned assistant boundary iff the durable tail is an open user row.

        The tail, not "did the gateway write the user row", is the key: on the primary path the
        agent's turn-start flush already persisted the row (so the platform-id dedupe skips the
        gateway write), and a platform redelivery of an already-closed turn must not stack a
        second assistant row."""
        if await self.async_session_store.transcript_tail_role(session_id) != "user":
            return
        await self.async_session_store.append_to_transcript(session_id, {
            "role": "assistant", "content": notice, "timestamp": time.time(),
        })

    def _hmwa_classify_turn_failure(self, agent_result, history, session_entry):
        """Classify a finished turn for transcript persistence. Returns
        ``(agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure)``.

        Context-overflow failures must NOT persist the user message (session would grow and
        reproduce the failure forever); transient failures (429/timeout/5xx) DO."""
        from gateway.run import _is_gateway_hidden_reasoning_incomplete_turn
        # Save the full conversation to the transcript, including tool calls. This preserves the complete
        # agent loop (tool_calls, tool results, intermediate reasoning) so sessions can be resumed with full
        # context and transcripts are useful for debugging and training data. IMPORTANT: For
        # context-overflow failures (compression exhausted, generic 400 on large sessions) we must NOT
        # persist the user's message — doing so would grow the session further and cause the same failure on
        # the next attempt, an infinite loop. (#1630, #9893) Transient failures (429, timeout, connection
        # error, provider 5xx) are different: the session is not oversized, and silently dropping the user
        # message causes severe context loss on retry — the agent forgets what was just asked. Persist the
        # user turn so the conversation is preserved. (#7100)
        agent_failed_early = bool(agent_result.get("failed"))
        hidden_reasoning_incomplete = _is_gateway_hidden_reasoning_incomplete_turn(agent_result)
        from gateway.run_turn import is_context_overflow_failure_result
        is_context_overflow_failure = is_context_overflow_failure_result(agent_result, len(history))
        if is_context_overflow_failure:
            logger.info(
                "Skipping transcript persistence for context-overflow "
                "failure in session %s to prevent session growth loop.", session_entry.session_id,
            )
        elif agent_failed_early:
            logger.info(
                "Transient agent failure in session %s — persisting user "
                "message so conversation context is preserved on retry.", session_entry.session_id,
            )
        elif hidden_reasoning_incomplete:
            logger.warning(
                "Suppressing hidden-reasoning-only incomplete gateway turn for session %s: %s",
                session_entry.session_id, agent_result.get("error", "processing incomplete"),
            )
        return agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure

    async def _hmwa_compression_exhaustion_reset(
        self, agent_result, response, session_entry, session_key, source,
    ):
        """Auto-reset a permanently oversized session so the next message starts fresh instead of
        replaying the oversized context forever. Never on a lock-contended defer — that is the
        OPPOSITE case (a concurrent path holds the lock and is shrinking it). Returns
        ``(response, session_entry)``."""
        # When compression is exhausted, the session is permanently too large to process. (#9893) Never wipe
        # the session for that — retry-next-message semantics apply (#69870 lock-skip consumer; salvaged
        # from #49874).
        if agent_result.get("compression_deferred"):
            logger.info(
                "Compression deferred for session %s — the compression "
                "lock is held by a concurrent compressor. Keeping the "
                "session intact; the next message retries normally.",
                session_entry.session_id if session_entry else "?",
            )
        elif agent_result.get("compression_exhausted") and session_entry and session_key:
            logger.info("Auto-resetting session %s after compression exhaustion.", session_entry.session_id)
            new_entry = await self.async_session_store.reset_session(session_key)
            self._evict_cached_agent(session_key)
            # Conversation boundary: the funnel clears every conversation-scoped per-session dict.
            self._clear_conversation_scope(session_key, reason="compression_exhausted_reset")
            if new_entry is not None:
                # Re-point the Telegram topic binding at the fresh session, or the binding-heal walk
                # switches the next message back onto the bloated child and re-triggers exhaustion
                # forever. No-op on non-topic lanes.
                # Compression rotated session_entry.session_id to the oversized compressed child earlier
                # this turn (the agent-result sync above), and that _sync also rewrote the (chat_id,
                # thread_id) -> bloated-child binding. reset_session swaps in a clean, parentless session,
                # but without re-syncing the binding the next inbound message in this topic gets
                # switch_session'd back onto the bloated child by the binding-heal walk, reloads the
                # oversized transcript, and re-triggers compression exhaustion forever (#35809 — regression
                # of the #9893/#10063 auto-reset).
                session_entry = new_entry
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="compression-exhausted-reset",
                )
            response = (response or "") + (
                "\n\n🔄 Session auto-reset — the conversation exceeded the maximum context size and "
                "could not be compressed further. Your next message will start a fresh session."
            )
        return response, session_entry

    @staticmethod
    def _hmwa_user_transcript_entry(event, prepared, ts):
        """Transcript row for the inbound user turn (clean text + event time when captured)."""
        # Transient failure (429/timeout/5xx): persist only the user message so the next message can load a
        # transcript that reflects what was said. Skip the assistant error text since it's a
        # gateway-generated hint, not model output. Hidden- reasoning-only incomplete turns follow the same
        # persistence rule so peer-agent channels don't ingest them as completed assistant turns. (#7100,
        # #51628)
        _user_entry = {
            "role": "user",
            "content": (
                prepared.persist_user_message if prepared.persist_user_message is not None
                else prepared.message_text
            ),
            "timestamp": prepared.persist_user_timestamp if prepared.persist_user_timestamp is not None else ts,
        }
        if prepared.persist_user_display_kind:
            _user_entry["display_kind"] = prepared.persist_user_display_kind
        if prepared.persistence_owner:
            _user_entry["display_metadata"] = {"gateway_input_owner": prepared.persistence_owner}
        if getattr(event, "message_id", None):
            _user_entry["message_id"] = str(event.message_id)
        return _user_entry

    async def _hmwa_persist_turn_transcript(
        self, *, event, source, session_entry, session_key, agent_result, agent_messages,
        prepared, response, agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure,
    ):
        """Persist this turn to the transcript (session_meta on first turn, user-only on transient
        failure, nothing on context overflow), update last_prompt_tokens, and re-baseline the
        cached agent's message count."""
        from gateway.run import _resolve_gateway_model
        ts = time.time()  # Unix epoch float — consistent with DB storage
        store = self.async_session_store
        sid = session_entry.session_id
        history = prepared.history
        # The agent already persisted this turn's rows (codex app-server reports agent_persisted=True
        # too); skip the DB write. Default = a session DB exists; non-persisting runtimes pass False.
        # The agent already persisted these messages to SQLite via _flush_messages_to_session_db(), so skip
        # the DB write here to prevent the duplicate-write bug (#860 / #42039). This holds for the codex
        # app-server runtime too: although it early-returns and bypasses conversation_loop's per-step
        # flushes, it flushes its own projected assistant/tool messages before returning and reports
        # agent_persisted=True (see agent/codex_runtime.py). Reading the flag (default = self._session_db is
        # not None) keeps the persistence contract explicit and lets any future non-persisting runtime opt
        # into a gateway-side write by returning False.
        agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)
        _user_row = self._hmwa_user_transcript_entry(event, prepared, ts)

        if is_context_overflow_failure:
            pass  # Skip all transcript writes — don't grow a broken session
        else:
            if not history:
                # Fresh session: the tool definitions (as sent in the API request) make the transcript
                # self-describing.
                await store.append_to_transcript(sid, {
                    "role": "session_meta",
                    "tools": agent_result.get("tools", []) or [],
                    "model": _resolve_gateway_model(),
                    "platform": source.platform.value if source.platform else "",
                    "timestamp": ts,
                })
            if agent_failed_early or hidden_reasoning_incomplete:
                # Transient failure / hidden-reasoning incomplete: persist only the user message (the
                # assistant error text is a gateway hint, not model output). Dedupe on platform
                # message_id (Telegram retries after transient failures).
                if event.message_id and await store.has_platform_message_id(sid, str(event.message_id)):
                    logger.info(
                        "Skipping duplicate user turn (message_id=%s) in session %s",
                        event.message_id, sid,
                    )
                else:
                    await store.append_to_transcript(sid, _user_row, skip_db=agent_persisted)
                # Close the failed turn: a user-only tail lets alternation repair merge this request
                # into an unrelated future message and replay stale side effects (#107070).
                await self._hmwa_close_failed_turn(sid, self._hmwa_failed_turn_notice(agent_result))
            else:
                # Only the NEW messages: history_offset (what the agent saw), not len(history), which
                # counts session_meta entries stripped before the agent saw them.
                history_len = agent_result.get("history_offset", len(history))
                new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
                if not new_messages:
                    # Edge case: fall back to simple user/assistant rows.
                    await store.append_to_transcript(sid, _user_row, skip_db=agent_persisted)
                    if response:
                        await store.append_to_transcript(
                            sid, {"role": "assistant", "content": response, "timestamp": ts},
                            skip_db=agent_persisted,
                        )
                else:
                    # Attach the inbound platform message_id to the first user entry so platform-level
                    # quote-resolution (e.g. Yuanbao) can find earlier @bot messages by original id.
                    _user_msg_id_attached = False
                    for msg in new_messages:
                        if msg.get("role") == "system":
                            continue  # rebuilt each run
                        entry = {**msg, "timestamp": ts}
                        if (
                            not _user_msg_id_attached
                            and msg.get("role") == "user"
                            and event.message_id
                            and "message_id" not in entry
                        ):
                            entry["message_id"] = str(event.message_id)
                            _user_msg_id_attached = True
                        await store.append_to_transcript(sid, entry, skip_db=agent_persisted)

        # The agent persists token counts/model itself; keep only last_prompt_tokens for hygiene.
        await store.update_session(
            session_entry.session_key, last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
            touch_activity=not bool(getattr(event, "internal", False)),
        )

        # Re-baseline the cached agent's message_count now that ALL of this turn's writes are done:
        # the coherence guard snapshots at agent-BUILD time, so our own writes would otherwise
        # trigger a rebuild next turn (destroying prompt caching).
        await self._refresh_agent_cache_message_count(session_key, sid)

    async def _hmwa_deliver_turn_response(
        self, event, source, session_entry, session_key, run_generation,
        agent_result, agent_messages, response, _footer_line, _intentional_silence,
    ):
        """Final delivery decisions: intentional silence, voice reply, streamed-turn media/footer.
        Returns the text for the adapter to send, or ``None`` when already delivered."""
        # Intentional silence is a delivery decision: the [SILENT] turn stays persisted (alternation).
        if _intentional_silence:
            logger.info("Suppressing intentional silence marker for session %s", session_entry.session_id)
            response = ""

        adapter = self._adapter_for_source(source)
        # Auto voice reply (TTS audio before the text) unless streaming TTS already delivered audio.
        _streaming_tts_done = adapter is not None and bool(
            getattr(adapter, "_streaming_tts_turn_completed", lambda *_a, **_k: False)(session_key, run_generation)
        )
        if not _streaming_tts_done and self._should_send_voice_reply(
            event, response, agent_messages, already_sent=bool(agent_result.get("already_sent")),
        ):
            await self._send_voice_reply(event, response)

        # Streamed responses still need MEDIA: files delivered (chunks carry the tags verbatim). Never
        # skip when the agent failed: the error text is new content streaming didn't show.
        if agent_result.get("already_sent") and not agent_result.get("failed"):
            if response and adapter:
                await self._deliver_media_from_response(response, event, adapter)
            # Streaming delivered the body, but the footer was held back (`not already_sent` gate).
            if _footer_line and adapter:
                try:
                    await adapter.send(source.chat_id, _footer_line, metadata=self._event_thread_metadata(event, source))
                except Exception as _e:
                    logger.debug("trailing footer send failed: %s", _e)
            # Return None so the body isn't sent twice; stash the delivered text on the event for the
            # /loop and /goal hooks that read the return value.
            with suppress(Exception):
                event._streamed_final_response = str(response or "")
            return None

        return response

    _STATUS_HINTS = {
        401: " Check your API key or run `claude /login` to refresh OAuth credentials.",
        402: " Your API balance or quota is exhausted. Check your provider dashboard.",
        529: " The API is temporarily overloaded. Please try again shortly.",
    }

    async def _hmwa_agent_error_reply(self, e, event, source, session_entry, session_key, prepared):
        """``except Exception`` body of the agent turn: stop typing, log, persist the inbound user
        turn once, and build the sanitized user-facing error reply."""
        # Retain Slack thread/workspace routing so a failed turn cannot leave its status visible.
        await self._hmwa_stop_typing_for_turn(event, source)
        logger.exception("Agent error in session %s", session_key)
        status_code = getattr(e, "status_code", None)
        if status_code in {400, 500} and len(prepared.history) > 50:
            # Context overflow / payload too large: a deterministic rejection (#107567), and the same
            # no-grow rule as the persist path (#1630) — nothing is written into an oversized session.
            return (
                "⚠️ Session too large for the model's context window.\nUse /compact to "
                "compress the conversation, or /reset to start fresh."
            )
        # Replay can coalesce inputs; only this input's durable marker establishes ownership.
        try:
            if prepared.message_text is not None and session_entry is not None:
                _owned = await self.async_session_store.has_input_owner(
                    prepared.persistence_session_id, prepared.persistence_owner,
                )
                if not _owned:
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id, self._hmwa_user_transcript_entry(event, prepared, time.time()),
                    )
                # Tool effects are unknown after an exception.
                await self._hmwa_close_failed_turn(session_entry.session_id, self._PARTIAL_FAILED_TURN_NOTICE)
        except Exception:
            logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
        # Never expose raw exception types/messages to end users (info-leakage risk).
        status_hint = self._STATUS_HINTS.get(status_code, "")
        if status_code == 429:
            # Plan usage limit (resets on a schedule) vs a transient rate limit
            _err_json = {}
            with suppress(Exception):
                _err_json = e.response.json().get("error", {})
            if not isinstance(_err_json, dict):
                _err_json = {}
            _resets_in = _err_json.get("resets_in_seconds")
            if _err_json.get("type") != "usage_limit_reached":
                status_hint = " You are being rate-limited. Please wait a moment and try again."
            elif _resets_in and _resets_in > 0:
                import math
                status_hint = f" Your plan's usage limit has been reached. It resets in ~{math.ceil(_resets_in / 3600)}h."
            else:
                status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
        elif status_code == 400:
            status_hint = " The request was rejected by the API."
        return self._hmwa_add_failed_turn_notice(
            f"Sorry, I encountered an unexpected error.{status_hint}\n"
            "Try again or use /reset to start a fresh session.",
            self._PARTIAL_FAILED_TURN_NOTICE,
        )

    def _hmwa_discard_stale_result(self, source, _quick_key, run_generation):
        """A newer run generation superseded this turn: drop its deferred post-delivery callback."""
        logger.info(
            "Discarding stale agent result for %s — generation %d is no longer current",
            _quick_key or "?", run_generation,
        )
        self._pop_post_delivery_callback(self._adapter_for_source(source), _quick_key, run_generation)

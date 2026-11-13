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
from hermes_agent.agent.i18n import t
from hermes_agent.agent.turn_failure_copy import (
    FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE, OPEN_TURN_TAIL_ROLES, PARTIAL_FAILED_TURN_NOTICE,
)
from hermes_agent.agent.session_activity import format_iteration_progress
from contextlib import nullcontext, suppress
from contextvars import copy_context
from hermes_agent.gateway.bot_peer_notices import automatic_notices_allowed
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.media_repair import repair_explicit_computer_use_media_paths
from hermes_agent.gateway.platforms.base import BasePlatformAdapter, _mark_notify_metadata
from hermes_agent.gateway.platforms.event import MessageEvent
from hermes_agent.gateway.session import (
    SessionSource, _session_key_namespace, build_channel_continuity_note,
    build_session_context,
)
from hermes_agent.gateway.session_detached import detached_route_of
from hermes_agent.gateway.session_transcript import TranscriptReadError
from hermes_agent.gateway.turn_context import TurnContext
from hermes_agent.gateway.warning_notifications import diagnostic_wake_muted
from hermes_agent.gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from hermes_agent.hermes_constants import get_hermes_home_override
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from hermes_agent.utils import base_url_hostname

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.run_turn_runner import TurnRunner

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_agent.gateway.run")


def _acp_launch(runner, source) -> bool:
    """An editor session: ACP shows no reasoning event yet, so its reply keeps the projected block."""
    from hermes_agent.gateway.session_setting_commands import local_policy
    from hermes_agent.hermes_state_runtime import RuntimeStoreError
    try:
        return getattr(local_policy(runner, source), "source", None) == "acp"
    except RuntimeStoreError:  # a local route with no restored policy is not an editor launch
        return False


class GatewayTurnPersistenceMixin:
    @staticmethod
    def _is_intentional_silence(agent_result, response) -> bool:
        try:
            from hermes_agent.gateway.response_filters import is_intentional_silence_agent_result
            return is_intentional_silence_agent_result(agent_result, response)
        except Exception:
            return False

    async def _record_room_silence_decision(
        self, session_key, agent_result, silent, *, display_kind=None, reply_expected=None,
    ):
        """Bind recovery permission to this exact persisted final and active-turn token.

        A raw persisted marker is not proof that a partial/interrupted turn completed. A crash
        before this receipt is durable keeps the human safeguard; transcript bytes stay intact.
        """
        from hermes_agent.gateway.response_filters import is_intentional_silence_response, silence_allowed
        kind = agent_result.get("queued_terminal_display_kind", display_kind)
        expected = agent_result.get("queued_terminal_reply_expected", reply_expected)
        if (not silent or not session_key or silence_allowed(kind, expected)
                or not is_intentional_silence_response(agent_result.get("final_response"))):
            return
        store = self.async_session_store
        try:
            entry = await store.lookup_by_session_key(session_key)
            token = entry.active_turn_token if entry is not None else None
            if not isinstance(token, str):
                return
            history = await store.load_transcript(entry.session_id)
            # The agent may have saved its final before gateway session metadata was appended.
            # Match recovery's visible tail rather than the literal last bookkeeping row.
            last = next((m for m in reversed(history) if m.get("role") not in ("session_meta", "system")), {})
            if (last.get("role") != "assistant" or last.get("timestamp") is None
                    or last.get("content") != agent_result.get("final_response")):
                return
            await store.set_session_metadata(session_key, "room_silence_decision", {
                "token": token, "timestamp": last.get("timestamp"), "silent": bool(silent),
            })
        except Exception:
            # The receipt only lets crash recovery honour this silence; without it recovery keeps
            # the human-turn notice. Live delivery must stay silent rather than become an error.
            logger.warning("Could not record the room silence decision for %s", session_key, exc_info=True)

    async def _hmwa_shape_agent_response(
        self, agent_result, source, history, session_entry, session_key,
        _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
        persist_user_display_kind: Optional[str] = None,
        reply_expected: Optional[bool] = None,
    ):
        """Turn the raw agent result into the outbound text: sentinel/silence handling, response
        logging, resume-pending clear, empty-response normalization, and identity-guarded
        post-compression session_id propagation. Returns
        ``(response, _intentional_silence, agent_messages)``."""
        from hermes_agent.gateway.run import (
            _is_gateway_hidden_reasoning_incomplete_turn, _normalize_empty_agent_response,
            _sanitize_gateway_final_response, _should_clear_resume_pending_after_turn,
        )
        response = agent_result.get("final_response") or ""
        # Hidden-reasoning-only retry exhaustion: the loop's sentinel text doubles as final_response
        # and would be delivered verbatim (peer agents would ingest it as a completed turn).
        if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
            response = ""
        _intentional_silence = self._is_intentional_silence(agent_result, response)
        # A queued (/queue) chain's TERMINAL turn owns the silence verdict, not the event that
        # opened the chain: an internal follow-up, or a message not addressed to the bot, may go
        # silent; any other human one must not.
        from hermes_agent.gateway.response_filters import is_machinery_display_kind, strip_trailing_silence_marker
        from hermes_agent.gateway.run_turn import _unexpected_silence_reply
        _silence_kind = agent_result.get("queued_terminal_display_kind", persist_user_display_kind)
        _silence_reply_expected = agent_result.get("queued_terminal_reply_expected", reply_expected)
        _silence_source = agent_result.get("queued_terminal_source", source)
        # A bot peer expects no reply even when it addressed us (a DM, a mention): its message may end
        # the exchange in silence, and the visible fallback would be runtime text it answers (#137027).
        if not automatic_notices_allowed(_silence_source):
            _silence_reply_expected = False
        if _intentional_silence and not self._silence_allowed_for_source(
            _silence_source, _silence_kind, _silence_reply_expected, agent_result,
        ):
            logger.warning(
                "silence marker rejected on a user turn: platform=%s chat=%s",
                _platform_name, source.chat_id or "unknown",
            )
            _intentional_silence = False
            response = _unexpected_silence_reply()
        elif _intentional_silence and not is_machinery_display_kind(_silence_kind):
            logger.debug(
                "silence marker suppressed on an unaddressed or opted-in turn: platform=%s chat=%s",
                _platform_name, source.chat_id or "unknown",
            )

        # "(empty)" = the model produced no visible content after exhausting all retries. One
        # text with the CLI explainer and the desktop (hermes_agent/agent/turn_explainers.py) so the user
        # reads the same words on every surface.
        if not automatic_notices_allowed(source) and (response == "(empty)" or agent_result.get("failed")):
            # A failed or empty turn's text is runtime failure copy, never the model's answer: a bot
            # peer would take it as a message to answer (#137027).
            response = ""
        if response == "(empty)" and not _intentional_silence:
            from hermes_agent.agent.turn_explainers import EMPTY_RESPONSE_EXPLANATION

            _model = str(agent_result.get("model") or "").strip() or t("gateway.errors.empty_response_model_label")
            response = t("gateway.shared.warn_passthrough", error=EMPTY_RESPONSE_EXPLANATION.format(model=_model))
        agent_messages = agent_result.get("messages", [])
        logger.info(
            "response ready: platform=%s chat=%s session=%s time=%.1fs api_calls=%d response=%d chars",
            _platform_name, source.chat_id or "unknown", session_key or "unknown",
            time.time() - _msg_start_time, agent_result.get("api_calls", 0), len(response),
        )

        # Successful turn: clear the consecutive-restart stuck-loop counter and resume_pending (set
        # by drain-timeout shutdown) so later messages don't get the restart-interruption note.
        if session_key and _should_clear_resume_pending_after_turn(agent_result):
            await self._clear_restart_failure_count(session_key)
            try:
                await self.async_session_store.clear_resume_pending(session_key)
            except Exception as _e:
                logger.debug("clear_resume_pending failed for %s: %s", session_key, _e, exc_info=True)

        # Normalize empty responses: surface errors, partial failures, and work-without-text.
        # Fix for #18765.
        if not _intentional_silence:
            response = strip_trailing_silence_marker(response)
            # A bot peer gets the model's own words only: the runtime's retry hints and "not
            # processed" lines are new inbound messages for its turn loop (#137027).
            if automatic_notices_allowed(source):
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

    # reasoning_style → (header catalog key, per-line quote prefix for blank / non-blank lines)
    _REASONING_QUOTE_STYLES = {
        "subtext": ("gateway.reasoning.quote_label_discord", "-# ", "-#"),
        "blockquote": ("gateway.reasoning.quote_label_md", "> ", ">"),
    }

    def _hmwa_prepend_reasoning(self, agent_result, response, source, _intentional_silence):
        """Prepend the last reasoning block when show_reasoning is on for this platform. Mattermost
        requires an explicit per-platform opt-in (scratch text, not final-answer content)."""
        block = self._hmwa_reasoning_block(agent_result, response, source, _intentional_silence)
        return f"{block}\n\n{response}" if block else response

    def _hmwa_reasoning_block(self, agent_result, response, source, _intentional_silence) -> str:
        """The rendered reasoning block for this turn's reply, or "" when none is shown."""
        if source.platform == Platform.LOCAL and not _acp_launch(self, source):
            # Local viewers receive reasoning.delta through their event stream; projecting it into
            # the final response duplicates it and leaks messaging-platform formatting into Desktop.
            return ""
        from hermes_agent.gateway.run import _load_gateway_config, _platform_config_key, _resolve_gateway_display_bool
        try:
            _show_reasoning_effective = _resolve_gateway_display_bool(
                _load_gateway_config(), _platform_config_key(source.platform), "show_reasoning",
                default=bool(getattr(self, "_show_reasoning", False)), platform=source.platform,
                require_platform_override_for={Platform.MATTERMOST},
            )
        except Exception:
            logger.debug("show_reasoning resolution failed; using the gateway default", exc_info=True)
            _show_reasoning_effective = (
                False if source.platform == Platform.MATTERMOST else getattr(self, "_show_reasoning", False)
            )
        last_reasoning = agent_result.get("last_reasoning")
        if not (_show_reasoning_effective and response and not _intentional_silence and last_reasoning):
            return ""
        from hermes_agent.gateway.stream_consumer_fences import escape_code_fences_for_display
        # Collapse long reasoning to keep messages readable
        lines = last_reasoning.strip().splitlines()
        if len(lines) > 15:
            display_reasoning = "\n".join(lines[:15]) + t("gateway.reasoning.more_lines", count=len(lines) - 15)
        else:
            display_reasoning = last_reasoning.strip()
        # Per-platform render style: Discord defaults to "-# " subtext, others keep the code block.
        try:
            from hermes_agent.gateway.display_config import resolve_display_setting
            _reasoning_style = resolve_display_setting(
                _load_gateway_config(), _platform_config_key(source.platform), "reasoning_style", "code",
            )
        except Exception:
            logger.debug("reasoning_style resolution failed; using code", exc_info=True)
            _reasoning_style = "code"
        _quote = self._REASONING_QUOTE_STYLES.get(_reasoning_style)
        if _quote:
            header_key, prefix, empty = _quote
            _quoted = "\n".join(f"{prefix}{ln}" if ln else empty for ln in display_reasoning.splitlines())
            return f"{t(header_key)}\n{_quoted}"
        # Escape ``` inside reasoning so inner fences don't break the outer code block.
        display_reasoning = escape_code_fences_for_display(display_reasoning)
        return t("gateway.reasoning.block", reasoning=display_reasoning, response="").rstrip()

    def _hmwa_runtime_footer_line(self, agent_result, source, _turn_seconds):
        """Runtime-metadata footer for the FINAL message of the turn; off by default
        (display.runtime_footer.enabled=false). Never on a finite (`-z`, `chat -q`) turn: its reply is
        the printed answer itself."""
        from hermes_agent.gateway.run import _load_gateway_config, _platform_config_key, _terminal_scope_cwd
        from hermes_agent.gateway.session_finite import finite_turn_required
        # A bot peer gets the model's words only: a streamed reply sends the footer as its own message.
        if finite_turn_required() is True or not automatic_notices_allowed(source):
            return ""
        try:
            from hermes_agent.gateway.runtime_footer import build_footer_line as _bfl
            return _bfl(
                user_config=_load_gateway_config(),
                platform_key=_platform_config_key(source.platform), model=agent_result.get("model"),
                context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                context_length=agent_result.get("context_length") or None,
                cwd=_terminal_scope_cwd(""), turn_seconds=_turn_seconds,
                requested_model=agent_result.get("requested_model"),
                served_model=agent_result.get("served_model"),
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
            from hermes_agent.tools.process_registry import process_registry
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
            from hermes_agent.tools.process_registry import process_registry as _pr
            await self._drain_watch_notifications(_pr.completion_queue)
        except Exception as e:
            logger.debug("Watch queue drain error: %s", e)

    # One owner for the boundary copy (hermes_agent/agent/turn_failure_copy.py): the core closer in
    # hermes_agent/agent/conversation_loop.py writes the same row on the paths that never reach this layer.

    def _hmwa_add_failed_turn_notice(self, response, notice):
        """Make failed-turn delivery explicit without replacing the provider-specific guidance."""
        response = str(response or "").strip()
        return f"{response}\n\n{notice}" if response else notice

    def _hmwa_failed_turn_notice(self, agent_result):
        """Choose retry guidance without assuming completed tool effects can be repeated safely."""
        from hermes_agent.gateway.media_repair import _current_turn_messages
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
            return PARTIAL_FAILED_TURN_NOTICE
        return FAILED_TURN_NOTICE

    async def _hmwa_close_failed_turn(self, session_id, notice):
        """Append the gateway-owned assistant boundary iff the durable tail is open (a user row, or the
        tool row of a call that ran before the failure).

        The tail, not "did the gateway write the user row", is the key: on the primary path the
        agent's turn-start flush already persisted the row (so the platform-id dedupe skips the
        gateway write), and a platform redelivery of an already-closed turn must not stack a
        second assistant row."""
        if await self.async_session_store.transcript_tail_role(session_id) not in OPEN_TURN_TAIL_ROLES:
            return
        await self.async_session_store.append_to_transcript(session_id, {
            "role": "assistant", "content": notice, "timestamp": time.time(), "display_kind": FAILED_TURN_DISPLAY_KIND,
        })

    def _hmwa_classify_turn_failure(self, agent_result, history, session_entry):
        """Classify a finished turn for transcript persistence. Returns
        ``(agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure)``.

        Context-overflow failures must NOT persist the user message (session would grow and
        reproduce the failure forever); transient failures (429/timeout/5xx) DO."""
        from hermes_agent.gateway.run import _is_gateway_hidden_reasoning_incomplete_turn
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
        from hermes_agent.gateway.run_turn import is_context_overflow_failure_result
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
        self, agent_result, response, session_entry, session_key, source, event=None,
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
        elif agent_result.get("compression_exhausted") and (
                getattr(source, "platform", None) == Platform.API_SERVER or detached_route_of(source)):
            # The caller addresses an API conversation by its id and every API door re-validates that
            # binding (``restore_api_session``): a rotated entry would refuse each later request to it
            # (``admission_conflict``). A conversation its chat moved past has no route of its own to
            # reset (hermes_agent/gateway/session_detached.py): its app viewer gets the start-fresh guidance a
            # failed overflow turn gives. Never a reset that did not happen.
            logger.info("Keeping session %s after compression exhaustion.",
                        session_entry.session_id if session_entry else "?")
            from hermes_agent.gateway.run import _context_overflow_reply
            guidance = _context_overflow_reply() if detached_route_of(source) else ""
            if guidance not in (response or ""):
                response = f"{response}\n\n{guidance}" if response else guidance
        elif agent_result.get("compression_exhausted") and session_entry and session_key:
            if not self._hmwa_defer_exhaustion_reset(session_entry, session_key, source):
                session_entry = await self._hmwa_rotate_exhausted_session(session_entry, session_key, source)
                self._hmwa_replay_exhausted_request(event, agent_result, session_entry, session_key, source)
            response = (response or "") + t("gateway.session.auto_reset_context_exhausted")
        return response, session_entry

    def _hmwa_replay_exhausted_request(self, event, agent_result, session_entry, session_key, source):
        """Re-run the request whose turn exhausted the context on the fresh session, once (#137565).

        The reset drops the old transcript, and with it the user's request: nothing answered it. It
        goes back in the head slot, ahead of any follow-up, so the adapter's post-handler drain runs
        it on the fresh session after this turn's lease and active slot are released. Never twice (a
        replay that exhausts again resets without another replay), never an internal event, and
        never when the turn already acted (a tool ran) or showed the user output: a re-run would
        repeat it. A voice note would re-run STT and its transcript echo, so it keeps the resend notice.
        Nor when follow-ups already wait on the fresh session (re-homed by the reset): the re-run would
        be admitted behind them and answer out of order, so the user, who has moved on, resends."""
        if (event is None or event.internal
                or agent_result.get("already_sent") or agent_result.get("response_previewed")
                or self._pending_event_audio_paths(event)):
            return
        from hermes_agent.gateway.session_ingress import executing_admission
        if executing_admission.get() and not getattr(event, "_adapter_drains_after", False):
            return  # an authority-run row no adapter task awaits: a parked retry would never run
        from hermes_agent.gateway.media_repair import _current_turn_messages
        messages = agent_result.get("messages") or []
        offset = agent_result.get("history_offset", 0) or 0
        new_messages = _current_turn_messages(messages, offset)
        if offset > len(messages) and not new_messages:
            return  # compaction kept no user message to bound the turn: what ran is unknown, resend
        if (any(m.get("role") in ("tool", "function") or m.get("tool_calls") for m in new_messages)
                or self._hmwa_fresh_session_has_pending_input(session_entry)):
            return
        adapter = self._delivery_adapter_for(source)
        persistent = self._session_state(session_key).persistent
        identity = f"{event.message_id or ''}\0{event.text or ''}"  # the drain re-runs a copy of the event
        if (adapter is None or not isinstance(getattr(adapter, "_pending_messages", None), dict)
                or persistent.exhaustion_replayed == identity):
            return
        persistent.exhaustion_replayed = identity
        self._requeue_event_at_front(session_key, event, adapter)
        logger.info("Re-running the request of session %s's exhausted turn on its fresh session", session_key)

    def _hmwa_fresh_session_has_pending_input(self, session_entry) -> bool:
        """Whether the authority FIFO of the fresh session already holds an admitted input."""
        from hermes_agent.gateway.session_authorities import active_authority
        from hermes_agent.hermes_state_runtime import list_session_admissions
        authority = active_authority(self)
        if authority is None or session_entry is None:
            return False
        owner = authority.logical_owner(session_entry.session_id)
        return bool(list_session_admissions(authority.db, session_id=owner))

    def _hmwa_defer_exhaustion_reset(self, session_entry, session_key, source):
        """True when the reset was handed to the authority drain. A LOCAL session (the Bot Chat a
        ``hermes peer dm`` lands in) resets its target in the store, which refuses while an admission
        on it is started (``session_busy``): this turn's own. The drain runs the reset right after it
        settles this admission, before it claims any successor (the Bot Chat's retry included)."""
        from hermes_agent.gateway.session_ingress import executing_admission
        if getattr(source, "platform", None) != Platform.LOCAL or not executing_admission.get():
            return False
        from hermes_agent.gateway.session_authorities import active_authority
        authority = active_authority(self)
        if authority is None:
            return False
        authority.deferred_resets[session_key] = lambda: self._hmwa_rotate_exhausted_session(
            session_entry, session_key, source)
        return True

    async def _hmwa_rotate_exhausted_session(self, session_entry, session_key, source):
        """The compression-exhaustion auto-reset itself; returns the route's (fresh) entry."""
        logger.info("Auto-resetting session %s after compression exhaustion.", session_entry.session_id)
        from hermes_agent.gateway.session_automation import move_route_wakes, reset_and_rehome_messages
        # Messages queued behind the exhausted turn still run, on the fresh session (unlike /new).
        new_entry, release_fresh = await reset_and_rehome_messages(self, session_key, session_entry.session_id)
        try:
            await move_route_wakes(self, session_key, session_entry.session_id)
            self._evict_cached_agent(session_key)
            # The /queue overflow behind the deferred head slot is queued user input, like the
            # re-homed native rows: it survives the boundary the funnel below otherwise clears.
            queued = list(self._overflow_queue(session_key) or ())
            # Conversation boundary: the funnel clears every conversation-scoped per-session dict.
            self._clear_conversation_scope(session_key, reason="compression_exhausted_reset")
            if queued:
                self._session_state(session_key).conversation.queued_events.extend(queued)
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
        finally:
            # Only now may the fresh session's FIFO claim (it is fenced until here, against every
            # scheduler): after eviction, scope clear and binding sync.
            release_fresh()
        return session_entry

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
        from hermes_agent.gateway.response_filters import persisted_user_metadata_for_event
        display_metadata = {**({"gateway_input_owner": prepared.persistence_owner} if prepared.persistence_owner else {}),
                            **persisted_user_metadata_for_event(event)}
        if display_metadata:
            _user_entry["display_metadata"] = display_metadata
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
        from hermes_agent.gateway.run import _resolve_gateway_model
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
        # agent_persisted=True (see hermes_agent/agent/codex_runtime.py). The turn runner now always sets the flag from
        # the handle the turn's agent actually holds; the fallback below (runner's CURRENT handle) only
        # serves direct callers — reading it here let a mid-turn recovery mask a no-handle turn whose
        # agent flush silently no-ops, dropping the turn from state.db entirely (#140823).
        agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)
        _user_row = self._hmwa_user_transcript_entry(event, prepared, ts)

        if is_context_overflow_failure:
            pass  # Skip all transcript writes — don't grow a broken session
        elif agent_result.get("failure_reason") == "content_policy_blocked":
            # A safety refusal is deterministic for the unchanged conversation (the filter judges the
            # whole thread), so a persisted refused turn re-trips the filter on every later message and
            # locks the chat — the "stuck session" report. The agent flushed the user row at turn start;
            # drop that turn now that the lease is released (soft-delete, the same rewind /undo uses) so
            # the next turn rebuilds from a clean transcript. The refusal was already delivered to the
            # user. Only when the agent persisted the turn (``agent_persisted``) is there a durable row to
            # drop; a non-persisting runtime wrote nothing, so rewinding would hit a prior accepted turn.
            # Mirrors the context_overflow no-grow rule (#1630); only a non-retryable block drops.
            if agent_persisted:
                await self._hmwa_retain_content_policy_refused_turn(
                    sid, session_key, agent_result.get("refused_turn_boundary_row_id"))
        else:
            if not history and source.platform != Platform.LOCAL:
                # Fresh session: the tool definitions (as sent in the API request) make the transcript
                # self-describing. A LOCAL route (one-shot, chat -q, ACP, TUI attach) keeps the classic
                # in-process CLI's transcript shape: the marker is gateway bookkeeping every local
                # reader filters out, and the session row already carries model/model_config.
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

    async def _hmwa_retain_content_policy_refused_turn(self, sid, session_key, boundary_row_id=None):
        """Retain a safety-refused turn for display while forcing the next turn to rebuild from the
        clean (model) transcript. The ONE owner-side refusal cleanup for every durable gateway owner:
        the in-process runner (this persist path), the managed worker (Bot Chat / safe-mode, via
        ``execute_managed``) and the standalone tui_gateway finish path all funnel here so none can be
        missed (#149708 review).

        The agent flushed the turn-start user row before the refusal was known (a managed worker's
        subprocess flushes to the SAME shared transcript, then exits); now that the turn lease is
        released, take that whole turn out of the MODEL context owner-side — the same soft-delete
        ``/undo`` uses — but KEEP the user's message visible in the chat as a greyed refused notice,
        and evict any cached agent so the next turn rebuilds from the active-only transcript.
        Otherwise the refused turn stays in the next build's in-memory history and the deterministic
        filter re-refuses every later message (the "stuck session" report). The refusal was already
        delivered to the user. ``boundary_row_id`` (the refused run's original durable boundary,
        captured by the refusal handler) is the precise rewind target; without it the newest user
        turn is guessed. ``session_key`` is ``None`` for the managed worker (its subprocess holds no
        cached owner agent to evict).

        ``retain_refused_turn_for_display`` reports failure by returning ``None`` (it swallows its own
        errors), so only evict on a confirmed retention and warn otherwise — a silently-failed
        un-stick leaves the chat locked, and that is the one outcome worth a visible signal.
        """
        from hermes_agent.agent.turn_failure_copy import CONTENT_REFUSED_NOTICE
        try:
            outcome = await self.async_session_store.retain_refused_turn_for_display(
                sid, notice=CONTENT_REFUSED_NOTICE, boundary_row_id=boundary_row_id)
        except Exception as e:  # health: allow BLE001 -- best-effort; a refusal must never crash turn teardown
            logger.warning("content-policy refusal: retaining the refused turn for %s raised: %s", sid, e)
            return
        if outcome is None:
            logger.warning(
                "content-policy refusal: could not retain the refused turn for %s (rewind found none); "
                "the chat may stay locked until the user runs /undo", sid)
            return
        if session_key is not None:
            with suppress(Exception):
                self._evict_cached_agent(session_key)

    @staticmethod
    def _usage_window_chat_wording(agent_result: dict, response: str) -> str:
        """A usage-window refusal sent to a chat: credits steps point at the Portal (a chat's /topup has no
        Keep going control) and the reset reads as a duration (the gateway's clock is not the reader's).
        Every other reply is returned unchanged."""
        if not (agent_result.get("usage_window_message") and response):
            return response
        from hermes_agent.agent.plan_limits import chat_refusal_wording
        return chat_refusal_wording(response, agent_result.get("failure_resets_at"))

    async def _deliver_turn_response_then_limit_offer(
        self, event, source, session_entry, session_key, run_generation,
        agent_result, agent_messages, response, footer_line, intentional_silence,
    ):
        """``_hmwa_deliver_turn_response``, then, for a usage-window refusal, the limit offer: the limit
        sentence is the reply, and the button follows it once it lands."""
        delivered = await self._hmwa_deliver_turn_response(
            event, source, session_entry, session_key, run_generation,
            agent_result, agent_messages, response, footer_line, intentional_silence,
        )
        if (delivered or agent_result.get("already_sent")) and not intentional_silence:
            self._note_bot_reply(source)
        if isinstance(agent_result.get("usage_limit"), dict):
            self._schedule_usage_limit_offer(event, agent_result["usage_limit"], session_key=session_key,
                                             generation=run_generation)
        return delivered

    async def _ledger_uncertain_streamed_final(self, event, session_key, response, adapter) -> bool:
        """A stream send of this answer timed out after the request left and was counted as shown: on
        screen or lost, the wire cannot tell. Sending it again now would be a silent second copy; the
        adapter ledgers it as a failed row instead, so the timed redelivery sends it once more WITH the
        may-be-a-duplicate marker and a lost send is not silence. That is what ``send_final_ledgered``
        gives a timed-out final on a non-streaming platform. Both the completion path and the queued
        first-response lane (whose result the follow-up replaces) ledger through here; an adapter
        without the base contract has no ledger and keeps the old behaviour. True only when a
        redelivery row was saved (``gateway.delivery_ledger: false`` saves none)."""
        if not callable(getattr(adapter, "ledger_attempted_final", None)):
            return False
        return await adapter.ledger_attempted_final(
            event, session_key, response, error="Timed out: the streamed send may not have landed") is True

    async def _hmwa_stop_typing_for_turn(self, event, source):
        """Stop the typing indicator (never raises), with the delivery path's thread/workspace routing
        metadata (Slack AI status is thread-scoped), on this turn's own message, not the chat's latest."""
        from hermes_agent.gateway.platforms.helpers import typing_metadata_for_turn
        with suppress(Exception):
            _typing_adapter = self._delivery_adapter_for(source)
            _kind = type(_typing_adapter)
            if _typing_adapter and callable(getattr(_kind, "_stop_typing_with_metadata", None)):
                await _typing_adapter._stop_typing_with_metadata(source.chat_id, typing_metadata_for_turn(_typing_adapter,
                    source.chat_id, self._reply_anchor_for_event(event), self._event_thread_metadata(event, source)))
            elif _typing_adapter and callable(getattr(_kind, "stop_typing", None)):
                await _typing_adapter.stop_typing(source.chat_id)

    def _hmwa_pause_typing_after_silence(self, event, source) -> None:
        """A turn that ends in intentional silence has nothing left to show: pause the adapter's typing
        keep-alive now. It refreshes until the handler returns, so hooks and transcript persistence
        after the agent finished kept "typing" on for a reply that never comes (#149948). Only inside
        the adapter's turn bracket (``_turn_marker_handoff``), whose ``finally`` lifts the pause; a turn
        run outside it has no keep-alive, and a pause there would mute the chat's next turn. Keyed on
        this turn's typing thread, as the adapter's loop computes it: another topic of the same chat
        is still working and keeps its own "typing"."""
        from hermes_agent.gateway.platforms.helpers import typing_metadata_for_turn, typing_pause_key
        adapter = self._delivery_adapter_for(source)
        paused = getattr(adapter, "_typing_paused", None)
        if getattr(event, "_turn_marker_handoff", False) is True and isinstance(paused, set):
            metadata = typing_metadata_for_turn(adapter, source.chat_id, self._reply_anchor_for_event(event),
                                                self._event_thread_metadata(event, event.source))
            paused.add(typing_pause_key(source.chat_id, metadata))

    async def _hmwa_deliver_turn_response(
        self, event, source, session_entry, session_key, run_generation,
        agent_result, agent_messages, response, _footer_line, _intentional_silence,
    ):
        """Final delivery decisions: intentional silence, voice reply, streamed-turn media/footer.
        Returns the text for the adapter to send, or ``None`` when already delivered."""
        if diagnostic_wake_muted(event):
            return None
        # Intentional silence is a delivery decision: the [SILENT] turn stays persisted (alternation).
        if _intentional_silence:
            logger.info("Suppressing intentional silence marker for session %s", session_entry.session_id)
            response = ""

        adapter = self._delivery_adapter_for(source)
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
            # The queued-follow-up lane uploads this response's attachments itself; re-scanning here
            # would upload every file a second time.
            _footer_captioned = False  # the footer rode the last photo's caption (#74547)
            if response and adapter and not agent_result.get("media_already_delivered"):
                # This media completes the user-visible reply: mark it notify-worthy like the
                # non-streamed attachments (#144897) — in `important` mode it must still ring.
                _footer_captioned = await self._deliver_media_from_response(
                    response, event, adapter, footer_line="" if _intentional_silence else _footer_line,
                    thread_metadata=_mark_notify_metadata(self._event_thread_metadata(event, source)))
            ledgered = bool(response and adapter and agent_result.get("delivery_uncertain")) and (
                await self._ledger_uncertain_streamed_final(event, session_key, response, adapter))
            # The stream showed only the answer: the reasoning block prepended to ``response`` for
            # show_reasoning never reached the chat, so it follows as its own message (#7251). A
            # ledgered final already carries it: its redelivery sends reasoning and answer together.
            reasoning_block = "" if ledgered else self._hmwa_reasoning_block(
                agent_result, response, source, _intentional_silence)
            # A bot peer would take a second message as fresh input: gated like the footer line.
            if reasoning_block and adapter and automatic_notices_allowed(source):
                try:
                    await adapter.send(source.chat_id, reasoning_block, metadata=self._event_thread_metadata(event, source))
                except Exception:
                    logger.debug("trailing reasoning send failed", exc_info=True)
            # Streaming delivered the body, but the footer was held back (`not already_sent` gate).
            if _footer_line and adapter and not _intentional_silence and not _footer_captioned:
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

    # Chat-side next steps keyed by HTTP status; Hermes commands only (/login is the gateway's own
    # sign-in, `{relogin}` the profile-aware host equivalent, filled from the turn's agent provider).
    # Values are catalog keys (``gateway.errors.hint_*``); 401 carries a ``{relogin}`` placeholder.
    _STATUS_HINTS = {
        401: "gateway.errors.hint_auth",
        402: "gateway.errors.hint_quota",
        529: "gateway.errors.hint_overloaded",
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
            from hermes_agent.gateway.run import _context_overflow_reply
            from hermes_agent.gateway.session_results import record_unexecuted_failure
            return record_unexecuted_failure(_context_overflow_reply())
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
                await self._hmwa_close_failed_turn(session_entry.session_id, PARTIAL_FAILED_TURN_NOTICE)
        except Exception:
            logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
        # Never expose raw exception types/messages to end users (info-leakage risk).
        _hint_key = self._STATUS_HINTS.get(status_code)
        status_hint = t(_hint_key) if _hint_key and status_code != 401 else ""
        if status_code == 401:
            from hermes_agent.agent.turn_failure_copy import relogin_command_hint

            _turn_agent = getattr(self._session_state(session_key).turn, "agent", None)
            status_hint = t(_hint_key, relogin=relogin_command_hint(getattr(_turn_agent, "provider", None)))
        elif status_code == 429:
            # Plan usage limit (resets on a schedule) vs a transient rate limit
            _err_json = {}
            with suppress(Exception):
                _err_json = e.response.json().get("error", {})
            if not isinstance(_err_json, dict):
                _err_json = {}
            _resets_in = _err_json.get("resets_in_seconds")
            if _err_json.get("type") != "usage_limit_reached":
                status_hint = t("gateway.errors.hint_rate_limited")
            elif _resets_in and _resets_in > 0:
                import math
                status_hint = t("gateway.errors.hint_usage_limit_resets", hours=math.ceil(_resets_in / 3600))
            else:
                status_hint = t("gateway.errors.hint_usage_limit")
        elif status_code == 400:
            status_hint = t("gateway.errors.hint_rejected")
        from hermes_agent.gateway.session_results import record_unexecuted_failure
        if not automatic_notices_allowed(source):
            # The failure is recorded for the operator; the bot peer gets no apology to answer (#137027).
            record_unexecuted_failure(t("gateway.errors.generic_failed"))
            return None
        if getattr(e, "reason", None) == "launch_credentials_unavailable":
            # The session's own route key is gone or changed: /retry repeats it, /new is the way out.
            from hermes_agent.agent.turn_failure_copy import launch_credentials_copy
            return record_unexecuted_failure(launch_credentials_copy(e), 'launch_credentials_unavailable')
        # A local client (CLI, TUI, Desktop) is the operator at this host: its result carries the
        # real cause, so its details line is not the chat apology with the log pointer cut off.
        local_error = f"{type(e).__name__}: {e}"[:500] if source.platform == Platform.LOCAL else None
        return record_unexecuted_failure(self._hmwa_add_failed_turn_notice(
            t("gateway.errors.generic_failed_with_hint", hint=status_hint), PARTIAL_FAILED_TURN_NOTICE,
        ), error=local_error)

    def _hmwa_discard_stale_result(self, source, _quick_key, run_generation):
        """A newer run generation superseded this turn: drop its deferred post-delivery callback."""
        logger.info(
            "Discarding stale agent result for %s — generation %d is no longer current",
            _quick_key or "?", run_generation,
        )
        self._pop_post_delivery_callback(self._delivery_adapter_for(source), _quick_key, run_generation)

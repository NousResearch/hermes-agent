"""Per-turn callback runner (progress/status/voice/run_sync) for the gateway agent turn.

``TurnRunner`` owns the per-turn callbacks ``GatewayRunner._run_agent_inner`` binds. ``gateway.run``
internals are imported lazily inside method bodies (import cycle), so ``patch("gateway.run.X")``
keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import queue
import re
import threading
import time
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.interrupt_compat import _accepts_keyword
from agent.replay_cleanup import strip_stale_dangerous_confirmations
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.turn_context import TurnContext
from hermes_cli.config import cfg_get
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class _ExecApprovalDeclined(RuntimeError):
    """The connector refused the approval card's destination.

    Raised (not returned) so it propagates out of `_approval_notify_sync` to
    `_await_gateway_decision`, whose notify-failure path drops the central
    approval queue entry and unblocks the waiting tool. A plain return
    suppressed the text fallback but left that entry pending.
    """


from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.session_execution import GatewaySessionAgentMixin


class TurnRunner(GatewayTurnProgressMixin, GatewaySessionAgentMixin):
    """Per-turn collaborator carrying ``GatewayRunner._run_agent_inner``'s tool-progress callbacks."""

    def __init__(self, runner: "GatewayRunner", ctx: TurnContext) -> None:
        self._runner = runner
        self._ctx = ctx
        from gateway.session_authorities import active_authority
        authority = active_authority(runner)
        self._approval_owner = None
        if authority is not None:
            source = getattr(ctx, 'source', None)
            owner_id = (source.chat_id if getattr(source, 'platform', None) == Platform.LOCAL
                        else authority.logical_owner(ctx.session_id))
            if owner_id in authority.sessions:
                generation = authority.db.get_session(owner_id)["runtime_generation"]
                self._approval_owner = (authority, owner_id, generation)


    def _publish_execution(self, event_type, payload):
        if self._approval_owner is not None:
            authority, session_id, generation = self._approval_owner
            return authority.publish_execution(session_id, generation, event_type, payload)
        return False

    def _publish_api_tool(self, event_type, call_id, tool_name, args, result=None):
        """The retained tool payload reaches the API observers of this exact admission only;
        the shared viewer stream keeps its ID-correlated frames."""
        if self._approval_owner is not None:
            from gateway.session_api_turn import publish_api_tool_event
            authority, session_id, generation = self._approval_owner
            publish_api_tool_event(authority, session_id, generation, event_type,
                                   str(call_id or ""), str(tool_name or "tool"), args, result)

    # ── stream consumer / interim commentary wiring ─────────────────────────────────────────

    def _setup_stream_consumer(self, platform_key):
        ctx = self._ctx
        stream_consumer = None
        # The streaming-TTS consumer is created on the outer loop thread before run_sync launches;
        # run_sync only reads it via the holder for delta-callback wiring.
        stts = ctx.streaming_tts_consumer_holder[0]
        scfg = getattr(getattr(self._runner, 'config', None), 'streaming', None)
        if scfg is None:
            from gateway.config import StreamingConfig
            scfg = StreamingConfig()
        # display.platforms.<plat>.streaming may disable streaming per platform; None = follow global.
        plat_streaming = ctx.resolve_display_setting(ctx.user_config, platform_key, "streaming")
        want_stream_deltas = (
            scfg.enabled and scfg.transport != "off" if plat_streaming is None else bool(plat_streaming)
        )
        want_interim_messages = ctx.interim_assistant_messages_enabled
        if want_stream_deltas or want_interim_messages:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                adapter = self._runner._adapter_for_source(ctx.source)
                if adapter:
                    consumer_cfg, pause_typing_before_finalize = self._runner._build_stream_consumer_config(
                        ctx.source, scfg, adapter, on_missing_cursor="raise",
                    )
                    stream_consumer = GatewayStreamConsumer(
                        adapter=adapter, chat_id=ctx.source.chat_id, config=consumer_cfg,
                        metadata=ctx._status_thread_metadata,
                        on_new_message=(
                            (lambda: ctx.progress_queue.put(("__reset__",))) if ctx.progress_queue is not None else None
                        ),
                        on_before_finalize=pause_typing_before_finalize,
                        initial_reply_to_id=ctx.event_message_id, run_still_current=ctx._run_still_current,
                    )
                    ctx.stream_consumer_holder[0] = stream_consumer
            except Exception as err:
                logger.debug("Could not set up stream consumer: %s", err)
        # Deltas tee to the stream consumer (when text streaming is on) and to streaming TTS.
        delta_sinks = [sc for sc in ((stream_consumer if want_stream_deltas else None), stts) if sc is not None]
        stream_delta_cb = None
        if delta_sinks or self._approval_owner is not None:
            def stream_delta_cb(text: Optional[str]) -> None:
                if ctx._run_still_current():
                    if text is not None:  # None closes only the native stream segment.
                        self._publish_execution("message.delta", {"text": text})
                    for sink in delta_sinks:
                        sink.on_delta(text)

        def interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            if stts is not None:
                # Flush accepted deltas; completed commentary is a separate speech segment.
                stts.on_delta(None)
                if not already_streamed:
                    stts.on_delta(text)
                    stts.on_delta(None)
            if stream_consumer is not None:
                stream_consumer.on_segment_break() if already_streamed else stream_consumer.on_commentary(text)
            elif not already_streamed and ctx._status_adapter and str(text or "").strip():
                self._send_status_text(text, ctx._status_thread_metadata, "interim_assistant_callback scheduling error")

        return stream_consumer, stream_delta_cb, interim_assistant_cb, want_interim_messages


    # ── per-turn agent wiring ───────────────────────────────────────────────────────────────

    def _notice_callback_sync(self, notice) -> None:
        """Credits / out-of-band notices (usage bands, depletion, restored) fire from the agent's
        sync worker thread; hop onto the gateway loop. Fired-once latch lives on the cached agent."""
        from gateway.run import render_notice_line
        if not self._status_live():
            return
        try:
            line = render_notice_line(notice)
        except Exception:
            logger.debug("render_notice_line failed", exc_info=True)
            return
        if line:
            self._schedule(self._runner._deliver_platform_notice(self._ctx.source, line), "notice_callback delivery scheduling error")

    def _make_bg_review_callbacks(self):
        """(send, release): background-review messages ("💾 Memory updated") are held until the
        adapter's post-delivery hook releases them after the main response lands."""
        from gateway.run import _interim_metadata, _non_conversational_metadata
        ctx = self._ctx
        release_evt = threading.Event()
        pending: list[str] = []
        pending_lock = threading.Lock()

        def deliver(message: str) -> None:
            if self._status_live():
                self._send_status_text(
                    message,
                    _interim_metadata(_non_conversational_metadata(ctx._status_thread_metadata, platform=ctx.source.platform)),
                    "background_review_callback scheduling error",
                )

        def release() -> None:
            release_evt.set()
            with pending_lock:
                queued = list(pending)
                pending.clear()
            for message in queued:
                deliver(message)

        def send(message: str) -> None:
            if not self._status_live():
                return
            if not release_evt.is_set():
                with pending_lock:
                    if not release_evt.is_set():
                        pending.append(message)
                        return
            deliver(message)

        return send, release

    @staticmethod
    def _merge_turn_request_overrides(agent, turn_route) -> None:
        """Merge, never overwrite: init-time request overrides (e.g. a custom provider's extra_body)
        must survive every reused-agent turn. Drop only the PREVIOUS turn's routing overrides before
        layering this turn's, so stale per-turn values never linger."""
        overrides = dict(getattr(agent, "request_overrides", {}) or {})
        for key, value in (getattr(agent, "_gateway_turn_request_overrides", {}) or {}).items():
            if overrides.get(key) == value:
                overrides.pop(key, None)
        turn_overrides = dict(turn_route.get("request_overrides") or {})
        overrides.update(turn_overrides)
        agent.request_overrides = overrides
        agent._gateway_turn_request_overrides = turn_overrides

    def _wire_turn_agent_callbacks(self, agent, turn_route, reasoning_config,
                                   stream_delta_cb, interim_assistant_cb, want_interim_messages):
        """Per-message state — callbacks and reasoning config change every turn, so they aren't
        baked into the cached agent."""
        ctx = self._ctx
        runner = self._runner
        # ALWAYS attached (never gated to None): its body gates each event class, and subagent-
        # failure notices must fire even with tool_progress/thinking off.
        agent.tool_progress_callback = ctx.progress_callback
        # Discord's one-time voice ack and Slack's task cards both ride the authoritative start
        # callback, so neither infers identity from tool names.
        agent.tool_start_callback = (
            (ctx.native_tool_start_callback or ctx.voice_ack_callback)
            if (ctx._voice_ack_guild[0] is not None or ctx._native_slack_task_cards) else None
        )
        agent.tool_complete_callback = ctx.native_tool_complete_callback if ctx._native_slack_task_cards else None
        if self._approval_owner is not None:
            agent.tool_start_callback = self.combined_tool_start_callback
            agent.tool_complete_callback = self.combined_tool_complete_callback
        agent.step_callback = ctx._step_callback_sync if (ctx._hooks_ref.loaded_hooks or self._approval_owner is not None) else None
        agent.stream_delta_callback = stream_delta_cb
        agent.interim_assistant_callback = interim_assistant_cb if want_interim_messages else None
        agent.status_callback, agent.notice_callback = ctx._status_callback_sync, self._notice_callback_sync
        agent.notice_clear_callback = None  # sends can't be retracted
        agent.event_callback = ctx._event_callback_sync
        agent.reasoning_config, agent.service_tier = reasoning_config, runner._service_tier
        self._merge_turn_request_overrides(agent, turn_route)
        # Must-deliver notes for THIS turn ride the current user message (api_content sidecar), never
        # the system prompt. Assigned unconditionally so a reused agent never replays a stale note.
        from gateway.session_surface import surface_turn_note
        agent._gateway_turn_context_notes = "\n\n".join(
            note for note in (*runner._consume_pending_turn_sidecar_notes(ctx.session_key), surface_turn_note(agent)) if note)
        agent.background_review_callback, bg_release = self._make_bg_review_callbacks()
        # Register the release hook on the adapter so base.py's finally block fires it after the
        # main response is delivered.
        if ctx._status_adapter and ctx.session_key:
            if getattr(type(ctx._status_adapter), "register_post_delivery_callback", None) is not None:
                ctx._status_adapter.register_post_delivery_callback(ctx.session_key, bg_release, generation=ctx.run_generation)
            else:
                pdc = getattr(ctx._status_adapter, "_post_delivery_callbacks", None)
                if pdc is not None:
                    pdc[ctx.session_key] = bg_release
        # display.memory_notifications: off | on (generic "💾 Memory updated", default) | verbose.
        # `display:` present-but-null yields None, not the {} default (same `or {}` guard as
        # display_config.py / runtime_footer.py).
        mem_notif = (ctx.user_config.get("display") or {}).get("memory_notifications")
        if isinstance(mem_notif, bool):
            mem_notif = "on" if mem_notif else "off"
        agent.memory_notifications = str(mem_notif).lower() if mem_notif else "on"
        agent.clarify_callback = self._clarify_callback_sync
        # Thinking between tool calls is independent of tool_progress mode (Mattermost opts in
        # per platform so global scratch-text doesn't leak into threads).
        agent.thinking_progress = ctx._thinking_enabled
        ctx.agent_holder[0] = agent  # interrupt support
        if self._approval_owner is not None:
            authority, owner_id, generation = self._approval_owner
            authority.adopt_agent(owner_id, generation, agent)
        # The titler fires from the turn prologue, so attach the rename lane before the run.
        self._attach_session_title_callback(agent, ctx)
        # Publish turn ownership for /stop, /new, disconnect and shutdown interrupts; older session
        # processes are outside this baseline and remain alive.
        agent._gateway_turn_process_task_id, agent._gateway_turn_process_baseline = ctx.process_task_id, ctx.process_baseline
        ctx.tools_holder[0] = getattr(agent, "tools", None)  # transcript logging

    # ── blocking prompts from the agent thread (approval / clarify) ─────────────────────────

    def _close_native_stream_boundary(self, reason: str, placeholder: str | None = None, reopen: bool = False) -> bool:
        """Native-streaming platforms (e.g. WeCom): an interrupting interaction (approval or clarify
        prompt) must finalize the current stream first, or post-interaction output keeps updating the
        OLD bubble above the prompt. Runs on the agent thread; the consumer serializes via its queue."""
        sc = self._stream_consumer()
        if not (sc and getattr(sc, "_use_native_streaming", False)):
            return True
        cancelled_flag = None
        try:
            boundary = sc.close_for_approval_prompt(placeholder, reason=reason, reopen=reopen)
            # Returns (future, cancelled_flag) or just a future.
            if isinstance(boundary, tuple):
                boundary, cancelled_flag = boundary
            if not hasattr(boundary, "result"):
                return True
            ok = boundary.result(timeout=10)
            if not ok:
                logger.warning(
                    "%s boundary failed to close stream properly — "
                    "prompt may still appear in typing bubble", reason,
                )
            return bool(ok)
        except (TimeoutError, Exception) as err:
            if cancelled_flag is not None:
                cancelled_flag["cancelled"] = True
            logger.warning("%s boundary timed out or failed: %s", reason, err)
            return False

    async def _send_shared_clarify(self, entry, **kwargs):
        result = await self._ctx._status_adapter.send_clarify(**kwargs)
        if self._approval_owner is not None and result.success:
            authority, session_id, generation = self._approval_owner
            authority.register_clarify(session_id, generation, entry)
        return result

    def _clarify_callback_sync(self, question: str, choices, multi_select: bool = False) -> str:
        """Present a clarify prompt and block on a response (clarify_tool's synchronous contract):
        schedule send_clarify on the gateway loop, block on the primitive's threading.Event with a
        timeout. Returns the response string, or a sentinel when none arrived."""
        from gateway.run import _clarify_send_then_wait
        from tools import clarify_gateway as clarify_mod
        import uuid
        ctx = self._ctx
        if not ctx._status_adapter:
            return ""
        session_key = ctx.session_key or ""
        clarify_id = uuid.uuid4().hex[:10]
        choices = list(choices) if choices else None
        entry = clarify_mod.register(
            clarify_id=clarify_id, session_key=session_key, question=question, choices=choices,
            multi_select=bool(multi_select),
        )
        # Unlike approval, clarify passes reopen=True so the continuation re-opens a native stream
        # below the question; if the re-seed fails the consumer degrades to send() automatically.
        self._close_native_stream_boundary("Clarify", "💬 等待你的选择...", reopen=True)
        # Pause typing: a "thinking..." status must not obscure the prompt or block an "Other" reply
        # on platforms that disable input while typing (Slack Assistant).
        with suppress(Exception):
            ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)
        # Ordering barrier: flush buffered assistant prose BEFORE the poll, which goes out on a
        # separate agent-thread-blocking path and would otherwise render ABOVE its own explanation.
        # Best-effort + short timeout so the agent thread never hangs if the consumer isn't running.
        flush = getattr(self._stream_consumer(), "flush_pending_sync", None)
        try:
            if callable(flush):
                flush(timeout=3.0)
        except Exception:
            logger.debug("Stream-consumer flush before clarify prompt failed", exc_info=True)
        fut = self._schedule(
            self._send_shared_clarify(entry,
                chat_id=ctx._status_chat_id, question=question, choices=choices, clarify_id=clarify_id,
                session_key=session_key, metadata=ctx._status_thread_metadata,
            ),
            "Clarify send failed to schedule",
        )
        # Boundary rule (see _approval_send_outcome): a send timeout is AMBIGUOUS — the card may
        # have posted with a late ack. Only a definitive failure tears down the registration;
        # ambiguous falls through to the bounded wait so a late reply resolves.
        response = _clarify_send_then_wait(fut, clarify_id=clarify_id, session_key=session_key, clarify_mod=clarify_mod)
        if self._approval_owner is not None:
            authority, session_id, generation = self._approval_owner
            authority.sessions[session_id].controls.snapshot(session_id, generation)
        # Only re-arm typing when the user actually answered — the undeliverable sentinel and the
        # timeout/cancellation strings start with '[' and must pass through untouched.
        if not (isinstance(response, str) and response.startswith("[")):
            # Reopen typing IMMEDIATELY, not on the LLM's first post-answer token (native streaming
            # otherwise re-seeds lazily on the first delta: ~48s of dead air). request_reopen_seed is
            # a no-op outside the reopen-pending native state.
            sc = self._stream_consumer()
            if sc is not None:
                try:
                    sc.request_reopen_seed()
                except Exception:
                    logger.debug("request_reopen_seed after clarify answer failed", exc_info=True)
            try:
                ctx._status_adapter.resume_typing_for_chat(ctx._status_chat_id)
            except Exception:
                logger.debug("resume_typing_for_chat after clarify answer failed", exc_info=True)
        return response

    def _approval_notify_sync(self, approval_data: dict) -> None:
        if self._approval_owner is not None:
            authority, session_id, generation = self._approval_owner
            authority.check_approval_generation(session_id, generation)
        # A native decline must finish before any observer can authorize work.
        self._render_approval_sync(approval_data)
        if self._approval_owner is not None:
            authority.register_approval(session_id, generation, self._ctx.session_key, approval_data)

    def _render_approval_sync(self, approval_data: dict) -> None:
        """Send the approval request from the agent thread: the adapter's interactive button
        approvals (``send_exec_approval``) when available, else plain text with ``/approve`` steps."""
        from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command
        ctx = self._ctx
        adapter = ctx._status_adapter
        # Slack's assistant_threads_setStatus disables the compose box, so the user can't type
        # /approve while "is thinking..." shows. Pausing stops _keep_typing re-setting it; resumed
        # in approve/deny.
        adapter.pause_typing_for_chat(ctx._status_chat_id)
        self._close_native_stream_boundary("Approval")
        # Redact credentials before display: Tirith's findings are already redacted, but the raw
        # command string still leaks secrets. Both the button and plain-text paths use this value.
        cmd = _redact_approval_command(approval_data.get("command", ""))
        desc = approval_data.get("description", "dangerous command")
        flags = {k: approval_data.get(k, d) for k, d in (("allow_permanent", True), ("allow_session", True), ("smart_denied", False))}
        # Check the *class*, not the instance — MagicMock auto-creates attributes in tests.
        if getattr(type(adapter), "send_exec_approval", None) is not None:
            try:
                fut = self._schedule(
                    adapter.send_exec_approval(
                        chat_id=ctx._status_chat_id, command=cmd, session_key=ctx.session_key or "",
                        description=desc, metadata=ctx._status_thread_metadata, **flags,
                    ),
                    "send_exec_approval scheduling error",
                )
                if fut is None:
                    raise RuntimeError("send_exec_approval: loop unavailable")
                outcome = _approval_send_outcome(fut, timeout=15)
                if outcome == "sent":
                    return
                if outcome == "ambiguous":
                    # Timeout ≠ failure: the card may have posted with a late ack. The prompt
                    # registration stays alive so a tap still resolves; re-sending made duplicate
                    # cards + orphaned "/approve: nothing pending".
                    logger.warning(
                        "Button-based approval send timed out — treating "
                        "as possibly-delivered (no re-send; the prompt "
                        "stays armed for a late tap)"
                    )
                    return
                if outcome == "declined":
                    # P5(b): the connector AUTHORIZED this destination and
                    # refused it. The text fallback below re-sends the same
                    # content to the same chat, which would turn a refused
                    # button card into a delivered plain-text one — the exact
                    # leak the egress guard exists to stop. A decline is
                    # definitive, so unlike `ambiguous` the registration is
                    # torn down; unlike `failed`, nothing is re-sent.
                    logger.warning(
                        "Button-based approval DECLINED by the connector's "
                        "egress guard — not falling back to text (the "
                        "destination is not approved for this connection)"
                    )
                    # RAISE, do not return. This function is the notify_cb for
                    # `_await_gateway_decision`, which already has a correct
                    # undeliverable path: a raising notify drops the queue entry
                    # and returns `notify_failed`, unblocking the tool. Returning
                    # quietly suppressed the text fallback (right) but left the
                    # CENTRAL approval entry pending (wrong) — the dangerous
                    # command then blocked until the approval timeout. My earlier
                    # comment claimed the registration was torn down; only the
                    # adapter's private prompt map was.
                    raise _ExecApprovalDeclined(
                        "exec approval undeliverable: connector egress declined "
                        "this destination"
                    )
                logger.warning("Button-based approval failed (send returned error), falling back to text")
            except _ExecApprovalDeclined:
                # Must escape this handler: the fallback below is a text send to
                # the destination the connector just refused.
                raise
            except Exception as e:
                logger.warning("Button-based approval failed, falling back to text: %s", e)
        # Plain-text prompt with the adapter's typed prefix (e.g. `!approve`): typed "/" is blocked
        # in Slack threads and reserved by Matrix clients.
        msg = _format_exec_approval_fallback(cmd, desc, getattr(adapter, "typed_command_prefix", "/"), **flags)
        try:
            # Mark as approval prompt so WeCom routes through the control lane.
            metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
            fut = self._schedule(
                adapter.send(ctx._status_chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
            )
            if fut is not None:
                fut.result(timeout=15)
        except Exception as e:
            logger.error("Failed to send approval request: %s", e)

    # ── run_sync phases ─────────────────────────────────────────────────────────────────────

    def _load_turn_history(self, agent, reused_cached_agent):
        from gateway.run import (
            _build_gateway_agent_history, _collect_history_media_paths, _message_timestamps_enabled,
            _select_cached_agent_history,
        )
        ctx = self._ctx
        from gateway.session_api_turn import api_execution
        api = api_execution.get()
        if api is not None and api['history'] is not None:
            from gateway.run import _collect_history_media_paths
            history = api['history']
            return history, None, _collect_history_media_paths(history)
        # Transcript rows ({role, content, timestamp}) lose timestamps; interrupt-path agent messages
        # (tool_calls/tool_call_id/reasoning) pass through intact so the API sees valid assistant→tool
        # sequences. Telegram observed=True rows are withheld from replayable history and attached to
        # the current addressed message as API-only context.
        agent_history, observed_group_context = _build_gateway_agent_history(
            ctx.history, channel_prompt=ctx.channel_prompt, inject_timestamps=_message_timestamps_enabled(ctx.user_config),
        )
        # FTS write-corruption guard: if persistence failed silently the reloaded transcript is stale
        # while the SAME cached agent still holds the live conversation (same-session amnesia). Only
        # for a reused agent bound to this exact session_id.
        # Replacing the live transcript with that shorter copy causes immediate same-session amnesia. See
        # #50502.
        if reused_cached_agent and getattr(agent, "session_id", None) == ctx.session_id:
            selected = _select_cached_agent_history(agent_history, getattr(agent, "_session_messages", None))
            if selected is not agent_history:
                logger.warning(
                    "Persisted transcript lagged live cached history for "
                    "session %s (disk=%d, memory=%d); preserving live "
                    "conversation context (possible FTS write corruption)",
                    ctx.session_key, len(agent_history), len(selected),
                )
                # The live history bypassed _build_gateway_agent_history's cleanup — re-apply the
                # stale-confirmation expiry so a dangerous confirmation can't slip through.
                agent_history = strip_stale_dangerous_confirmations(selected, now=time.time())
        # MEDIA paths already in history are excluded from this turn's extraction (compression-safe).
        return agent_history, observed_group_context, _collect_history_media_paths(agent_history)

    def _prepend_pending_note(self, attr: str) -> None:
        """Consume a one-shot per-session note (model switch, /reload-skills) into the NEXT user
        message. Nothing hits the transcript out-of-band, so alternation stays intact."""
        ctx = self._ctx
        notes = getattr(self._runner, attr, None)
        note = notes.pop(ctx.session_key, None) if notes and ctx.session_key and ctx.session_key in notes else None
        if note:
            ctx.message = note + "\n\n" + ctx.message

    def _resume_note_interactive(self) -> bool:
        """Interactive platforms report the restore and ask what next; event platforms (webhook,
        API server) continue the work — nobody is present to answer."""
        return bool(getattr(self._runner._adapter_for_source(self._ctx.source), "interactive_resume", True))

    def _prepare_turn_message(self, agent_history):
        """Prepend recovery/notice guidance to ``ctx.message``.

        Returns (persist_user_message_override, persist_user_timestamp_override): real user text is
        kept separate from API-only recovery guidance so stale guidance never replays as user text.
        """
        from gateway.run import (
            _auto_continue_freshness_window, _is_fresh_gateway_interruption,
            _last_transcript_timestamp, _prepare_resume_pending_message, build_resume_recovery_note,
        )
        ctx = self._ctx
        persist_override: Optional[Any] = ctx.persist_user_message
        self._prepend_pending_note("_pending_model_notes")
        # Auto-continue: history ending with a tool result means the previous turn was cut off
        # (restart, crash, SIGTERM). Session-level resume_pending (drain-timeout shutdown) uses
        # stronger reason-aware wording that subsumes this case. Both gate on the age of
        # ``history[-1]`` (not agent_history, which stripped tool-row timestamps); no stamp = fresh.
        window = _auto_continue_freshness_window()
        interruption_is_fresh = _is_fresh_gateway_interruption(_last_transcript_timestamp(ctx.history), window_secs=window)
        entry = None
        if ctx.session_key:
            with suppress(Exception):
                entry = self._runner.session_store._entries.get(ctx.session_key)
        resume_pending = entry is not None and getattr(entry, "resume_pending", False)
        resume_reason = (getattr(entry, "resume_reason", None) or "restart_timeout") if resume_pending else None
        # resume_pending freshness ALSO uses the restart watchdog's ``last_resume_marked_at`` (the true
        # interruption stamp): the transcript clock can be hours older for an active thread, and the
        # startup auto-resume turn has empty text, so gating on it alone yields a blank user message.
        mark_is_fresh = resume_pending and _is_fresh_gateway_interruption(
            getattr(entry, "last_resume_marked_at", None), window_secs=window,
        )
        if resume_pending and (interruption_is_fresh or mark_is_fresh):
            # Empty message = the startup auto-resume turn; there is no NEW user message.
            ctx.message, persist_override = _prepare_resume_pending_message(
                resume_reason, ctx.message, interactive=self._resume_note_interactive(),
            )
        elif agent_history and agent_history[-1].get("role") == "tool" and interruption_is_fresh:
            persist_override = ctx.message
            ctx.message = (
                "[System note: A new message has arrived. The conversation "
                "history contains pending tool outputs from an interrupted turn. "
                "IGNORE those pending results. Address the user's NEW message "
                "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                + ctx.message
            )
        self._prepend_pending_note("_pending_skills_reload_notes")
        # Safety net: a startup auto-resume event carries empty text; if the resume_pending branch
        # did not fire (freshness signals disagreed, marker cleared) we must NOT hand the model a blank
        # user turn. Restricted to resume_pending sessions so caption-less image turns are untouched.
        if isinstance(ctx.message, str) and not ctx.message.strip() and resume_pending:
            ctx.message = build_resume_recovery_note(resume_reason, "", interactive=self._resume_note_interactive())
        return persist_override, ctx.persist_user_timestamp

    def _native_image_run_message(self):
        """Wrap the user turn as an OpenAI-style multimodal content list when
        _prepare_inbound_message_text buffered image paths; consume-and-clear so later turns on the
        same runner never re-attach stale images. Falls back to plain text when nothing is readable."""
        ctx = self._ctx
        from gateway.session_api_turn import api_execution
        api = api_execution.get()
        if api is not None and isinstance(api.get('content'), list):
            return api['content']
        native_imgs = self._runner._consume_pending_native_image_paths(ctx.session_key)
        if not native_imgs:
            return ctx.message
        try:
            from agent.image_routing import build_native_content_parts
            parts, skipped = build_native_content_parts(ctx.message, native_imgs)
            if skipped:
                logger.warning("Native image attachment: skipped %d unreadable path(s): %s", len(skipped), skipped)
            if any(p.get("type") == "image_url" for p in parts):
                return parts
        except Exception as exc:
            logger.warning("Native image attachment failed, falling back to text: %s", exc)
        return ctx.message

    def _run_conversation_with_approval(self, agent, agent_history, observed_group_context,
                                        persist_user_message_override, persist_user_timestamp_override):
        """Run the turn with the per-session gateway approval callback registered: dangerous-command
        approval blocks the agent thread (mirrors CLI input()); the callback bridges sync→async."""
        from gateway.run import _wrap_current_message_with_observed_context
        from tools.approval import register_gateway_notify, unregister_gateway_notify
        from tools.approval_context import reset_current_session_key, set_current_session_key
        ctx = self._ctx
        session_key = ctx.session_key or ""
        token = set_current_session_key(session_key)
        register_gateway_notify(session_key, self._approval_notify_sync)
        try:
            api_message = _wrap_current_message_with_observed_context(self._native_image_run_message(), observed_group_context)
            kwargs = {"conversation_history": agent_history, "task_id": ctx.session_id}
            if _accepts_keyword(agent.run_conversation, "turn_author"):
                # Sent on every transport: a provider gating durable writes needs the bot flag in a DM too.
                from gateway.session_api_turn import api_execution
                from gateway.session_ingress import admission_author
                api = api_execution.get()
                kwargs["turn_author"] = (api.get('turn_author') if api is not None else
                    admission_author.get() or {"id": ctx.source.user_id or None, "name": ctx.source.user_name or None,
                                               "is_bot": bool(getattr(ctx.source, "is_bot", False))})
            if persist_user_message_override is not None:
                kwargs["persist_user_message"] = persist_user_message_override
            elif observed_group_context:
                kwargs["persist_user_message"] = ctx.message
            if ctx.persist_user_display_kind:
                # Internal self-injected turn: type the persisted user row so UIs render it as a
                # timeline notice, not a user bubble (stripped from provider payloads downstream).
                kwargs["persist_user_display_kind"] = ctx.persist_user_display_kind
            if ctx.persist_user_display_metadata:
                kwargs["persist_user_display_metadata"] = ctx.persist_user_display_metadata
            if ctx.moa_config is not None:
                kwargs["moa_config"] = ctx.moa_config
            if persist_user_timestamp_override is not None:
                kwargs["persist_user_timestamp"] = persist_user_timestamp_override
            # The RAW inbound id (not event_message_id, the reply anchor) rides the persisted user
            # turn so a restart-interrupted turn is recorded WITH its id for drain-window dedup.
            if ctx.inbound_message_id is not None:
                kwargs["persist_user_platform_id"] = str(ctx.inbound_message_id)
            from gateway.session_results import execution_result
            captured = execution_result.get()
            before = (getattr(agent, 'session_prompt_tokens', 0) or 0,
                      getattr(agent, 'session_completion_tokens', 0) or 0)
            result = agent.run_conversation(api_message, **kwargs)
            if captured is not None:
                incoming = max(0, (getattr(agent, 'session_prompt_tokens', 0) or 0) - before[0])
                outgoing = max(0, (getattr(agent, 'session_completion_tokens', 0) or 0) - before[1])
                captured['usage'] = {'input_tokens': incoming, 'output_tokens': outgoing,
                                     'total_tokens': incoming + outgoing}
            return result
        finally:
            unregister_gateway_notify(session_key)
            # Cancel pending clarify entries so blocked agent threads don't hang past the end of the
            # run (interrupt, completion, gateway shutdown). Idempotent.
            with suppress(Exception):
                from tools.clarify_gateway import clear_session
                clear_session(session_key)
            reset_current_session_key(token)

    def _finish_stream_consumer(self, result, agent_history, stream_consumer):
        ctx = self._ctx
        # Canonicalize a model-emitted computer-use screenshot path at the common result boundary so
        # the streaming finalizer and the non-streaming delivery path see the same response.
        if isinstance(result, dict) and isinstance(result.get("final_response"), str):
            result["final_response"] = repair_explicit_computer_use_media_paths(
                result["final_response"], result.get("messages", []), history_offset=len(agent_history),
            )
        ctx.result_holder[0] = result
        if stream_consumer is None:
            return
        # Pass final_response as the authoritative finalize payload: it includes post-stream
        # augmentation (verifier footer, explainer) the accumulator never saw. Adopt ONLY a genuinely
        # completed final: interrupt paths return {interrupted: True, completed: False} with a
        # DIAGNOSTIC final_response — adopting it would seal the partial answer over with the
        # diagnostic AND suppress the gateway's own error delivery.
        _final_for_stream = None
        if (
            isinstance(result, dict) and not result.get("failed") and not result.get("interrupted")
            and result.get("completed") is not False
        ):
            fr = result.get("final_response")
            if isinstance(fr, str) and fr.strip() and fr != "(empty)":
                _final_for_stream = fr
        if _final_for_stream is None:
            stream_consumer.finish()
            return
        # Duck-type safe: test doubles / older consumers may expose a zero-arg finish().
        try:
            stream_consumer.finish(_final_for_stream)
        except TypeError:
            stream_consumer.finish()

    def _restore_telegram_thread_id_after_split(self, agent_session_id) -> None:
        """Telegram DM whose source.thread_id was lost in the session split (synthetic/recovered
        event): restore it from the binding so _thread_metadata_for_source yields the right
        message_thread_id instead of the General thread (non-fatal)."""
        ctx = self._ctx
        try:
            # run_sync is off-loop (executor); sync DB is fine.
            binding = self._runner._session_db._db.get_telegram_topic_binding_by_session(session_id=agent_session_id)
            if binding and binding.get("thread_id"):
                ctx.source.thread_id = str(binding["thread_id"])
                logger.debug(
                    "Restored source.thread_id=%s from binding after session split %s → %s",
                    ctx.source.thread_id, ctx.session_id, agent_session_id,
                )
        except Exception:
            logger.debug("Failed to restore thread_id from binding after session split", exc_info=True)

    def _sync_session_after_run(self, agent_history):
        """Sync session_id right after run_conversation(): compression can rotate before a follow-up
        model call fails, and the failure return must still point at the compressed child.
        Returns (compacted_in_place, effective_session_id, effective_history_offset)."""
        ctx = self._ctx
        runner = self._runner
        agent = ctx.agent_holder[0]
        # In-place compaction compacts the transcript WITHOUT rotating the id, so the id-change diff
        # can't see it; compress_context() sets this flag and the gateway re-baselines as for a split.
        compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
        agent_session_id = getattr(agent, 'session_id', ctx.session_id) if agent else ctx.session_id
        session_was_split = bool(agent and ctx.session_key and agent_session_id != ctx.session_id)
        if session_was_split:
            logger.info("Session split detected: %s → %s (compression)", ctx.session_id, agent_session_id)
            entry = runner.session_store._entries.get(ctx.session_key)
            persisted = False
            if entry:
                entry_session_id = getattr(entry, "session_id", None)
                if not ctx._run_still_current():
                    logger.info(
                        "Skipping session split sync for stale run %s — "
                        "generation %s is no longer current",
                        ctx.session_key or "?", ctx.run_generation,
                    )
                elif entry_session_id == agent_session_id:
                    persisted = True
                elif entry_session_id != ctx.session_id:
                    logger.info(
                        "Skipping session split sync for %s because the "
                        "session binding moved from %s to %s before "
                        "compression finished",
                        ctx.session_key or "?", ctx.session_id, entry_session_id,
                    )
                else:
                    entry.session_id = agent_session_id
                    runner.session_store._save()
                    runner.session_store._record_gateway_session_peer(agent_session_id, ctx.session_key, ctx.source)
                    persisted = True
            # Only after this run published its split — a stale /stop→/new predecessor must not
            # mutate routing state.
            if persisted:
                src = ctx.source
                if (
                    getattr(src, "platform", None) == Platform.TELEGRAM and getattr(src, "chat_type", None) == "dm"
                    and getattr(src, "thread_id", None) is None and runner._session_db is not None
                ):
                    self._restore_telegram_thread_id_after_split(agent_session_id)
                runner._sync_telegram_topic_binding(src, entry, reason="agent-run-compression")
        runner._sync_session_model_from_agent(agent_session_id, agent)
        # history_offset=0 whenever the agent's message list lost the original history prefix
        # (split OR in-place compaction): the returned `messages` is the compacted set, persist all
        # of it; slicing past the pre-compaction length would drop everything.
        offset = 0 if (session_was_split or compacted_in_place) else len(agent_history)
        return compacted_in_place, agent_session_id, offset

    def _combined_ephemeral_prompt(self) -> str:
        """Platform context + YAML channel_prompts hint + channel_overrides system_prompt (or global
        ephemeral) + the gateway ephemeral prompt."""
        ctx = self._ctx
        from gateway.session_api_turn import api_execution
        api = api_execution.get()
        if api is not None:
            return api['settings'].get('ephemeral_system_prompt') or ''
        combined = ctx.context_prompt or ""
        for extra in (
            (ctx.channel_prompt or "").strip(),
            self._runner._get_system_prompt_for_channel(
                ctx.source.platform, ctx.source.chat_id or "", thread_id=getattr(ctx.source, "thread_id", None),
                parent_id=getattr(ctx.source, "parent_chat_id", None),
            ),
        ):
            if extra:
                combined = (combined + "\n\n" + extra).strip()
        return combined

    def _append_auto_media_tags(self, final_response: str, result, agent_history, history_media_paths) -> str:
        """Append MEDIA:<path> tags from tool results (e.g. TTS) that the model's final text omits, so
        extract_media() delivers each file once. Scoped to THIS turn (slice at len(agent_history)) so
        a stale MEDIA: path from an earlier turn never rides a later reply; the history-path dedup is
        the secondary guard — and the sole one when mid-run compression shrank the list."""
        from gateway.run import _collect_auto_append_media_tags
        if "MEDIA:" in final_response:
            return final_response
        # Scan tool results for MEDIA:<path> tags that need to be delivered as native audio/file
        # attachments. The TTS tool embeds MEDIA: tags in its JSON response, but the model's final text
        # reply usually doesn't include them. We collect unique tags from tool results and append any that
        # aren't already present in the final response, so the adapter's extract_media() can find and
        # deliver the files exactly once. Scope the scan to THIS turn's tool results only. ``agent_history``
        # was passed into run_conversation as ``conversation_history``, so the agent's returned ``messages``
        # list is ``agent_history`` followed by the messages produced this turn. Slicing at
        # ``len(agent_history)`` isolates the current turn precisely, so a stale MEDIA: path emitted by a
        # tool several turns earlier (still present in the full message list) can never leak onto a later
        # text-only reply. (Fixes #34608) Path-based deduplication against _history_media_paths (collected
        # before run_conversation) is retained as a secondary guard. It is also the sole guard on the
        # fallback branch taken when mid-run context compression shrinks the message list below the original
        # history length, preserving the compression-safe behaviour of #160.
        media_tags, has_voice_directive = _collect_auto_append_media_tags(
            result.get("messages", []), history_offset=len(agent_history), history_media_paths=history_media_paths,
        )
        if not media_tags:
            return final_response
        unique_tags = (["[[audio_as_voice]]"] if has_voice_directive else []) + list(dict.fromkeys(media_tags))
        return final_response + "\n" + "\n".join(unique_tags)

    def run_sync(self):
        from gateway.session_policy import policy_for_source, policy_scope
        from gateway.session_api_turn import api_policy_scope
        from gateway.session_authorities import active_authority
        with policy_scope(policy_for_source(self._runner, self._ctx.source),
                          authority=active_authority(self._runner)), api_policy_scope():
            result = self._run_sync_scoped()
            from gateway.session_results import execution_result
            captured = execution_result.get()
            if captured is not None:
                captured['result'] = result
            return result

    def _run_sync_scoped(self):
        """Executor-thread body of the turn; returns the gateway result dict.

        The turn message lives on the shared TurnContext (``ctx.message``) so ``_run_agent_inner`` sees
        every rebind. session_key propagates via contextvars (_set_session_env / set_current_session_key)
        — never os.environ["HERMES_SESSION_KEY"], which would misroute approvals across sessions.
        """
        from gateway.run import _current_max_iterations, _normalize_empty_agent_response, _sanitize_gateway_final_response
        ctx = self._ctx
        runner = self._runner
        # Platform.LOCAL ("local") maps to the "cli" hint key the agent understands.
        # session_key is propagated via contextvars in _set_session_env() (_SESSION_KEY) and via
        # set_current_session_key() (_approval_session_key) below — both concurrency-safe and inherited by
        # tool worker threads. We deliberately do NOT write os.environ["HERMES_SESSION_KEY"] here:
        # os.environ is process-global, so concurrent gateway sessions (e.g. two Discord threads) would
        # clobber each other's value, and a tool thread whose contextvar is unset would fall back to
        # os.environ and read the wrong session key — misrouting command-approval prompts to the wrong
        # thread (#24100). The non-gateway surfaces don't depend on this write: CLI and cron bind the
        # session via contextvars (set_current_session_key / session context), and only the TUI slash-worker
        # *subprocess* exports HERMES_SESSION_KEY (from its own --session-key argv, a separate process) — so
        # removing this in-process gateway write does not affect any of them.
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(runner, ctx.source)
        platform_key = policy.platform if policy else ("cli" if ctx.source.platform == Platform.LOCAL else ctx.source.platform.value)
        combined_ephemeral = self._combined_ephemeral_prompt()
        max_iterations = policy.max_turns if policy else _current_max_iterations()
        from gateway.hosted_room_execution_policy import current_room_execution_policy
        room = current_room_execution_policy()
        if room is not None:
            max_iterations = room.max_iterations
            ctx.enabled_toolsets = list(room.enabled_toolsets)
        try:
            model, runtime_kwargs = runner._resolve_session_agent_runtime(
                source=ctx.source, session_key=ctx.session_key, user_config=ctx.user_config,
            )
            from gateway.session_api_turn import prepare_api_runtime
            model, runtime_kwargs = prepare_api_runtime(model, runtime_kwargs)
            if policy and policy.model:
                model = policy.model
            logger.debug(
                "run_agent resolved: model=%s provider=%s session=%s",
                model, runtime_kwargs.get("provider"), ctx.session_key or "",
            )
        except Exception as exc:
            return {"final_response": f"⚠️ Provider authentication failed: {exc}", "messages": [], "api_calls": 0, "tools": []}
        pr = runner._provider_routing
        reasoning_config = (policy.reasoning_config if policy else
            runner._resolve_session_reasoning_config(source=ctx.source, session_key=ctx.session_key, model=model))
        runner._service_tier = runner._resolve_session_service_tier(source=ctx.source, session_key=ctx.session_key)
        from gateway.session_api_turn import api_execution
        api = api_execution.get()
        if api is not None:
            from gateway.platforms.api_server import _request_reasoning_config, _request_service_tier, _REQUEST_OPTION_MISSING
            requested_reasoning = _request_reasoning_config(api['settings'].get('model_options'))
            if requested_reasoning is not None:
                reasoning_config = requested_reasoning
            tier = _request_service_tier(api['settings'].get('model_options'))
            if tier is not _REQUEST_OPTION_MISSING:
                runner._service_tier = tier
        runner._reasoning_config = reasoning_config
        stream_consumer, stream_delta_cb, interim_cb, want_interim = self._setup_stream_consumer(platform_key)
        turn_route = runner._resolve_turn_agent_config(ctx.message, model, runtime_kwargs)
        agent, reused_cached_agent = self._resolve_turn_agent(
            turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr,
        )
        self._wire_turn_agent_callbacks(agent, turn_route, reasoning_config, stream_delta_cb, interim_cb, want_interim)
        agent_history, observed_group_context, history_media_paths = self._load_turn_history(agent, reused_cached_agent)
        persist_msg, persist_ts = self._prepare_turn_message(agent_history)
        result = self._run_conversation_with_approval(agent, agent_history, observed_group_context, persist_msg, persist_ts)
        self._finish_stream_consumer(result, agent_history, stream_consumer)
        # The streaming-TTS consumer's finish() runs on the outer loop thread after the executor
        # returns, so early run_sync returns are also finalised.
        # See the outer finally/completion section below. See #60671.
        final_response = result.get("final_response")
        # Actual token counts from the agent instance used for this run.
        agent = ctx.agent_holder[0]
        has_comp = bool(agent) and hasattr(agent, "context_compressor")
        comp = agent.context_compressor if has_comp else None
        usage = {
            "last_prompt_tokens": getattr(comp, "last_prompt_tokens", 0) if has_comp else 0,
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) if has_comp else 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) if has_comp else 0,
            "model": getattr(agent, "model", None) if agent else None,
            "context_length": (getattr(comp, "context_length", 0) or 0) if has_comp else 0,
        }
        compacted_in_place, effective_session_id, history_offset = self._sync_session_after_run(agent_history)
        # failure_reason must survive the empty-response path too (TUI billing, transient-failure
        # persistence). compression_deferred (soft lock-contention defer) is distinct from
        # compression_exhausted so the gateway never auto-resets a session a concurrent compressor is
        # about to shrink.
        common = {
            "messages": result.get("messages", []), "api_calls": result.get("api_calls", 0),
            "failed": result.get("failed", False), "failure_reason": result.get("failure_reason"),
            "partial": result.get("partial", False), "completed": result.get("completed"),
            "interrupted": result.get("interrupted", False), "interrupt_message": result.get("interrupt_message"),
            "error": result.get("error"),
            "compression_exhausted": result.get("compression_exhausted", False),
            "compression_deferred": result.get("compression_deferred", False),
            "tools": ctx.tools_holder[0] or [],
            "history_offset": history_offset, "compacted_in_place": compacted_in_place, "session_id": effective_session_id,
            **usage,
        }
        if not final_response:
            final_response = _normalize_empty_agent_response(result, final_response or "", history_len=len(agent_history))
            final_response = _sanitize_gateway_final_response(ctx.source.platform, final_response)
            if not final_response:
                final_response = f"⚠️ {result['error']}" if result.get("error") else ""
            # NOTE: deliberately omits agent_persisted/last_reasoning/response_* — the caller
            # defaults agent_persisted differently when the key is absent.
            return {"final_response": final_response, **common}
        final_response = self._append_auto_media_tags(final_response, result, agent_history, history_media_paths)
        # Auto-titling runs at TURN START (agent/turn_context.py) from the user's message alone, so a
        # failed/interrupted turn is still titled.
        return {
            "final_response": final_response, "last_reasoning": result.get("last_reasoning"), **common,
            "response_previewed": result.get("response_previewed", False),
            "response_transformed": result.get("response_transformed", False),
            # Lets the persistence block tell whether the codex app-server path self-persisted (it
            # didn't — see codex_runtime.py); default True keeps skip-db for the standard runtime.
            "agent_persisted": result.get("agent_persisted", True),
        }

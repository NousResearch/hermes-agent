"""Busy-session queueing, slot claims, slash dispatch tables and destructive-slash confirmation
for GatewayRunner (mixin bound via the MRO).

``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import contextlib
import json
import os
import time
from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent, MessageType
from gateway.session import SessionSource
from typing import Any, Dict, Optional, Union

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayBusySessionMixin:
    """Busy-session queueing, slot claims, slash dispatch tables, destructive-slash confirmation."""

    def _queue_during_drain_enabled(self, busy_input_mode: Optional[str] = None) -> bool:
        # "queue"/"steer" mean messages survive a restart (queued for the new process); "interrupt" drops.
        mode = busy_input_mode or self._busy_input_mode
        return self._restart_requested and mode in {"queue", "steer"}

    def _overflow_queue(self, session_key: str):
        """The session's FIFO overflow list, or None when no session state exists yet."""
        state = self._peek_session_state(session_key)
        return state.conversation.queued_events if state else None

    def _enqueue_fifo(self, session_key: str, queued_event: "MessageEvent", adapter: Any) -> None:
        """Append a /queue event to the FIFO chain for a session."""
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if pending_slot is None:
            return
        if session_key in pending_slot:
            self._session_state(session_key).conversation.queued_events.append(queued_event)
        else:
            pending_slot[session_key] = queued_event

    def _promote_queued_event(
        self, session_key: str, adapter: Any, pending_event: Optional["MessageEvent"]
    ) -> Optional["MessageEvent"]:
        """Promote the next overflow item after the slot drained.

        ``pending_event`` None → the overflow head becomes the pending event; otherwise the head is
        staged into the slot for the NEXT recursion. Returns the (possibly updated) pending_event.
        """
        overflow = self._overflow_queue(session_key)
        if not overflow:
            return pending_event
        if pending_event is None:
            return overflow.pop(0)
        if adapter is not None and hasattr(adapter, "_pending_messages"):
            adapter._pending_messages[session_key] = overflow.pop(0)
        # else: no adapter — leave the head in place so we don't silently drop it.
        return pending_event

    def _queue_depth(self, session_key: str, *, adapter: Any = None) -> int:
        """Total pending /queue items for a session — slot + overflow."""
        depth = len(self._overflow_queue(session_key) or ())
        if adapter is not None and session_key in getattr(adapter, "_pending_messages", {}):
            depth += 1
        return depth

    def _rescue_orphaned_overflow(self, session_key: str, adapter: Any) -> Optional["MessageEvent"]:
        """Pop the oldest orphaned FIFO overflow event for an idle session (None if nothing to rescue).

        ``queued_events`` drains only at the post-turn promotion site; a busy window ending without
        it (early exit, exception/interrupt/generation-bump) orphans the overflow. On a NEW event for
        a NON-busy session the oldest orphan runs as THIS turn, the next is staged into the slot so
        arrival order holds, and the caller enqueues the incoming event behind it. The returned
        event is REMOVED from both stores, else the post-turn dequeue would run it twice.

        See #28503.
        """
        try:
            overflow = self._overflow_queue(session_key)
            if not overflow:
                return None
            pending_slot = getattr(adapter, "_pending_messages", None)
            if not isinstance(pending_slot, dict) or pending_slot.get(session_key):
                return None  # slot occupied (busy) or no slot storage — promotion owns this
            head = overflow.pop(0)
            # Keep the slot occupied so the drain promotes in order and a mid-chain arrival routes
            # to overflow instead of jumping the queue (same invariant as _promote_queued_event).
            if overflow:
                pending_slot[session_key] = overflow.pop(0)
            logger.warning(
                "Rescued orphaned FIFO overflow event for idle session "
                "%s — it was queued during a busy window but the post-turn "
                "drain never promoted it (#99882)", session_key,
            )
            if overflow:
                logger.warning(
                    "%d overflow event(s) still queued for session %s after "
                    "rescue staging (will drain via normal promotion)", len(overflow), session_key,
                )
            return head
        except Exception:
            logger.debug("FIFO overflow rescue failed for %s", session_key, exc_info=True)
            return None

    @staticmethod
    def _is_goal_continuation_event(event_or_text: Any) -> bool:
        """True for synthetic /goal continuation turns (so pause/clear can spare real /queue items)."""
        text = getattr(event_or_text, "text", event_or_text) or ""
        return str(text).startswith("[Continuing toward your standing goal]\nGoal:")

    def _clear_goal_pending_continuations(self, session_key: str, adapter: Any) -> int:
        """Remove queued synthetic /goal continuations for one session; real /queue items are kept."""
        removed = 0
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if isinstance(pending_slot, dict):
            pending_event = pending_slot.get(session_key)
            if self._is_goal_continuation_event(pending_event):
                pending_slot.pop(session_key, None)
                removed += 1

        overflow = self._overflow_queue(session_key)
        if overflow:
            kept = [e for e in overflow if not self._is_goal_continuation_event(e)]
            removed += len(overflow) - len(kept)
            self._peek_session_state(session_key).conversation.queued_events = kept
        return removed

    def _goal_still_active_for_session(self, session_id: str) -> bool:
        """Best-effort fresh DB check before running a queued continuation."""
        if not session_id:
            return False
        try:
            from hermes_cli.goals import GoalManager
            return GoalManager(session_id=session_id).is_active()
        except Exception as exc:
            logger.debug("goal continuation: active-state recheck failed: %s", exc)
            return False

    def _get_max_concurrent_sessions(self) -> Optional[int]:
        """Return the configured active chat session cap, if enabled."""
        try:
            from hermes_cli.active_sessions import resolve_max_concurrent_sessions
            return resolve_max_concurrent_sessions(getattr(self, "config", None))
        except Exception:
            return None

    def _active_session_limit_message(self, session_key: str) -> Optional[str]:
        """Return a user-facing rejection when starting a new session exceeds the cap."""
        max_sessions = self._get_max_concurrent_sessions()
        if max_sessions is None or self._is_session_running(session_key):
            return None
        active_count = self._running_agent_count()
        if active_count < max_sessions:
            return None
        from hermes_cli.active_sessions import active_session_limit_message
        return active_session_limit_message(active_count, max_sessions)

    def _claim_active_session_slot(
        self, session_key: str, source: SessionSource
    ) -> tuple[Any, Optional[str]]:
        """Claim a cross-process active-session slot for a new gateway turn."""
        if self._is_session_running(session_key):
            return None, None
        limit_message = self._active_session_limit_message(session_key)
        if limit_message is not None:
            return None, limit_message
        try:
            from hermes_cli.active_sessions import try_acquire_active_session
            platform = source.platform.value if source and source.platform else "gateway"
            return try_acquire_active_session(
                session_id=session_key,
                surface=f"gateway:{platform}",
                config=getattr(self, "config", None),
                metadata={
                    "platform": platform,
                    "chat_id": getattr(source, "chat_id", "") or "",
                    "user_id": getattr(source, "user_id", "") or "",
                    # Writer identity: a leaked lease from this process is re-acquired by the next
                    # turn rather than fencing it out forever (pruning only reclaims dead PROCESSES).
                    # Writer identity for re-entrancy (#94595): if this process leaks a lease for this
                    # session (exception path skipped release), the next turn re-acquires its own entry
                    # instead of being fenced out of it forever — pruning only reclaims entries whose
                    # PROCESS died.
                    "live_session_id": str(session_key),
                },
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return None, None

    @staticmethod
    def _agent_has_active_subagents(running_agent: Any) -> bool:
        """True when *running_agent* is driving subagents (callers demote interrupt → queue;
        ``interrupt()`` would cascade through ``_active_children``). Fail-safe False on any error."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        if running_agent is None or running_agent is _AGENT_PENDING_SENTINEL:
            return False
        current_tool = getattr(running_agent, "_current_tool", None)
        if not isinstance(current_tool, str):
            try:
                summary = running_agent.get_activity_summary()
            except Exception:
                summary = None
            if isinstance(summary, dict):
                current_tool = summary.get("current_tool")
        if isinstance(current_tool, str):
            active_tools = {part.strip() for part in current_tool.split(",")}
            if "delegate_task" in active_tools:
                return True
        children = getattr(running_agent, "_active_children", None)
        # Real collections only — a ``MagicMock()._active_children`` auto-attr must not demote.
        if not isinstance(children, (list, tuple, set)) or not children:
            return False
        lock = getattr(running_agent, "_active_children_lock", None)
        try:
            with lock if lock is not None else contextlib.nullcontext():
                return bool(children)
        except Exception:
            return False

    async def _session_has_compression_in_flight(self, session_key: str) -> bool:
        """True when a compression lock is held for this session's id (callers demote interrupt →
        queue, else a follow-up against the pre-rotation parent orphans compression siblings).
        Both blocking reads run in a worker thread so a large state.db never freezes the loop.

        Context compression is interrupt-protected (#23975) but gateway ``interrupt`` busy-input mode can
        still start a follow-up turn against the pre-rotation parent while compression is mid-flight,
        producing orphaned compression siblings (#56391).
        """
        session_store = getattr(self, "session_store", None)
        if not session_key or session_store is None:
            return False
        def _assume_active(what: str, ident) -> bool:
            logger.warning(
                "Compression in-flight check failed while reading %s %s; treating compression as "
                "active to avoid interrupting a possible parent-session rotation", what, ident, exc_info=True,
            )
            return True

        try:
            session_id = await asyncio.to_thread(
                self._lookup_session_id_under_store_lock, session_store, session_key
            )
        except (AttributeError, TypeError):
            return False
        except Exception:
            return _assume_active("session", session_key)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return False
        raw_db = getattr(session_db, "_db", session_db)
        try:
            holder = await asyncio.to_thread(raw_db.get_compression_lock_holder, str(session_id))
            # Production returns Optional[str]. Reject non-strings so a MagicMock auto-attr (or any
            # unexpected truthy) cannot look like a held lock and skip hygiene.
            # See #96953.
            return isinstance(holder, str) and bool(holder)
        except (AttributeError, TypeError):
            return False
        except Exception:
            return _assume_active("lock holder for session", session_id)

    @staticmethod
    def _lookup_session_id_under_store_lock(session_store, session_key: str):
        """Sync helper run in the thread pool: read session_id under the store lock."""
        # noqa: SLF001 — intentional private access; runs off the event loop.
        with session_store._lock:  # noqa: SLF001
            session_store._ensure_loaded_locked()  # noqa: SLF001
            entry = session_store._entries.get(session_key)  # noqa: SLF001
        return getattr(entry, "session_id", None) if entry is not None else None

    # Metadata that must match for two pending events to merge into one slot.
    _SECURITY_METADATA_KEYS = (
        "hermes_plugin_id", "hermes_plugin_injection", "gateway_session_key",
        "gateway_session_id", "gateway_session_strict",
    )

    def _queue_or_replace_pending_event(self, session_key: str, event: MessageEvent) -> None:
        from gateway.platforms.base import merge_pending_message_event
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return
        # FIFO so each follow-up gets its own turn in arrival order (the single pending slot used to
        # be silently OVERWRITTEN). Photo bursts still merge into the head slot (album semantics).
        pending_slot = getattr(adapter, "_pending_messages", None)
        # #28503 — Previously this called ``merge_pending_message_event`` with the default
        # ``merge_text=False``, which silently OVERWROTE the single pending slot when consecutive text
        # messages arrived in ``busy_input_mode: queue``.
        existing = pending_slot.get(session_key) if isinstance(pending_slot, dict) else None
        same_security_context = existing is not None and (
            getattr(existing, "internal", False) == getattr(event, "internal", False)
            and getattr(existing, "allow_gateway_control", True)
            == getattr(event, "allow_gateway_control", True)
            and all(
                (getattr(existing, "metadata", None) or {}).get(key)
                == (getattr(event, "metadata", None) or {}).get(key)
                for key in self._SECURITY_METADATA_KEYS
            )
        )
        if same_security_context and (
            getattr(existing, "message_type", None) == MessageType.PHOTO
            or event.message_type == MessageType.PHOTO
            or bool(getattr(existing, "media_urls", None))
            or bool(getattr(event, "media_urls", None))
        ):
            # Preserve photo-burst / media-merge semantics for the head slot.
            merge_pending_message_event(
                adapter._pending_messages, session_key, event,
                merge_text=event.message_type == MessageType.TEXT,
            )
            return

        if self._queue_depth(session_key, adapter=adapter) >= self._BUSY_QUEUE_MAX_PENDING:
            logger.warning(
                "Dropping busy-mode follow-up for session %s — pending queue at cap (%d).",
                session_key, self._BUSY_QUEUE_MAX_PENDING,
            )
            return

        self._enqueue_fifo(session_key, event, adapter)

    async def _prepare_busy_steer_text(self, event: MessageEvent) -> str:
        """Steerable text for a busy follow-up, transcribing voice-message media first.

        Steer bypasses the inbound STT queue, so a media-only voice follow-up would otherwise
        silently degrade to queue mode. Uses the single out-of-band STT choke point, so STT runs at
        most once per message; on failure the caption (if any) is kept.
        """
        text = (event.text or "").strip()
        if not self._pending_event_audio_paths(event):
            return text
        enriched_text, successful_transcripts = await self._transcribe_and_echo_pending_voice(
            event, self._adapter_for_source(event.source), event.source, text, log_context="Busy-steer"
        )
        return (enriched_text or text).strip() if successful_transcripts else text

    @staticmethod
    def _busy_reply_to(event: MessageEvent, reply_anchor):
        # Telegram DM topics anchor on the thread; other Telegram threads send unanchored.
        return (
            reply_anchor
            if event.source.platform == Platform.TELEGRAM
            and event.source.chat_type == "dm"
            and event.source.thread_id
            else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
        )

    async def _send_busy_reply(self, event: MessageEvent, adapter, content: str, *, plain_anchor: bool = False) -> None:
        """Send a busy-path reply anchored to the event (thread metadata included)."""
        reply_anchor = self._reply_anchor_for_event(event)
        await adapter._send_with_retry(
            chat_id=event.source.chat_id, content=content,
            reply_to=reply_anchor if plain_anchor else self._busy_reply_to(event, reply_anchor),
            metadata=self._thread_metadata_for_source(event.source, reply_anchor),
        )

    async def _send_busy_drain_notice(self, event: MessageEvent, session_key: str, effective_mode: str) -> None:
        """Busy path while the gateway is restarting/stopping: queue (if allowed) and tell the user."""
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return
        if self._queue_during_drain_enabled(effective_mode):
            self._queue_or_replace_pending_event(session_key, event)
            message = f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
        else:
            message = f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."
        await self._send_busy_reply(event, adapter, message)

    # Bare-word approval replies → (verb, args) for the synthesized slash command.
    _PLAINTEXT_APPROVAL_WORDS: Dict[str, tuple] = {
        **{w: ("approve", "") for w in ("approve", "yes", "ok", "okay", "confirm", "y", "👍")},
        **{w: ("deny", "") for w in ("deny", "no", "reject", "cancel", "n", "👎")},
        **{w: ("approve", "always") for w in ("always", "approve always", "always approve")},
        **{w: ("approve", "session") for w in ("session", "approve session", "session approve")},
    }

    async def _route_plaintext_approval_while_busy(self, event: MessageEvent, session_key: str) -> bool:
        """Route a bare "yes"/"no" to the approval handlers while a dangerous-command approval blocks.

        Returns True when the message was consumed as an approval response.
        """
        # A bare "yes" while blocked on a dangerous-command approval must reach the approval handler,
        # not queue behind a turn that can't start until it resolves (auto-deny deadlock). Gated on
        # has_blocking_approval so a conversational "yes" never fires a command.
        try:
            from tools.approval import has_blocking_approval
            # --- Approval response routing (#46866) --- When the agent is blocked waiting for a
            # dangerous-command approval, plain-text responses like "yes" or "approve" must be routed to the
            # approval handler instead of being steered/queued/interrupted. Slash forms (/approve, /deny)
            # already bypass to the runner at the base-adapter guard. This handles the bare-word forms
            # (Signal/SMS users naturally type "yes" rather than "/approve"). Gating on
            # has_blocking_approval(session_key) is the disambiguator that keeps a conversational "yes" from
            # triggering a dangerous command when no approval is actually pending (design intent — see
            # run.py "Pending exec approvals are handled by /approve and /deny" note). We reuse the
            # canonical /approve and /deny handlers rather than re-deriving the resolution + i18n messaging:
            # they resolve the waiting thread, resume typing, AND return a localized confirmation string.
            # The busy-handler path does not auto-send that return, so we deliver it ourselves (mirroring
            # the draining-case send above).
            if event.allow_gateway_control and has_blocking_approval(session_key):
                _raw_text = (event.text or "").strip().lower()
                _match = self._PLAINTEXT_APPROVAL_WORDS.get(_raw_text)
                if _match is not None:
                    _verb, _normalized_args = _match
                    _approval_handler = (
                        self._handle_approve_command if _verb == "approve" else self._handle_deny_command
                    )
                    # Synthesize "/approve [args]" / "/deny" so the slash handlers parse modifiers via
                    # event.get_command_args(). Always a literal "/": is_command()/get_command_args()
                    # don't recognize per-platform display prefixes ("!" on Slack/Matrix).
                    event.text = f"/{_verb} {_normalized_args}".rstrip()
                    _reply = await _approval_handler(event)
                    logger.info(
                        "Approval response via plain text: session=%s verb=%s args=%r",
                        session_key, _verb, _normalized_args,
                    )
                    _adapter = self._adapter_for_source(event.source)
                    if _adapter and _reply:
                        _text, _eph_ttl = _adapter._unwrap_ephemeral(_reply)
                        if _text:
                            await self._send_busy_reply(event, _adapter, _text, plain_anchor=True)
                    return True
        except Exception:
            logger.warning(
                "Plain-text approval routing failed for session %s; "
                "falling through to busy handling", session_key, exc_info=True,
            )
        return False

    async def _resolve_busy_steer_or_redirect(
        self, event: MessageEvent, session_key: str, effective_mode: str, running_agent: Any
    ) -> "GatewayRunner._BusySteerOutcome":
        """Apply interrupt->queue demotions, then attempt steer (steer mode) or redirect (interrupt mode)."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        # Steer injects mid-run via running_agent.steer(), falling back to queue (nothing lost) when
        # the agent isn't running yet, lacks steer(), or the payload is empty. Interrupt is demoted
        # to queue while subagents run (interrupt() would abort them); /stop and /new still cancel all.
        demoted_for_subagents = (
            effective_mode == "interrupt" and self._agent_has_active_subagents(running_agent)
        )
        if demoted_for_subagents:
            effective_mode = self._demote_interrupt(session_key, "the running agent has active subagents (#30170)")
        demoted_for_compression = (
            effective_mode == "interrupt" and await self._session_has_compression_in_flight(session_key)
        )
        if demoted_for_compression:
            effective_mode = self._demote_interrupt(session_key, "context compression is in flight (#56391)")
        steered = redirected = False
        agent_live = running_agent is not None and running_agent is not _AGENT_PENDING_SENTINEL
        plain_text = (
            event.message_type == MessageType.TEXT and not event.media_urls and not event.media_types
        )
        if effective_mode == "steer":
            steer_text = await self._prepare_busy_steer_text(event)
            # Steerable: plain text, OR every attachment is voice media folded into steer_text.
            # A follow-up qualifies for steering when it is plain text, OR when every attachment is
            # STT-eligible voice media whose transcript was just folded into steer_text — otherwise a voice
            # note in steer mode silently degrades to queue mode (#58780).
            _steer_media_urls = getattr(event, "media_urls", None) or []
            _steer_all_voice = bool(_steer_media_urls) and (
                len(self._pending_event_audio_paths(event)) == len(_steer_media_urls)
            )
            if steer_text and (plain_text or _steer_all_voice) and agent_live and hasattr(running_agent, "steer"):
                steered = self._try_agent_verb(running_agent, "steer", steer_text, session_key)
            if not steered:
                effective_mode = "queue"
        elif (
            effective_mode == "interrupt" and plain_text and agent_live
            and getattr(running_agent, "_supports_active_turn_redirect", False) is True
            and hasattr(running_agent, "redirect")
        ):
            redirected = self._try_agent_verb(
                running_agent, "redirect", (event.text or "").strip(), session_key
            )
        return self._BusySteerOutcome(
            effective_mode=effective_mode, demoted_for_subagents=demoted_for_subagents,
            demoted_for_compression=demoted_for_compression, steered=steered, redirected=redirected,
        )

    @staticmethod
    def _demote_interrupt(session_key: str, why: str) -> str:
        logger.info("Demoting busy_input_mode 'interrupt' to 'queue' for session %s because %s", session_key, why)
        return "queue"

    @staticmethod
    def _try_agent_verb(running_agent, verb: str, text: str, session_key: str) -> bool:
        """Call ``running_agent.<verb>(text)`` (steer/redirect); False + warning on failure."""
        try:
            return bool(getattr(running_agent, verb)(text))
        except Exception as exc:
            logger.warning("Gateway %s failed for session %s: %s", verb, session_key, exc)
            return False

    async def _interrupt_running_agent_for_busy_event(self, event: MessageEvent, adapter, running_agent) -> None:
        """Interrupt mode: abort in-flight tool calls; the agent loop exits at its next check point."""
        from gateway.run import _build_media_placeholder
        try:
            _interrupt_text = event.text
            _media_urls = getattr(event, "media_urls", None) or []
            if self._pending_event_audio_paths(event):
                _interrupt_text, _ = await self._transcribe_and_echo_pending_voice(
                    event, adapter, event.source, event.text or "", log_context="Voice-busy-interrupt",
                )
            elif not _interrupt_text and _media_urls:
                _interrupt_text = _build_media_placeholder(event)
            running_agent.interrupt(_interrupt_text)
        except Exception:
            pass  # don't let interrupt failure block the ack

    def _busy_steer_ack_enabled(self, event: MessageEvent, session_key: str) -> bool:
        # Some mobile chat setups want silent steering — keep the behavior, drop the bubble.
        from gateway.run import _load_gateway_config, _platform_config_key
        from gateway.display_config import resolve_display_setting
        steer_ack_env = os.environ.get("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED")
        if steer_ack_env is not None:
            steer_ack_enabled = steer_ack_env.strip().lower() in {"1", "true", "yes", "on"}
        else:
            steer_ack_enabled = bool(
                resolve_display_setting(
                    _load_gateway_config(), _platform_config_key(event.source.platform),
                    "busy_steer_ack_enabled", True,
                )
            )
        if not steer_ack_enabled:
            logger.debug("Busy steer ack suppressed for session %s", session_key)
        return steer_ack_enabled

    _BUSY_DEMOTED_TAIL = (
        " — your message is queued for when it finishes (use /stop to cancel everything)."
    )

    def _compose_busy_ack_message(
        self, event: MessageEvent, now: float, _busy_state, running_agent: Any, *,
        is_steer_mode: bool, is_queue_mode: bool, is_redirect_mode: bool,
        demoted_for_subagents: bool, demoted_for_compression: bool,
    ) -> str:
        from gateway.run import (
            _AGENT_PENDING_SENTINEL, _hermes_home, _load_gateway_config, _platform_config_key
        )
        from gateway.display_config import resolve_display_setting

        # Terse by default; iteration/tool detail opts in via display.platforms.<p>.busy_ack_detail.
        status_parts = []
        busy_ack_detail_enabled = bool(
            resolve_display_setting(
                _load_gateway_config(), _platform_config_key(event.source.platform),
                "busy_ack_detail", True,
            )
        )
        if busy_ack_detail_enabled and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                summary = running_agent.get_activity_summary()
                elapsed_min = 0
                if _busy_state and _busy_state.turn.started_ts:
                    elapsed_min = int((now - _busy_state.turn.started_ts) / 60)
                if elapsed_min > 0:
                    status_parts.append(f"{elapsed_min} min elapsed")
                if summary.get("max_iterations", 0):
                    status_parts.append(
                        f"iteration {summary.get('api_call_count', 0)}/{summary.get('max_iterations', 0)}"
                    )
                if summary.get("current_tool"):
                    status_parts.append(f"running: {summary.get('current_tool')}")
            except Exception:
                pass
        status_detail = f" ({', '.join(status_parts)})" if status_parts else ""
        if is_steer_mode:
            head, tail = "⏩ Steered into current run", ". Your message arrives after the next tool call."
        elif is_redirect_mode:
            head, tail = "↪ Redirected current run", ". I'll adjust using your correction."
        elif is_queue_mode and demoted_for_subagents:
            # Explain the demotion: the follow-up didn't kill the subagent; /stop is the escape hatch.
            head, tail = "⏳ Subagent working", self._BUSY_DEMOTED_TAIL
        elif is_queue_mode and demoted_for_compression:
            head, tail = "⏳ Compressing context", self._BUSY_DEMOTED_TAIL
        elif is_queue_mode:
            head, tail = "⏳ Queued for the next turn", ". I'll respond once the current task finishes."
        else:
            head, tail = "⚡ Interrupting current task", ". I'll respond to your message shortly."
        message = f"{head}{status_detail}{tail}"

        # One-time onboarding hint about the queue/interrupt knob (flag persisted to config.yaml).
        try:
            from agent.onboarding import (BUSY_INPUT_FLAG, busy_input_hint_gateway, is_seen, mark_seen)
            if not is_seen(_load_gateway_config(), BUSY_INPUT_FLAG):
                _hint_mode = (
                    "steer" if is_steer_mode
                    else "queue" if is_queue_mode
                    else "redirect" if is_redirect_mode
                    else "interrupt"
                )
                message = f"{message}\n\n{busy_input_hint_gateway(_hint_mode)}"
                mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
        except Exception as _onb_err:
            logger.debug("Failed to apply busy-input onboarding hint: %s", _onb_err)
        return message

    async def _send_busy_ack_reply(self, event: MessageEvent, adapter, message: str) -> None:
        try:
            await self._send_busy_reply(event, adapter, message)
        except Exception as e:
            logger.debug("Failed to send busy-ack: %s", e)

    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        # Same authorization gate as the cold path, else unauthorized users in shared threads
        # inject messages into a session they don't own.
        from gateway.run import _AGENT_PENDING_SENTINEL
        # See #17775.
        if not self._is_user_authorized(event.source):
            logger.warning(
                "Dropping message from unauthorized user in active session: "
                "user=%s (%s), platform=%s, session=%s", event.source.user_id, event.source.user_name,
                event.source.platform.value if event.source.platform else "unknown", session_key,
            )
            return True  # handled (silently dropped); do not fall through

        effective_mode = self._effective_busy_input_mode(event.source)
        if self._draining:  # gateway restarting/stopping
            await self._send_busy_drain_notice(event, session_key, effective_mode)
            return True
        if await self._route_plaintext_approval_while_busy(event, session_key):
            return True
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return False  # let default path handle it
        # Internal synthetic events (delegation / background completions) must never interrupt or
        # steer; they surface as a NEW turn when idle. Plugin events carry untrusted payload text, so
        # queue them through the FIFO (security metadata kept apart).
        if getattr(event, "internal", False):
            if event.allow_gateway_control:
                return False
            self._queue_or_replace_pending_event(session_key, event)
            return True
        if (
            event.message_type == MessageType.TEXT
            and self._effective_busy_text_mode(event.source) == "queue"
            and effective_mode != "steer"
        ):
            return False

        _busy_state = self._peek_session_state(session_key)
        running_agent = _busy_state.turn.agent if _busy_state else None
        _steer = await self._resolve_busy_steer_or_redirect(event, session_key, effective_mode, running_agent)
        effective_mode, redirected = _steer.effective_mode, _steer.redirected
        # Queue as the next turn — skipped after a successful steer/redirect (the text is already in
        # the run and must NOT replay). FIFO gives each text its own turn (raw merge would join them).
        if not _steer.steered and not redirected:
            self._queue_or_replace_pending_event(session_key, event)
        # Store the message so it's processed as the next turn after the current run finishes (or is
        # interrupted). Skip this for a successful steer — the text already landed inside the run and must
        # NOT also be replayed as a next-turn user message. Route through _queue_or_replace_pending_event
        # (the same FIFO infrastructure used by busy queue-mode and /queue) rather than a raw
        # merge_pending_message_event(merge_text=True). The raw merge newline-joins consecutive TEXT
        # follow-ups into a SINGLE pending turn, destroying message boundaries — so two separate user
        # messages sent while the agent was busy (interrupt mode, or a steer that fell back to queue)
        # arrived as one mashed-together turn (#43066 sub-bug 2). The FIFO path gives each text its own turn
        # in arrival order while still preserving photo-burst / album merge semantics for media.
        is_queue_mode = effective_mode == "queue"
        is_steer_mode = effective_mode == "steer"
        is_redirect_mode = effective_mode == "interrupt" and redirected
        if (
            effective_mode == "interrupt" and not redirected
            and running_agent and running_agent is not _AGENT_PENDING_SENTINEL
        ):
            await self._interrupt_running_agent_for_busy_event(event, adapter, running_agent)

        # Disabled ack: still process input. Checked before debounce so an undelivered ack never
        # stamps the "last ack" timestamp.
        if os.environ.get("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true").lower() != "true":
            logger.debug("Busy ack suppressed for session %s", session_key)
            return True  # input still processed, just no ack sent

        # Debounce (30s) before the config-heavy display lookup.
        now = time.time()
        if now - (_busy_state.turn.busy_ack_ts if _busy_state else 0) < 30:
            return True  # interrupt sent (if not queue), ack already delivered recently

        if is_steer_mode and not self._busy_steer_ack_enabled(event, session_key):
            return True

        self._session_state(session_key).turn.busy_ack_ts = now

        message = self._compose_busy_ack_message(
            event, now, _busy_state, running_agent, is_steer_mode=is_steer_mode,
            is_queue_mode=is_queue_mode, is_redirect_mode=is_redirect_mode,
            demoted_for_subagents=_steer.demoted_for_subagents,
            demoted_for_compression=_steer.demoted_for_compression,
        )
        await self._send_busy_ack_reply(event, adapter, message)
        return True

    # Slash name → handler method is ``_handle_<name>_command`` (``-`` → ``_``) except these.
    _COMMAND_HANDLER_ALIASES = {"bg": "_handle_background_command", "sethome": "_handle_set_home_command"}
    # Ordinary slash handlers shared by idle and busy dispatch.
    _PLAIN_COMMANDS = (
        "status", "context", "restart", "approve", "deny", "pause", "agents", "bg", "btw",
        "kanban", "subgoal", "heartbeat", "busy", "yolo", "verbose", "footer", "help",
        "commands", "profile", "update", "version",
    )
    # Dispatched only on the idle path (busy dispatch has its own allowlist).
    _IDLE_COMMANDS = (
        "topic", "whoami", "platform", "stop", "reasoning", "memory", "skills", "fast",
        "approvals", "model", "codex-runtime", "personality", "suggestions", "save", "retry",
        "sethome", "compress", "usage", "topup", "insights", "reload-mcp", "reload-skills",
        "bundles", "debug", "title", "resume", "sessions", "branch", "rollback", "diff", "goal",
        "loop", "refine", "review", "voice",
    )

    def _command_handler_table(self, names) -> Dict[str, Any]:
        return {
            name: getattr(
                self, self._COMMAND_HANDLER_ALIASES.get(name, f"_handle_{name.replace('-', '_')}_command"),
            )
            for name in names
        }

    def _gateway_plain_command_handlers(self):
        """Return ordinary slash handlers shared by idle and busy dispatch."""
        return self._command_handler_table(self._PLAIN_COMMANDS)

    async def _send_command_ack(self, source, text: str, label: str) -> None:
        """Best-effort acknowledgment for a slash command that falls through to agent processing."""
        try:
            adapter = self._adapter_for_source(source)
            if adapter:
                await adapter.send(
                    str(source.chat_id), text, metadata=self._thread_metadata_for_source(source)
                )
        except Exception:
            logger.debug("%s ack send failed", label, exc_info=True)

    def _gateway_idle_command_handlers(self):
        """Slash handlers dispatched only on the idle path (busy dispatch has its own allowlist)."""
        return self._command_handler_table(self._IDLE_COMMANDS)

    # busy_handler key (hermes_cli/commands.py CommandDef) → mid-run variant ``_busy_<key>_command``.
    _BUSY_SPECIAL_HANDLERS: Dict[str, str] = {
        k: f"_busy_{k}_command" for k in ("start", "stop", "new", "queue", "steer", "egress", "goal", "loop")
    }

    async def _dispatch_busy_slash_command(self, event: MessageEvent, cmd_def, quick_key: str, source):
        """Dispatch a recognized slash command while an agent is running.

        Order: ``busy_handler`` (mid-run variant) → ``busy_policy == "dispatch"`` (normal handler)
        → catch-all reject text. Rejecting is required rather than falling through to
        interrupt + discard: commands like /model, /reasoning, /voice, /insights, /title,
        /resume, /retry, /undo, /compress, /usage, /reload-mcp, /sethome, /reset (all
        registered as Discord slash commands) would interrupt the agent AND get silently
        discarded by the slash-command safety net, producing a zero-char response.
        See #5057, #6252, #10370.

        1. ``busy_handler`` — special mid-run variant (e.g. /goal's control-verb whitelist, /queue's FIFO
        enqueue, /model's custom reject text). 2. 3. See #5057, #6252, #10370.
        """
        name = cmd_def.name
        policy = getattr(cmd_def, "busy_policy", "reject")
        handler_key = getattr(cmd_def, "busy_handler", None)
        if handler_key:
            special = self._BUSY_SPECIAL_HANDLERS.get(handler_key)
            if special is not None:
                return await getattr(self, special)(event, quick_key, source)
            reject_text = self._BUSY_REJECT_TEXT.get(handler_key)
            if reject_text is not None:
                return reject_text
        if policy in ("dispatch", "interrupt_then_dispatch"):
            plain = self._gateway_plain_command_handlers().get(name)
            if plain is not None:
                return await plain(event)
            logger.warning(
                "busy_policy=%s for /%s has no mid-run handler — "
                "falling back to busy-reject", policy, name,
            )

        return (
            f"⏳ Agent is running — `/{name}` can't run "
            f"mid-turn. Wait for the current response or `/stop` first."
        )

    async def _handle_pause_command(self, event: MessageEvent):
        """`/pause [reason]` engages the global emergency stop; `/pause off` lifts it (the estop gate
        lets slash commands through while paused so messaging-only operators are never locked out)."""
        from agent import estop
        args = (event.get_command_args() or "").strip()
        if args.lower() in {"off", "resume", "stop", "disengage"}:
            if estop.disengage():
                return "▶️ Resumed — new work is accepted again."
            return "Hermes wasn't paused."
        state = estop.get_state()
        if state is not None and not args:
            suffix = f" (reason: {state.get('reason')})" if state.get("reason") else ""
            return f"⏸️ Hermes is already paused{suffix}. Use `/pause off` to resume."
        estop.engage(reason=args or None)
        suffix = f" (reason: {args})" if args else ""
        return (
            f"⏸️ Paused{suffix}. New cron/kanban/gateway work is on hold; "
            "in-flight work finishes normally. Use `/pause off` to resume."
        )

    async def _busy_start_command(self, event: MessageEvent, quick_key: str, source):
        # Telegram's /start is a platform ping (bot launch/deep-link), not a user command.
        logger.info("Ignoring /start platform ping for active session %s", quick_key)
        return ""

    async def _busy_egress_command(self, event: MessageEvent, quick_key: str, source):
        from hermes_cli.proxy_cli import format_status_text
        return format_status_text()

    async def _busy_stop_command(self, event: MessageEvent, quick_key: str, source):
        # Hard-kill: a soft interrupt can't reach a truly hung executor thread.
        from gateway.run import _INTERRUPT_REASON_STOP
        await self._interrupt_and_clear_session(
            quick_key, source, interrupt_reason=_INTERRUPT_REASON_STOP, invalidation_reason="stop_command",
        )
        logger.info("STOP for session %s — agent interrupted, session lock released", quick_key)
        return EphemeralReply(t("gateway.stop.stopped"))

    async def _busy_new_command(self, event: MessageEvent, quick_key: str, source):
        # /reset and /new bypass the running-agent guard (else they'd queue as user text and replay
        # into the same broken history); clear pending messages so the old text doesn't replay.
        from gateway.run import _INTERRUPT_REASON_RESET
        # Interrupt the agent first, then clear the adapter's pending queue so the stale "/reset" text
        # doesn't get re-processed as a user message after the interrupt completes. See #2170.
        await self._interrupt_and_clear_session(
            quick_key, source, interrupt_reason=_INTERRUPT_REASON_RESET, invalidation_reason="new_command",
        )
        return await self._handle_reset_command(event)

    async def _busy_queue_command(self, event: MessageEvent, quick_key: str, source):
        # Each /queue is its own full agent turn, run FIFO after the current run; never merged.
        queued_text = event.get_command_args().strip()
        # A /queue carrying media or reply context is valid with no prompt text (image caption).
        has_media = bool(getattr(event, "media_urls", None))
        if not queued_text and not has_media:
            return "Usage: /queue <prompt>"
        adapter = self._adapter_for_source(source)
        if adapter:
            self._enqueue_fifo(quick_key, MessageEvent(
                text=queued_text, message_type=event.message_type if has_media else MessageType.TEXT,
                source=event.source, raw_message=event.raw_message, message_id=event.message_id,
                media_urls=list(getattr(event, "media_urls", []) or []),
                media_types=list(getattr(event, "media_types", []) or []),
                media_text_inlined=list(getattr(event, "media_text_inlined", []) or []),
                reply_to_message_id=event.reply_to_message_id, reply_to_text=event.reply_to_text,
                reply_to_author_id=event.reply_to_author_id,
                reply_to_author_name=event.reply_to_author_name,
                reply_to_is_own_message=event.reply_to_is_own_message, auto_skill=event.auto_skill,
                channel_prompt=event.channel_prompt, channel_context=event.channel_context,
                internal=event.internal, timestamp=event.timestamp,
            ), adapter)
        depth = self._queue_depth(quick_key, adapter=adapter)
        return "Queued for the next turn." + (f" ({depth} queued)" if depth > 1 else "")

    async def _busy_steer_command(self, event: MessageEvent, quick_key: str, source):
        # /steer lands BETWEEN tool-call iterations of the same run (appended to the last tool
        # result) — no interrupt, no new user turn, no role-alternation violation.
        from gateway.run import _AGENT_PENDING_SENTINEL
        steer_text = event.get_command_args().strip()
        if not steer_text:
            return "Usage: /steer <prompt>"
        _steer_state = self._peek_session_state(quick_key)
        running_agent = _steer_state.turn.agent if _steer_state else None

        def _queue_fallback(reply: str) -> str:
            # Turn-boundary fallback: queue the steer text as its own follow-up turn.
            adapter = self._adapter_for_source(source)
            if adapter:
                self._enqueue_fifo(quick_key, MessageEvent(
                    text=steer_text, message_type=MessageType.TEXT, source=event.source,
                    message_id=event.message_id, channel_prompt=event.channel_prompt,
                    channel_context=event.channel_context,
                ), adapter)
            return reply

        if running_agent is _AGENT_PENDING_SENTINEL:
            return _queue_fallback("Agent still starting — /steer queued for the next turn.")
        if not running_agent or not hasattr(running_agent, "steer"):
            return _queue_fallback("No active agent — /steer queued for the next turn.")
        try:
            accepted = running_agent.steer(steer_text)
        except Exception as exc:
            logger.warning("Steer failed for session %s: %s", quick_key, exc)
            return f"⚠️ Steer failed: {exc}"
        if not accepted:
            return "Steer rejected (empty payload)."
        preview = steer_text[:60] + ("..." if len(steer_text) > 60 else "")
        return f"⏩ Steer queued — arrives after the next tool call: '{preview}'"

    async def _busy_goal_command(self, event: MessageEvent, quick_key: str, source):
        # Control verbs are safe mid-run (state only); setting new goal text is rejected so we don't
        # race a second continuation against the current turn. wait/gate take an argument.
        _goal_arg = (event.get_command_args() or "").strip().lower()
        _goal_verb = _goal_arg.split(None, 1)[0] if _goal_arg else ""
        if (
            not _goal_arg
            or _goal_arg in {"status", "pause", "resume", "clear", "stop", "done", "unwait"}
            or _goal_verb in {"wait", "gate"}
        ):
            return await self._handle_goal_command(event)
        return "Agent is running — use /goal status / pause / clear / wait mid-run, or /stop before setting a new goal."

    async def _busy_loop_command(self, event: MessageEvent, quick_key: str, source):
        # Mirrors /goal: control verbs are safe mid-run; a new loop is rejected.
        _loop_arg = (event.get_command_args() or "").strip().lower()
        if not _loop_arg or _loop_arg in {"status", "pause", "resume", "stop", "clear", "cancel", "help", "--help", "-h"}:
            return await self._handle_loop_command(event)
        return "Agent is running — use /loop status / pause / stop mid-run, or /stop before setting a new loop."

    def _check_slash_access(self, source: SessionSource, canonical_cmd: str) -> Optional[str]:
        """Denial message if ``source`` cannot run ``canonical_cmd``, else None (both dispatch paths
        use it so an in-flight agent can't bypass admin gating; no ``allow_admin_from`` → None)."""
        from gateway.slash_access import policy_for_source as _policy_for_source
        if not canonical_cmd:
            return None
        policy = _policy_for_source(self.config, source)
        if not policy.enabled or policy.can_run(source.user_id, canonical_cmd):
            return None
        logger.info(
            "Slash command /%s denied for %s:%s (not admin, not in user_allowed_commands)",
            canonical_cmd, source.platform.value if source.platform else "?", source.user_id,
        )
        allowed_preview = sorted(policy.user_allowed_commands)
        if allowed_preview:
            suffix = (
                "You can run: " + ", ".join(f"/{c}" for c in allowed_preview[:12])
                + ("…" if len(allowed_preview) > 12 else "") + ". Use /whoami for the full list."
            )
        else:
            suffix = (
                "No slash commands are enabled for non-admins on this platform. Ask an admin to "
                "add you to allow_admin_from or to set user_allowed_commands."
            )
        return f"⛔ /{canonical_cmd} is admin-only here. {suffix}"

    def _sibling_thread_run_keys(self, source: SessionSource, own_key: str) -> list:
        """Running-agent keys of OTHER participants in the same thread (per-user thread mode keys
        are ``...:{thread_id}:{user_id}``, so another user's run is invisible to the caller's own
        ``/stop``). Excludes the pending sentinel and ``own_key``; callers still gate on authz."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        thread_id = getattr(source, "thread_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not thread_id or not chat_id:
            return []
        platform = source.platform.value
        chat_type = getattr(source, "chat_type", None) or ""
        # Match the exact key or prefix + ":" so a thread id that merely starts with this one
        # is not matched.
        prefix = ":".join(["agent:main", platform, chat_type, str(chat_id), str(thread_id)])
        return [
            key
            for key, agent in self._running_agent_items()
            if key != own_key
            and agent is not _AGENT_PENDING_SENTINEL and agent
            and (key == prefix or key.startswith(prefix + ":"))
        ]

    def _is_stale_restart_redelivery(self, event: MessageEvent) -> bool:
        """True if this /restart is a Telegram re-delivery we already handled.

        The previous gateway wrote ``.restart_last_processed.json`` (platform + update_id). A
        /restart with update_id <= that value is a redelivery when this process booted from that
        restart; otherwise the marker must be < 5 minutes old. Telegram only (numeric ordering).
        """
        from gateway.run import _hermes_home
        if event is None or event.source is None or event.platform_update_id is None:
            return False
        try:
            if event.source.platform.value != "telegram":
                return False
        except Exception:
            return False

        try:
            marker_path = _hermes_home / ".restart_last_processed.json"
            if not marker_path.exists():
                # Missing marker: a redelivered /restart would otherwise re-restart forever. Suppress
                # ONLY when this process booted from a chat /restart AND is within a short post-boot
                # window; consume the flag one-shot so a later legitimate /restart is honored.
                if (
                    # Belt-and-suspenders for when the dedup marker goes missing (manually cleaned up, or
                    # the previous cycle's write failed). Without a marker the update_id comparison below
                    # can't run, so a redelivered /restart would sail through and re-restart the gateway —
                    # an infinite loop (issue #18528).
                    getattr(self, "_booted_from_restart", False)
                    and time.time() - getattr(self, "_startup_time", 0.0) < 60
                ):
                    self._booted_from_restart = False
                    return True
                return False
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        recorded_uid = data.get("update_id")
        if (
            data.get("platform") != "telegram"
            or not isinstance(recorded_uid, int)
            or event.platform_update_id > recorded_uid
        ):
            return False

        # A service-managed restart can outlast the 5-minute trust window; consume the boot
        # signal one-shot.
        if getattr(self, "_booted_from_restart", False):
            self._booted_from_restart = False
            return True

        # Staleness guard: an old marker (crash recovery) must not swallow a fresh /restart.
        requested_at = data.get("requested_at")
        return not (isinstance(requested_at, (int, float)) and time.time() - requested_at > 300)

    async def _handle_suggestions_command(self, event: MessageEvent) -> str:
        """/suggestions via the shared handler (origin = event source so jobs deliver back here)."""
        from gateway.run import _command_origin_for_source
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command
            return handle_suggestions_command(
                (event.get_command_args() or "").strip(),
                origin=_command_origin_for_source(event.source), surface="gateway",
            )
        except Exception as e:
            logger.debug("suggestions command failed: %s", e)
            return f"Suggestions command failed: {e}"

    async def _handle_blueprint_command(self, event: MessageEvent):
        """/blueprint via the shared handler (origin = event source so jobs deliver back here)."""
        from gateway.run import _command_origin_for_source
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command
            return handle_blueprint_command(
                (event.get_command_args() or "").strip(),
                origin=_command_origin_for_source(event.source), surface="gateway",
            )
        except Exception as e:
            logger.debug("blueprint command failed: %s", e)
            from hermes_cli.blueprint_cmd import BlueprintCommandResult
            return BlueprintCommandResult(f"Cron blueprint command failed: {e}")

    async def _maybe_confirm_destructive_slash(
        self, *, event: MessageEvent, command: str, title: str, detail: str, execute
    ) -> Union[str, "EphemeralReply", None]:
        """Gate a destructive session slash command (/new, /reset, /undo).

        ``execute()`` (async → str | EphemeralReply) runs immediately when
        ``approvals.destructive_slash_confirm`` is off; otherwise via ``_request_slash_confirm``:
        ``once`` runs it, ``always`` persists the opt-out then runs it, ``cancel`` skips it.
        """
        confirm_required = True
        try:
            approvals = self._read_user_config().get("approvals")
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass
        if not confirm_required:
            return await execute()

        session_key = self._session_key_for_source(event.source)

        async def _on_confirm(choice: str):
            # Via the class, not ``self``: tests drive this gate on a bare SimpleNamespace runner.
            return await GatewayBusySessionMixin._run_confirmed_destructive_slash(
                choice, command, execute, session_key
            )

        _p = self._typed_command_prefix_for(event.source.platform)
        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            f"_Text fallback: reply `{_p}approve`, `{_p}always`, or `{_p}cancel`._"
        )
        return await self._request_slash_confirm(
            event=event, command=command, title=title, message=prompt_message, handler=_on_confirm
        )

    _DESTRUCTIVE_OPTOUT_NOTE = {
        True: (
            "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
            "without confirmation. Re-enable via "
            "`approvals.destructive_slash_confirm: true` in config.yaml."
        ),
        # The user did approve this run, so the action still goes ahead, but the preference did
        # not stick and the prompt will be back next time. Say so rather than promising an
        # opt-out that was never written.
        False: (
            "\n\n⚠️ Could not save that preference (config.yaml is not "
            "writable), so /clear, /new, /reset, and /undo will ask "
            "again next time. To silence it permanently, set "
            "`approvals.destructive_slash_confirm: false` in config.yaml."
        ),
    }

    @staticmethod
    async def _run_confirmed_destructive_slash(choice: str, command: str, execute, session_key: str):
        """Confirm-callback body: ``cancel`` → message; ``always`` persists the opt-out, then runs."""
        if choice == "cancel":
            return f"🟡 /{command} cancelled. Conversation unchanged."
        persisted = False
        if choice == "always":
            try:
                from cli import save_config_value
                # save_config_value swallows its own errors and reports the outcome in the return
                # value, so the try block alone says nothing about whether the write landed.
                persisted = bool(save_config_value("approvals.destructive_slash_confirm", False))
                if persisted:
                    logger.info("User opted out of destructive slash confirm (session=%s)", session_key)
                else:
                    logger.warning(
                        "Could not persist destructive_slash_confirm=false "
                        "(session=%s); config.yaml is not writable", session_key,
                    )
            except Exception as exc:
                logger.warning("Failed to persist destructive_slash_confirm=false: %s", exc)
        result = await execute()
        # Only plain-string results get the note: it would mangle an EphemeralReply.
        if choice == "always" and isinstance(result, str):
            return result + GatewayBusySessionMixin._DESTRUCTIVE_OPTOUT_NOTE[persisted]
        return result

    async def _request_slash_confirm(
        self, *, event: MessageEvent, command: str, title: str, message: str, handler
    ) -> Optional[str]:
        """Ask the user to confirm a slash command; ``handler(choice)`` runs on "once"/"always"/
        "cancel" and its return is sent as a message. Returns None if buttons rendered, else the
        text-fallback message (which IS the ack)."""
        from tools import slash_confirm as _slash_confirm_mod
        source = event.source
        session_key = self._session_key_for_source(source)
        # object.__new__ test runners lack the counter; fall back to a local one.
        counter = getattr(self, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = self._slash_confirm_counter = _itertools.count(1)
        confirm_id = f"{next(counter)}"

        # Register FIRST so a fast button click cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self._adapter_for_source(source)
        metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

        if adapter is not None:
            try:
                button_result = await adapter.send_slash_confirm(
                    chat_id=source.chat_id, title=title, message=message, session_key=session_key,
                    confirm_id=confirm_id, metadata=metadata,
                )
                if button_result and getattr(button_result, "success", False):
                    return None  # buttons rendered — no redundant text ack
            except Exception as exc:
                logger.debug("send_slash_confirm failed for %s on %s: %s", command, source.platform, exc)
        # Text fallback — the prompt message itself is the direct reply.
        return message

    def _read_user_config(self) -> Dict[str, Any]:
        """Raw config.yaml for gate lookups that must see on-disk changes without a restart."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception:
            return {}
        return cfg if isinstance(cfg, dict) else {}

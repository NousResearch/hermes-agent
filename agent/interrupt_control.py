"""Interrupt / steer / redirect control surface for ``AIAgent``.

Soft/hard interrupt requests, tool-thread interrupt propagation, pending steer/redirect queues.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import contextlib
import logging
import threading
from typing import Optional

from agent.interrupt_compat import request_hard_interrupt
from tools.interrupt import set_interrupt as _set_interrupt

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


def _fence_cancel_before_commit(fence, *, when_in_flight: bool, failure_log: str) -> None:
    """Call ``type(fence).cancel_before_commit(fence)`` when ``commit_in_flight`` matches.

    Hard-cancel admission has two halves (#99758 P1). BEFORE the generation claim is
    consumed only a commit already in flight is waited out (the call blocks on the fence
    lock and returns False WITHOUT setting ``_cancelled``) — cancelling a still-pending
    fence there would be irreversible for an abort that may yet be declined. AFTER the
    claim survived, only a still-pending commit is cancelled; one that started meanwhile
    owns the fence and completes on its own."""
    if fence is None or bool(getattr(fence, "commit_in_flight", False)) is not when_in_flight:
        return
    cancel_before_commit = getattr(type(fence), "cancel_before_commit", None)
    if callable(cancel_before_commit):
        try:
            cancel_before_commit(fence)
        except Exception:
            logger.debug(failure_log, exc_info=True)


def _ic_lock(agent, attr: str):
    """``with`` the lock stored at ``attr`` when present; __init__-less test stubs run unlocked."""
    lock = getattr(agent, attr, None)
    return contextlib.nullcontext() if lock is None else lock


def _ic_slot(agent, lock_attr: str, slot: str):
    """Read the pending-text ``slot`` guarded by ``lock_attr``. An initialized agent always has both
    attributes, so under the lock the slot is read directly and a missing one fails loud (a real bug);
    only ``__init__``-less test stubs (no lock) get the ``getattr`` fallback."""
    if getattr(agent, lock_attr, None) is None:
        return getattr(agent, slot, None)
    return getattr(agent, slot)


def _ic_codex_method(agent, name: str):
    """Codex app-server owns its model/tool loop; return its ``name`` hook or None."""
    if getattr(agent, "api_mode", None) != "codex_app_server":
        return None
    method = getattr(getattr(agent, "_codex_session", None), name, None)
    return method if callable(method) else None


def _ic_abort_active_request(agent, reason: str, failure_log: str) -> None:
    """Shut the registered in-flight request's sockets (cron turns register their client here)."""
    abort = getattr(agent, "_active_request_abort", None)
    if callable(abort):
        try:
            abort(reason)
        except Exception:
            logger.debug(failure_log, exc_info=True)


def _ic_signal_tool_workers(agent, active: bool, **kw) -> None:
    """Fan the tool interrupt bit out to concurrent-tool worker tids.

    ``is_interrupted()`` inside a tool only sees its own tid, so without this a hung
    concurrent tool runs to its own timeout (and a stale bit could survive a turn
    boundary onto a recycled tid). getattr covers __init__-less stubs."""
    tracker = getattr(agent, "_tool_worker_threads", None)
    tracker_lock = getattr(agent, "_tool_worker_threads_lock", None)
    if tracker is None or tracker_lock is None:
        return
    with tracker_lock:
        worker_tids = list(tracker)
    for tid in worker_tids:
        try:
            _set_interrupt(active, tid, **kw)
        except Exception:
            pass


class InterruptControlMixin:
    """interrupt()/hard_interrupt()/clear_interrupt()/steer()/redirect() (see module docstring)."""

    def interrupt(
        self, message: Optional[str] = None, *, hard_cancel: bool = False,
        tool_reason: Optional[str] = None, require_generation: Optional[int] = None,
    ) -> bool:
        """Request the agent to interrupt its current tool-calling loop (call from another thread).

        ``hard_cancel``: explicit stop; compression may honor it even while ordinary interrupts are masked.
        ``tool_reason``: trusted fixed category safe for tool output. ``require_generation``: activity-
        generation claim — published only if the turn's generation still matches at the final mutation edge;
        returns False if the turn resumed meanwhile.
        """
        if require_generation is not None:
            # RESERVE the claim under the SAME lock `_touch_activity` stamps with; real progress invalidates
            # it and it is CONSUMED at the final mutation edge, so a resumed turn abandons the abort.
            with self._liveness_activity_lock():
                if getattr(self, "_turn_liveness_activity_generation", 0) != require_generation:
                    return False
                self._turn_liveness_abort_claim = require_generation

        # Tool cancellation attribution stays separate from _interrupt_message, which may carry the user's
        # full next message.
        tool_interrupt_reason = (
            (tool_reason or "explicit stop requested") if hard_cancel
            else ("user sent a new message" if message else "user interrupt")
        )

        def _publish_interrupt_state() -> None:
            self._interrupt_requested = True
            self._interrupt_message = message
            self._tool_interrupt_reason = tool_interrupt_reason
            _hard_event = getattr(self, "_hard_interrupt_requested", None) if hard_cancel else None
            if _hard_event is not None:
                _hard_event.set()

        def _fence():  # re-read each time: a finished commit may replace or clear the slot
            return vars(self).get("_active_compression_commit_fence") if hard_cancel else None

        # A hard stop and redirect share one lock so /stop cannot race with an accepted correction and
        # accidentally turn itself into a retry. The blocking in-flight-commit wait runs BEFORE the atomic
        # claim edge (redirect lock still held); the destructive pending-commit cancel runs AFTER the claim
        # survives (#99758 P1).
        with _ic_lock(self, "_pending_redirect_lock"):
            _fence_cancel_before_commit(
                _fence(), when_in_flight=True, failure_log="Compression hard-cancel fence wait failed"
            )
            if require_generation is None:
                # No claim to race: publish WITHOUT the liveness lock (bare AIAgent stand-ins in other
                # suites lack the liveness seam and would AttributeError).
                _publish_interrupt_state()
            else:
                # Final mutation edge: claim consumption and the FIRST observable publication are ONE
                # activity-lock critical section, so either the claim survives and commits before any later
                # activity stamp, or the stamp landed first and the abort declines without publishing.
                with self._liveness_activity_lock():
                    if getattr(self, "_turn_liveness_abort_claim", None) != require_generation:
                        return False
                    self._turn_liveness_abort_claim = None
                    _publish_interrupt_state()
            _fence_cancel_before_commit(
                _fence(), when_in_flight=False, failure_log="Compression hard-cancel fence admission failed"
            )
            self._pending_redirect = None

        # Codex watches a private interrupt event rather than Hermes' per-thread flag.
        _request_interrupt = _ic_codex_method(self, "request_interrupt")
        if _request_interrupt is not None:
            try:
                _request_interrupt()
            except Exception:
                logger.debug("Failed to interrupt Codex app-server turn", exc_info=True)

        # Cron turns request on the conversation thread (no nested interrupt-worker deadlock); their client
        # is registered here so this cross-thread interrupt can still shut the sockets.
        _ic_abort_active_request(self, "interrupt_abort", "Failed to abort active inline request")
        # Scope the tool interrupt to this agent's execution thread so other in-process agents are unaffected.
        if self._execution_thread_id is not None:
            _set_interrupt(True, self._execution_thread_id, reason=tool_interrupt_reason)
            self._interrupt_thread_signal_pending = False
        else:
            # Interrupt arrived before run_conversation bound the execution thread: defer the tool-level
            # signal instead of targeting the caller thread.
            self._interrupt_thread_signal_pending = True
        _ic_signal_tool_workers(self, True, reason=tool_interrupt_reason)
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                if hard_cancel:
                    request_hard_interrupt(child, message, tool_reason=tool_interrupt_reason)
                else:
                    child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
        return True

    def hard_interrupt(self, message: Optional[str] = None, *, tool_reason: Optional[str] = None) -> None:
        """Explicit stop preserving the ``interrupt()`` ABI (frontends feature-detect this and fall back to
        legacy ``interrupt()`` for third-party agents). Bypasses dynamic dispatch: legacy subclasses may
        override interrupt(message=None) without hard_cancel."""
        InterruptControlMixin.interrupt(self, message, hard_cancel=True, tool_reason=tool_reason)

    def clear_interrupt(self, *, preserve_redirect: bool = False) -> bool:
        """Clear the interrupt request and per-thread tool signal. ``preserve_redirect`` is only for the
        conversation loop rebuilding the same logical turn after cancelling a model request."""
        with _ic_lock(self, "_pending_redirect_lock"):
            if preserve_redirect and not _ic_slot(self, "_pending_redirect_lock", "_pending_redirect"):
                return False
            self._interrupt_requested = False
            self._interrupt_message = self._tool_interrupt_reason = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            if not preserve_redirect:
                self._pending_redirect = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        _ic_signal_tool_workers(self, False)
        # A hard interrupt supersedes any pending /steer — its target iteration will no longer happen.
        with _ic_lock(self, "_pending_steer_lock"):
            self._pending_steer = None
        return True

    def steer(self, text: str) -> bool:
        """Append user text to the LAST tool result once the batch finishes (no interrupt); multiple calls
        concatenate with newlines. Returns False for empty text."""
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        with _ic_lock(self, "_pending_steer_lock"):
            existing = _ic_slot(self, "_pending_steer_lock", "_pending_steer")
            self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
        return True

    def redirect(self, text: str) -> bool:
        """Redirect the active turn without converting it into a new task: during a model request only that
        request is cancelled (completed messages kept, partial reasoning becomes assistant context, the
        correction is appended as a real user message, the loop retries); during tool execution it degrades
        to ``steer()``; Codex app-server uses native ``turn/steer``. False when no live turn / empty text."""
        if not text or not text.strip():
            return False
        cleaned = text.strip()

        _native_steer = _ic_codex_method(self, "request_steer")
        if _native_steer is not None:
            with _ic_lock(self, "_pending_redirect_lock"):
                if self._interrupt_requested:
                    return False
            try:
                return bool(_native_steer(cleaned))
            except Exception:
                logger.debug("Codex app-server turn/steer failed", exc_info=True)
                return False

        # Never kill a tool to deliver guidance; the steer drain puts it on the final tool result.
        if getattr(self, "_executing_tools", False):
            return self.steer(cleaned)

        _model_active = getattr(self, "_model_request_active", None)
        with _ic_lock(self, "_pending_redirect_lock"):
            if _model_active is None or not _model_active.is_set():
                return False  # response completed before we got the lock: surface queues a new turn
            existing = _ic_slot(self, "_pending_redirect_lock", "_pending_redirect")
            if self._interrupt_requested and not existing:
                return False
            self._pending_redirect = (
                f"{existing}\n\n[Additional user correction]\n{cleaned}" if existing else cleaned
            )
            self._interrupt_requested = True
            self._interrupt_message = None

        # Interrupt only the model request — no fan-out to tool workers / child agents as interrupt() does.
        _execution_thread_id = getattr(self, "_execution_thread_id", None)
        if _execution_thread_id is not None:
            _set_interrupt(True, _execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_thread_signal_pending = True
        _ic_abort_active_request(self, "redirect_abort", "Failed to abort request for redirect")
        return True

    def _has_pending_redirect(self) -> bool:
        """Return whether an active-turn redirect is waiting to be applied."""
        with _ic_lock(self, "_pending_redirect_lock"):
            return bool(_ic_slot(self, "_pending_redirect_lock", "_pending_redirect"))

    def _drain_pending_redirect(self) -> Optional[str]:
        """Return and clear pending active-turn correction text."""
        with _ic_lock(self, "_pending_redirect_lock"):
            text = _ic_slot(self, "_pending_redirect_lock", "_pending_redirect")
            self._pending_redirect = None
        return text

    def _drain_pending_steer(self) -> Optional[str]:
        """Return the pending steer text (if any) and clear the slot; None when nothing is pending."""
        with _ic_lock(self, "_pending_steer_lock"):
            text = _ic_slot(self, "_pending_steer_lock", "_pending_steer")
            self._pending_steer = None
        return text

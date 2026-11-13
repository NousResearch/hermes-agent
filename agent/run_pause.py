"""Pause / resume a running turn at its safe boundaries.

``pause()`` only asks: the turn keeps running until it reaches the next iteration boundary
(after the current tool batch, before the next model request), so the in-flight tool call or
model request always finishes and nothing is aborted. ``begin_iteration`` then calls
``_wait_while_paused()``, which blocks the turn's own thread until ``resume()``, an interrupt
(``hard_interrupt`` / ``request_hard_interrupt`` / ``interrupt``) or the max-pause guard. A
steer accepted while paused stays buffered and drains in ``prepare_iteration`` after resume.

States: ``None`` (running) -> ``"pausing"`` (asked, not yet at a boundary) -> ``"paused"``
(blocked) -> ``None``. A pause belongs to the turn it was asked in: a turn that ends before
reaching a boundary drops it, so a cached agent never starts its next turn paused.

Wired in by the mixin base in ``run_agent.py`` and one call in ``agent/turn_iteration_prep.py``.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from agent.interrupt_compat import request_hard_interrupt

logger = logging.getLogger("run_agent")

# ``tool_reason`` of the auto-stop; the turn exit reads ``interrupted_by_system(run_pause_timeout)``.
PAUSE_TIMEOUT_REASON = "run pause timeout"
# How often a paused turn re-checks for an interrupt, and how often it stamps activity so the
# gateway's inactivity watchdog does not take a deliberate pause for a hung turn.
_POLL_SECONDS = 0.2
_ACTIVITY_TOUCH_SECONDS = 30.0

PauseListener = Callable[[str, dict], None]

_init_lock = threading.Lock()


def _cond(agent) -> threading.Condition:
    """The agent's pause condition, created on first use (no ``__init__`` change upstream)."""
    cond = vars(agent).get("_run_pause_cond")
    if cond is None:
        with _init_lock:
            cond = vars(agent).get("_run_pause_cond")
            if cond is None:
                cond = agent._run_pause_cond = threading.Condition()
    return cond


def _notify(listener: Optional[PauseListener], event: str, fields: dict) -> None:
    if listener is None:
        return
    try:
        listener(event, fields)
    except Exception:
        logger.debug("run pause listener failed for %s", event, exc_info=True)


class RunPauseMixin:
    """``pause()`` / ``resume()`` / ``pause_state()`` and the boundary wait (see module docstring)."""

    def pause_state(self) -> Optional[str]:
        """``"pausing"``, ``"paused"`` or ``None``. A pause left over from an earlier turn reads ``None``."""
        with _cond(self):
            status = vars(self).get("_run_pause_status")
            if status is None or vars(self).get("_run_pause_turn") != getattr(self, "_current_turn_id", None):
                return None
            return status

    def pause(self, *, max_seconds: Optional[float] = None, listener: Optional[PauseListener] = None) -> bool:
        """Ask the running turn to pause at its next boundary (call from another thread).

        ``max_seconds`` > 0 auto-stops the turn once it has been paused that long. ``listener(event, fields)``
        receives ``run.paused`` (from the turn's thread, when it blocks) and ``run.resumed``. False when the
        turn is already pausing/paused, is being interrupted, or runs a loop Hermes does not own (Codex
        app-server)."""
        if getattr(self, "api_mode", None) == "codex_app_server":
            return False
        with _cond(self):
            if getattr(self, "_interrupt_requested", False) or self.pause_state() is not None:
                return False
            self._run_pause_status = "pausing"
            self._run_pause_turn = getattr(self, "_current_turn_id", None)
            self._run_pause_max_seconds = max_seconds if max_seconds and max_seconds > 0 else None
            self._run_pause_listener = listener
            self._run_pause_epoch = vars(self).get("_run_pause_epoch", 0) + 1
        logger.info("Run pause requested (turn %s)", self._run_pause_turn)
        return True

    def resume(self) -> bool:
        """Let a pausing or paused turn continue. False when nothing is pausing or paused."""
        cond = _cond(self)
        with cond:
            previous = self.pause_state()
            if previous is None:
                return False
            listener = vars(self).get("_run_pause_listener")
            self._run_pause_status = None
            self._run_pause_listener = None
            cond.notify_all()
        logger.info("Run resumed (was %s)", previous)
        _notify(listener, "run.resumed", {"previous_status": previous})
        return True

    def _wait_while_paused(self) -> None:
        """Iteration boundary: block while this turn is paused. Returns on resume, on an interrupt (the
        caller's interrupt check then ends the turn) or after the max-pause guard stopped the turn."""
        cond = _cond(self)
        with cond:
            if vars(self).get("_run_pause_status") is None:
                return
            if self.pause_state() is None or getattr(self, "_interrupt_requested", False):
                self._run_pause_status = None  # earlier turn's pause, or a stop already won
                self._run_pause_listener = None
                return
            self._run_pause_status = "paused"
            epoch = self._run_pause_epoch
            max_seconds = self._run_pause_max_seconds
            listener = self._run_pause_listener
        paused_at = time.time()
        started = last_touch = time.monotonic()
        _notify(listener, "run.paused", {"paused_at": paused_at, "max_pause_seconds": max_seconds})
        logger.info("Run paused at iteration boundary")
        timed_out = False
        while True:
            with cond:
                if vars(self).get("_run_pause_status") != "paused" or self._run_pause_epoch != epoch:
                    return  # resumed (resume() already notified)
                if getattr(self, "_interrupt_requested", False):
                    break
                now = time.monotonic()
                if max_seconds is not None and now - started >= max_seconds:
                    timed_out = True
                    break
                cond.wait(_POLL_SECONDS)
            if time.monotonic() - last_touch >= _ACTIVITY_TOUCH_SECONDS:
                last_touch = time.monotonic()
                touch = getattr(self, "_touch_activity", None)
                if callable(touch):
                    try:
                        touch("paused via run pause")
                    except Exception:
                        logger.debug("activity touch while paused failed", exc_info=True)
        with cond:
            if self._run_pause_epoch == epoch:
                self._run_pause_status = None
                self._run_pause_listener = None
        if timed_out:
            logger.warning("Run paused longer than %ss; stopping it", max_seconds)
            request_hard_interrupt(self, tool_reason=PAUSE_TIMEOUT_REASON)


def wait_while_paused(agent: Any) -> None:
    """``begin_iteration``'s hook; a no-op for agents without the mixin (test doubles)."""
    wait = getattr(type(agent), "_wait_while_paused", None)
    if callable(wait):
        wait(agent)

"""``/after sleep|hibernate|shutdown|cancel`` — one power action once all agent work is done.

Armed per task, never persisted: a forgotten setting must not power the machine off after every
turn, and a restart disarms it. "Done" is the same picture scale-to-zero uses before it suspends
a gateway — no live turn, no deferred background review, no async delegation, no background
process or pending watcher — held for a grace window, because follow-up turns (``process_complete``,
background review) start seconds after a turn ends. Then a countdown is announced to every surface
that armed it; ``/after cancel`` or any new work during it aborts.

The live-turn count is process-wide (``agent.review_idle_queue.QUEUE``), so this only works in the
process that runs turns: the classic CLI and the messaging gateway, not the TUI/Desktop slash worker.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ACTIONS = ("sleep", "hibernate", "shutdown")
_GRACE_S = 120.0
_COUNTDOWN_S = 60.0
_POLL_S = 5.0
_VERB = {"sleep": "Sleeping", "hibernate": "Hibernating", "shutdown": "Shutting down"}
_USAGE = "Usage: /after sleep | hibernate | shutdown | cancel"

Notifier = Callable[[str], None]


def _work_in_flight() -> bool:
    """Background work a turn left running (it can start a follow-up turn when it finishes)."""
    from agent.review_idle_queue import QUEUE
    from tools.async_delegation import active_count
    from tools.process_registry import process_registry
    return bool(QUEUE.pending_count() or active_count() or process_registry.has_any_active()
                or process_registry.pending_watchers)


def _quiet_for() -> float:
    from agent.review_idle_queue import QUEUE
    return QUEUE.quiet_for()


def _command_for(action: str) -> Optional[list[str]]:
    """argv for ``action`` on this host; None where Windows does it in-process. Graceful only:
    apps with unsaved work may still veto, which is the point (no forced power-off)."""
    from hermes_platform.host import facts
    os_family = facts.os_family()
    if os_family == "win32":
        return None if action != "shutdown" else ["shutdown", "/s", "/t", "0"]
    if os_family == "darwin":
        return {"sleep": ["pmset", "sleepnow"],
                "shutdown": ["osascript", "-e", 'tell application "System Events" to shut down']}[action]
    return ["systemctl", {"sleep": "suspend", "hibernate": "hibernate", "shutdown": "poweroff"}[action]]


def unsupported_reason(action: str) -> Optional[str]:
    """Why ``action`` cannot run on this host, or None."""
    from hermes_platform.host import facts
    from hermes_platform.host.runtime import is_container, is_termux, is_wsl
    if is_wsl() or is_container() or is_termux():
        return "Hermes is running inside WSL, a container or Termux, so it cannot power off the host."
    if action == "hibernate" and facts.os_family() == "darwin":
        return "macOS has no on-demand hibernate; use /after sleep or /after shutdown."
    return None


def _run_os_action(action: str) -> None:
    argv = _command_for(action)
    if argv is None:
        import ctypes
        # SetSuspendState(hibernate, force, disable_wake_events)
        ctypes.windll.powrprof.SetSuspendState(action == "hibernate", False, False)
        return
    subprocess.run(argv, stdin=subprocess.DEVNULL, check=False, timeout=30)


def _spawn_watcher(target: Callable[[], None]) -> None:
    from agent.memory_provider import spawn_context_thread
    spawn_context_thread(target, name="power-after").start()


class PowerAfter:
    """Process-wide armed action + the watcher thread that fires it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action: Optional[str] = None
        self._notifiers: list[Notifier] = []
        self._wake = threading.Event()
        self._running = False  # a watcher thread owns the armed action
        # Test seams — replaced by unit tests, never in production.
        self._now: Callable[[], float] = time.monotonic
        self._quiet_for: Callable[[], float] = _quiet_for
        self._busy: Callable[[], bool] = _work_in_flight
        self._fire: Callable[[str], None] = _run_os_action
        self._start: Callable[[Callable[[], None]], None] = _spawn_watcher

    def armed(self) -> Optional[str]:
        with self._lock:
            return self._action

    def arm(self, action: str, notify: Optional[Notifier] = None) -> None:
        with self._lock:
            self._action = action
            if notify is not None:
                self._notifiers.append(notify)
            spawn, self._running = not self._running, True
        self._wake.set()
        if spawn:
            self._start(self._watch)

    def cancel(self) -> Optional[str]:
        with self._lock:
            action, self._action = self._action, None
            self._notifiers.clear()
        self._wake.set()
        return action

    def _idle(self) -> bool:
        return self._quiet_for() >= _GRACE_S and not self._busy()

    def _notify(self, text: str) -> None:
        with self._lock:
            notifiers = list(self._notifiers)
        for notify in notifiers:
            try:
                notify(text)
            except Exception:
                logger.debug("/after notification failed", exc_info=True)

    def _sleep(self, seconds: float) -> None:
        self._wake.wait(seconds)
        self._wake.clear()

    def _countdown(self, action: str) -> bool:
        """True when the countdown ran out still armed for ``action`` and still idle."""
        deadline = self._now() + _COUNTDOWN_S
        while self._now() < deadline:
            self._sleep(min(_POLL_S, deadline - self._now()))
            if self.armed() != action:
                return False
            if not self._idle():
                self._notify(f"New work started, so /after {action} waits until it is done.")
                return False
        return True

    def _take(self, action: str) -> Optional[list[Notifier]]:
        """Disarm and release the watcher iff still armed for ``action`` (atomic with cancel/arm);
        returns the notifiers so a failed action can still be reported."""
        with self._lock:
            if self._action != action:
                return None
            notifiers, self._notifiers = self._notifiers, []
            self._action, self._running = None, False
            return notifiers

    def _watch_until_fired(self) -> bool:
        while (action := self.armed()) is not None:
            if not self._idle():
                self._sleep(_POLL_S)
                continue
            self._notify(f"All agent work is done. {_VERB[action]} in {_COUNTDOWN_S:.0f}s; "
                         "send /after cancel to stop.")
            if self._countdown(action) and (notifiers := self._take(action)) is not None:
                logger.info("/after: running %s", action)
                try:
                    self._fire(action)
                except Exception as exc:
                    logger.warning("/after %s failed", action, exc_info=True)
                    for notify in notifiers:
                        notify(f"/after {action} failed: {exc}")
                return True
        return False

    def _watch(self) -> None:
        while True:
            try:
                if self._watch_until_fired():
                    return
            except Exception:
                logger.warning("/after watcher failed; disarming", exc_info=True)
                self.cancel()
            with self._lock:  # re-armed after the loop's last check: keep watching
                if self._action is None:
                    self._running = False
                    return


POWER_AFTER = PowerAfter()


def run_command(args: str, notify: Optional[Notifier] = None) -> str:
    """The one handler behind ``/after`` on the CLI and messaging surfaces."""
    arg = (args or "").strip().lower()
    if arg in ("", "status"):
        armed = POWER_AFTER.armed()
        return (f"Armed: /after {armed} runs once all agent work is done." if armed
                else f"Nothing armed. {_USAGE}")
    if arg in ("cancel", "off", "stop"):
        cancelled = POWER_AFTER.cancel()
        return f"Cancelled /after {cancelled}." if cancelled else "Nothing was armed."
    if arg not in ACTIONS:
        return _USAGE
    if reason := unsupported_reason(arg):
        return reason
    POWER_AFTER.arm(arg, notify)
    return (f"Armed: {_VERB[arg].lower()} once all agent work is done "
            f"(after {_GRACE_S / 60:.0f} min idle and a {_COUNTDOWN_S:.0f}s countdown). "
            "Send /after cancel to stop.")

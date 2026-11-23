"""Hold start-of-session maintenance until the first frame is on screen.

Python's GIL means a background thread is not free: while it imports or scans files, the main
thread that is building the first frame waits its turn. Work the first screen does not need
(skill sync, prewarms for the model picker and the agent runtime) therefore starts here, after
the prompt has been painted once, instead of racing the paint.

Nothing is dropped: a process that never paints (a one-shot ``-q`` run, a handed-off TUI, a broken
terminal) runs the held work after ``FALLBACK_SECONDS``, and an unarmed process runs it at once.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

FALLBACK_SECONDS = 6.0

_lock = threading.Lock()
_armed = False
_released = False
_pending: list[tuple[Callable[[], None], contextvars.Context]] = []
_timer: threading.Timer | None = None


def arm() -> None:
    """An interactive prompt is about to paint: hold work until :func:`release` (its first frame)."""
    global _armed
    with _lock:
        _armed = True


def arm_for(args: Any) -> None:
    """:func:`arm` for an interactive launch. A ``-q`` / ``--query-file`` run answers and exits, so it never paints."""
    if not (getattr(args, "query", None) or getattr(args, "query_file", None)):
        arm()


def defer_thread(target: Callable[[], None], *, name: str) -> None:
    """Start a daemon thread running *target* once the first frame is out, under the caller's profile scope.

    The scope is the one in force HERE: ``defer_until_first_paint`` runs the start in the context captured
    at this call, so it holds whether the frame's ``after_render`` or the fallback timer's thread releases it."""
    def _start() -> None:
        from agent.memory_provider import spawn_context_thread

        spawn_context_thread(target, name=name).start()

    defer_until_first_paint(_start)


def defer_until_first_paint(start: Callable[[], None]) -> None:
    """Call *start* after the first frame, or now when no frame is coming (unarmed) or it was painted.

    A held *start* runs in a copy of the caller's contextvars (the profile scope among them): the thread that
    releases it, the fallback timer's in particular, has none of its own."""
    global _timer
    with _lock:
        hold = _armed and not _released
        if hold:
            _pending.append((start, contextvars.copy_context()))
            if _timer is None:
                _timer = threading.Timer(FALLBACK_SECONDS, release)
                _timer.daemon = True
                _timer.start()
    if not hold:
        _run(start)


def release(_app: Any = None) -> None:
    """Run the held work once. Usable as a prompt_toolkit ``after_render`` handler."""
    global _released, _timer
    with _lock:
        if _released:
            return
        _released = True
        held = list(_pending)
        _pending.clear()
        timer, _timer = _timer, None
    if timer is not None:
        timer.cancel()
    for start, context in held:
        _run(start, context)


def _run(start: Callable[[], None], context: contextvars.Context | None = None) -> None:
    try:
        start() if context is None else context.run(start)
    except Exception:  # health: allow BLE001 -- maintenance must never break a launch
        logger.debug("deferred startup work failed", exc_info=True)

"""Locate and format the event-loop thread's stack for watchdog / stall dumps.

``faulthandler.dump_traceback(all_threads=True)`` emits at most 100 threads, newest first
(CPython caps that list), so in a gateway process running a few hundred threads the loop thread —
created before every worker, hence the oldest — is dropped from each dump. That is exactly the
thread whose stack explains a frozen loop, and the dump then leads with a collateral thread
(a terminal wait, an idle reader) that says nothing about the stall.

:func:`capture_loop_thread_ident` records the ident while the loop can still answer,
:func:`format_loop_thread_block` renders that thread's stack, and dump sites emit it **first**,
ahead of the truncated all-threads dump. Everything here is best-effort: a diagnostics path must
never raise into the code it is diagnosing.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
from typing import IO, Optional

__all__ = [
    "LOOP_THREAD_HEADER",
    "capture_loop_thread_ident",
    "format_loop_thread_block",
    "guess_loop_thread_ident",
    "thread_name",
    "write_loop_thread_block",
]

LOOP_THREAD_HEADER = "=== LOOP THREAD ==="

# A running loop is always inside one of these frames, however deep the blocking call is, so a
# stack search finds it when no ident was captured beforehand.
_LOOP_FRAME_NAMES = frozenset({"_run_once", "run_forever"})


def _current_frames() -> dict:
    try:
        return sys._current_frames()
    except Exception:  # pragma: no cover - every supported interpreter has _current_frames
        return {}


def guess_loop_thread_ident(frames: Optional[dict] = None) -> Optional[int]:
    """Ident of the thread whose stack contains the asyncio run loop, when one is present.

    This is the fallback for a dump taken before any ident was captured: the loop thread is a
    frame inside ``run_forever`` / ``_run_once`` even while it is blocked in a synchronous call.
    """
    for ident, frame in (_current_frames() if frames is None else frames).items():
        current = frame
        while current is not None:
            if current.f_code.co_name in _LOOP_FRAME_NAMES:
                return ident
            current = current.f_back
    return None


def capture_loop_thread_ident(loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[int]:
    """Ident of ``loop``'s thread, resolved without scheduling anything on it.

    Called ON the loop it is exact — the common case, since the gateway arms its watchdog from a
    startup phase running on the loop. From any other thread (or before the loop has started) it
    falls back to the running-loop stack search, and to ``None`` when no loop thread exists yet;
    the dump path repeats that search at dump time, which is when it actually matters.
    """
    try:
        if loop is not None and asyncio.get_running_loop() is loop:
            return threading.get_ident()
    except RuntimeError:
        pass  # no loop on this thread
    return guess_loop_thread_ident()


def thread_name(ident: Optional[int]) -> Optional[str]:
    """Name of the live thread with ``ident``, if it is still around."""
    if ident is None:
        return None
    for thread in threading.enumerate():
        if thread.ident == ident:
            return thread.name
    return None


def format_loop_thread_block(ident: Optional[int] = None, *, frames: Optional[dict] = None) -> str:
    """``LOOP_THREAD_HEADER`` plus the loop thread's stack; always returns a block.

    A dump that could not locate the thread says so explicitly instead of silently omitting it,
    so the absence is never mistaken for a healthy loop.
    """
    snapshot = _current_frames() if frames is None else frames
    resolved = ident if ident is not None else guess_loop_thread_ident(snapshot)
    if resolved is None:
        return (
            f"{LOOP_THREAD_HEADER}\n"
            "  (loop thread not located: no ident supplied and no thread is inside the asyncio "
            "run loop)\n"
        )
    name = thread_name(resolved)
    label = f"ident={resolved}" + (f" name={name!r}" if name else "")
    frame = snapshot.get(resolved)
    if frame is None:
        label += " (no Python frame: thread already exited, or is blocked in a C call)"
        return f"{LOOP_THREAD_HEADER}\n{label}\n"
    return f"{LOOP_THREAD_HEADER}\n{label}\n" + "".join(traceback.format_stack(frame))


def write_loop_thread_block(stream: IO[str], ident: Optional[int] = None) -> None:
    """Write :func:`format_loop_thread_block` to ``stream``; never raises."""
    try:
        stream.write(format_loop_thread_block(ident))
        stream.flush()
    except Exception:
        pass

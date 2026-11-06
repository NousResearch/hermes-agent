"""Content-free attribution for a canonical RPC sampled during a missed loop probe.

One lazy process-wide daemon; no tasks, request objects, locals or source lines are
retained. A suspended RPC is NOT the current caller: match its frame identity on
the owner's live thread stack before emitting. Elapsed is dispatch wall time,
not a measurement of continuous on-loop time or proof of the historical blocker.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import logging
from pathlib import Path
import re
import sys
import threading
import time
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)
_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_MAX_ACTIVE = 128
_MAX_WALK = 128
_MAX_FRAMES = 16
_POLL_SECONDS = 0.25
_STALL_SECONDS = 1.0
_LOG_INTERVAL_SECONDS = 30.0
_NAME = re.compile(r"[A-Za-z0-9_.<>-]{1,80}\Z")


def _location(frame):
    """Never call traceback/linecache or format a frame's locals/arguments."""
    code = frame.f_code
    try:
        filename = Path(code.co_filename).relative_to(_SOURCE_ROOT).as_posix()
    except ValueError:
        return "<external>:<frame>:0"
    name = code.co_name
    if len(filename) > 160 or not all(_NAME.fullmatch(p) for p in filename.split("/")):
        return "<external>:<frame>:0"
    name = name if _NAME.fullmatch(name) else "<frame>"
    return f"{filename}:{name}:{frame.f_lineno}"


class _DispatchWatchdog:
    def __init__(self):
        self._condition = threading.Condition()
        self._active = {}
        self._thread = None
        self._closed = False
        self._last_log = float("-inf")

    def begin(self, method, frame, loop):
        with self._condition:
            if self._closed or len(self._active) >= _MAX_ACTIVE:
                return None
            if self._thread is None:
                thread = threading.Thread(
                    target=self._run, name="rpc-slow-dispatch", daemon=True
                )
                thread.start()
                self._thread = thread
            key = object()
            self._active[key] = (
                method,
                id(frame),
                threading.get_ident(),
                time.monotonic(),
                loop,
            )
            self._condition.notify()
            return key

    def end(self, key):
        with self._condition:
            self._active.pop(key, None)

    def close(self):
        with self._condition:
            self._closed = True
            self._active.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(1.0)
        return thread is None or not thread.is_alive()

    def _run(self):
        probes = WeakKeyDictionary()
        next_sample = time.monotonic() + _POLL_SECONDS
        while True:
            with self._condition:
                if self._closed:
                    return
                if not self._active:
                    self._condition.wait()
                    continue
                remaining = next_sample - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                active = dict(self._active)
            try:
                self._sample(active, probes)
                next_sample = time.monotonic() + _POLL_SECONDS
            except Exception:  # health: allow BLE001 -- optional diagnostics must never recursively log failures
                # A failed frame API or logging handler is not a handler failure.
                # Disable this daemon rather than repeatedly entering a broken sink.
                with self._condition:
                    self._closed = True
                    self._active.clear()
                return

    def _sample(self, active, probes):
        now = time.monotonic()
        loops = {record[4] for record in active.values()}
        for loop, (_, ack) in tuple(probes.items()):
            if loop not in loops and (loop.is_closed() or ack.is_set()):
                del probes[loop]
        overdue = set()
        for loop in loops:
            if loop not in probes and len(probes) >= _MAX_ACTIVE:
                continue
            sent, ack = probes.get(loop, (now, None))
            if ack is None or ack.is_set():
                ack = threading.Event()
                probes[loop] = (now, ack)
                if not loop.is_closed():
                    loop.call_soon_threadsafe(
                        ack.set
                    )  # at most one outstanding probe per loop
            elif now - sent >= _STALL_SECONDS:
                overdue.add(loop)
        if not overdue or now - self._last_log < _LOG_INTERVAL_SECONDS:
            return
        by_thread = {}
        for key, record in active.items():
            if record[4] in overdue:
                by_thread.setdefault(record[2], {})[record[1]] = (key, record)
        frames = sys._current_frames()
        try:
            for tid, candidates in by_thread.items():
                frame, locations = frames.get(tid), []
                for _ in range(_MAX_WALK):
                    if frame is None:
                        break
                    if len(locations) < _MAX_FRAMES:
                        locations.append(_location(frame))
                    found = candidates.get(id(frame))
                    if found is not None:
                        key, (method, _, _, started, _) = found
                        with self._condition:
                            if key not in self._active or self._closed:
                                break
                            self._last_log = now
                        logger.warning(
                            "slow_dispatch method=%s dispatch_elapsed_ms=%d stack=%s",
                            method,
                            max(0, int((now - started) * 1000)),
                            " <- ".join(locations),
                        )
                        return
                    frame = frame.f_back
        finally:
            # Do not retain frames (and their request-bearing locals) between polls.
            frames.clear()
            frame = None


_watchdog = _DispatchWatchdog()
atexit.register(_watchdog.close)


@contextmanager
def watch_dispatch(method):
    """``method`` is a source-declared contract name, never the incoming wire value."""
    key = None
    try:
        if method is not None and logger.isEnabledFor(logging.WARNING):
            import asyncio

            key = _watchdog.begin(method, sys._getframe(2), asyncio.get_running_loop())
    except Exception:  # health: allow BLE001 -- no request or diagnostic failure may change RPC behavior
        pass
    try:
        yield
    finally:
        if key is not None:
            _watchdog.end(key)

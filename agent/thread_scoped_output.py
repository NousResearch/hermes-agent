"""Thread-scoped stdout/stderr silencing for background worker threads.

``contextlib.redirect_stdout``/``redirect_stderr`` reassign the *process-global*
``sys.stdout``/``sys.stderr``.  When a daemon worker thread (e.g. the background
memory/skill review) wraps its whole body in those context managers, every other
thread in the process — including a gateway's asyncio event-loop thread driving a
Telegram long-poll — sees ``sys.stdout``/``sys.stderr`` pointing at ``devnull``
for the full duration.  Any bare ``print`` / ``sys.stderr.write`` from those other
threads is silently lost during that window (see issue #55769 / #55925).

This module installs a thin proxy as ``sys.stdout``/``sys.stderr`` that routes
writes per-thread: threads registered as "silenced" go to a sink; every other
thread passes through to the *original* stream.  The proxy is installed once,
idempotently, and is never uninstalled (uninstalling would race other threads
mid-write), so the only observable effect for unregistered threads is one extra
attribute lookup per write.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Iterator, TextIO

__all__ = ["thread_scoped_silence"]

_install_lock = threading.Lock()
# Maps the proxy we installed for a given attribute ("stdout"/"stderr") so we
# never double-wrap and so we can recover the original stream.
_installed: dict[str, "_ThreadRoutingStream"] = {}


class _ThreadRoutingStream:
    """A ``sys.stdout``/``sys.stderr`` stand-in that routes writes per-thread.

    Threads whose ident is in ``_silenced`` write to ``_sink``; all other
    threads write to ``_passthrough`` (the original stream captured at install
    time).  Attribute access for anything other than the methods we override
    is delegated to the *current* target so things like ``.encoding`` /
    ``.fileno()`` behave like the underlying stream for the calling thread.
    """

    def __init__(self, passthrough: TextIO, sink: TextIO) -> None:
        self._passthrough = passthrough
        self._sink = sink
        # ident -> nesting depth.  A thread is silenced while depth > 0, so
        # nested ``thread_scoped_silence()`` on the same thread composes
        # correctly (the inner exit decrements rather than fully clearing).
        self._silenced: dict[int, int] = {}
        self._lock = threading.Lock()

    def _target(self) -> TextIO:
        if self._silenced.get(threading.get_ident(), 0) > 0:
            return self._sink
        return self._passthrough

    # --- registration -----------------------------------------------------
    def silence(self, ident: int) -> None:
        with self._lock:
            self._silenced[ident] = self._silenced.get(ident, 0) + 1

    def unsilence(self, ident: int) -> None:
        with self._lock:
            depth = self._silenced.get(ident, 0) - 1
            if depth > 0:
                self._silenced[ident] = depth
            else:
                self._silenced.pop(ident, None)

    # --- file-like surface ------------------------------------------------
    def write(self, data):  # type: ignore[no-untyped-def]
        try:
            return self._target().write(data)
        except Exception:
            return len(data) if isinstance(data, str) else 0

    def flush(self):  # type: ignore[no-untyped-def]
        try:
            return self._target().flush()
        except Exception:
            return None

    def writelines(self, lines):  # type: ignore[no-untyped-def]
        target = self._target()
        try:
            return target.writelines(lines)
        except Exception:
            return None

    def isatty(self) -> bool:
        try:
            return bool(self._target().isatty())
        except Exception:
            return False

    def fileno(self):  # type: ignore[no-untyped-def]
        return self._target().fileno()

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        # Delegate everything we don't override (encoding, buffer, mode, ...)
        # to the calling thread's current target.
        return getattr(self._target(), name)


def _ensure_installed(attr: str, sink: TextIO) -> "_ThreadRoutingStream":
    """Install (idempotently) a routing proxy as ``sys.<attr>`` and return it."""
    with _install_lock:
        proxy = _installed.get(attr)
        current = getattr(sys, attr, None)
        if proxy is not None and current is proxy:
            return proxy
        # Capture whatever is currently bound as the passthrough.  If a prior
        # global redirect_stdout is active we deliberately route non-silenced
        # threads to *that* (matching prior behaviour) rather than guessing at
        # the "real" stream.
        passthrough = current if current is not None else sink
        proxy = _ThreadRoutingStream(passthrough, sink)
        setattr(sys, attr, proxy)
        _installed[attr] = proxy
        return proxy


@contextlib.contextmanager
def thread_scoped_silence() -> Iterator[None]:
    """Silence ``stdout``/``stderr`` for the *current thread only*.

    Other threads keep writing to the real streams.  Use this around a worker
    thread's body instead of ``contextlib.redirect_stdout(devnull)`` when the
    process is multi-threaded and another thread must keep its console output.
    """
    sink = open(os.devnull, "w", encoding="utf-8")
    ident = threading.get_ident()
    out_proxy = _ensure_installed("stdout", sink)
    err_proxy = _ensure_installed("stderr", sink)
    out_proxy.silence(ident)
    err_proxy.silence(ident)
    try:
        yield
    finally:
        out_proxy.unsilence(ident)
        err_proxy.unsilence(ident)
        try:
            sink.close()
        except Exception:
            pass

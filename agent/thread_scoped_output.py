"""Thread-scoped stdout/stderr silencing for background worker threads.

``contextlib.redirect_stdout`` reassigns the *process-global* stream, so a daemon worker
silencing itself also silences every other thread (gateway event loop included). This
module installs a per-thread routing proxy as ``sys.stdout``/``sys.stderr``: silenced
threads write to a sink, everyone else passes through to the original stream. Installed
once, idempotently, and never uninstalled (that would race other threads mid-write).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Iterator, TextIO

__all__ = ["stdio_chain", "stdio_chain_find", "thread_scoped_silence"]

# Only these Hermes proxy types have known passthrough semantics. An unrelated
# wrapper may expose the same attribute names without forwarding writes through it.
_PROXY_LINKS = (
    ("agent.process_bootstrap", "_SafeWriter", "_inner"),
    ("hermes_logging", "_RedactingStdio", "_stream"),
)
_MAX_CHAIN = 64


def _proxy_link_attr(stream) -> str | None:  # type: ignore[no-untyped-def]
    if isinstance(stream, _ThreadRoutingStream):
        return "_passthrough"
    # Read loaded modules only: importing process_bootstrap here would activate
    # bootstrap dependencies on the thread-silencing hot path.
    for module_name, class_name, attr in _PROXY_LINKS:
        module = sys.modules.get(module_name)
        cls = vars(module).get(class_name) if module is not None else None
        if isinstance(cls, type) and isinstance(stream, cls):
            return attr
    return None


def stdio_chain(stream):  # type: ignore[no-untyped-def]
    """Walk known Hermes stdio proxies, rejecting cycles and excessive depth."""
    seen: set[int] = set()
    for _ in range(_MAX_CHAIN):
        if stream is None:
            return
        identity = id(stream)
        if identity in seen:
            raise ValueError("stdio proxy cycle detected")
        seen.add(identity)
        yield stream
        attr = _proxy_link_attr(stream)
        if attr is None:
            return
        try:
            stream = object.__getattribute__(stream, attr)
        except AttributeError as exc:
            raise ValueError(f"stdio proxy is missing {attr}") from exc
    raise ValueError(f"stdio proxy depth exceeds {_MAX_CHAIN}")


def stdio_chain_find(stream, cls):  # type: ignore[no-untyped-def]
    """Find a proxy while validating the full known chain before reuse."""
    found = None
    for link in stdio_chain(stream):
        if found is None and isinstance(link, cls):
            found = link
    return found

_install_lock = threading.Lock()
# Proxy installed per attribute ("stdout"/"stderr"): never double-wrap.
_installed: dict[str, _ThreadRoutingStream] = {}
# One process-lifetime sink per stream: global redirects that displace and
# restore a proxy must not leak a new /dev/null descriptor each time.
_sinks: dict[str, TextIO] = {}
_routing_states: dict[str, _RoutingState] = {}


class _RoutingState:
    """Silencing registry shared by every proxy generation for one stream."""

    def __init__(self, sink: TextIO) -> None:
        self.sink = sink
        self.silenced: dict[int, int] = {}
        self.lock = threading.Lock()


class _ThreadRoutingStream:
    """``sys.stdout``/``sys.stderr`` stand-in routing writes per calling thread;
    unknown attributes delegate to the current thread's target."""

    def __init__(self, passthrough: TextIO, state: _RoutingState) -> None:
        self._passthrough = passthrough
        self._state = state

    def _target(self) -> TextIO:
        return self._state.sink if self._state.silenced.get(threading.get_ident(), 0) > 0 else self._passthrough

    def silence(self, ident: int) -> None:
        with self._state.lock:
            self._state.silenced[ident] = self._state.silenced.get(ident, 0) + 1

    def unsilence(self, ident: int) -> None:
        with self._state.lock:
            depth = self._state.silenced.get(ident, 0) - 1
            if depth > 0:
                self._state.silenced[ident] = depth
            else:
                self._state.silenced.pop(ident, None)

    def _forward(self, name: str, fallback, *args):  # type: ignore[no-untyped-def]
        """Call ``name`` on the current target; a dead target yields ``fallback(*args)`` instead of raising."""
        try:
            return getattr(self._target(), name)(*args)
        except (OSError, ValueError):
            return fallback(*args)

    def write(self, data):  # type: ignore[no-untyped-def]
        return self._forward("write", lambda d: len(d) if isinstance(d, str) else 0, data)

    def flush(self):  # type: ignore[no-untyped-def]
        return self._forward("flush", lambda: None)

    def writelines(self, lines):  # type: ignore[no-untyped-def]
        return self._forward("writelines", lambda _l: None, lines)

    def isatty(self) -> bool:
        try:
            return bool(self._target().isatty())
        except Exception:
            return False

    def fileno(self):  # type: ignore[no-untyped-def]
        return self._target().fileno()

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        return getattr(self._target(), name)


def _ensure_installed(attr: str, passthrough: TextIO) -> _ThreadRoutingStream:
    """Install (idempotently) a routing proxy as ``sys.<attr>`` and return it."""
    with _install_lock:
        proxy = _installed.get(attr)
        current = getattr(sys, attr, None)
        existing = stdio_chain_find(current, _ThreadRoutingStream)
        if existing is not None:
            # A redirect context may restore an older proxy, or another installer
            # may have wrapped ours; adopt it (writes still pass through it)
            # rather than wrapping the chain again.
            _installed[attr] = existing
            _routing_states[attr] = existing._state
            return existing
        if proxy is not None and current is proxy:
            return proxy
        # Route non-silenced threads to whatever is currently bound (an active
        # global redirect keeps its old behavior).
        passthrough = current if current is not None else passthrough
        sink = _sinks.get(attr)
        if sink is None or sink.closed:
            sink = _sinks[attr] = open(os.devnull, "w", encoding="utf-8")
        state = _routing_states.get(attr)
        if state is None or state.sink is not sink:
            state = _routing_states[attr] = _RoutingState(sink)
        proxy = _installed[attr] = _ThreadRoutingStream(passthrough, state)
        setattr(sys, attr, proxy)
        return proxy


@contextlib.contextmanager
def thread_scoped_silence() -> Iterator[None]:
    """Silence ``stdout``/``stderr`` for the *current thread only*."""
    ident = threading.get_ident()
    proxies = (_ensure_installed("stdout", sys.__stdout__ or sys.stdout), _ensure_installed("stderr", sys.__stderr__ or sys.stderr))
    for proxy in proxies:
        proxy.silence(ident)
    try:
        yield
    finally:
        for proxy in proxies:
            proxy.unsilence(ident)

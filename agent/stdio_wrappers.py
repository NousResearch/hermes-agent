"""Crash-resistant stdio wrappers shared by the two stdio installers.

Dependency-free leaf: ``agent.process_bootstrap`` and
``agent.thread_scoped_output`` both import this module at module scope, so it
must not import either of them (or anything reaching ``hermes_bootstrap``).
Importing the bootstrap chain from inside a stdio installer runs
module-scope dependency activation — real-HERMES_HOME filesystem I/O and
sys.path mutation — at an arbitrary hot per-thread moment, and risks
import-lock deadlock under threads.
"""

from __future__ import annotations


class _SafeWriter:
    """Transparent stdio wrapper swallowing OSError/ValueError from broken pipes.

    Headless runs (systemd, Docker) lose the stdout pipe → ``OSError: [Errno 5]``;
    subagent threads can see the shared handle close → ``ValueError``. Either
    would otherwise crash the agent (often via double-fault in an except handler).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _unwrap_stdio_stream(stream, routing_stream_type):
    """Descend through Hermes stdio wrappers to the innermost actual stream.

    Returns ``(innermost, routing)`` where ``routing`` is the outermost
    ``routing_stream_type`` layer seen (None if none), so callers can preserve
    thread-routing semantics instead of wrapping over them. Cycle-safe: descent
    stops at an already-visited object, so a corrupt chain cannot hang the
    bootstrap. The routing type is passed in because importing
    ``agent.thread_scoped_output`` from here would make this leaf part of a
    cycle.
    """
    visited = set()
    routing = None
    while isinstance(stream, (_SafeWriter, routing_stream_type)) and id(stream) not in visited:
        visited.add(id(stream))
        if routing is None and isinstance(stream, routing_stream_type):
            routing = stream
        stream = stream._inner if isinstance(stream, _SafeWriter) else stream._passthrough
    return stream, routing

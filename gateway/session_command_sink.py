"""What a messaging command handler would have SENT, captured for a local ``slash.exec``.

The messaging gateway's handlers deliver two things outside their return value: an ack line
(``_send_command_ack``) and a next turn pushed onto the adapter FIFO (``/goal``'s kickoff,
``_enqueue_goal_turn``). A local session's adapter has no transport and no FIFO reader: its turns
are client-submitted admissions with the client's own durable input id. So while ``slash.exec``
runs a handler for a local session, both are captured here and returned to the client as one
``send`` directive (``notice`` + ``message``); the client submits it, the gateway never does.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_SINK: ContextVar = ContextVar('local_command_sink', default=None)


@dataclass
class CommandSink:
    notices: list[str] = field(default_factory=list)
    prompt: str | None = None
    display: str | None = None


def active_sink() -> CommandSink | None:
    return _SINK.get()


@contextmanager
def capture():
    sink = CommandSink()
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)

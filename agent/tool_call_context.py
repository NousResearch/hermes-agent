"""Task-local identity of a persisted tool-call occurrence, separate from provider IDs."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agent.message_metadata import index_tool_call_uids, resolve_tool_call_uid


@dataclass(frozen=True)
class ToolCallOccurrence:
    tool_name: str
    call_id: str
    uid: str


_PENDING: ContextVar[ToolCallOccurrence | None] = ContextVar("pending_tool_occurrence", default=None)
CURRENT_TOOL_CALL_UID: ContextVar[str] = ContextVar("current_tool_call_uid", default="")


def persisted_occurrence(messages: list, tool_name: str, call_id: str) -> ToolCallOccurrence | None:
    """Resolve once before worker handoff; never mint an identity during execution."""
    index: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            index_tool_call_uids(index, message)
    uid = resolve_tool_call_uid(index, call_id)
    return ToolCallOccurrence(tool_name, call_id, uid) if uid else None


@contextmanager
def bind_occurrence(occurrence: ToolCallOccurrence | None):
    token = _PENDING.set(occurrence)
    try:
        yield
    finally:
        _PENDING.reset(token)


@contextmanager
def dispatch_occurrence(tool_name: str, call_id: str):
    """Expose only this dispatch's UID; nested programmatic calls cannot inherit it."""
    occurrence = _PENDING.get()
    uid = occurrence.uid if occurrence and (occurrence.tool_name, occurrence.call_id) == (tool_name, call_id) else ""
    token = CURRENT_TOOL_CALL_UID.set(uid)
    try:
        with bind_occurrence(None):
            yield
    finally:
        CURRENT_TOOL_CALL_UID.reset(token)

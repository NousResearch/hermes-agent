"""Bounded, detached completed-turn handoff for opt-in memory providers.

This is host provenance, not authentication against hostile in-process Python.
Providers must independently verify committed rows before using durable evidence.
"""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class CompletedTurnSnapshot:
    session_id: str
    hermes_home: str
    payload: str

    def messages(self):
        return json.loads(self.payload)


def snapshot_completed_turn(messages, *, session_id, user_content, assistant_content):
    from hermes_constants import get_hermes_home

    if not isinstance(messages, list) or not messages:
        return None
    tail = messages[-128:]
    if any(type(m) is not dict for m in tail):
        return None
    start = next((i for i in range(len(tail) - 1, -1, -1)
                  if tail[i].get("role") == "user"), None)
    if start is None:
        return None
    turn = tail[start:]
    if (turn[0].get("content") != user_content
            or turn[-1].get("role") != "assistant"
            or turn[-1].get("content") != assistant_content
            or turn[-1].get("tool_calls")):
        return None
    if any(m.get("content") is not None and
           (not isinstance(m["content"], str) or len(m["content"]) > 64000)
           for m in turn):
        return None
    keys = ("id", "_row_id", "_db_persisted", "role", "content", "tool_name",
            "tool_call_id", "tool_calls", "session_id", "scope", "source_id", "source_event_id")
    selected = [{k: m[k] for k in keys if k in m} for m in turn]
    # Streaming encoding bounds the copy, including nested tool-call arguments.
    chunks, size = [], 0
    try:
        for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(selected):
            size += len(chunk)
            if size > 1_000_000:
                return None
            chunks.append(chunk)
    except (ValueError, TypeError, RecursionError):
        return None
    return CompletedTurnSnapshot(session_id, str(get_hermes_home().resolve()), "".join(chunks))

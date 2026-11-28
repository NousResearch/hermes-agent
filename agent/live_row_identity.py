"""The stored identity a running turn's live interim and tool events carry.

A tool round persists its assistant row before it emits the row's interim commentary or runs its calls
(agent/turn_tool_round.py), and the flush copies the store's outcome back onto the live dict
(``_row_id``, ``message_uid``, ``_tool_call_uids``; agent/session_persistence._db_flush_write). A viewer
that drew those events live can then hand each narration and tool card the stored row it is, and a
later history read replaces exactly those pieces instead of drawing them again.

``message_uid`` is the identity to key on: ``_row_id`` is a physical id that an in-place compaction
during the turn re-issues (agent/message_metadata.py), while the uid survives every copy. The row id
rides along for stores and readers that predate uids. Nothing is stamped for a dict that is not
durable yet: an event without these fields is what every older host sent.

The identity travels on an agent attribute instead of the callbacks' signatures, which many hosts
implement: the round sets it for the span of its interim emit and its tool execution and clears it after,
so a callback outside that span (a codex runtime's own tool items, a verify-on-stop interim) reads none.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from agent.message_metadata import MESSAGE_UID, TOOL_CALL_UIDS
from session_rules.turns import _DB_PERSISTED_MARKER

LIVE_ROW_ATTR = "_live_assistant_row"


def _durable(msg: Any) -> bool:
    return isinstance(msg, dict) and msg.get(_DB_PERSISTED_MARKER) is True and type(msg.get("_row_id")) is int


@contextmanager
def live_assistant_row(agent: Any, msg: Any) -> Iterator[None]:
    """Name ``msg`` as the assistant row the events fired inside this block belong to."""
    previous = getattr(agent, LIVE_ROW_ATTR, None)
    setattr(agent, LIVE_ROW_ATTR, msg if _durable(msg) else None)
    try:
        yield
    finally:
        setattr(agent, LIVE_ROW_ATTR, previous)


def _row_fields(msg: dict) -> dict[str, Any]:
    uid = msg.get(MESSAGE_UID)
    return {"row_id": msg["_row_id"], **({"message_uid": uid} if isinstance(uid, str) and uid else {})}


def interim_row_identity(agent: Any) -> dict[str, Any]:
    """``row_id`` / ``message_uid`` of the row an interim emitted now seals, or nothing."""
    msg = getattr(agent, LIVE_ROW_ATTR, None)
    return _row_fields(msg) if _durable(msg) else {}


def final_reply_row_identity(messages: Any, history: Any, raw: Any, status: str) -> dict[str, Any]:
    """The stored row of the reply ``message.complete`` delivers, or nothing.

    The completion receipt (agent/persisted_turn_receipt.py) withholds itself when it cannot address
    the whole turn, yet the reply's own row can still be durable: this names it on its own. Only a row
    this turn wrote qualifies (never one of ``history``'s), the transcript's last row, a plain assistant
    reply whose stored text is the delivered, non-empty text."""
    last = messages[-1] if isinstance(messages, list) and messages else None
    if (status != "complete" or not isinstance(raw, str) or not raw or not _durable(last) or last.get("role") != "assistant" or last.get("tool_calls")
            or last.get("content") != raw):
        return {}
    if any(before is last or (isinstance(before, dict) and before.get("_row_id") == last["_row_id"])
           for before in history or ()):
        return {}
    return _row_fields(last)


def reply_row_result_fields(result: Any, history: Any) -> dict[str, Any]:
    """The reply's stored row as scalar turn-result fields (``reply_row_id``, ``reply_message_uid``).

    A host that settles the turn away from the agent (the session authority's completion, a managed
    worker's result frame) carries it this way, and ``message.complete`` names it again through
    ``reply_row_payload``."""
    if not isinstance(result, dict):
        return {}
    status = "error" if result.get("failed") or result.get("interrupted") else "complete"
    named = final_reply_row_identity(result.get("messages"), history, result.get("final_response"), status)
    if not named:
        return {}
    return {"reply_row_id": named["row_id"],
            **({"reply_message_uid": named["message_uid"]} if "message_uid" in named else {})}


def reply_row_payload(result: Any) -> dict[str, Any]:
    """``row_id`` / ``message_uid`` for ``message.complete`` from ``reply_row_result_fields``, or nothing."""
    row = result.get("reply_row_id") if isinstance(result, dict) else None
    if type(row) is not int or row <= 0:
        return {}
    uid = result.get("reply_message_uid")
    return {"row_id": row, **({"message_uid": uid} if isinstance(uid, str) and uid else {})}


def tool_call_row_identity(agent: Any, tool_call_id: Any) -> dict[str, Any]:
    """The stored row that holds call ``tool_call_id`` (plus the call's own ``tool_call_uid``), or nothing
    when the running round's row does not hold that call."""
    msg = getattr(agent, LIVE_ROW_ATTR, None)
    if not _durable(msg) or not isinstance(tool_call_id, str) or not any(
            isinstance(call, dict) and call.get("id") == tool_call_id for call in msg.get("tool_calls") or ()):
        return {}
    uids = msg.get(TOOL_CALL_UIDS)
    call_uid = uids.get(tool_call_id) if isinstance(uids, dict) else None
    return {**_row_fields(msg), **({"tool_call_uid": call_uid} if isinstance(call_uid, str) and call_uid else {})}

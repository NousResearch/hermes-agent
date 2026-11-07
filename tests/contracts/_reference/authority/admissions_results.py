"""Stored-result codecs and the discarded-turn closer the admission reference runs inside settle and resolve.

Private pure helpers: ``compact_result``, ``encode_result`` and ``load_result``, the current-turn user re-anchor (its turn view through the conversation reference's policy shim) and ``close_discarded_turn`` over the reference's transcript tables, with the failed-turn copy of ``session_rules.turns``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from persistence.conversation.messages import NewMessage
from session_rules.turns import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE

from ..conversation import _turn_policy as policy
from ..conversation.transcript_store import _Transcript, insert_rows

_STORED = "_stored"


def _reanchor(messages: list[Any], user_message: Any) -> int:
    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        if msg.get("content") == user_message:
            return i
        view = policy.user_originated_turn_view(msg)
        if view is None:
            continue
        if view.get("content") == user_message:
            return i
        if fallback < 0:
            fallback = i
    return fallback


def compact_result(result: Mapping[str, Any], *, user_message: Any = None) -> dict[str, Any]:
    """Keep replay and accounting data and this turn's output, never cumulative history."""
    out = dict(result)
    value = out.get("result")
    if not isinstance(value, dict):
        return out
    value = dict(value)
    value.pop("tools", None)
    messages = value.get("messages")
    if isinstance(messages, list) and not value.get("_compressed") and not value.get("_messages_are_turn_suffix"):
        anchor = value.get("current_turn_user_idx")
        if not (type(anchor) is int and 0 <= anchor < len(messages) and isinstance(messages[anchor], dict) and messages[anchor].get("role") == "user"):
            anchor = _reanchor(messages, user_message)
        value["messages"] = messages[anchor + 1:]
        if anchor >= 0 and isinstance(messages[anchor], dict) and messages[anchor].get("api_content"):
            value["_turn_user_row"] = {k: messages[anchor][k] for k in ("role", "content", "api_content")}
        value["_messages_are_turn_suffix"] = True
        value.pop("current_turn_user_idx", None)
    out["result"] = value
    return out


def encode_result(result: Any) -> Any:
    """Store the reply once: the closing assistant message's content becomes null with a ``reply_ref``."""
    if not isinstance(result, dict):
        return result
    if _STORED in result:
        return {**result, _STORED: {"v": 1, "raw": result[_STORED]}}
    value = result.get("result")
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        return result
    reply, messages = value.get("final_response"), list(value["messages"])
    last = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], dict) and messages[i].get("role") == "assistant"), None)
    if not (isinstance(reply, str) and reply and last is not None and messages[last].get("content") == reply):
        return result
    messages[last] = {**messages[last], "content": None}
    return {**result, "result": {**value, "messages": messages}, _STORED: {"v": 1, "reply_ref": last}}


def load_result(text: str) -> Any:
    """Decode a stored result; only ``encode_result``'s exact envelope is decoded."""
    result = json.loads(text)
    if not isinstance(result, dict) or not isinstance(envelope := result.get(_STORED), dict):
        return result
    if envelope.keys() == {"v", "raw"} and type(envelope["v"]) is int and envelope["v"] == 1:
        result[_STORED] = envelope["raw"]
        return result
    value, ref = result.get("result"), envelope.get("reply_ref")
    messages = value.get("messages") if isinstance(value, dict) else None
    if not (envelope.keys() == {"v", "reply_ref"} and type(envelope["v"]) is int and envelope["v"] == 1 and type(ref) is int
            and isinstance(messages, list) and 0 <= ref < len(messages) and isinstance(value.get("final_response"), str)
            and isinstance(messages[ref], dict) and messages[ref].get("role") == "assistant" and messages[ref].get("content", 0) is None):
        return result
    messages[ref] = {k: value["final_response"] if k == "content" else v for k, v in messages[ref].items()}
    del result[_STORED]
    return result


def _open_tail(tail: Any) -> bool:
    message = tail.message
    return message.role == "user" or (message.role == "assistant" and bool(message.tool_calls))


def close_discarded_turn(tables: dict[str, Any], target: str, notice: str | None = None, *, replace_after: Any = None, display_metadata: Mapping[str, Any] | None = None) -> bool:
    """Close a discarded turn on the physical transcript ``target`` like a failed turn."""
    from dataclasses import replace

    namespace = tables.setdefault("messages", {"seq": 0, "transcripts": {}, "cooldowns": {}})
    transcript = namespace["transcripts"].get(target)
    live = [row for row in transcript.active() if row.message.role not in ("session_meta", "system")] if transcript is not None else []
    tail = max(live, key=lambda row: row.id) if live else None
    if (notice is not None and type(replace_after) is int and tail is not None and tail.message.role == "assistant"
            and tail.message.display_kind == FAILED_TURN_DISPLAY_KIND and tail.id > replace_after):
        tail.message = replace(tail.message, content=notice, display_metadata=dict(display_metadata) if display_metadata else None)
        return True
    if notice is None and (tail is None or not _open_tail(tail)):
        return False
    if transcript is None:
        transcript = namespace["transcripts"][target] = _Transcript()
    boundary = NewMessage(role="assistant", content=notice or PARTIAL_FAILED_TURN_NOTICE, timestamp=time.time(), display_kind=FAILED_TURN_DISPLAY_KIND,
                          display_metadata=dict(display_metadata) if display_metadata else None)
    insert_rows(namespace, transcript, [boundary])
    return True

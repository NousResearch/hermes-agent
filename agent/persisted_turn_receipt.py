"""``message.complete.persisted_turn``: the committed row addresses of the agent's current turn.

Shared by every host that publishes a turn's completion (the TUI gateway's in-process turn and
the session authority's TurnRunner turn), so a client sees one receipt contract whichever host
ran the turn. Clients key on it to bind a streamed reply to its stored row (``SessionMessage.row_id``)
instead of comparing words.
"""
from __future__ import annotations

from typing import Any, Callable

from agent.context_compressor import _DB_PERSISTED_MARKER


def committed_row_id(message: Any) -> int | None:
    row_id = message.get("_row_id") if isinstance(message, dict) else None
    return row_id if message.get(_DB_PERSISTED_MARKER) and type(row_id) is int and row_id > 0 else None


def persisted_turn_receipt(
    messages: Any, start: Any, history: list, raw: Any, status: str, *,
    compression_unchanged: bool, prefix_row_matches: Callable[[Any, Any], bool],
) -> dict | None:
    """Report committed row addresses, never a text/timestamp search for a matching turn.

    ``start`` is the agent's persistence cursor (``_persist_user_message_idx``), re-anchored during
    compaction. A partial receipt can address its surviving rows, but cannot retire a client's
    entire streamed turn. Full coverage additionally requires the unchanged pre-turn prefix
    (``prefix_row_matches(history_row, turn_row)`` for every row before ``start``), no compaction
    and no redirected user boundary.
    """
    if (not isinstance(messages, list) or type(start) is not int or not 0 <= start < len(messages)
            or not isinstance(messages[start], dict) or messages[start].get("role") != "user"):
        return None
    tail = messages[start:]
    anchor_id = committed_row_id(tail[0])
    if any(before is tail[0] or (anchor_id is not None and anchor_id == committed_row_id(before))
           for before in history):
        return None  # preflight returned the old transcript, not a new turn
    row_ids = [rid for message in tail if (rid := committed_row_id(message)) is not None]
    if not row_ids:
        return None
    receipt: dict = {"row_ids": row_ids, "complete": False}
    if (user_row_id := committed_row_id(tail[0])) is not None:
        receipt["user_row_id"] = user_row_id
    # Every rendered user row of the turn in order (the prompt, then each steer/redirect row), so a
    # client binds its optimistic bubbles even when the turn holds more than one prompt row. Only a
    # fully committed list is published: a gap would shift every later pairing.
    users = [message for message in tail if message.get("role") == "user" and message.get("display_kind") != "hidden"]
    user_row_ids = [committed_row_id(message) for message in users]
    if users and None not in user_row_ids:
        receipt["user_row_ids"] = user_row_ids
    last = tail[-1]
    # Equality only verifies the structurally selected final row's body: it never selects an identity.
    if (status == "complete" and last.get("role") == "assistant" and not last.get("tool_calls")
            and last.get("content") == raw and (final_id := committed_row_id(last)) is not None):
        receipt["final_assistant_row_id"] = final_id
    prefix_unchanged = start == len(history) and all(
        prefix_row_matches(before, after) for before, after in zip(history, messages[:start]))
    receipt["complete"] = bool(
        prefix_unchanged and compression_unchanged
        and len(row_ids) == len(tail)
        and sum(message.get("role") == "user" for message in tail) == 1
        and "final_assistant_row_id" in receipt)
    return receipt

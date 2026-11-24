"""Display reads paint a compacted tool result once, as its archived original (#132939, #135655).

Compaction can carry a tool result forward as a shortened stub: same ``message_uid`` and timestamp, short
body. A tool result keeps its payload in its display key on purpose (the archived output must never fold
INTO its stub), so the stub lands in a display slot (``display_order``) of its own and both copies painted.
The reads therefore skip a tool row when an archived (``compacted``) display-visible tool row of the same
session, ``message_uid`` and timestamp holds an EARLIER slot: that slot already paints the full original.

The rule uses identity only (session, uid, timestamp, slot), never bodies, and changes no stored row. An
in-place prune keeps its row's slot, so its stub shares the slot with the archived copies and the per-slot
rule (the live row wins) keeps showing the stub. A row without a slot (legacy, never backfilled) is never
skipped. The timestamp lets the probe seek ``(session_id, timestamp)``, so a page read stays per-row cheap.
"""

from __future__ import annotations

from typing import Any, Iterable

from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY
from hermes_state_common import _sql_json_extract

_ORIGINAL_VISIBLE_SQL = (
    f"COALESCE({_sql_json_extract('original.display_metadata', '$.' + MODEL_ONLY_DISPLAY_METADATA_KEY)}, 0) = 0")


def not_a_later_tool_copy_sql(alias: str) -> str:
    """Filter that drops *alias* rows that are a later copy of an archived tool result."""
    return (f" AND NOT ({alias}.role = 'tool' AND {alias}.message_uid IS NOT NULL AND EXISTS ("
            f"SELECT 1 FROM messages AS original WHERE original.session_id = {alias}.session_id "
            f"AND original.timestamp = {alias}.timestamp AND original.message_uid = {alias}.message_uid "
            "AND original.role = 'tool' AND original.active = 0 AND original.compacted = 1 "
            f"AND +original.display_order < {alias}.display_order AND {_ORIGINAL_VISIBLE_SQL}))")


def later_tool_copy_ids(db: Any, rows: Iterable[Any]) -> set[int]:
    """Python twin of :func:`not_a_later_tool_copy_sql` over display-set *rows*: the ids it skips.
    Empty when the rows carry no slot (``display_order``) or archive flag (``compacted``)."""
    tools = [row for row in rows if row["role"] == "tool"]
    if not tools or "compacted" not in tools[0].keys():
        return set()
    candidates = [row for row in tools if row["active"] or row["compacted"]]
    if not candidates or "display_order" not in candidates[0].keys():
        return set()

    def owner(row):
        return row["session_id"], row["message_uid"], row["timestamp"]

    first_slot: dict[Any, int] = {}
    for row in candidates:
        slot = row["display_order"]
        if row["message_uid"] and slot is not None and not row["active"] and not db._is_model_only_row(row):
            first_slot[owner(row)] = min(first_slot.get(owner(row), slot), slot)
    return {row["id"] for row in candidates if row["message_uid"] and row["display_order"] is not None
            and first_slot.get(owner(row), row["display_order"]) < row["display_order"]}

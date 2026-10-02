"""Exact held-row coverage for an in-place archive.

A watermark cap archives every active row up to the newest id the caller held.
That deletes a gap below that id, and a trailing unpersisted turn makes the cap
fall back to the lease watermark. When the caller can name the rows the
compressor actually held, the commit archives those and clones the rest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

ABSORBED_ROW_IDS = "_absorbed_row_ids"
# How many durable rows an alternation repair folded into one user dict. A reload without row ids
# has no other way to tell that dict from a prompt that was never persisted.
MERGED_DURABLE_ROWS = "_merged_durable_rows"
# How many more durable rows a dict stands for on a reload without row ids, besides a merged user
# run: rows the repair dropped behind it (a tool call with no result, a result with no call) or
# folded into an assistant turn. The caller was handed them, and nothing else names them.
UNNAMED_DURABLE_ROWS = "_unnamed_durable_rows"
# What names those rows at commit: role, content and tool-call ids of each, one per counted row.
# The commit matches each to one active row, so the rest of the held history keeps exact coverage.
RETIRED_DURABLE_ROWS = "_retired_durable_rows"
# Tags an entry of RETIRED_DURABLE_ROWS once coverage_for_commit hands it to the commit as unresolved.
RETIRED_ROW = "_retired_row"
# Marks the entry for the dict's own row: an assistant fold rewrote its text, so content no longer names it.
# It stands in the dict's own tail slot and is not one of the counted rows behind it.
OWN_ROW = "_own_row"


def retired_row_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the commit matches a retired id-less row by."""
    payload: Dict[str, Any] = {"role": message.get("role"), "content": message.get("content")}
    if message.get("tool_call_id"):
        payload["tool_call_id"] = message["tool_call_id"]
    calls = [call.get("id") for call in message.get("tool_calls") or () if isinstance(call, dict)]
    if calls:
        payload["tool_call_ids"] = calls
    return payload


def _positive_id(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def held_archive_coverage(
    messages: Sequence[Any], verbatim_tail: Optional[Sequence[Any]] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """``(covered ids, unresolved dicts)`` from the history the compressor was handed.

    A positive ``_row_id`` is covered, including ids a repair merged into that dict.
    A dict with no id is unresolved: the commit matches one durable row, and a
    marker-less miss is an unpersisted turn rather than a reason to archive the
    lease watermark.
    """
    covered: List[int] = []
    unresolved: List[Dict[str, Any]] = []
    for batch in (messages or (), verbatim_tail or ()):
        for message in batch:
            if not isinstance(message, dict):
                continue
            row_id = _positive_id(message.get("_row_id"))
            if row_id is None:
                unresolved.append(message)
            else:
                covered.append(row_id)
            for absorbed in message.get(ABSORBED_ROW_IDS) or ():
                absorbed_id = _positive_id(absorbed)
                if absorbed_id is not None:
                    covered.append(absorbed_id)
    return list(dict.fromkeys(covered)), unresolved


def durable_rows_behind_tail(session_db: Any, session_id: str, verbatim_tail: Sequence[Any]) -> int:
    """How many durable rows the ``here N`` exchanges stand for, for the rewind's ``tail_count``.

    A dict the alternation repair merged stands for every row behind it, and ``compress_now`` moves a
    rewritten copy's own id into ``_absorbed_row_ids`` beside the rows it absorbed. Counted once per
    dict, the rewind window ends one row short per merge, and the oldest carried original stays
    compacted=1 beside its live copy: recalled twice. Ids an earlier compaction archived are skipped.
    """
    total = 0
    for message in verbatim_tail or ():
        ids = set()
        if isinstance(message, dict):
            ids = {_positive_id(message.get("_row_id")), *map(_positive_id, message.get(ABSORBED_ROW_IDS) or ())}
            ids.discard(None)
        total += max(1, sum(1 for row_id in ids if session_db.get_message_role(session_id, row_id) is not None))
    return total


def newest_exact_held_id(
    messages: Sequence[Any], verbatim_tail: Optional[Sequence[Any]] = None,
) -> Optional[int]:
    """Newest held id that still names the row the compressor saw.

    A head row counts only while it still carries the persist marker. A ``here N``
    tail is marker-swept copies; its id counts when the copy kept one.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    exact: List[int] = []
    for message in messages or ():
        if not isinstance(message, dict) or not message.get(_DB_PERSISTED_MARKER):
            continue
        row_id = _positive_id(message.get("_row_id"))
        if row_id is not None:
            exact.append(row_id)
    for message in verbatim_tail or ():
        if isinstance(message, dict):
            row_id = _positive_id(message.get("_row_id"))
            if row_id is not None:
                exact.append(row_id)
    return max(exact) if exact else None


def coverage_for_commit(
    session_db: Any, session_id: str, messages: Sequence[Any],
    verbatim_tail: Optional[Sequence[Any]] = None,
) -> Tuple[Optional[List[int]], Optional[List[Dict[str, Any]]]]:
    """Coverage to pass into ``archive_and_compact``, or ``(None, None)`` to keep the watermark.

    ``None`` when the newest exact held row is already inactive (another compaction
    won: archiving only the held ids would clone the winner), when nothing held
    is a durable row, or when a held dict counts more rows than it can name: naming
    the rest would clone those behind the running turn. A trailing unpersisted turn
    does not take this branch: the rows above it stay unnamed and are cloned.
    The rows a dict names in ``RETIRED_DURABLE_ROWS`` join the unresolved set, so a
    dropped row does not throw away the coverage of everything else: the watermark
    would also archive rows another surface appended that the compressor never saw.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    newest = newest_exact_held_id(messages, verbatim_tail)
    role_of = getattr(session_db, "get_message_role", None)
    if newest is not None and callable(role_of) and role_of(session_id, newest) is None:
        return None, None
    covered, unresolved = held_archive_coverage(messages, verbatim_tail)
    retired: List[Dict[str, Any]] = []
    for message in unresolved:
        named = [row for row in message.get(RETIRED_DURABLE_ROWS) or () if isinstance(row, dict)]
        if int(message.get(UNNAMED_DURABLE_ROWS) or 0) > sum(1 for row in named if not row.get(OWN_ROW)):
            return None, None
        retired.extend({**row, _DB_PERSISTED_MARKER: True, RETIRED_ROW: True} for row in named)
    unresolved = [*unresolved, *retired]
    marked = [
        message for message in unresolved
        if isinstance(message, dict) and message.get(_DB_PERSISTED_MARKER)
    ]
    if not covered and not marked:
        return None, None
    return covered, unresolved

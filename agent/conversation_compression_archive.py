"""Exact held-row coverage for an in-place archive.

A watermark cap archives every active row up to the newest id the caller held.
That deletes a gap below that id, and a trailing unpersisted turn makes the cap
fall back to the lease watermark. When the caller can name the rows the
compressor actually held, the commit archives those and clones the rest.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from persistence.conversation.messages import HeldCoverage, HeldMessage, NewMessage, Watermark, stamp_live
from persistence.core.identity import MessageId, SessionId

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


def _positive_id(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def held_message(message: dict[str, Any]) -> HeldMessage:
    """One live message dict as the held record the store compares against its row.

    Reads what the store's held compare and coverage proof read off the dict, the repair's markers
    included: a missing or blank role is ``unknown``, ``tool_calls`` given as JSON text is decoded
    (undecodable text is empty), and a key of the wrong type is left unset.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    role, tool_call_id = message.get("role"), message.get("tool_call_id")
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except ValueError:
            tool_calls = []
    timestamp, row_id = message.get("timestamp"), _positive_id(message.get("_row_id"))
    snapshot, uid = message.get("_db_row_snapshot"), message.get("message_uid")
    absorbed = message.get(ABSORBED_ROW_IDS)
    merged, retired = message.get(MERGED_DURABLE_ROWS), message.get(RETIRED_DURABLE_ROWS)
    call_ids = message.get("tool_call_ids")
    return HeldMessage(
        role=role if isinstance(role, str) and role else "unknown", content=message.get("content"),
        tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
        tool_calls=tuple(tool_calls) if isinstance(tool_calls, (list, tuple)) else (),
        timestamp=float(timestamp) if isinstance(timestamp, float) else (
            timestamp if type(timestamp) is int else None),
        row_id=None if row_id is None else MessageId(int(row_id)),
        persisted=bool(message.get(_DB_PERSISTED_MARKER)),
        db_row_snapshot=snapshot if isinstance(snapshot, str) else None,
        message_uid=uid if isinstance(uid, str) and uid else None,
        absorbed_row_ids=tuple(
            MessageId(int(i)) for i in (absorbed if isinstance(absorbed, (list, tuple)) else ())
            if _positive_id(i) is not None),
        merged_durable_rows=merged if type(merged) is int and merged > 0 else 0,
        retired_durable_rows=tuple(
            held_message(row) for row in (retired if isinstance(retired, (list, tuple)) else ()) if isinstance(row, dict)),
        tool_call_ids=tuple(call_ids) if isinstance(call_ids, (list, tuple)) else (),
        own_row=bool(message.get(OWN_ROW)), retired_row=bool(message.get(RETIRED_ROW)))


def commit_compaction(
    session_db: Any, persistence: Any, session_id: str, compacted_messages: list[dict[str, Any]], **kwargs: Any,
) -> None:
    """``archive_and_compact(session_id, compacted_messages, **kwargs)``: ``MessageRepository.compact`` on the
    storage interface handle *persistence*, which stamps each dict with the row state the commit gave it as
    the store's own writer does, else the method of *session_db*, a client of the file's owner. Held dicts
    become held records; a ``None`` ``covered_ids`` keeps the watermark path."""
    if persistence is None:
        session_db.archive_and_compact(session_id, compacted_messages, **kwargs)
        return

    def held(key: str) -> Optional[tuple[HeldMessage, ...]]:
        messages = kwargs.get(key)
        return None if messages is None else tuple(held_message(m) for m in messages if isinstance(m, dict))

    watermark, covered_ids = kwargs.get("watermark"), kwargs.get("covered_ids")
    result = persistence.messages.compact(
        SessionId(session_id), [NewMessage.from_live(m) for m in compacted_messages],
        watermark=None if watermark is None else Watermark(MessageId(int(watermark)) if watermark else None),
        carried_tail_count=kwargs.get("tail_count", 0), lock_holder=kwargs.get("lock_holder"),
        model_config_patch=kwargs.get("model_config_patch"), carried=held("carried_messages") or (),
        coverage=None if covered_ids is None else HeldCoverage(
            covered_ids=tuple(MessageId(int(i)) for i in covered_ids), unresolved=held("unresolved_held") or ()),
        held_for_cas=held("held_for_cas"))
    stamp_live(compacted_messages, result)


def held_archive_coverage(
    messages: Sequence[Any], verbatim_tail: Optional[Sequence[Any]] = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """``(covered ids, unresolved dicts)`` from the history the compressor was handed.

    A positive ``_row_id`` is covered, including ids a repair merged into that dict.
    A dict with no id is unresolved: the commit matches one durable row, and a
    marker-less miss is an unpersisted turn rather than a reason to archive the
    lease watermark.
    """
    covered: list[int] = []
    unresolved: list[dict[str, Any]] = []
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


def newest_exact_held_id(
    messages: Sequence[Any], verbatim_tail: Optional[Sequence[Any]] = None,
) -> Optional[int]:
    """Newest held id that still names the row the compressor saw.

    A head row counts only while it still carries the persist marker. A ``here N``
    tail is marker-swept copies; its id counts when the copy kept one.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    exact: list[int] = []
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
    verbatim_tail: Optional[Sequence[Any]] = None, *, persistence: Any = None,
) -> tuple[Optional[list[int]], Optional[list[dict[str, Any]]]]:
    """Coverage to pass into ``archive_and_compact``, or ``(None, None)`` to keep the watermark.

    ``None`` when the newest exact held row is already inactive (another compaction
    won: archiving only the held ids would clone the winner), when nothing held
    is a durable row, or when a held dict counts more rows than it can name: naming
    the rest would clone those behind the running turn. A trailing unpersisted turn
    does not take this branch: the rows above it stay unnamed and are cloned.
    The rows a dict names in ``RETIRED_DURABLE_ROWS`` join the unresolved set, so a
    dropped row does not throw away the coverage of everything else: the watermark
    would also archive rows another surface appended that the compressor never saw.
    ``persistence`` is the storage interface handle the liveness read goes through when the store has one.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    newest = newest_exact_held_id(messages, verbatim_tail)
    if persistence is not None:
        if newest is not None and persistence.messages.message_role(session_id, MessageId(newest)) is None:
            return None, None
    else:
        role_of = getattr(session_db, "get_message_role", None)
        if newest is not None and callable(role_of) and role_of(session_id, newest) is None:
            return None, None
    covered, unresolved = held_archive_coverage(messages, verbatim_tail)
    retired: list[dict[str, Any]] = []
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

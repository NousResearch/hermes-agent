"""What ``session.transcript`` reads beyond the display pages: the compression lineage a delegated child
displays, the state check a poller's cursor holds (it tells an append from every other change), and the
compaction handoffs a page shows as rows. Split out of ``hermes_state_messages`` and bound onto SessionDB
through ``SessionMessagesMixin``. Each method's docstring names its storage-interface port.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_state_common import _placeholders

# The rows that can hold a compaction handoff (``ContextCompressor.classify_summary_content``'s prefixes,
# the merged delimiter, or the stored flag) for each shown row: the row itself, or an earlier copy of its
# message (same ``message_uid``: a later compaction carried, re-wrapped or unwrapped the carrier, so the
# display dedupe shows one copy and the older handoff would be lost). ``compaction_summary`` confirms each.
_HANDOFF_ROWS_SQL = (
    "SELECT w.id AS _shown_id, g.* FROM messages w JOIN messages g"
    " ON g.session_id IN (SELECT value FROM json_each(:segments)) AND (g.id = w.id OR ("
    "w.message_uid IS NOT NULL AND w.message_uid <> '' AND g.message_uid = w.message_uid AND g.id < w.id))"
    " WHERE w.id IN (SELECT value FROM json_each(:shown)) AND g.role IN ('user', 'assistant')"
    " AND (g._compressed_summary = 1 OR instr(g.content, '[CONTEXT COMPACTION') > 0"
    " OR instr(g.content, '[CONTEXT SUMMARY]') > 0) ORDER BY w.id, g.id")


class SessionTranscriptCursorMixin:
    """Poll one session's transcript by appended rows only, while nothing older changed."""

    def transcript_lineage_ids(self, session_id: str) -> list[str]:
        """The compression lineage a delegated child's transcript displays: ``_resume_lineage_ids``, or the
        session alone while its row carries no ``_delegate_from``. A queued child's row is reserved with
        no ``model_config`` (``_reserve_child_transcript``); its delegation marker stays in memory until its
        first turn writes it, and a delegator that rotated through compression meanwhile would read as
        the child's own earlier segment. Every continuation of a child copies the marker.

        Storage interface: port as ``delegated_child_lineage_ids``."""
        row = self.get_session(session_id)
        cfg = row.get("model_config") if row else None
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except ValueError:
                cfg = None
        if not isinstance(cfg, dict) or not cfg.get("_delegate_from"):
            return [session_id]
        return self._resume_lineage_ids(session_id)

    def transcript_cursor_state(self, session_id: str, through_id: int,
                                lineage_ids: list[str] | None = None) -> tuple[int, int, int]:
        """``(live rows with id <= through_id, inactive rows, transcript_revision)``. The live count is the
        session's own (only it is appended to); the other two span *lineage_ids* (by default
        ``transcript_lineage_ids``), whose earlier segments' rows a full read shows. An append moves none
        of them. A compaction or rewind (anywhere, including rows appended after *through_id*) moves the
        inactive count, deleting the rows held moves the first, an in-place rewrite of any displayed row
        (an ancestor's reaction included) the last; while all three hold, the rows after *through_id* are
        all that changed.

        Storage interface: port as ``transcript_cursor_state``."""
        ids = list(lineage_ids) if lineage_ids else self.transcript_lineage_ids(session_id)
        marks = _placeholders(ids)
        row = self._read_one(
            "SELECT (SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1 AND id <= ?),"
            f" (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 0),"
            f" (SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks}))",
            (session_id, int(through_id), *ids, *ids))
        return int(row[0]), int(row[1]), int(row[2])

    def transcript_handoff_rows(self, shown_ids: list[int], segment_ids: list[str]) -> dict[int, list[dict[str, Any]]]:
        """``{shown row id: [candidate handoff message, ...]}`` in id order, for display rows *shown_ids*
        of the lineage *segment_ids*: the shown row when it can hold a handoff, and every earlier stored
        copy of its message that can (``_HANDOFF_ROWS_SQL``). Decoded as ``_row_to_message_dict`` does,
        each with its own ``id``; the caller confirms a handoff with ``compaction_summary``.

        Storage interface: port as ``transcript_handoff_rows``."""
        if not shown_ids:
            return {}
        with self._read_ctx() as conn:
            rows = conn.execute(_HANDOFF_ROWS_SQL, {'segments': json.dumps(list(segment_ids)),
                                                    'shown': json.dumps(sorted(set(shown_ids)))}).fetchall()
        found: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            message = self._row_to_message_dict(row, warn_context='session.transcript', summary_flag=True)
            found.setdefault(int(message.pop('_shown_id')), []).append(message)
        return found

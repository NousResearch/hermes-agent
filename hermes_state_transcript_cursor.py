"""The cursor a transcript poller holds (``subagent.transcript``): a cheap state check and the keyset
read of the rows after it. Split out of ``hermes_state_messages`` and bound onto SessionDB through
``SessionMessagesMixin``; built on its _fetch_conversation_rows/_rows_to_conversation primitives."""

from __future__ import annotations

from typing import Any

from hermes_state_common import _placeholders


class SessionTranscriptCursorMixin:
    """Poll one session's live transcript by appended rows only, while nothing older changed."""

    def transcript_cursor_state(self, session_id: str, through_id: int) -> tuple[int, int, int]:
        """``(live rows with id <= through_id, inactive rows, transcript_revision)``. The live count is the
        session's own (only it is appended to); the other two span the displayed compression lineage
        (``_resume_lineage_ids``), whose earlier segments' rows a full read shows archived. An append moves
        none of them. A compaction or rewind (anywhere, including rows appended after *through_id*) moves
        the inactive count, deleting the rows held moves the first, an in-place rewrite of any displayed
        row (an ancestor's reaction included) the last; while all three hold, the rows after *through_id*
        are all that changed."""
        ids = self._resume_lineage_ids(session_id)
        marks = _placeholders(ids)
        row = self._read_one(
            "SELECT (SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1 AND id <= ?),"
            f" (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 0),"
            f" (SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks}))",
            (session_id, int(through_id), *ids, *ids))
        return int(row[0]), int(row[1]), int(row[2])

    def conversation_tail(self, session_id: str, after_id: int) -> list[dict[str, Any]]:
        """``get_messages_as_conversation(include_row_ids=True)``'s live rows of one session with an id
        greater than *after_id*: the keyset read a ``transcript_cursor_state`` holder polls."""
        rows = self._fetch_conversation_rows([session_id], self._active_clause(False, False), with_session_id=False,
                                             after_id=after_id)
        return self._rows_to_conversation(rows, session_id=session_id, include_ancestors=False,
                                          repair_alternation=False, include_row_ids=True)

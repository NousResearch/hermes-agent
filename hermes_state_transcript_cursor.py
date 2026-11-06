"""The cursor a transcript poller holds: a cheap state check that tells an append from every other
change. Split out of ``hermes_state_messages`` and bound onto SessionDB through ``SessionMessagesMixin``."""

from __future__ import annotations

import json

from hermes_state_common import _placeholders


class SessionTranscriptCursorMixin:
    """Poll one session's live transcript by appended rows only, while nothing older changed."""

    def transcript_lineage_ids(self, session_id: str) -> list[str]:
        """The compression lineage a delegated child's transcript displays: ``_resume_lineage_ids``, or the
        session alone while its row carries no ``_delegate_from``. A queued child's row is reserved with
        no ``model_config`` (``_reserve_child_transcript``); its delegation marker stays in memory until its
        first turn writes it, and a delegator that rotated through compression meanwhile would read as
        the child's own earlier segment. Every continuation of a child copies the marker."""
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

    def transcript_cursor_state(self, session_id: str, through_id: int) -> tuple[int, int, int]:
        """``(live rows with id <= through_id, inactive rows, transcript_revision)``. The live count is the
        session's own (only it is appended to); the other two span the displayed compression lineage
        (``transcript_lineage_ids``), whose earlier segments' rows a full read shows archived. An append moves
        none of them. A compaction or rewind (anywhere, including rows appended after *through_id*) moves
        the inactive count, deleting the rows held moves the first, an in-place rewrite of any displayed
        row (an ancestor's reaction included) the last; while all three hold, the rows after *through_id*
        are all that changed."""
        ids = self.transcript_lineage_ids(session_id)
        marks = _placeholders(ids)
        row = self._read_one(
            "SELECT (SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1 AND id <= ?),"
            f" (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 0),"
            f" (SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks}))",
            (session_id, int(through_id), *ids, *ids))
        return int(row[0]), int(row[1]), int(row[2])

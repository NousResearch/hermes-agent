"""Archive/unarchive persistence for session rows, lineage-wide.

Mixin bound onto ``SessionDB`` via the MRO; built on its ``_write_rowcount`` /
``_read_all`` / ``_write_sql`` primitives. Extracted from
``hermes_state_sessions.py`` (moved code keeps its cap; see root AGENTS.md)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from hermes_state_common import _RECOVERABLE_END_REASONS

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typed only
    from hermes_state import SessionDB

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

# ``lineage(id)``: the compression lineage of the session bound twice as ``(?, ?)`` —
# ancestors through compression-ended parents plus compression continuations after it.
_LINEAGE_CTE_SQL = """
            WITH RECURSIVE
              ancestors(id) AS (
                SELECT ?
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE parent.end_reason = 'compression'
              ),
              descendants(id) AS (
                SELECT ?
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions parent ON parent.id = d.id
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.end_reason = 'compression'
              ),
              lineage(id) AS (
                SELECT id FROM ancestors
                UNION
                SELECT id FROM descendants
              )"""


def _log_session_archive(db_path: Optional[Path], target_session_id: str, archived: bool,
                         preview: Dict[str, Any], trigger: str) -> None:
    """Append one JSON line per successful archive/unarchive to ``<db dir>/logs/archives.jsonl``.

    The state.db row is the only record of what a cascade hid, so an unexpected archive
    leaves no recoverable trail; the JSONL file sits next to state.db (the profile's own home,
    so two profiles don't share one log) and survives it. Best-effort by design — a filesystem
    hiccup must never fail the user's archive call (#70185)."""
    try:
        from hermes_constants import get_hermes_home
        log_dir = (Path(db_path).resolve().parent if db_path else get_hermes_home()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(), "trigger": trigger, "target": target_session_id,
            "archived": bool(archived),
            "cascade_count": int(preview.get("cascade_count", 0)),
            "affected_ids": list(preview.get("affected_ids", ())),
        }
        with (log_dir / "archives.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("archive audit log append failed", exc_info=True)


class SessionArchiveMixin:
    """Archive/unarchive of sessions and their compression lineages, plus audit logging."""

    if TYPE_CHECKING:  # pragma: no cover
        _write_rowcount: Any
        _read_all: Any
        _write_sql: Any
        db_path: Any
        get_session: Any
        get_compression_tip: Any

    def _set_lineage_column(self, column: str, session_id: str, value: Any, *,
                            extra_set_sql: str = "") -> bool:
        """Set one ``sessions`` column across a whole compression lineage: Desktop projects roots
        forward to their tip, so updating only the tip would let the root resurrect it on refresh.
        *extra_set_sql* (trusted literal, ``, col = expr``) rides the same UPDATE."""
        return self._write_rowcount(
            _LINEAGE_CTE_SQL + f"""
            UPDATE sessions
            SET {column} = ?{extra_set_sql}
            WHERE id IN (SELECT id FROM lineage)
            """,
            (session_id, session_id, value),
        ) > 0

    def preview_session_archive_lineage(self, session_id: str, archived: bool = True) -> Dict[str, Any]:
        """Rows :meth:`set_session_archived` would flip, read-only (the confirmation gate's data).

        Runs the same lineage CTE as the archive UPDATE, filtered to rows whose ``archived``
        flag would actually change, so an idempotent re-archive previews an empty cascade.
        Returns ``cascade_count`` (rows that would flip), ``cascade_extra`` (rows beyond the
        targeted one), ``affected_ids``, and the oldest/newest ``started_at`` in the set
        (#70185)."""
        rows = self._read_all(
            _LINEAGE_CTE_SQL + """
            SELECT id, started_at FROM sessions
            WHERE id IN (SELECT id FROM lineage) AND archived IS NOT ?
            ORDER BY started_at
            """,
            (session_id, session_id, 1 if archived else 0),
        )
        affected_ids = [row["id"] for row in rows]
        started = [row["started_at"] for row in rows if row["started_at"] is not None]
        return {
            "cascade_count": len(affected_ids),
            "cascade_extra": max(0, len(affected_ids) - 1),
            "affected_ids": affected_ids,
            "oldest_started_at": min(started) if started else None,
            "newest_started_at": max(started) if started else None,
        }

    def set_session_archived(
        self, session_id: str, archived: bool, *, trigger: str = "api",
    ) -> bool:
        """Soft-hide (or unhide) a session and its compression lineage; messages are kept.
        This is the DELIBERATE archive (user, CLI, API): it clears the ``auto_archived``
        provenance, so re-activation never un-hides it on the user's behalf.

        One call can flip a whole compression lineage (the user's one click hides N rows,
        #70185): callers that need the blast radius first run
        :meth:`preview_session_archive_lineage`, and every successful call appends an audit
        record (trigger, timestamp, ids) to ``<db dir>/logs/archives.jsonl``."""
        preview = self.preview_session_archive_lineage(session_id, archived)
        result = self._set_lineage_column(
            "archived", session_id, int(archived), extra_set_sql=", auto_archived = 0")
        if result:
            _log_session_archive(self.db_path, session_id, archived, preview, trigger)
        return result

    def _auto_archive_lineage(self, session_id: str) -> bool:
        """The idle sweep's archive: like :meth:`set_session_archived` but stamps
        ``auto_archived`` on the rows IT hides. A row already archived keeps its provenance, so a
        deliberately archived ancestor is never relabelled as sweep-owned (SQLite evaluates every
        SET expression against the pre-update row)."""
        return self._set_lineage_column(
            "archived", session_id, 1,
            extra_set_sql=", auto_archived = CASE WHEN archived = 0 THEN 1 ELSE auto_archived END")

    @staticmethod
    def _unarchive_auto_archived_lineage(conn, session_id: str) -> bool:
        """Un-hide a lineage the idle sweep archived, on *conn* (caller's write txn). A tip that is
        published (compression) or reopened (resume) under it is live again, and the listing admits
        a lineage by its ROOT's flag, so leaving the sweep's stamp would hide an active chat (#117713).
        A lineage with ANY deliberately archived row (``archived = 1 AND auto_archived = 0``) is left
        alone: a manual archive stays until the user un-archives it. True when rows were un-hidden."""
        params = (session_id, session_id)
        manual = conn.execute(
            _LINEAGE_CTE_SQL + """
            SELECT 1 FROM sessions
            WHERE id IN (SELECT id FROM lineage) AND archived <> 0 AND COALESCE(auto_archived, 0) = 0
            LIMIT 1
            """, params).fetchone()
        if manual is not None:
            return False
        return conn.execute(
            _LINEAGE_CTE_SQL + """
            UPDATE sessions SET archived = 0, auto_archived = 0
            WHERE id IN (SELECT id FROM lineage) AND archived <> 0
            """, params).rowcount > 0

    # Accidental end reasons recovery treats as resumable (also interpolated into
    # the recovery/promotion SQL so literals cannot drift).
    RECOVERABLE_END_REASONS = _RECOVERABLE_END_REASONS

    def unarchive_recoverable_session(self, session_id: str) -> bool:
        """Un-archive a session archived by a recoverable accident (ws_orphan_reap, agent_close);
        deliberate archives are left alone. True when un-archived.

        Registry-style lookups (Bot Mode's canonical "Bot Chat") use this to resurrect a row the ws-orphan
        reaper (``ws_orphan_reap``) or older agent cleanup (``agent_close``) archived: those ends are
        accidents, not user intent, so the identity-scoped canonical chat must survive them (#92687).
        Sessions archived with no end_reason or an explicit boundary reason (user archived deliberately,
        ``session_reset``, …) are left untouched — returns ``False`` for those, ``True`` only when the row
        was archived for a recoverable reason and is now un-archived (whole compression lineage, via
        :meth:`set_session_archived`).
        """
        if not session_id:
            return False
        try:
            row = self.get_session(session_id)
        except Exception:
            return False
        if not row or not row.get("archived"):
            return False
        # The accidental stamp lives on the live TIP; judge recoverability there.
        tip = row
        try:
            tip_id = self.get_compression_tip(session_id) or session_id
            if tip_id != session_id:
                tip = self.get_session(tip_id) or row
        except Exception:
            pass
        if (tip.get("end_reason") or "") not in self.RECOVERABLE_END_REASONS:
            return False
        if not self.set_session_archived(session_id, False):
            return False
        # Clear the accidental end stamp, or a LATER deliberate archive (which never
        # writes end_reason) would auto-resurrect on the next lookup.
        self._write_sql(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (tip["id"],),
        )
        return True

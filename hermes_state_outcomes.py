"""External session outcomes (#138653): what happened to a session's work after it ended — a PR
merged or reverted, CI passed or failed, a review accepted, a ticket reopened.

Bound onto SessionDB through ``SessionOutcomesMixin``. Rows store the physical session id and go with it
(``ON DELETE CASCADE``). Reads match across the session's compression lineage, so a root id and a tip id see
the same outcomes; a delegate subagent's id reads its parent's. /reset and /branch are never crossed.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

from hermes_state_common import EPOCH_MAX, EPOCH_MIN, _sql_json_extract
from hermes_state_sessions import _LINEAGE_CTE_SQL

# Fixed vocabulary -> default verdict. ``custom`` has none, so its verdict is required.
OUTCOME_SIGNALS: dict[str, Optional[str]] = {
    "merged": "good", "reverted": "bad", "review_accepted": "good", "review_rejected": "bad",
    "ci_passed": "good", "ci_failed": "bad", "ticket_closed": "good", "ticket_reopened": "bad", "custom": None,
}
OUTCOME_VERDICTS = ("good", "bad", "neutral")
OUTCOME_COLUMNS = ("id", "session_id", "signal", "verdict", "source", "recorder", "ref", "note",
                   "observed_at", "recorded_at")
MAX_OUTCOME_REF_CHARS = 512
MAX_OUTCOME_NOTE_CHARS = 500
_LABEL = r"[a-z0-9][a-z0-9_.-]{0,63}"
_SOURCE_RE = re.compile(_LABEL)
_RECORDER_RE = re.compile(rf"cli|agent|plugin:{_LABEL}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_DELEGATE_OF = _sql_json_extract("{a}.model_config", "$._delegate_from")
# A delegate subagent reads its parent's outcomes: one hop, marker bound to the actual parent.
_SEED_SQL = (f"SELECT CASE WHEN {_DELEGATE_OF.format(a='s')} = s.parent_session_id THEN s.parent_session_id "
             "ELSE s.id END FROM sessions s WHERE s.id = ?")
# Every segment of the compression lineage, plus delegate subagents run from any of them.
_SCOPE_SQL = (f"{_LINEAGE_CTE_SQL},\n scope(id) AS (SELECT id FROM lineage UNION SELECT c.id FROM sessions c "
              f"JOIN lineage l ON c.parent_session_id = l.id WHERE {_DELEGATE_OF.format(a='c')} = l.id)\n")


def _missing_outcomes_table(exc: sqlite3.OperationalError) -> bool:
    """A read-only open of a store no writer has opened since upgrade has no outcomes table (yet)."""
    return "no such table: session_outcomes" in str(exc)


def _validate_outcome(signal, verdict, source, recorder, ref, note, observed_at) -> tuple[str, float]:
    """``(verdict, observed_at)`` to store, or ValueError naming the bad field and what it accepts."""
    if signal not in OUTCOME_SIGNALS:
        raise ValueError(f"Unknown outcome signal {signal!r}: use one of {', '.join(OUTCOME_SIGNALS)}.")
    verdict = verdict or OUTCOME_SIGNALS[signal]
    if verdict not in OUTCOME_VERDICTS:
        raise ValueError(f"Outcome verdict {verdict!r} is invalid: use one of {', '.join(OUTCOME_VERDICTS)}"
                         + (" (signal 'custom' has no default)." if signal == "custom" else "."))
    if not isinstance(source, str) or not _SOURCE_RE.fullmatch(source):
        raise ValueError(f"Outcome source {source!r} is not a plain label: use 1-64 of a-z 0-9 _ . - "
                         "(for example 'github').")
    if not isinstance(recorder, str) or not _RECORDER_RE.fullmatch(recorder):
        raise ValueError(f"Outcome recorder {recorder!r} is invalid: use 'cli', 'agent' or 'plugin:<id>'.")
    for field, text, cap in (("ref", ref, MAX_OUTCOME_REF_CHARS), ("note", note, MAX_OUTCOME_NOTE_CHARS)):
        if not isinstance(text, str) or len(text) > cap or _CONTROL_RE.search(text):
            raise ValueError(f"Outcome {field} must be a single line of at most {cap} characters.")
    if note:
        from tools.threat_patterns import scan_for_threats
        if findings := scan_for_threats(note, scope="strict"):
            raise ValueError(f"Outcome note matches threat pattern '{findings[0]}': describe the outcome "
                             "in plain words.")
    observed_at = time.time() if observed_at is None else observed_at
    if (isinstance(observed_at, bool) or not isinstance(observed_at, (int, float))
            or not math.isfinite(observed_at) or not EPOCH_MIN <= observed_at <= EPOCH_MAX):
        raise ValueError(f"Outcome observed_at {observed_at!r} is not a supported Unix timestamp.")
    return verdict, float(observed_at)


class SessionOutcomesMixin:
    """Record, list, remove and batch-read ``session_outcomes`` rows."""

    def record_session_outcome(self, session_id: str, *, signal: str, source: str, recorder: str,
                               verdict: Optional[str] = None, ref: str = "", note: str = "",
                               observed_at: Optional[float] = None) -> dict[str, Any]:
        """Record (or update) one outcome on the physical *session_id*; returns the stored row.

        ``(session_id, signal, source, ref, recorder)`` is the identity: recording it again updates verdict,
        note and both timestamps, so a poller can re-report idempotently. *verdict* defaults to the signal's.
        *recorder* is ``cli``, ``agent`` or ``plugin:<id>``. Raises ValueError for an unknown session or an
        invalid field.
        """
        verdict, observed_at = _validate_outcome(signal, verdict, source, recorder, ref, note, observed_at)

        def _write(conn: sqlite3.Connection) -> dict[str, Any]:
            if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
                raise ValueError(f"Unknown session {session_id!r}: see `hermes sessions list`.")
            row = conn.execute(
                f"INSERT INTO session_outcomes ({', '.join(OUTCOME_COLUMNS[1:])}) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (session_id, signal, source, ref, recorder) DO UPDATE SET verdict = excluded.verdict, "
                "note = excluded.note, observed_at = excluded.observed_at, recorded_at = excluded.recorded_at "
                f"RETURNING {', '.join(OUTCOME_COLUMNS)}",
                (session_id, signal, verdict, source, recorder, ref, note, observed_at, time.time())).fetchone()
            return dict(zip(OUTCOME_COLUMNS, row))
        return self._execute_write(_write)

    def delete_session_outcome(self, outcome_id: int, *, recorder: Optional[str] = None) -> bool:
        """Delete one outcome by id (only *recorder*'s rows when given); True when a row went."""
        def _delete(conn: sqlite3.Connection) -> bool:
            sql, params = "DELETE FROM session_outcomes WHERE id = ?", [outcome_id]
            if recorder is not None:
                sql, params = sql + " AND recorder = ?", [outcome_id, recorder]
            return conn.execute(sql, params).rowcount > 0
        return bool(self._execute_write(_delete))

    def list_session_outcomes(self, session_id: Optional[str] = None, *, since: Optional[float] = None,
                              signal: Optional[str] = None, verdict: Optional[str] = None,
                              limit: int = 200) -> list[dict[str, Any]]:
        """Outcomes newest-observed first: of *session_id*'s lineage (see module doc) or of every session,
        observed at/after *since*, of one *signal* / *verdict*. A store without the table has none."""
        if limit < 1:
            raise ValueError(f"Outcome list limit must be at least 1, got {limit}.")
        clauses, params = [], []
        for clause, param in (("observed_at >= ?", since), ("signal = ?", signal), ("verdict = ?", verdict)):
            if param is not None:
                clauses.append(f" AND {clause}")
                params.append(param)
        tail = f"{''.join(clauses)} ORDER BY observed_at DESC, id DESC LIMIT ?"
        cols = ", ".join(OUTCOME_COLUMNS)

        def _read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            if session_id is None:
                return conn.execute(f"SELECT {cols} FROM session_outcomes WHERE 1{tail}", (*params, limit)).fetchall()
            seed = conn.execute(_SEED_SQL, (session_id,)).fetchone()
            if seed is None:
                return []
            return conn.execute(f"{_SCOPE_SQL} SELECT {cols} FROM session_outcomes "
                                f"WHERE session_id IN (SELECT id FROM scope){tail}",
                                (seed[0], seed[0], *params, limit)).fetchall()
        try:
            return [dict(row) for row in self._read_retrying_ioerr(_read)]
        except sqlite3.OperationalError as exc:
            if _missing_outcomes_table(exc):
                return []
            raise

    def outcomes_for_sessions(self, session_ids: Iterable[str], *, per_session: int = 3) -> dict[str, list[dict]]:
        """``{session_id: [outcome, ...]}`` newest first, at most *per_session* each, lineage-matched like
        :meth:`list_session_outcomes`; sessions without outcomes are absent. For short id lists (a search
        page): one lineage query per id."""
        found = {}
        for sid in dict.fromkeys(s for s in session_ids if isinstance(s, str) and s):
            if rows := self.list_session_outcomes(sid, limit=per_session):
                found[sid] = rows
        return found

    # ── portability (export / import_sessions) ─────────────────────────────

    def _outcomes_by_session(self, session_ids: Iterable[str]) -> dict[str, list[dict]]:
        """Physical rows per session for export (no lineage matching: each segment exports its own)."""
        from hermes_state_common import _id_chunks
        found: dict[str, list[dict]] = {}
        try:
            for chunk in _id_chunks(sorted(set(session_ids))):
                for row in self._read_all(f"SELECT {', '.join(OUTCOME_COLUMNS)} FROM session_outcomes WHERE "
                                          f"session_id IN ({','.join('?' * len(chunk))}) ORDER BY id", chunk):
                    found.setdefault(row["session_id"], []).append(dict(row))
        except sqlite3.OperationalError as exc:
            if not _missing_outcomes_table(exc):
                raise
        return found

    @staticmethod
    def _normalize_import_outcomes(raw: Any) -> list[tuple]:
        """Validate an imported session's ``outcomes`` exactly like a fresh record; ValueError on any bad one."""
        if raw is None:
            return []
        if not isinstance(raw, list) or any(not isinstance(o, dict) for o in raw):
            raise ValueError("outcomes must be a list of objects")
        rows = []
        for o in raw:
            fields = {k: o.get(k, "") for k in ("signal", "source", "recorder", "ref", "note")}
            verdict, observed_at = _validate_outcome(fields["signal"], o.get("verdict"), fields["source"],
                fields["recorder"], fields["ref"], fields["note"], o.get("observed_at"))
            recorded_at = o.get("recorded_at")
            if not isinstance(recorded_at, (int, float)) or isinstance(recorded_at, bool) or not math.isfinite(recorded_at):
                recorded_at = observed_at
            rows.append((fields["signal"], verdict, fields["source"], fields["recorder"], fields["ref"],
                         fields["note"], observed_at, float(recorded_at)))
        return rows

    @staticmethod
    def _insert_import_outcomes(conn: sqlite3.Connection, session_id: str, rows: list[tuple]) -> None:
        conn.executemany(f"INSERT OR IGNORE INTO session_outcomes ({', '.join(OUTCOME_COLUMNS[1:])}) "
                         "VALUES (?,?,?,?,?,?,?,?,?)", [(session_id, *row) for row in rows])

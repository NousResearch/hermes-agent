"""External session outcomes (#138653): what happened to a session's work after it ended — a PR
merged or reverted, CI passed or failed, a review accepted, a ticket reopened.

Bound onto SessionDB through ``SessionOutcomesMixin``. Rows store the physical session id and go with it
(``ON DELETE CASCADE``). Reads match across the session's compression lineage, so a root id and a tip id see
the same outcomes; a delegate subagent's id reads its parent's. /reset and /branch are never crossed.

Supersession: rows with the same session, source, domain and non-empty ref form one artifact; default reads
return only its latest-observed row (ties: higher id), so a later ``pr.reverted`` replaces ``pr.merged``.
``agent`` rows are their own group: they never cover, or are covered by, a ``cli`` or ``plugin:`` row.
History is kept; ``all_rows=True`` returns the journal.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

from hermes_state_common import EPOCH_MAX, EPOCH_MIN, _sql_json_extract
from hermes_state_sessions import _LINEAGE_CTE_SQL

# Signals are open labels (``pr.merged``, ``review.comment_applied``); only these well-known ones default
# the verdict, every other signal needs one. Unprefixed names fold into a domain like their dotted forms.
_LEGACY_SIGNALS = {"merged": "pr.merged", "reverted": "pr.reverted", "review_accepted": "review.accepted",
                   "review_rejected": "review.rejected", "ci_passed": "ci.passed", "ci_failed": "ci.failed",
                   "ticket_closed": "task.done", "ticket_reopened": "task.reopened"}
DEFAULT_VERDICTS: dict[str, str] = {
    "pr.merged": "good", "pr.reverted": "bad", "review.accepted": "good", "review.rejected": "bad",
    "ci.passed": "good", "ci.failed": "bad", "task.done": "good", "task.reopened": "bad",
}
DEFAULT_VERDICTS.update({old: DEFAULT_VERDICTS[new] for old, new in _LEGACY_SIGNALS.items()})
OUTCOME_VERDICTS = ("good", "bad", "neutral")
OUTCOME_COLUMNS = ("id", "session_id", "signal", "domain", "verdict", "source", "recorder", "ref", "note",
                   "observed_at", "recorded_at")
MAX_OUTCOME_REF_CHARS = 512
MAX_OUTCOME_NOTE_CHARS = 500
_LABEL = r"[a-z0-9][a-z0-9_.-]{0,63}"
_SIGNAL_RE = _SOURCE_RE = re.compile(_LABEL)
_RECORDER_RE = re.compile(rf"cli|agent|plugin:{_LABEL}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_DELEGATE_OF = _sql_json_extract("{a}.model_config", "$._delegate_from")
# A delegate subagent reads its parent's outcomes: one hop, marker bound to the actual parent.
_SEED_SQL = (f"SELECT CASE WHEN {_DELEGATE_OF.format(a='s')} = s.parent_session_id THEN s.parent_session_id "
             "ELSE s.id END FROM sessions s WHERE s.id = ?")
# Every segment of the compression lineage, plus delegate subagents run from any of them.
_SCOPE_SQL = (f"{_LINEAGE_CTE_SQL},\n scope(id) AS (SELECT id FROM lineage UNION SELECT c.id FROM sessions c "
              f"JOIN lineage l ON c.parent_session_id = l.id WHERE {_DELEGATE_OF.format(a='c')} = l.id)\n")
# ``head`` is the id of the row's artifact's effective row; an empty ref is its own artifact. The
# ``recorder = 'agent'`` partition term keeps agent rows from covering, or being covered by, anyone else's.
_RANKED_SQL = ("ranked AS (SELECT o.*, CASE WHEN o.ref = '' THEN o.id ELSE first_value(o.id) OVER (PARTITION BY "
               "o.session_id, o.source, o.domain, o.ref, o.recorder = 'agent' ORDER BY o.observed_at DESC, o.id DESC) "
               "END AS head FROM session_outcomes o {where})")


def _missing_outcomes_table(exc: sqlite3.OperationalError) -> bool:
    """A read-only open of a store no writer has opened since upgrade has no outcomes table (yet)."""
    return "no such table: session_outcomes" in str(exc)


def outcome_domain(signal: str) -> str:
    """The artifact kind a signal speaks about: the prefix before the first dot (``pr.merged`` -> ``pr``)."""
    return _LEGACY_SIGNALS.get(signal, signal).split(".", 1)[0]


def _validate_outcome(signal, verdict, source, recorder, ref, note, observed_at) -> tuple[str, float]:
    """``(verdict, observed_at)`` to store, or ValueError naming the bad field and what it accepts."""
    if not isinstance(signal, str) or not _SIGNAL_RE.fullmatch(signal) or signal == "custom":
        raise ValueError(f"Outcome signal {signal!r} is invalid: name what happened with a label matching "
                         f"^{_LABEL}$ (for example 'review.comment_applied') and give a verdict.")
    verdict = verdict or DEFAULT_VERDICTS.get(signal)
    if verdict is None:
        raise ValueError(f"Outcome signal {signal!r} has no default verdict: give one of "
                         f"{', '.join(OUTCOME_VERDICTS)} (--verdict).")
    if verdict not in OUTCOME_VERDICTS:
        raise ValueError(f"Outcome verdict {verdict!r} is invalid: use one of {', '.join(OUTCOME_VERDICTS)}.")
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
                f"INSERT INTO session_outcomes ({', '.join(OUTCOME_COLUMNS[1:])}) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (session_id, signal, source, ref, recorder) DO UPDATE SET verdict = excluded.verdict, "
                "note = excluded.note, observed_at = excluded.observed_at, recorded_at = excluded.recorded_at "
                f"RETURNING {', '.join(OUTCOME_COLUMNS)}",
                (session_id, signal, outcome_domain(signal), verdict, source, recorder, ref, note, observed_at,
                 time.time())).fetchone()
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
                              source: Optional[str] = None, all_rows: bool = False, include_agent: bool = True,
                              limit: int = 200) -> list[dict[str, Any]]:
        """Effective outcomes newest-observed first (see module doc): of *session_id*'s lineage or of every
        session, filtered by *since* (observed at/after), *signal*, *verdict*, *source*. *all_rows* adds the
        superseded rows; each row carries ``effective`` and ``superseded_by`` (the effective row's id, or
        None). Filters apply after supersession, so a superseded merge never reads as effective. A store
        without the table has none."""
        if limit < 1:
            raise ValueError(f"Outcome list limit must be at least 1, got {limit}.")
        clauses, params = ["" if all_rows else " AND head = id", "" if include_agent else " AND recorder != 'agent'"], []
        for clause, param in (("observed_at >= ?", since), ("signal = ?", signal), ("verdict = ?", verdict),
                              ("source = ?", source)):
            if param is not None:
                clauses.append(f" AND {clause}")
                params.append(param)
        tail = (f"SELECT {', '.join(OUTCOME_COLUMNS)}, head = id AS effective, NULLIF(head, id) AS superseded_by "
                f"FROM ranked WHERE 1{''.join(clauses)} ORDER BY observed_at DESC, id DESC LIMIT ?")

        def _read(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            if session_id is None:
                return conn.execute(f"WITH {_RANKED_SQL.format(where='')} {tail}", (*params, limit)).fetchall()
            seed = conn.execute(_SEED_SQL, (session_id,)).fetchone()
            if seed is None:
                return []
            ranked = _RANKED_SQL.format(where="WHERE o.session_id IN (SELECT id FROM scope)")
            return conn.execute(f"{_SCOPE_SQL}, {ranked} {tail}", (seed[0], seed[0], *params, limit)).fetchall()
        try:
            rows = self._read_retrying_ioerr(_read)
        except sqlite3.OperationalError as exc:
            if _missing_outcomes_table(exc):
                return []
            raise
        return [{**dict(row), "effective": bool(row["effective"])} for row in rows]

    def outcomes_for_sessions(self, session_ids: Iterable[str], *, per_session: int = 3,
                              include_agent: bool = True) -> dict[str, list[dict]]:
        """``{session_id: [outcome, ...]}`` of effective rows newest first, at most *per_session* each,
        lineage-matched like :meth:`list_session_outcomes`; sessions without outcomes are absent. For short
        id lists (a search page): one lineage query per id."""
        found = {}
        for sid in dict.fromkeys(s for s in session_ids if isinstance(s, str) and s):
            if rows := self.list_session_outcomes(sid, include_agent=include_agent, limit=per_session):
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
            rows.append((fields["signal"], outcome_domain(fields["signal"]), verdict, fields["source"],
                         fields["recorder"], fields["ref"], fields["note"], observed_at, float(recorded_at)))
        return rows

    @staticmethod
    def _insert_import_outcomes(conn: sqlite3.Connection, session_id: str, rows: list[tuple]) -> None:
        conn.executemany(f"INSERT OR IGNORE INTO session_outcomes ({', '.join(OUTCOME_COLUMNS[1:])}) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)", [(session_id, *row) for row in rows])

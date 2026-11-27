"""The candidate database.

A candidate is a proposed lesson: a rule, an explicit instruction, or a
reusable procedure the pipeline has classified from transcript evidence. It
is stored *structured* — scope, target, confidence, evidence count, source
hashes, status, conflict data — never as raw transcript text (requirement:
avoid raw transcript storage). ``body`` is the already-redacted, model- or
heuristic-written lesson text; nothing upstream of this module hands it raw
transcript excerpts.

One SQLite file, one table, matching the pattern the existing cron
scheduler already uses for its own run history
(``~/.hermes/cron/executions.db``). SQLite gives this pipeline exactly what
an append-only log doesn't: fast dedup lookups by key, and in-place status
transitions as a candidate collects more evidence across nightly runs.

Status lifecycle::

    pending -> applied              (threshold met, no conflict, written)
    pending -> conflict_blocked     (contradicts existing content; never auto-applied)
    pending -> excluded_repo_scope  (repo-specific; global auto-application forbidden)
    pending -> rejected             (operator rejected via the CLI)
    applied -> superseded           (rolled back, or replaced by a newer candidate)

Merging evidence for an already-``rejected`` or already-``applied`` row is a
deliberate no-op: a human decision (reject) or a completed action (applied)
is not silently re-opened by a new occurrence of the same lesson.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    canonical_key TEXT NOT NULL,
    category TEXT NOT NULL,
    scope TEXT NOT NULL,
    target_kind TEXT,
    target_path TEXT,
    title TEXT,
    body TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    session_ids TEXT NOT NULL DEFAULT '[]',
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    source_hashes TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    conflict_reason TEXT,
    reject_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_key ON candidates(canonical_key, scope);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
"""

_ACTIVE_STATUSES = ("pending", "conflict_blocked")


def candidate_id(canonical_key: str, scope: str) -> str:
    raw = f"{scope}|{canonical_key}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:24]


def source_hash(file_path: str, session_id: str, snippet: str) -> str:
    """A stable, non-reversible reference to *where* evidence came from.

    Deliberately hashes the file path + session id + snippet together
    rather than storing the snippet — this is a source *reference* for
    audit/debugging, not a way to reconstruct transcript content.
    """
    raw = f"{file_path}|{session_id}|{snippet}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CandidateStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or paths.candidates_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(_SCHEMA)
        self._conn.commit()
        # Owner read/write only, matching every other state file this
        # pipeline writes (checkpoints.json, audit.log.jsonl, backup JSON,
        # the notify queue). Rows here hold lesson text about the user's
        # real work, so this file gets the same floor as the others
        # regardless of the process umask. Applied every open, not just on
        # creation, so a file that predates this fix self-heals.
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CandidateStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writes ---------------------------------------------------------

    def upsert_occurrence(
        self,
        *,
        canonical_key: str,
        category: str,
        scope: str,
        target_kind: str | None,
        target_path: str | None,
        title: str,
        body: str,
        confidence: float,
        session_id: str,
        source_hash_value: str,
    ) -> str:
        """Record one occurrence of a lesson. Inserts or merges evidence.

        Returns the candidate's stable id either way.
        """
        cid = candidate_id(canonical_key, scope)
        now = _now()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT * FROM candidates WHERE id = ?", (cid,)
            )
            row = cur.fetchone()

            if row is None:
                cur.execute(
                    """
                    INSERT INTO candidates (
                        id, canonical_key, category, scope, target_kind,
                        target_path, title, body, confidence, evidence_count,
                        session_ids, occurrence_count, source_hashes, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 1, ?, 'pending', ?, ?)
                    """,
                    (
                        cid, canonical_key, category, scope, target_kind,
                        target_path, title, body, confidence,
                        json.dumps([session_id]),
                        json.dumps([source_hash_value]),
                        now, now,
                    ),
                )
                self._conn.commit()
                return cid

            status = row["status"]
            if status in ("applied", "rejected"):
                # Deliberate no-op: a completed action or a human rejection
                # is not silently reopened by another occurrence.
                return cid

            session_ids = set(json.loads(row["session_ids"] or "[]"))
            session_ids.add(session_id)
            source_hashes = set(json.loads(row["source_hashes"] or "[]"))
            # A replay-safe check: this exact source hash names one physical
            # transcript location (file + session + snippet). A crashed scan
            # that dies after this write commits but before the checkpoint
            # saves will re-read the same bytes on its next run and arrive
            # here again with the identical source_hash_value. Bumping the
            # counters unconditionally would count that single real event
            # twice, silently inflating a candidate toward its evidence
            # threshold from a crash, not from independent confirmation.
            # Only a hash this candidate has never seen before is new
            # evidence.
            is_new_evidence = source_hash_value not in source_hashes
            source_hashes.add(source_hash_value)

            if is_new_evidence:
                cur.execute(
                    """
                    UPDATE candidates SET
                        evidence_count = evidence_count + 1,
                        occurrence_count = occurrence_count + 1,
                        session_ids = ?,
                        source_hashes = ?,
                        confidence = MAX(confidence, ?),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(sorted(session_ids)),
                        json.dumps(sorted(source_hashes)),
                        confidence,
                        now,
                        cid,
                    ),
                )
            else:
                # Same evidence replayed - refresh bookkeeping fields only,
                # never the counters that feed thresholds.is_eligible().
                cur.execute(
                    """
                    UPDATE candidates SET
                        session_ids = ?,
                        source_hashes = ?,
                        confidence = MAX(confidence, ?),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(sorted(session_ids)),
                        json.dumps(sorted(source_hashes)),
                        confidence,
                        now,
                        cid,
                    ),
                )
            self._conn.commit()
            return cid

    def set_status(
        self, cid: str, status: str, *, reason: str | None = None
    ) -> None:
        now = _now()
        with closing(self._conn.cursor()) as cur:
            if status == "conflict_blocked":
                cur.execute(
                    "UPDATE candidates SET status=?, conflict_reason=?, updated_at=? WHERE id=?",
                    (status, reason, now, cid),
                )
            elif status == "rejected":
                cur.execute(
                    "UPDATE candidates SET status=?, reject_reason=?, updated_at=? WHERE id=?",
                    (status, reason, now, cid),
                )
            else:
                cur.execute(
                    "UPDATE candidates SET status=?, updated_at=? WHERE id=?",
                    (status, now, cid),
                )
        self._conn.commit()

    def mark_applied(self, cid: str, *, target_kind: str, target_path: str | None) -> None:
        """Set status to applied and record the target actually written.

        ``target_kind``/``target_path`` may differ from the row's original
        classification (the CLAUDE.md block overflow-to-rule routing case),
        so this always overwrites them with what was actually used.
        """
        now = _now()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE candidates SET status='applied', target_kind=?, target_path=?, updated_at=? WHERE id=?",
                (target_kind, target_path, now, cid),
            )
        self._conn.commit()

    def clear_rejection(self, cid: str) -> None:
        """Operator override: allow a previously rejected candidate to be
        reconsidered on its next occurrence."""
        now = _now()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE candidates SET status='pending', reject_reason=NULL, updated_at=? WHERE id=?",
                (now, cid),
            )
        self._conn.commit()

    # -- reads ------------------------------------------------------------

    def get(self, cid: str) -> dict[str, Any] | None:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT * FROM candidates WHERE id = ?", (cid,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def list_by_status(self, status: str) -> list[dict[str, Any]]:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT * FROM candidates WHERE status = ? ORDER BY updated_at",
                (status,),
            )
            rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_pending_global(self) -> list[dict[str, Any]]:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT * FROM candidates WHERE status = 'pending' AND scope = 'global' "
                "ORDER BY updated_at"
            )
            rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def all(self) -> list[dict[str, Any]]:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT * FROM candidates ORDER BY updated_at")
            rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def counts_by_status(self) -> dict[str, int]:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT status, COUNT(*) as n FROM candidates GROUP BY status")
            rows = cur.fetchall()
        return {r["status"]: r["n"] for r in rows}


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["session_ids"] = json.loads(d.get("session_ids") or "[]")
    d["source_hashes"] = json.loads(d.get("source_hashes") or "[]")
    return d

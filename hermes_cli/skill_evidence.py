"""P3.4 — Canonical skill health evidence (schema, writer, read API).

Single source contract for skill health scorecards.  Supersedes the
unowned live ``skill_health_scores`` scaffold (see
``docs/governance/skill-evidence-provenance.md``).

Evidence dimensions:
  * trigger    — skill loaded/used when relevant
  * compliance — required procedure followed
  * boundary   — forbidden overreach/bypass did not occur

Rules encoded here:
  * ``insufficient_evidence`` whenever the event taxonomy cannot support
    a dimension — never a fabricated 0/1/100%/neutral value;
  * every value carries window_start/window_end, skill_id, n_samples,
    exact source-event references/counts, method and version;
  * no invented denominators, no relevance inferred from free text;
  * idempotent writes per (window, skill_id) at an explicit db path;
  * this build computes only against disposable copies — the live ledger
    is never written by this module during the governance build.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
METHOD = "skill-evidence"
INSUFFICIENT = "insufficient_evidence"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skill_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    trigger_evidence TEXT NOT NULL,
    compliance_evidence TEXT NOT NULL,
    boundary_evidence TEXT NOT NULL,
    flags_json TEXT NOT NULL DEFAULT '{}',
    method TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload_sha256 TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(window_start, window_end, skill_id)
)
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


def _aggregate_counts(
    conn: sqlite3.Connection, skill_id: str, window_start: int, window_end: int,
) -> dict[str, int]:
    """Safe aggregate counts from skill.* taxonomy.  Summaries only."""
    rows = conn.execute(
        "SELECT event_type, COUNT(*) FROM activity_events "
        "WHERE event_type LIKE 'skill.%' AND object_id = ? "
        "AND occurred_at >= ? AND occurred_at <= ? GROUP BY event_type",
        (skill_id, int(window_start), int(window_end)),
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def compute_scores(
    ledger_conn: sqlite3.Connection,
    skill_id: str,
    *,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    """Compute evidence-bearing scores from safe aggregate evidence.

    Trigger: supportable from skill.loaded / skill.borrowed events.
    Compliance / Boundary: the current taxonomy has no event type that
    proves procedure-following or absence-of-overreach, so both emit
    ``insufficient_evidence`` rather than a fabricated number.
    """
    counts = _aggregate_counts(ledger_conn, skill_id, window_start, window_end)
    samples = counts.get("skill.loaded", 0) + counts.get("skill.borrowed", 0)
    trigger = {
        "value": "evident" if samples > 0 else INSUFFICIENT,
        "n_samples": samples,
        "evidence_refs": {
            "skill.loaded": counts.get("skill.loaded", 0),
            "skill.borrowed": counts.get("skill.borrowed", 0),
        },
        "reason": (
            "loaded/borrowed events present in window"
            if samples > 0
            else "no trigger-supporting events in window and denominators "
                 "cannot be derived from the taxonomy"
        ),
    }
    compliance = {
        "value": INSUFFICIENT,
        "n_samples": 0,
        "evidence_refs": {},
        "reason": "no event type proving required-procedure compliance exists in the taxonomy",
    }
    boundary = {
        "value": INSUFFICIENT,
        "n_samples": 0,
        "evidence_refs": {},
        "reason": "no event type proving absence-of-overreach exists in the taxonomy",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "skill_id": skill_id,
        "window_start": window_start,
        "window_end": window_end,
        "trigger": trigger,
        "compliance": compliance,
        "boundary": boundary,
        "flags": {"compliance": "unsupported_by_taxonomy",
                  "boundary": "unsupported_by_taxonomy"},
    }


def write_scores(
    records: list[dict[str, Any]],
    *,
    db_path: str | Path,
) -> list[int]:
    """Write computed records to an EXPLICIT db path (idempotent).

    Never called against the live ledger during the governance build.
    Returns the row ids written (existing ids on idempotent re-run).
    """
    import hashlib
    import time

    target = sqlite3.connect(str(db_path))
    try:
        ensure_schema(target)
        ids: list[int] = []
        for record in records:
            payload = json.dumps(
                {k: record[k] for k in ("trigger", "compliance", "boundary", "flags")},
                sort_keys=True,
            )
            digest = hashlib.sha256(payload.encode()).hexdigest()
            now = int(time.time())
            cur = target.execute(
                "INSERT INTO skill_evidence "
                "(window_start, window_end, skill_id, n_samples, trigger_evidence, "
                " compliance_evidence, boundary_evidence, flags_json, method, version, "
                " payload_sha256, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(window_start, window_end, skill_id) DO NOTHING",
                (
                    str(record["window_start"]), str(record["window_end"]),
                    record["skill_id"],
                    int(record["trigger"].get("n_samples", 0)),
                    json.dumps(record["trigger"]), json.dumps(record["compliance"]),
                    json.dumps(record["boundary"]), json.dumps(record["flags"]),
                    record["method"], record["schema_version"], digest, now,
                ),
            )
            if cur.lastrowid:
                ids.append(int(cur.lastrowid))
            else:
                existing = target.execute(
                    "SELECT id FROM skill_evidence WHERE window_start = ? AND "
                    "window_end = ? AND skill_id = ?",
                    (str(record["window_start"]), str(record["window_end"]),
                     record["skill_id"]),
                ).fetchone()
                if existing:
                    ids.append(int(existing[0]))
        target.commit()
        return ids
    finally:
        target.close()


def read_scores(
    db_path: str | Path,
    *,
    skill_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Read API over an explicit database path.  Read-only."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        if skill_id:
            rows = con.execute(
                "SELECT * FROM skill_evidence WHERE skill_id = ? ORDER BY window_end, skill_id",
                (skill_id,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM skill_evidence ORDER BY window_end, skill_id"
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "window_start": r["window_start"],
                "window_end": r["window_end"],
                "skill_id": r["skill_id"],
                "n_samples": r["n_samples"],
                "trigger": json.loads(r["trigger_evidence"]),
                "compliance": json.loads(r["compliance_evidence"]),
                "boundary": json.loads(r["boundary_evidence"]),
                "flags": json.loads(r["flags_json"]),
                "method": r["method"],
                "version": r["version"],
                "payload_sha256": r["payload_sha256"],
            })
        return out
    finally:
        con.close()
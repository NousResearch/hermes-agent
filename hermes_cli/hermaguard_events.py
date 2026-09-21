"""P4.3 — Event-driven Hermaguard requirement/evidence path.

Corrected per audit correction handoff (C1–C4):

  C1 evidence binds to the EXACT requirement: a hermaguard_required event
     must exist for the same task with the SAME review_event_id, and the
     referenced review_requested event must exist and belong to that task.
  C2 gate-time tamper evidence: the gate re-opens the report under the
     supplied artifact_dir (realpath containment), recomputes SHA-256 and
     compares to the stored digest — modified/deleted/moved/symlink-
     escaped reports fail the gate.
  C3 exactly-once is enforced ATOMICALLY in SQLite via unique indexes on
     (task_id, kind, review_event_id); races resolve as idempotent
     replay, never duplicates, never errors.
  C4 default-off is enforced at every mutation boundary: emit, record,
     reconcile and hook all require explicit enabled mode; absent/off/
     invalid config produces zero writes.

Task-event interfaces (bounded payloads, no title/body/prompt text):

    kind: hermaguard_required
    payload:
        review_event_id: <int id of governing review_requested event>
        policy_version:  <bounded structured token>
        task_tier:       full | fast
        task_kind:       <bounded structured token>
        reason_codes:    [<bounded structured token>]

    kind: hermaguard_evidence_recorded
    payload:
        review_event_id: <same governing review_requested event id>
        report:          <safe relative path in the task artifact dir>
        report_sha256:   <64 hex lowercase>
        status:          pass | fail | error
        version:         <semantic version>

Identity and idempotence:
  * The review cycle identity is the ``review_requested`` event id —
    never a timestamp.
  * Uniqueness invariant: UNIQUE(task_id, kind, review_event_id) on the
    event identity index; races are idempotent replay.

Behaviour gating:
  * Eligibility derives from structured tier/task_kind + the review
    transition, never keyword inference.
  * Full/high-risk tasks cannot pass review without valid evidence when
    the new mode is explicitly enabled (default OFF gate); fast tasks
    keep the existing sampling/failure policy.
  * Deterministic reconciliation finds missed requirements exactly once,
    only when the mode is explicitly enabled.  No cron, no cadence, no
    live schedule mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

POLICY_VERSION = "hermaguard-event-2"
MODE_CONFIG_KEY = "kanban.hermaguard_event_mode"  # default OFF

KIND_REQUIRED = "hermaguard_required"
KIND_EVIDENCE = "hermaguard_evidence_recorded"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# R2-2/R2-3: atomic uniqueness invariants (installed via execute(), never
# executescript — Python SQLite's executescript implicitly COMMITS the
# caller's pending transaction).
_UNIQUENESS_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_hermaguard_task_kind_review "
    "ON task_events (task_id, kind, "
    "CAST(json_extract(payload, '$.review_event_id') AS INTEGER)) "
    "WHERE kind IN ('hermaguard_required', 'hermaguard_evidence_recorded')"
)


def _bounded(token: str) -> bool:
    return bool(re.match(r"^[a-z0-9][a-z0-9_\-.]{0,63}$", token))


def ensure_uniqueness_invariants(conn: sqlite3.Connection) -> None:
    """Install the atomic uniqueness invariants.

    R2-2: uses execute() (transaction-preserving) so a caller's open
    transaction is NOT implicitly committed.  Raises MigrationBlocked on
    legacy duplicate residue (R2-3) — never silently dedupes.
    """
    result = ensure_uniqueness_invariants_safe(conn)
    if not result["installed"]:
        raise MigrationBlocked(result)


def ensure_uniqueness_invariants_safe(conn: sqlite3.Connection) -> dict[str, Any]:
    """R2-3: fail-closed migration for the uniqueness invariant.

    Returns {"installed": bool, "duplicate_groups": int, "detail": str}.
    Detects legacy duplicate (task_id, kind, review_event_id) groups BEFORE
    index creation; never deletes or rewrites append-only governance
    evidence.  Use :func:`migration_plan` for a separately callable report;
    applying destructive dedupe requires later approval.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='ux_hermaguard_task_kind_review'"
    ).fetchone()
    if row is not None:
        return {"installed": True, "duplicate_groups": 0,
                "detail": "invariant already installed"}
    dupes = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT task_id, kind, "
        "  CAST(json_extract(payload, '$.review_event_id') AS INTEGER) AS rid"
        "  FROM task_events WHERE kind IN (?, ?)"
        "  GROUP BY task_id, kind, rid HAVING COUNT(*) > 1)",
        (KIND_REQUIRED, KIND_EVIDENCE),
    ).fetchone()[0]
    if dupes:
        return {
            "installed": False,
            "duplicate_groups": int(dupes),
            "detail": (
                "legacy duplicate hermaguard evidence rows block the unique "
                "index; run migration_plan() and reconcile separately"
            ),
        }
    try:
        conn.execute(_UNIQUENESS_SQL)
        return {"installed": True, "duplicate_groups": 0,
                "detail": "invariant installed"}
    except sqlite3.IntegrityError:
        # Race with another installer that found duplicates first.
        return {"installed": False, "duplicate_groups": -1,
                "detail": "index creation failed; inspect task_events residue"}


def migration_plan(conn: sqlite3.Connection) -> dict[str, Any]:
    """R2-3: read-only reconciliation plan for legacy duplicates.

    Reports ONLY safe identifiers/counts — never payloads.  Applying any
    dedupe is outside this build and requires later approval.
    """
    rows = conn.execute(
        "SELECT task_id, kind, "
        "CAST(json_extract(payload, '$.review_event_id') AS INTEGER) AS rid, "
        "COUNT(*) AS n FROM task_events WHERE kind IN (?, ?) "
        "GROUP BY task_id, kind, rid HAVING COUNT(*) > 1 ORDER BY task_id",
        (KIND_REQUIRED, KIND_EVIDENCE),
    ).fetchall()
    affected = sorted({r["task_id"] for r in rows})
    return {
        "duplicate_groups": len(rows),
        "affected_task_ids": list(affected),
        "total_surplus_rows": int(sum(max(0, r["n"] - 1) for r in rows)),
        "plan": "dedupe requires separate operator approval; this build "
                "fails closed and preserves append-only history",
    }


class MigrationBlocked(RuntimeError):
    """R2-3: uniqueness invariant cannot install over duplicate residue."""


def _resolve_mode(force_mode: Any) -> bool:
    """R2-9: strict, consistent mode resolution.

    Accepts only None (defer to config), True or False.  Any other type
    (strings, ints, etc.) fails closed → False.  The explicit boolean is
    honoured consistently at every boundary.
    """
    if force_mode is None:
        return event_mode_enabled()
    if force_mode is True:
        return True
    if force_mode is False:
        return False
    return False  # invalid type fails closed


def event_mode_enabled(kanban_cfg: Optional[dict] = None) -> bool:
    """Dedicated default-OFF gate protecting the new behaviour.

    R3-3: when no explicit ``kanban_cfg`` is supplied the mode is resolved
    through the CANONICAL config loader (``hermes_cli.config.load_config``).
    The previous implementation imported ``get_kanban_config`` — a symbol
    that does not exist in this repo — inside a try/except, so the ImportError
    was swallowed and the mode was hard-locked to OFF: a real
    ``kanban.hermaguard_event_mode: true`` in ``config.yaml`` could never
    enable emission.

    Absent, empty, malformed or falsy config → False.  Only an explicit
    truthy ``kanban.hermaguard_event_mode`` enables mutations.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            loaded = load_config()
        except Exception:
            return False
        if not isinstance(loaded, dict):
            return False
        kanban_cfg = loaded.get("kanban")
    if not isinstance(kanban_cfg, dict):
        return False
    return kanban_cfg.get("hermaguard_event_mode") is True


def _review_eligible(task_tier: str, task_kind: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    tier = (task_tier or "").lower().strip()
    if tier not in ("full", "fast"):
        return False, ["tier:unclassified"]
    reasons.append("tier:" + tier)
    if task_kind and _bounded(str(task_kind)):
        reasons.append("kind:" + str(task_kind))
    return True, reasons


def _review_event_exists(
    conn: sqlite3.Connection, task_id: str, review_event_id: int,
) -> bool:
    """The referenced review_requested event must exist AND belong to the task."""
    row = conn.execute(
        "SELECT 1 FROM task_events WHERE id = ? AND task_id = ? "
        "AND kind = 'review_requested'",
        (int(review_event_id), task_id),
    ).fetchone()
    return row is not None


def emit_requirement_on_review(
    conn: sqlite3.Connection,
    task_id: str,
    review_event_id: int,
    *,
    policy_version: str = POLICY_VERSION,
    force_mode: Optional[bool] = None,
) -> Optional[int]:
    """Record exactly one hermaguard_required event for a review cycle.

    C4: writes ONLY when the event mode is explicitly enabled (via config
    or the ``force_mode`` opt-in argument).  Absent/off/invalid config →
    zero writes, returns None.
    Returns the new event row id, or None when disabled/ineligible/
    idempotent replay.
    """
    enabled = _resolve_mode(force_mode)
    if not enabled:
        return None
    trow = conn.execute(
        "SELECT tier, task_kind FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if trow is None:
        return None
    tier = trow[0] if not isinstance(trow, sqlite3.Row) else trow["tier"]
    task_kind = trow[1] if not isinstance(trow, sqlite3.Row) else trow["task_kind"]
    eligible, reasons = _review_eligible(tier, task_kind)
    if not eligible:
        return None
    # C1: the review event must exist and belong to this task.
    if not _review_event_exists(conn, task_id, review_event_id):
        return None
    ensure_uniqueness_invariants(conn)
    payload = {
        "review_event_id": int(review_event_id),
        "policy_version": policy_version,
        "task_tier": str(tier),
        "task_kind": str(task_kind or "task"),
        "reason_codes": reasons,
    }
    try:
        cur = conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, ?)",
            (task_id, KIND_REQUIRED, json.dumps(payload, sort_keys=True), int(time.time())),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        # C3: concurrent/raced writer won the uniqueness race — idempotent replay.
        return None


def record_evidence(
    db_path: str | Path,
    task_id: str,
    *,
    review_event_id: int,
    artifact_dir: str | Path,
    report_relative: str,
    status: str,
    version: str,
    force_mode: Optional[bool] = None,
) -> Optional[int]:
    """Record an evidence event with bounded path + SHA-256 of the report.

    C1: requires an existing hermaguard_required event for THIS task with
    the SAME review_event_id, and that the referenced review_requested
    event exists and belongs to this task.
    C3: atomic via the uniqueness invariant; races are idempotent replay.
    Returns the event row id or None on rejection/idempotent replay.
    """
    enabled = _resolve_mode(force_mode)
    if not enabled:
        return None
    if status not in ("pass", "fail", "error"):
        return None
    if not _VERSION_RE.match(version or ""):
        return None
    if not isinstance(report_relative, str) or not report_relative or os.path.isabs(report_relative):
        return None
    base = os.path.realpath(str(artifact_dir))
    candidate = os.path.realpath(os.path.join(base, report_relative))
    if candidate != base and not candidate.startswith(base + os.sep):
        return None
    if not os.path.isfile(candidate) or os.path.getsize(candidate) == 0:
        return None
    with open(candidate, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    conn = sqlite3.connect(str(db_path))
    try:
        # C1: requirement must exist for THIS task + SAME review cycle.
        requirement = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? "
            "AND CAST(json_extract(payload, '$.review_event_id') AS INTEGER) = ?",
            (task_id, KIND_REQUIRED, int(review_event_id)),
        ).fetchone()
        if requirement is None:
            return None
        # C1: the referenced review event must exist and belong to the task.
        if not _review_event_exists(conn, task_id, review_event_id):
            return None
        # C1 (stale): refuse evidence bound to a superseded review cycle —
        # a newer review_requested event for this task means the cycle is
        # no longer current (reject/fix/re-review superseded it).
        newer = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
            "AND id > ? LIMIT 1",
            (task_id, int(review_event_id)),
        ).fetchone()
        if newer is not None:
            return None
        ensure_uniqueness_invariants(conn)
        payload = {
            "review_event_id": int(review_event_id),
            "report": report_relative,
            "report_sha256": digest,
            "status": status,
            "version": version,
        }
        try:
            cur = conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, ?)",
                (task_id, KIND_EVIDENCE, json.dumps(payload, sort_keys=True), int(time.time())),
            )
            conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            conn.rollback()
            return None  # idempotent replay
    finally:
        conn.close()


def evidence_valid_for_review(
    db_path: str | Path,
    task_id: str,
    *,
    review_event_id: int,
    artifact_dir: str | Path,
) -> bool:
    """R2-4: full chain validation at gate time.

    Evidence passes ONLY when ALL hold:
      1. the referenced review_requested event exists for the same task;
      2. a matching hermaguard_required event exists for the same task and
         review ID (evidence without a requirement never opens the gate);
      3. the evidence belongs to that exact cycle (not superseded);
      4. the report is contained, current, and its SHA-256 matches the
         stored digest;
      5. status == pass and the digest is well-formed.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Chain link 1: review event exists for this task.
        review_ok = conn.execute(
            "SELECT 1 FROM task_events WHERE id = ? AND task_id = ? "
            "AND kind = 'review_requested'",
            (int(review_event_id), task_id),
        ).fetchone()
        if review_ok is None:
            return False
        # Chain link 2: a requirement exists for the same task + cycle.
        requirement = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? "
            "AND CAST(json_extract(payload, '$.review_event_id') AS INTEGER) = ?",
            (task_id, KIND_REQUIRED, int(review_event_id)),
        ).fetchone()
        if requirement is None:
            return False
        # Chain link 3: evidence bound to the exact (non-superseded) cycle.
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "AND CAST(json_extract(payload, '$.review_event_id') AS INTEGER) = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id, KIND_EVIDENCE, int(review_event_id)),
        ).fetchone()
        if not row or not row[0]:
            return False
        payload = json.loads(row[0])
        if payload.get("status") != "pass":
            return False
        if int(payload.get("review_event_id", -1)) != int(review_event_id):
            return False
        stored = str(payload.get("report_sha256", ""))
        if not HEX64_RE.match(stored):
            return False
        # Chain link 4 (anti-supersession): no newer review cycle exists.
        newer = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
            "AND id > ? LIMIT 1",
            (task_id, int(review_event_id)),
        ).fetchone()
        if newer is not None:
            return False
        relative = payload.get("report")
        if not isinstance(relative, str) or not relative or os.path.isabs(relative):
            return False
        base = os.path.realpath(str(artifact_dir))
        candidate = os.path.realpath(os.path.join(base, relative))
        if candidate != base and not candidate.startswith(base + os.sep):
            return False  # traversal / symlink escape
        if not os.path.isfile(candidate) or os.path.getsize(candidate) == 0:
            return False  # missing / moved / empty
        with open(candidate, "rb") as handle:
            current = hashlib.sha256(handle.read()).hexdigest()
        return current == stored
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    finally:
        conn.close()


def gate_review_transition(
    db_path: str | Path,
    conn: sqlite3.Connection,
    task_id: str,
    *,
    review_event_id: int,
    artifact_dir: str | Path,
    kanban_cfg: Optional[dict] = None,
) -> tuple[bool, Optional[str]]:
    """Hermaguard gate consulted at the existing review transition.

    Only when the new mode is explicitly enabled: full-tier tasks need
    evidence bound to THIS review cycle whose report bytes STILL match
    the recorded digest.  Fast tasks keep the existing policy.  Default
    off never blocks.
    """
    if not event_mode_enabled(kanban_cfg):
        return True, None
    trow = conn.execute("SELECT tier FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if trow is None:
        return False, "task not found"
    tier = (trow["tier"] or "").lower()
    if tier != "full":
        return True, None  # fast tasks: existing sampling/failure policy
    if evidence_valid_for_review(db_path, task_id, review_event_id=review_event_id,
                                 artifact_dir=artifact_dir):
        return True, None
    return False, "full-tier task lacks valid tamper-evident hermaguard evidence for this review cycle"


def reconcile_missed_requirements(
    db_path: str | Path,
    *,
    now: Optional[int] = None,
    force_mode: Optional[bool] = None,
) -> dict[str, Any]:
    """Deterministic reconciliation of missed requirements.

    C4: writes ONLY when the event mode is explicitly enabled; off/absent
    config produces zero writes.  Each miss repaired exactly once via the
    uniqueness invariant.  Read-only otherwise.
    """
    enabled = _resolve_mode(force_mode)
    if not enabled:
        return {"created": [], "repaired": 0, "policy_version": POLICY_VERSION}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    created: list[dict[str, Any]] = []
    try:
        ensure_uniqueness_invariants(conn)
        rows = conn.execute(
            "SELECT e.task_id, e.id AS review_event_id "
            "FROM task_events e "
            "WHERE e.kind = 'review_requested' "
            "AND EXISTS (SELECT 1 FROM tasks t WHERE t.id = e.task_id "
            "            AND t.tier IN ('full','fast')) "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM task_events r WHERE r.task_id = e.task_id "
            "  AND r.kind = ? AND CAST(json_extract(r.payload, '$.review_event_id') AS INTEGER) = e.id)",
            (KIND_REQUIRED,),
        ).fetchall()
        for row in rows:
            event_id = emit_requirement_on_review(
                conn, row["task_id"], row["review_event_id"], force_mode=True
            )
            if event_id is not None:
                created.append({"task_id": row["task_id"],
                                "review_event_id": row["review_event_id"],
                                "event_id": event_id})
        conn.commit()
    finally:
        conn.close()
    return {"created": created, "repaired": len(created), "policy_version": POLICY_VERSION}


def hook_request_review(
    conn: sqlite3.Connection, task_id: str, review_event_id: int,
    *, force_mode: Optional[bool] = None,
) -> None:
    """Call-site hook for request_review: best-effort, never raises.

    C4/R2-9: resolves the effective mode ONCE via _resolve_mode — an
    explicit True/False is honoured consistently (config off + explicit
    True still records; config on + explicit False still no-ops); invalid
    types fail closed.
    """
    try:
        if not _resolve_mode(force_mode):
            return
        emit_requirement_on_review(conn, task_id, review_event_id, force_mode=True)
    except Exception:
        return
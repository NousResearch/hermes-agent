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

# C3: atomic exactly-once invariants (created idempotently).
_UNIQUENESS_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_hermaguard_task_kind_review
    ON task_events (
        task_id,
        kind,
        CAST(json_extract(payload, '$.review_event_id') AS INTEGER)
    )
    WHERE kind IN ('hermaguard_required', 'hermaguard_evidence_recorded')
"""


def _bounded(token: str) -> bool:
    return bool(re.match(r"^[a-z0-9][a-z0-9_\-.]{0,63}$", token))


def ensure_uniqueness_invariants(conn: sqlite3.Connection) -> None:
    """Create the atomic uniqueness invariants (idempotent)."""
    conn.executescript(_UNIQUENESS_SQL)


def event_mode_enabled(kanban_cfg: Optional[dict] = None) -> bool:
    """Dedicated default-OFF gate protecting the new behaviour.

    Absent, empty, malformed or falsy config → False.  Only an explicit
    truthy ``kanban.hermaguard_event_mode`` enables mutations.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import get_kanban_config
            kanban_cfg = get_kanban_config()
        except Exception:
            return False
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
    enabled = event_mode_enabled() if force_mode is None else force_mode
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
    enabled = event_mode_enabled() if force_mode is None else force_mode
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
    """C2: valid pass evidence for THIS cycle, with the report's CURRENT
    bytes still matching the stored digest (tamper-evident at gate time)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
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
    enabled = event_mode_enabled() if force_mode is None else force_mode
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

    C4: no-ops unless the event mode is explicitly enabled.
    """
    try:
        if not event_mode_enabled():
            return
        emit_requirement_on_review(conn, task_id, review_event_id, force_mode=True)
    except Exception:
        return
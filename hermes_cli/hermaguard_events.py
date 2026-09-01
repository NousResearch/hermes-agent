"""P4.3 — Event-driven Hermaguard requirement/evidence path.

Replaces keyword polling as the INTENDED primary path with explicit
structured task/review events, while the existing poller
(``scripts/hermaguard-gate.py``) remains a low-frequency reconciliation
capability.

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
  * Idempotence is (task_id, event kind, review_event_id), enforced in
    the same write transaction as the transition append.

Behaviour gating:
  * Eligibility derives from structured tier/task_kind + the review
    transition, never keyword inference over titles/bodies.
  * Exactly one requirement event per eligible task/review cycle.
  * Full/high-risk tasks cannot pass review without valid evidence when
    the new mode is explicitly enabled (default OFF gate); fast tasks
    keep the existing sampling/failure policy.
  * Deterministic reconciliation finds missed requirements exactly once;
    a second run emits nothing.  No production cadence is encoded here —
    cadence is a later scheduler/deployment decision.  No new cron is
    created and no live schedule is touched.
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

POLICY_VERSION = "hermaguard-event-1"
MODE_CONFIG_KEY = "kanban.hermaguard_event_mode"  # default OFF

KIND_REQUIRED = "hermaguard_required"
KIND_EVIDENCE = "hermaguard_evidence_recorded"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _bounded(token: str) -> bool:
    return bool(re.match(r"^[a-z0-9][a-z0-9_\-.]{0,63}$", token))


def event_mode_enabled(kanban_cfg: Optional[dict] = None) -> bool:
    """Dedicated default-OFF gate protecting the new behaviour."""
    if kanban_cfg is None:
        try:
            from hermes_cli.config import get_kanban_config
            kanban_cfg = get_kanban_config()
        except Exception:
            kanban_cfg = {}
    return bool(kanban_cfg.get("hermaguard_event_mode", False))


def _review_eligible(task_tier: str, task_kind: str) -> tuple[bool, list[str]]:
    """Structured eligibility: tier in (full, fast); kinds exclude pure
    system subtasks?  Kept permissive: eligibility is tier-driven, with
    reason codes explaining the decision."""
    reasons: list[str] = []
    tier = (task_tier or "").lower().strip()
    if tier not in ("full", "fast"):
        return False, ["tier:unclassified"]
    reasons.append("tier:" + tier)
    if task_kind and _bounded(str(task_kind)):
        reasons.append("kind:" + str(task_kind))
    return True, reasons


def emit_requirement_on_review(
    conn: sqlite3.Connection,
    task_id: str,
    review_event_id: int,
    *,
    policy_version: str = POLICY_VERSION,
) -> Optional[int]:
    """Record exactly one hermaguard_required event for a review cycle.

    Called at the existing review transition (request_review).  Eligibility
    derives from structured tier/task_kind.  Idempotent by
    (task_id, kind, review_event_id) inside the caller's transaction.
    Returns the new event row id, or None when ineligible/duplicate.
    """
    trow = conn.execute(
        "SELECT tier, task_kind FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if trow is None:
        return None
    eligible, reasons = _review_eligible(trow["tier"], trow["task_kind"])
    if not eligible:
        return None
    duplicate = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? "
        "AND json_extract(payload, '$.review_event_id') = ?",
        (task_id, KIND_REQUIRED, int(review_event_id)),
    ).fetchone()
    if duplicate:
        return None
    payload = {
        "review_event_id": int(review_event_id),
        "policy_version": policy_version,
        "task_tier": str(trow["tier"]),
        "task_kind": str(trow["task_kind"] or "task"),
        "reason_codes": reasons,
    }
    cur = conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (task_id, KIND_REQUIRED, json.dumps(payload, sort_keys=True), int(time.time())),
    )
    return int(cur.lastrowid)


def record_evidence(
    db_path: str | Path,
    task_id: str,
    *,
    review_event_id: int,
    artifact_dir: str | Path,
    report_relative: str,
    status: str,
    version: str,
) -> Optional[int]:
    """Record an evidence event with bounded path + SHA-256 of the report.

    The report must exist inside the task artifact directory; its content
    is NEVER stored — only the relative path and digest.  Idempotent per
    (task, kind, review_event_id).  Returns the event row id or None on
    rejection/duplicate.
    """
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
        duplicate = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? "
            "AND json_extract(payload, '$.review_event_id') = ?",
            (task_id, KIND_EVIDENCE, int(review_event_id)),
        ).fetchone()
        if duplicate:
            return None
        row = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, KIND_REQUIRED),
        ).fetchone()
        if row is None:
            return None  # no governing requirement: refuse
        payload = {
            "review_event_id": int(review_event_id),
            "report": report_relative,
            "report_sha256": digest,
            "status": status,
            "version": version,
        }
        cur = conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, ?)",
            (task_id, KIND_EVIDENCE, json.dumps(payload, sort_keys=True), int(time.time())),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def evidence_valid_for_review(
    db_path: str | Path,
    task_id: str,
    *,
    review_event_id: int,
) -> bool:
    """Valid (pass) evidence present for this review cycle?"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id, KIND_EVIDENCE),
        ).fetchone()
        if not row or not row[0]:
            return False
        payload = json.loads(row[0])
        return (
            payload.get("status") == "pass"
            and int(payload.get("review_event_id", -1)) == int(review_event_id)
            and bool(HEX64_RE.match(str(payload.get("report_sha256", ""))))
        )
    except Exception:
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
    valid evidence; fast tasks follow the existing sampling/failure
    policy (fast tasks always pass this gate — sampling is upstream).
    Default-off mode never blocks anything.
    """
    if not event_mode_enabled(kanban_cfg):
        return True, None
    trow = conn.execute("SELECT tier FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if trow is None:
        return False, "task not found"
    tier = (trow["tier"] or "").lower()
    if tier != "full":
        return True, None  # fast tasks: existing sampling/failure policy
    if evidence_valid_for_review(db_path, task_id, review_event_id=review_event_id):
        return True, None
    return False, "full-tier task lacks valid hermaguard evidence for this review cycle"


def reconcile_missed_requirements(
    db_path: str | Path,
    *,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Deterministic reconciliation of missed requirements.

    Finds tasks whose latest review_requested event has no matching
    hermaguard_required event and repairs each exactly once.  A second
    invocation emits nothing (idempotent).  Read-only otherwise; no cron,
    no cadence, no live schedule mutation.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    created: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT e.task_id, e.id AS review_event_id, t.tier, t.task_kind "
            "FROM task_events e JOIN tasks t ON t.id = e.task_id "
            "WHERE e.kind = 'review_requested' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM task_events r WHERE r.task_id = e.task_id "
            "  AND r.kind = ? AND json_extract(r.payload, '$.review_event_id') = e.id)",
            (KIND_REQUIRED,),
        ).fetchall()
        for row in rows:
            event_id = emit_requirement_on_review(
                conn, row["task_id"], row["review_event_id"]
            )
            if event_id is not None:
                created.append({"task_id": row["task_id"],
                                "review_event_id": row["review_event_id"],
                                "event_id": event_id})
        conn.commit()
    finally:
        conn.close()
    return {"created": created, "repaired": len(created), "policy_version": POLICY_VERSION}


def hook_request_review(conn: sqlite3.Connection, task_id: str, review_event_id: int) -> None:
    """Call-site hook for request_review: best-effort, never raises.

    Runs inside the review transition's write transaction when a caller
    opts in; failure to record a requirement never blocks the legacy
    transition (default-off behaviour preserved).
    """
    try:
        if not event_mode_enabled():
            return
        emit_requirement_on_review(conn, task_id, review_event_id)
    except Exception:
        return
"""P4.4 — Pilot evidence aggregation (instrumentation only).

Builds the evidence contract needed for later 2–3 REAL full-pipeline
pilots.  No pilot tasks are created, no gate is manually advanced, no
stage cron or bypass path is added, and no second source of task truth
exists: every metric derives from existing task/run/event records.

The pilot record captures:
  * task/pilot id and exact pipeline contract version;
  * elapsed time and human waiting time;
  * model/provider/token/cost where available — explicit
    ``unavailable`` fields otherwise (never fabricated);
  * handoff count;
  * revision/council loops;
  * failures/rework;
  * artifact completeness;
  * reviewer verdict and quality outcome;
  * exact evidence refs and final status.

Aggregation is deterministic and report generation idempotent.  A
SIMULATED fixture is labelled as such and can never satisfy the
real-pilot gate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

PILOT_EVIDENCE_VERSION = "pilot-evidence-1"
SIMULATED_LABEL = "SIMULATED"

REQUIRED_RECORD_FIELDS = (
    "pilot_id", "task_id", "pipeline_contract_version",
    "started_at", "ended_at", "human_waiting_seconds",
    "handoff_count", "revision_loops", "failures", "rework",
    "artifact_completeness", "reviewer_verdict", "final_status",
    "evidence_refs",
)


def _count_events(conn: sqlite3.Connection, task_id: str, kinds: tuple[str, ...]) -> int:
    placeholders = ",".join("?" * len(kinds))
    row = conn.execute(
        f"SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind IN ({placeholders})",
        (task_id, *kinds),
    ).fetchone()
    return int(row[0]) if row else 0


def build_pilot_record(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    pipeline_contract_version: str,
    artifact_dir: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """Aggregate a pilot record from existing task/run/event records.

    Deterministic: the same underlying records produce the same record.
    Returns None when the task does not exist.
    """
    trow = conn.execute(
        "SELECT id, created_at, completed_at, status FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if trow is None:
        return None

    started = trow["created_at"]
    ended = trow["completed_at"]
    elapsed = (
        int(ended) - int(started) if started and ended else None
    )

    runs_row = conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    run_count = int(runs_row[0]) if runs_row else 0

    handoffs = _count_events(conn, task_id, ("claimed",))
    revisions = _count_events(conn, task_id, ("changes_requested",))
    failures = _count_events(conn, task_id, ("gate_failed", "review_rejected"))
    rework = _count_events(conn, task_id, ("council_revise", "audit_revise"))
    human_gates = _count_events(conn, task_id, ("human_approved", "human_gate_stale_nudge"))

    completeness: dict[str, Any] = {"sources_used": ["tasks", "task_runs", "task_events"]}
    if artifact_dir is not None:
        base = Path(artifact_dir)
        expected = ["decompose-output.md", "decompose-tasks.json", "audit-report.md"]
        present = sorted(name for name in expected if (base / name).exists())
        completeness["expected"] = expected
        completeness["present"] = present
        completeness["complete"] = present == sorted(expected)

    # Model/provider/token/cost: the LIVE task_runs schema has no dedicated
    # model/provider/token columns (verified at the corrected base); the
    # fields are explicitly 'unavailable' — never fabricated.  If a future
    # schema adds them, read each field directly (do not select-then-discard
    # any column such as tokens_used).
    if _has_columns(conn, "task_runs", ("model_used", "provider_used")):
        usage = conn.execute(
            "SELECT model_used, provider_used FROM task_runs "
            "WHERE task_id = ? AND (model_used IS NOT NULL OR provider_used IS NOT NULL) "
            "LIMIT 1",
            (task_id,),
        ).fetchone()
    else:
        usage = None
    if usage and (usage[0] or usage[1]):
        model = {"model": usage[0], "provider": usage[1], "token_cost": "unavailable"}
    else:
        model = {"model": "unavailable", "provider": "unavailable", "token_cost": "unavailable"}

    return {
        "pilot_evidence_version": PILOT_EVIDENCE_VERSION,
        "pilot_id": f"pilot-{task_id}",
        "task_id": task_id,
        "pipeline_contract_version": pipeline_contract_version,
        "started_at": int(started) if started else None,
        "ended_at": int(ended) if ended else None,
        "elapsed_seconds": elapsed,
        "human_waiting_seconds": None,  # derivable only with gate-entry/exit event pairs
        "human_gate_events": human_gates,
        "human_waiting_evidence": "human_gate_events count only; per-wait durations "
                                  "require gate-entry/exit event pairs (recorded gap)",
        "handoff_count": handoffs,
        "revision_loops": revisions,
        "failures": failures,
        "rework": rework,
        "artifact_completeness": completeness,
        "model_provider": model,
        "reviewer_verdict": reviewer_verdict(conn, task_id),
        "final_status": trow["status"],
        "evidence_refs": {
            "task_rows": 1,
            "task_runs": run_count,
            "task_events": handoffs + revisions + failures + rework + human_gates,
        },
        "label": None,  # real pilot: unset; SIMULATED fixtures set their own label
    }


def reviewer_verdict(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_passed' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0]).get("verdict") or "pass"
    except (json.JSONDecodeError, TypeError):
        return None


def _has_columns(conn: sqlite3.Connection, table: str, cols: tuple[str, ...]) -> bool:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    return all(c in existing for c in cols)


def build_simulated_pilot_record(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    pipeline_contract_version: str,
) -> Optional[dict[str, Any]]:
    """A SIMULATED pilot record — labelled, never satisfies the real gate.

    C7: stamps ``ever_simulated`` so post-hoc relabelling cannot flip the
    real-pilot gate.
    """
    record = build_pilot_record(conn, task_id, pipeline_contract_version=pipeline_contract_version)
    if record is None:
        return None
    record["label"] = SIMULATED_LABEL
    record["ever_simulated"] = True
    return record


def mint_live_pilot_provenance(
    task_id: str,
    *,
    review_event_id: Optional[int],
    evidence_event_id: Optional[int],
    pipeline_contract_version: str,
    authorised_by: str,
    signing_key: Optional[bytes] = None,
) -> dict[str, Any]:
    """Mint trusted live-pilot provenance — HMAC-bound, NOT self-authorising.

    R2-5: the previous unkeyed SHA-256 let any caller self-mint authority.
    Provenance is now signed with HMAC over the bound identity using a
    caller-supplied secret key; the real-pilot gate verifies the signature
    with the SAME trusted key.  Calling the mint function without holding
    the trusted key confers nothing — a forgery with the wrong key fails
    verification at gate time.  No key may be committed, logged or read
    from live config in this build (tests supply fixture keys).
    """
    if not authorised_by or not str(authorised_by).strip():
        raise ValueError("live pilot provenance requires an explicit authorised_by operator identity")
    if not signing_key:
        raise ValueError(
            "live pilot provenance requires a trusted signing key; "
            "self-minted provenance confers no authority"
        )
    identity = f"{task_id}|{review_event_id}|{evidence_event_id}|{pipeline_contract_version}|{authorised_by}"
    import hmac as _hmac
    signature = _hmac.new(signing_key, identity.encode(), hashlib.sha256).hexdigest()
    return {
        "kind": "live-pilot-authorisation",
        "task_id": task_id,
        "review_event_id": review_event_id,
        "evidence_event_id": evidence_event_id,
        "pipeline_contract_version": pipeline_contract_version,
        "authorised_by": str(authorised_by),
        "provenance_signature": signature,
        # No key material stored; the key id is informational only.
        "provenance_sha256": hashlib.sha256(identity.encode()).hexdigest(),
    }


def real_pilot_gate_satisfied(
    record: dict[str, Any],
    *,
    expected_task_id: Optional[str] = None,
    expected_contract_version: Optional[str] = None,
    signing_key: Optional[bytes] = None,
) -> bool:
    """The real-pilot gate requires POSITIVE trusted live provenance.

    R2-5: the provenance block must carry an HMAC signature computed over
    the record's own task/review/evidence/contract/operator identities
    with the SAME trusted signing key the gate verifies with.  Calling the
    public mint helper without the trusted key cannot forge a passing
    record; transplanted provenance fails identity binding; SIMULATED /
    relabelled records stay permanently ineligible via ``ever_simulated``.
    """
    if record.get("label") == SIMULATED_LABEL or record.get("ever_simulated"):
        return False
    provenance = record.get("live_provenance")
    if not isinstance(provenance, dict):
        return False
    if provenance.get("kind") != "live-pilot-authorisation":
        return False
    if not signing_key:
        return False  # no trusted verifier → no authority (fail closed)
    import hmac as _hmac
    identity = (
        f"{provenance.get('task_id')}|{provenance.get('review_event_id')}|"
        f"{provenance.get('evidence_event_id')}|{provenance.get('pipeline_contract_version')}|"
        f"{provenance.get('authorised_by')}"
    )
    expected_sig = _hmac.new(
        signing_key, identity.encode(), hashlib.sha256
    ).hexdigest()
    supplied = provenance.get("provenance_signature")
    if not isinstance(supplied, str) or not _hmac.compare_digest(supplied, expected_sig):
        return False
    task_id = record.get("task_id")
    contract = record.get("pipeline_contract_version")
    if provenance.get("task_id") != task_id:
        return False
    if provenance.get("pipeline_contract_version") != contract:
        return False
    if expected_task_id is not None and task_id != expected_task_id:
        return False
    if expected_contract_version is not None and contract != expected_contract_version:
        return False
    return record.get("final_status") == "done" and record.get("reviewer_verdict") is not None


def report_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def render_report(records: list[dict[str, Any]], *, simulated: bool = False) -> str:
    """Deterministic markdown report; idempotent for identical inputs."""
    lines = ["# Pipeline pilot evidence", ""]
    if simulated:
        lines.append(f"**Label: {SIMULATED_LABEL}** — does not satisfy the real-pilot gate.")
        lines.append("")
    header = "| pilot | task | contract | elapsed_s | handoffs | revisions | failures | rework | verdict | status |"
    lines += [header, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r['pilot_id']} | {r['task_id']} | {r['pipeline_contract_version']} | "
            f"{r['elapsed_seconds']} | {r['handoff_count']} | {r['revision_loops']} | "
            f"{r['failures']} | {r['rework']} | {r['reviewer_verdict']} | {r['final_status']} |"
        )
    lines.append("")
    return "\n".join(lines)
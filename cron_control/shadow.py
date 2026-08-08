"""No-agent shadow scanner for the cron control-plane seam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import (
    collect_delivery_evidence,
    collect_dead_man_switch_evidence,
    collect_execution_evidence,
    collect_heartbeat_evidence,
    collect_job_registry_evidence,
    collect_provider_evidence,
    collect_state_store_evidence,
    read_execution_rows,
    read_job_rows,
)
from .store import append_audit_event, open_control_plane_db
from .store import record_incident
from .normalizer import utc_now_iso


def collect_shadow_snapshot(
    *,
    jobs_path: Path | None = None,
    executions_path: Path | None = None,
    control_plane_path: Path | None = None,
) -> dict[str, Any]:
    try:
        jobs = read_job_rows(jobs_path, include_disabled=True)
    except Exception:
        jobs = []
    try:
        executions = read_execution_rows(executions_path)
    except Exception:
        executions = []
    evidence = []
    evidence.extend(collect_job_registry_evidence(jobs_path))
    evidence.extend(collect_execution_evidence(executions_path))
    evidence.extend(collect_delivery_evidence(jobs, executions))
    evidence.extend(collect_provider_evidence(jobs))
    evidence.extend(collect_heartbeat_evidence())
    evidence.extend(collect_dead_man_switch_evidence())
    evidence.extend(collect_state_store_evidence((jobs_path, executions_path, control_plane_path)))
    return {
        "collected_at": utc_now_iso(),
        "jobs": jobs,
        "executions": executions,
        "evidence": evidence,
    }


def persist_shadow_snapshot(snapshot: dict[str, Any], control_plane_path: Path | None = None) -> None:
    conn = open_control_plane_db(control_plane_path)
    try:
        for evidence in snapshot.get("evidence", []):
            record_incident(
                conn,
                incident_id=str(evidence["incident_id"]),
                job_id=str(evidence["job_id"]),
                state="observed",
                evidence_state=str(evidence.get("validation") or "valid"),
                summary=f"{evidence['kind']} from {evidence['source']}",
                classifier_version="cron_control.shadow/v1",
            )
        for evidence in snapshot.get("evidence", []):
            record = dict(evidence)
            record.setdefault("source", "shadow")
            record.setdefault("validation", "valid")
            # Evidence tables and audit sink are the write boundary. Runtime
            # state stays untouched.
            from .store import record_evidence
            record_evidence(conn, record)
        record_incident(
            conn,
            incident_id="shadow-scan",
            job_id="shadow-scan",
            state="observed",
            evidence_state="valid",
            summary="shadow scan audit trail",
            classifier_version="cron_control.shadow/v1",
        )
        append_audit_event(
            conn,
            {
                "audit_id": f"au_{snapshot['collected_at'].replace(':', '').replace('-', '')}",
                "timestamp": snapshot["collected_at"],
                "incident_id": "shadow-scan",
                "job_id": "shadow-scan",
                "event_type": "incident_opened",
                "actor": {"type": "system", "id": "shadow-scanner"},
                "evidence_refs": [e["evidence_id"] for e in snapshot.get("evidence", [])[:5]],
                "result": "planned",
                "before_state": {},
                "after_state": {"evidence_count": len(snapshot.get("evidence", []))},
            },
        )
        conn.commit()
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Hermes Cron shadow scanner")
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--executions", type=Path, default=None)
    parser.add_argument("--control-plane", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    snapshot = collect_shadow_snapshot(
        jobs_path=args.jobs,
        executions_path=args.executions,
        control_plane_path=args.control_plane,
    )
    if args.persist:
        persist_shadow_snapshot(snapshot, args.control_plane)
    if args.json or True:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

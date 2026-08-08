"""Deterministic verdict evaluation for Hermes Cron control-plane evidence."""

from __future__ import annotations

import argparse
import json
import hashlib
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cron.provider_recovery as provider_recovery

from .normalizer import canonical_json, sha256_json, utc_now_iso
from .store import (
    append_audit_event,
    open_control_plane_db,
    record_evidence,
    record_incident,
    record_verdict,
)

CLASSIFIER_VERSION = "cron_control.evaluator/v1"


def _iso_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return (0, text)
    return (1, dt.astimezone(timezone.utc).isoformat())


def _sort_evidence(evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(item) for item in evidence],
        key=lambda item: (
            _iso_key(item.get("observed_at")),
            _iso_key(item.get("source_time")),
            str(item.get("evidence_id") or ""),
        ),
    )


def _latest_by_kind(evidence: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [item for item in _sort_evidence(evidence) if item.get("kind") == kind]


def _job_policy(job: dict[str, Any]) -> dict[str, Any]:
    policy = job.get("control_policy")
    return policy if isinstance(policy, dict) else {}


def _job_schedule_minutes(job: dict[str, Any]) -> int:
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        for key in ("minutes", "every_minutes", "interval_minutes"):
            value = schedule.get(key)
            if isinstance(value, int) and value > 0:
                return value
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return 30


def _max_runtime_seconds(job: dict[str, Any]) -> int:
    policy = _job_policy(job)
    for key in ("max_runtime_seconds", "max_runtime", "timeout_seconds"):
        value = policy.get(key)
        if isinstance(value, int) and value > 0:
            return value
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return max(900, _job_schedule_minutes(job) * 60)


def _policy_allows_auto_rerun(job: dict[str, Any]) -> bool:
    policy = _job_policy(job)
    rerun_policy = str(policy.get("rerun_policy") or "").strip().lower()
    return rerun_policy not in {"manual_only", "never", "blocked"}


def _is_idempotent(job: dict[str, Any]) -> bool:
    policy = _job_policy(job)
    if policy.get("idempotent") is False:
        return False
    side_effect_class = str(policy.get("side_effect_class") or "").strip().lower()
    if side_effect_class == "non_idempotent_write":
        return False
    return True


def _delivery_contract(job: dict[str, Any]) -> dict[str, Any]:
    policy = _job_policy(job)
    contract = policy.get("delivery_contract")
    return contract if isinstance(contract, dict) else {}


def _evidence_state(evidence: list[dict[str, Any]]) -> str:
    validations = {str(item.get("validation") or "").strip().lower() for item in evidence}
    if "conflicting" in validations:
        return "conflicting"
    if "unavailable" in validations:
        return "unavailable"
    if "malformed" in validations:
        return "partial"
    if not evidence:
        return "partial"
    return "complete"


def _sorted_refs(evidence: list[dict[str, Any]]) -> list[str]:
    refs = [str(item.get("evidence_id") or "").strip() for item in evidence]
    return [ref for ref in dict.fromkeys(refs) if ref]


def _provider_error_category(execution: dict[str, Any]) -> provider_recovery.FailureCategory:
    error = execution.get("error")
    return provider_recovery.classify_cron_error(str(error) if error is not None else None)


def _has_policy_block(evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        kind = item.get("kind")
        value = item.get("value")
        if kind == "execution_state" and isinstance(value, dict):
            error = str(value.get("error") or "").lower()
            if "policy block" in error or "route disallowed" in error:
                return True
        if kind == "provider_probe" and isinstance(value, dict):
            result = str(value.get("result") or "").lower()
            detail = str(value.get("detail") or "").lower()
            if result == "blocked" or "policy" in detail:
                return True
    return False


def _state_store_issue(evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest = _latest_by_kind(evidence, "state_store")
    if not latest:
        return None
    for item in reversed(latest):
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        status = str(value.get("status") or "").lower()
        if status in {"unavailable", "malformed", "conflicting"}:
            return item
        if str(item.get("validation") or "").lower() in {"unavailable", "malformed", "conflicting"}:
            return item
    return None


def _manual_repair_readback_failed(evidence: list[dict[str, Any]]) -> bool:
    has_repair = False
    for item in evidence:
        if item.get("kind") != "manual_override":
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        if str(value.get("action") or "").strip() != "repair_state_store":
            continue
        if str(value.get("status") or "").strip() == "completed":
            has_repair = True
    if not has_repair:
        return False
    state_store = _state_store_issue(evidence)
    return state_store is not None


def _has_successful_completion(evidence: list[dict[str, Any]]) -> bool:
    for item in reversed(_latest_by_kind(evidence, "execution_state")):
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        if str(value.get("status") or "").strip() == "completed":
            return True
    return False


def _latest_execution(evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = _latest_by_kind(evidence, "execution_state")
    return rows[-1] if rows else None


def _stale_running(evidence: list[dict[str, Any]], job: dict[str, Any]) -> dict[str, Any] | None:
    latest = _latest_execution(evidence)
    if latest is None:
        return None
    value = latest.get("value")
    if not isinstance(value, dict):
        return None
    status = str(value.get("status") or "").strip()
    if status not in {"running", "claimed"}:
        return None

    age = latest.get("freshness_seconds")
    try:
        age_seconds = int(age)
    except (TypeError, ValueError):
        age_seconds = 0
    if age_seconds < _max_runtime_seconds(job):
        return None

    worker_heartbeats = _latest_by_kind(evidence, "worker_heartbeat")
    if worker_heartbeats:
        heartbeat_value = worker_heartbeats[-1].get("value")
        if isinstance(heartbeat_value, dict):
            if str(heartbeat_value.get("status") or "").strip().lower() != "missing":
                return None
    return latest


def _recoverable_failure(evidence: list[dict[str, Any]], job: dict[str, Any]) -> dict[str, Any] | None:
    latest = _latest_execution(evidence)
    if latest is None:
        return None
    value = latest.get("value")
    if not isinstance(value, dict):
        return None
    if str(value.get("status") or "").strip() != "failed":
        return None
    if _has_policy_block(evidence):
        return None
    category = _provider_error_category(value)
    if category not in provider_recovery.RECOVERABLE_CATEGORIES:
        return None
    if not _policy_allows_auto_rerun(job):
        return None
    provider = str(job.get("provider") or "").strip()
    fallback = provider_recovery.find_fallback(provider) if provider else None
    if fallback is None:
        return None
    return latest


def _healthy_completion(evidence: list[dict[str, Any]], job: dict[str, Any]) -> bool:
    if not _has_successful_completion(evidence):
        return False
    delivery_contract = _delivery_contract(job)
    for item in reversed(_latest_by_kind(evidence, "execution_state")):
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        delivery_state = str(value.get("delivery_state") or "").strip().lower()
        if delivery_state == "suppressed":
            return bool(delivery_contract.get("suppressed_receipt_allowed", False))
    return True


def _delivery_conflict(
    evidence: list[dict[str, Any]],
    evidence_state: str,
) -> dict[str, Any] | None:
    latest_execution = _latest_execution(evidence)
    if latest_execution is None:
        return None
    value = latest_execution.get("value")
    if not isinstance(value, dict):
        return None
    if str(value.get("status") or "").strip() != "completed":
        return None

    provider_probes = _latest_by_kind(evidence, "provider_probe")
    if provider_probes:
        probe = provider_probes[-1]
        if str(probe.get("validation") or "").strip().lower() == "conflicting":
            probe_value = probe.get("value")
            if isinstance(probe_value, dict):
                status_code = probe_value.get("status_code")
                detail = str(probe_value.get("detail") or "").lower()
                if status_code == 429 or "rate limit" in detail:
                    return probe
                if str(probe_value.get("result") or "").strip().lower() == "blocked":
                    return probe

    execution_delivery_state = str(value.get("delivery_state") or "").strip().lower()
    if execution_delivery_state in {"", "null", "none"} and evidence_state == "conflicting":
        return latest_execution
    return None


def evaluate_job_verdict(
    job: dict[str, Any],
    evidence: Iterable[dict[str, Any]],
    *,
    incident_id: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic verdict for one job/evidence bundle."""
    job = dict(job or {})
    evidence_rows = _sort_evidence(evidence)
    evidence_refs = _sorted_refs(evidence_rows)
    if not evidence_refs:
        raise ValueError("verdict evaluation requires at least one evidence reference")

    incident = incident_id or str(evidence_rows[0].get("incident_id") or f"inc_{job.get('id') or 'unknown'}")
    job_id = str(job.get("id") or "unknown")
    evidence_state = _evidence_state(evidence_rows)
    blocked_by: list[str] = []

    verdict_state = "quarantined"
    recommended_action = "escalate_to_human"
    automatic_action_allowed = False
    rule_id = "SUSPECT_TO_QUARANTINED_PARTIAL_V1"

    if _manual_repair_readback_failed(evidence_rows):
        verdict_state = "quarantined"
        recommended_action = "escalate_to_human"
        automatic_action_allowed = False
        rule_id = "REPAIR_IN_PROGRESS_TO_QUARANTINED_FAILED_V1"
        blocked_by = ["repair_readback_failed", "state_store_unavailable"]
    elif _state_store_issue(evidence_rows) is not None:
        verdict_state = "systemic_failure"
        recommended_action = "repair_state_store"
        automatic_action_allowed = False
        rule_id = "HEALTHY_TO_SYSTEMIC_FAILURE_LEDGER_V1"
        blocked_by = ["state_store_unavailable"]
    elif _has_policy_block(evidence_rows):
        verdict_state = "human_required"
        recommended_action = "escalate_to_human"
        automatic_action_allowed = False
        rule_id = "TRANSIENT_TO_HUMAN_REQUIRED_POLICY_BLOCK_V1"
        blocked_by = ["policy_block", f"rerun_policy={_job_policy(job).get('rerun_policy', 'unknown')}"]
    elif _delivery_conflict(evidence_rows, evidence_state) is not None:
        verdict_state = "quarantined"
        recommended_action = "escalate_to_human"
        automatic_action_allowed = False
        rule_id = "SUSPECT_TO_QUARANTINED_PARTIAL_V1"
        blocked_by = ["delivery_receipt_conflict", "provider_429"]
    else:
        stale = _stale_running(evidence_rows, job)
        if stale is not None:
            if _is_idempotent(job) and _policy_allows_auto_rerun(job):
                verdict_state = "stale_running"
                recommended_action = "reset_job"
                automatic_action_allowed = False
                rule_id = "SUSPECT_TO_STALE_RUNNING_MAX_RUNTIME_V1"
                blocked_by = ["max_runtime_exceeded"]
            else:
                verdict_state = "human_required"
                recommended_action = "escalate_to_human"
                automatic_action_allowed = False
                rule_id = "STALE_RUNNING_TO_HUMAN_REQUIRED_NONIDEMPOTENT_V1"
                blocked_by = [
                    "non_idempotent" if not _is_idempotent(job) else "manual_rerun_policy",
                    "worker_heartbeat_missing",
                ]
        else:
            recoverable = _recoverable_failure(evidence_rows, job)
            if recoverable is not None:
                verdict_state = "recoverable"
                recommended_action = "switch_provider"
                automatic_action_allowed = False
                rule_id = "TRANSIENT_TO_RECOVERABLE_FALLBACK_OK_V1"
                blocked_by = ["provider_fallback_required"]
            elif _healthy_completion(evidence_rows, job):
                verdict_state = "healthy"
                recommended_action = "none"
                automatic_action_allowed = True
                rule_id = "SUSPECT_TO_HEALTHY_EVIDENCE_RECOVERED_V1"
                blocked_by = []
            else:
                verdict_state = "suspect"
                recommended_action = "none"
                automatic_action_allowed = True
                rule_id = "HEALTHY_TO_SUSPECT_2X_INTERVAL_V1"
                blocked_by = ["evidence_incomplete"]

    verdict = {
        "verdict_id": "vd_" + hashlib.sha256(
            canonical_json(
                {
                    "incident_id": incident,
                    "job_id": job_id,
                    "state": verdict_state,
                    "evidence_refs": evidence_refs,
                    "rule_id": rule_id,
                    "recommended_action": recommended_action,
                }
            ).encode("utf-8")
        ).hexdigest()[:16],
        "incident_id": incident,
        "job_id": job_id,
        "state": verdict_state,
        "evidence_state": evidence_state,
        "rule_id": rule_id,
        "evidence_refs": evidence_refs,
        "recommended_action": recommended_action,
        "automatic_action_allowed": automatic_action_allowed,
        "blocked_by": [item for item in dict.fromkeys(blocked_by) if item],
        "classified_at": utc_now_iso(),
        "classifier_version": CLASSIFIER_VERSION,
    }
    return verdict


def evaluate_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = snapshot.get("jobs")
    evidence = snapshot.get("evidence")
    if not isinstance(jobs, list):
        jobs = []
    if not isinstance(evidence, list):
        evidence = []

    evidence_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_by_job[str(item.get("job_id") or "unknown")].append(item)

    verdicts: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("id") or "unknown")
        job_evidence = evidence_by_job.get(job_id, [])
        if not job_evidence:
            continue
        verdicts.append(evaluate_job_verdict(job, job_evidence))
    verdicts.sort(key=lambda item: (str(item["job_id"]), str(item["verdict_id"])))
    return verdicts


def persist_verdicts(
    snapshot: dict[str, Any],
    verdicts: list[dict[str, Any]],
    control_plane_path: Path | None = None,
) -> None:
    conn = open_control_plane_db(control_plane_path)
    try:
        evidence_rows = snapshot.get("evidence") if isinstance(snapshot.get("evidence"), list) else []
        for verdict in verdicts:
            incident_id = str(verdict["incident_id"])
            job_id = str(verdict["job_id"])
            record_incident(
                conn,
                incident_id=incident_id,
                job_id=job_id,
                state=str(verdict["state"]),
                evidence_state=str(verdict["evidence_state"]),
                summary=f"{verdict['rule_id']} => {verdict['recommended_action']}",
                classifier_version=str(verdict["classifier_version"]),
            )
            for evidence in evidence_rows:
                if not isinstance(evidence, dict):
                    continue
                if str(evidence.get("job_id") or "") != job_id:
                    continue
                try:
                    record_evidence(conn, evidence)
                except sqlite3.IntegrityError:
                    pass
            try:
                record_verdict(conn, verdict)
            except sqlite3.IntegrityError:
                pass
            append_audit_event(
                conn,
                {
                    "audit_id": "au_" + uuid.uuid4().hex,
                    "timestamp": verdict["classified_at"],
                    "incident_id": incident_id,
                    "job_id": job_id,
                    "execution_id": verdict["evidence_refs"][0] if verdict["evidence_refs"] else None,
                    "event_type": "verdict_recorded",
                    "actor": {"type": "system", "id": "cron-control-evaluator"},
                    "evidence_refs": verdict["evidence_refs"],
                    "verdict_ref": verdict["verdict_id"],
                    "action": verdict["recommended_action"],
                    "result": "verified" if verdict["automatic_action_allowed"] else "planned",
                    "before_state": {},
                    "after_state": {"state": verdict["state"], "rule_id": verdict["rule_id"]},
                },
            )
        conn.commit()
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    from .shadow import collect_shadow_snapshot

    parser = argparse.ArgumentParser(description="Evaluate Hermes Cron control-plane evidence")
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--executions", type=Path, default=None)
    parser.add_argument("--control-plane", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args(argv)

    snapshot = collect_shadow_snapshot(
        jobs_path=args.jobs,
        executions_path=args.executions,
        control_plane_path=args.control_plane,
    )
    verdicts = evaluate_snapshot(snapshot)
    if args.persist:
        persist_verdicts(snapshot, verdicts, args.control_plane)
    print(json.dumps({"verdicts": verdicts}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

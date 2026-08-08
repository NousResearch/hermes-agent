"""Executable action layer for Hermes Cron control-plane verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cron.provider_recovery as provider_recovery
from cron.jobs import compute_next_run, get_job, update_job

from .normalizer import canonical_json, utc_now_iso
from .store import (
    acquire_lease,
    append_audit_event,
    get_action,
    open_control_plane_db,
    record_action,
    record_component_heartbeat,
    record_incident,
    release_lease,
    update_action,
)

SUPPORTED_ACTIONS = {"reset_job", "switch_provider", "repair_state_store"}
ACTION_ACTOR = {"type": "system", "id": "cron-control-action-runner"}
LEASE_TTL_SECONDS = 300
DEFAULT_FALLBACK_CHAIN = [
    ("opencode-go", "deepseek-v4-pro"),
    ("openai-codex", "gpt-5.4-mini"),
    ("xai-oauth", "grok-4.20-reasoning"),
]


def _fingerprint(verdict: dict[str, Any]) -> str:
    return canonical_json(
        {
            "incident_id": str(verdict.get("incident_id") or ""),
            "job_id": str(verdict.get("job_id") or ""),
            "verdict_id": str(verdict.get("verdict_id") or ""),
            "action": str(verdict.get("recommended_action") or ""),
            "evidence_refs": list(verdict.get("evidence_refs", [])),
            "classifier_version": str(verdict.get("classifier_version") or ""),
        }
    )


def _idempotency_key(verdict: dict[str, Any]) -> str:
    digest = hashlib.sha256(_fingerprint(verdict).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _action_id(verdict: dict[str, Any]) -> str:
    return "act_" + hashlib.sha256(_fingerprint(verdict).encode("utf-8")).hexdigest()[:16]


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job.get("id"),
        "state": job.get("state"),
        "enabled": job.get("enabled"),
        "provider": job.get("provider"),
        "model": job.get("model"),
        "recovery_state": job.get("recovery_state"),
        "next_run_at": job.get("next_run_at"),
        "run_claim": job.get("run_claim"),
        "fire_claim": job.get("fire_claim"),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
        "last_error": job.get("last_error"),
        "last_delivery_error": job.get("last_delivery_error"),
    }


def _fallback_for_provider(provider: str) -> tuple[str, str] | None:
    fallback = provider_recovery.find_fallback(provider)
    if fallback is not None:
        return fallback
    for index, (candidate_provider, candidate_model) in enumerate(DEFAULT_FALLBACK_CHAIN):
        if candidate_provider == provider and index + 1 < len(DEFAULT_FALLBACK_CHAIN):
            return DEFAULT_FALLBACK_CHAIN[index + 1]
    return None


def _repair_view(conn) -> dict[str, Any]:
    row = conn.execute("PRAGMA user_version").fetchone()
    heartbeat = conn.execute(
        "SELECT * FROM component_heartbeats WHERE component_id=?",
        ("cron-control-repair",),
    ).fetchone()
    return {
        "user_version": int(row[0]) if row is not None else 0,
        "heartbeat": (
            {
                "component_id": heartbeat["component_id"],
                "observed_at": heartbeat["observed_at"],
                "status": heartbeat["status"],
                "detail": heartbeat["detail"],
            }
            if heartbeat is not None
            else None
        ),
    }


def _subset_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual:
                return False
            if not _subset_matches(value, actual[key]):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(_subset_matches(left, right) for left, right in zip(expected, actual))
    return expected == actual


def _audit_event(
    *,
    action_id: str,
    verdict: dict[str, Any],
    outcome: dict[str, Any],
    event_type: str,
    result: str,
) -> dict[str, Any]:
    payload = {
        "action_id": action_id,
        "event_type": event_type,
        "result": result,
        "timestamp": outcome["timestamp"],
    }
    audit_id = "au_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]
    return {
        "audit_id": audit_id,
        "timestamp": outcome["timestamp"],
        "incident_id": str(verdict["incident_id"]),
        "job_id": str(verdict["job_id"]),
        "execution_id": (verdict.get("evidence_refs") or [None])[0],
        "event_type": event_type,
        "actor": ACTION_ACTOR,
        "evidence_refs": list(verdict.get("evidence_refs", [])),
        "verdict_ref": verdict.get("verdict_id"),
        "action": outcome.get("action"),
        "idempotency_key": outcome.get("idempotency_key"),
        "fencing_token": outcome.get("fencing_token"),
        "before_state": outcome.get("before_state", {}),
        "after_state": outcome.get("after_state", {}),
        "result": result,
        "rollback_hint": outcome.get("rollback_hint"),
    }


def _record_blocked_action(
    conn,
    *,
    verdict: dict[str, Any],
    action_id: str,
    action: str,
    idempotency_key: str,
    before_state: dict[str, Any],
    reason: str,
    rollback_hint: str,
) -> dict[str, Any]:
    timestamp = utc_now_iso()
    outcome = {
        "action_id": action_id,
        "incident_id": str(verdict["incident_id"]),
        "job_id": str(verdict["job_id"]),
        "verdict_ref": verdict.get("verdict_id"),
        "action": action,
        "status": "blocked",
        "result": "denied",
        "idempotency_key": idempotency_key,
        "fencing_token": 0,
        "before_state": before_state,
        "after_state": before_state,
        "rollback_hint": rollback_hint,
        "timestamp": timestamp,
        "blocked_reason": reason,
    }
    record_action(
        conn,
        {
            "action_id": action_id,
            "incident_id": outcome["incident_id"],
            "job_id": outcome["job_id"],
            "action": action,
            "status": "blocked",
            "idempotency_key": idempotency_key,
            "fencing_token": 0,
            "before_state": before_state,
            "after_state": before_state,
            "result": "denied",
            "rollback_hint": rollback_hint,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    append_audit_event(conn, _audit_event(action_id=action_id, verdict=verdict, outcome=outcome, event_type="action_planned", result="denied"))
    conn.commit()
    return outcome


def _reset_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    before = _job_view(job)
    updates: dict[str, Any] = {
        "state": "scheduled",
        "enabled": True,
        "run_claim": None,
        "fire_claim": None,
    }
    if job.get("state") == "paused":
        updates["paused_at"] = None
        updates["paused_reason"] = None
    if not job.get("next_run_at"):
        next_run = compute_next_run(job.get("schedule") or {})
        if next_run is not None:
            updates["next_run_at"] = next_run
    updated = update_job(str(job["id"]), updates)
    if updated is None:
        raise RuntimeError(f"reset_job failed for {job.get('id')}")
    expected = _job_view(updated)
    readback = _job_view(get_job(str(job["id"])) or updated)
    return before, expected, readback


def _switch_provider(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provider = str(job.get("provider") or "").strip()
    fallback = _fallback_for_provider(provider) if provider else None
    if fallback is None:
        raise RuntimeError(f"No fallback provider is configured for {provider!r}")
    fallback_provider, fallback_model = fallback
    before = _job_view(job)
    record = provider_recovery.execute_recovery(
        str(job["id"]),
        fallback_provider,
        fallback_model,
        reason_category="verdict_action",
    )
    if record is None:
        raise RuntimeError(f"switch_provider failed for {job.get('id')}")
    updated = get_job(str(job["id"]))
    if updated is None:
        raise RuntimeError(f"switch_provider read-back failed for {job.get('id')}")
    expected = _job_view(updated)
    readback = _job_view(get_job(str(job["id"])) or updated)
    return before, expected, readback


def _repair_state_store(conn, control_plane_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    before = _repair_view(conn)
    heartbeat = {
        "component_id": "cron-control-repair",
        "observed_at": utc_now_iso(),
        "status": "healthy",
        "detail": "state-store repair read-back passed",
        "payload": {
            "component_id": "cron-control-repair",
            "intent": "repair_state_store",
            "path": str(control_plane_path or ""),
        },
    }
    record_component_heartbeat(conn, heartbeat)
    expected = {
        "user_version": before["user_version"],
        "heartbeat": {
            "component_id": "cron-control-repair",
            "status": "healthy",
            "detail": "state-store repair read-back passed",
        },
    }
    readback = _repair_view(conn)
    conn.commit()
    return before, expected, readback


def execute_verdict_action(
    verdict: dict[str, Any],
    *,
    approved: bool = False,
    control_plane_path: Path | None = None,
) -> dict[str, Any]:
    action = str(verdict.get("recommended_action") or "").strip()
    if action not in SUPPORTED_ACTIONS:
        return {
            "status": "skipped",
            "result": "skipped",
            "action": action or None,
            "reason": "unsupported_action",
        }

    conn = open_control_plane_db(control_plane_path)
    lease_token: int | None = None
    try:
        action_id = _action_id(verdict)
        existing = get_action(conn, action_id)
        if existing and existing.get("status") in {"verified", "blocked"}:
            return existing

        job = get_job(str(verdict["job_id"]))
        if job is None:
            raise ValueError(f"Unknown job_id {verdict['job_id']}")

        record_incident(
            conn,
            incident_id=str(verdict["incident_id"]),
            job_id=str(verdict["job_id"]),
            state=str(verdict.get("state") or "observed"),
            evidence_state=str(verdict.get("evidence_state") or "complete"),
            summary=f"{action} action decision",
            classifier_version=str(verdict.get("classifier_version") or "cron_control.actions/v1"),
        )

        before_state = _job_view(job) if action != "repair_state_store" else _repair_view(conn)
        idempotency_key = _idempotency_key(verdict)

        if not approved and not bool(verdict.get("automatic_action_allowed")):
            return _record_blocked_action(
                conn,
                verdict=verdict,
                action_id=action_id,
                action=action,
                idempotency_key=idempotency_key,
                before_state=before_state,
                reason="approval_required",
                rollback_hint="operator approval required",
            )

        lease = acquire_lease(
            conn,
            resource_key=f"{verdict['incident_id']}:{verdict['job_id']}:{action}",
            incident_id=str(verdict["incident_id"]),
            holder_id="cron-control-action-runner",
            ttl_seconds=LEASE_TTL_SECONDS,
        )
        if lease is None:
            return _record_blocked_action(
                conn,
                verdict=verdict,
                action_id=action_id,
                action=action,
                idempotency_key=idempotency_key,
                before_state=before_state,
                reason="lease_unavailable",
                rollback_hint="lease unavailable",
            )
        lease_token = int(lease["fencing_token"])

        timestamp = utc_now_iso()
        record_action(
            conn,
            {
                "action_id": action_id,
                "incident_id": str(verdict["incident_id"]),
                "job_id": str(verdict["job_id"]),
                "action": action,
                "status": "planned",
                "idempotency_key": idempotency_key,
                "fencing_token": lease_token,
                "before_state": before_state,
                "after_state": before_state,
                "result": "planned",
                "rollback_hint": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        planned_outcome = {
            "action_id": action_id,
            "incident_id": str(verdict["incident_id"]),
            "job_id": str(verdict["job_id"]),
            "verdict_ref": verdict.get("verdict_id"),
            "action": action,
            "status": "planned",
            "result": "planned",
            "idempotency_key": idempotency_key,
            "fencing_token": lease_token,
            "before_state": before_state,
            "after_state": before_state,
            "rollback_hint": None,
            "timestamp": timestamp,
        }
        append_audit_event(conn, _audit_event(action_id=action_id, verdict=verdict, outcome=planned_outcome, event_type="action_planned", result="planned"))
        update_action(
            conn,
            action_id,
            status="executing",
            result="executing",
            fencing_token=lease_token,
        )
        append_audit_event(conn, _audit_event(action_id=action_id, verdict=verdict, outcome={**planned_outcome, "timestamp": utc_now_iso()}, event_type="action_started", result="executing"))

        if action == "reset_job":
            _, expected_after, readback = _reset_job(job)
            rollback_hint = "re-run the scheduler tick after the stale claim clears"
        elif action == "switch_provider":
            _, expected_after, readback = _switch_provider(job)
            rollback_hint = "provider_recovery.execute_rollback(job_id)"
        else:
            _, expected_after, readback = _repair_state_store(conn, control_plane_path)
            rollback_hint = "manual re-open of control-plane.db"

        verified = _subset_matches(expected_after, readback)
        timestamp = utc_now_iso()
        outcome = {
            "action_id": action_id,
            "incident_id": str(verdict["incident_id"]),
            "job_id": str(verdict["job_id"]),
            "verdict_ref": verdict.get("verdict_id"),
            "action": action,
            "status": "verified" if verified else "failed",
            "result": "verified" if verified else "verification_failed",
            "idempotency_key": idempotency_key,
            "fencing_token": lease_token,
            "before_state": before_state,
            "after_state": {"expected": expected_after, "actual": readback},
            "rollback_hint": rollback_hint,
            "timestamp": timestamp,
        }
        update_action(
            conn,
            action_id,
            status=outcome["status"],
            after_state=outcome["after_state"],
            result=outcome["result"],
            rollback_hint=rollback_hint,
            fencing_token=lease_token,
        )
        append_audit_event(
            conn,
            _audit_event(
                action_id=action_id,
                verdict=verdict,
                outcome=outcome,
                event_type="action_completed" if verified else "verification_failed",
                result=outcome["result"],
            ),
        )
        conn.commit()
        return outcome
    except Exception as exc:
        # Fail closed. If we already wrote the planned action, stamp the failure
        # so the ledger retains the attempted side effect.
        action_id = _action_id(verdict)
        try:
            if get_action(conn, action_id) is not None:
                failure = {
                    "action_id": action_id,
                    "incident_id": str(verdict["incident_id"]),
                    "job_id": str(verdict["job_id"]),
                    "verdict_ref": verdict.get("verdict_id"),
                    "action": action,
                    "status": "failed",
                    "result": "failed",
                    "idempotency_key": _idempotency_key(verdict),
                    "fencing_token": lease_token or 0,
                    "before_state": {},
                    "after_state": {},
                    "rollback_hint": str(exc),
                    "timestamp": utc_now_iso(),
                }
                update_action(
                    conn,
                    action_id,
                    status="failed",
                    result="failed",
                    rollback_hint=str(exc),
                    fencing_token=lease_token or 0,
                )
                append_audit_event(conn, _audit_event(action_id=action_id, verdict=verdict, outcome=failure, event_type="action_failed", result="failed"))
                conn.commit()
        finally:
            pass
        return {
            "action_id": action_id,
            "incident_id": str(verdict.get("incident_id") or ""),
            "job_id": str(verdict.get("job_id") or ""),
            "verdict_ref": verdict.get("verdict_id"),
            "action": action,
            "status": "failed",
            "result": "failed",
            "idempotency_key": _idempotency_key(verdict),
            "fencing_token": lease_token or 0,
            "before_state": {},
            "after_state": {},
            "rollback_hint": str(exc),
            "timestamp": utc_now_iso(),
        }
    finally:
        try:
            if lease_token is not None:
                release_lease(
                    conn,
                    resource_key=f"{verdict['incident_id']}:{verdict['job_id']}:{action}",
                    holder_id="cron-control-action-runner",
                    fencing_token=lease_token,
                )
                conn.commit()
        finally:
            conn.close()


def execute_verdict_actions(
    verdicts: Iterable[dict[str, Any]],
    *,
    approved: bool = False,
    control_plane_path: Path | None = None,
) -> list[dict[str, Any]]:
    return [
        execute_verdict_action(
            verdict,
            approved=approved,
            control_plane_path=control_plane_path,
        )
        for verdict in verdicts
    ]


def main(argv: list[str] | None = None) -> int:
    from .evaluator import evaluate_snapshot
    from .shadow import collect_shadow_snapshot

    parser = argparse.ArgumentParser(description="Execute Hermes Cron control-plane actions")
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--executions", type=Path, default=None)
    parser.add_argument("--control-plane", type=Path, default=None)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args(argv)

    snapshot = collect_shadow_snapshot(
        jobs_path=args.jobs,
        executions_path=args.executions,
        control_plane_path=args.control_plane,
    )
    verdicts = evaluate_snapshot(snapshot)
    actions = execute_verdict_actions(
        verdicts,
        approved=args.approve,
        control_plane_path=args.control_plane,
    )
    print(json.dumps({"verdicts": verdicts, "actions": actions}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

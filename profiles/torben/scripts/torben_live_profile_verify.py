from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.signal_coo.automation_policy import (
    ea_automation_decisions,
    load_torben_automation_policy,
    write_automation_policy_artifact,
)
from hermes_cli.signal_coo.gmail_realtime import write_json
from hermes_cli.signal_coo.live_profile_verify import (
    clear_live_profile_investigation_request,
    render_verification_failure,
    stage_live_profile_investigation_request,
    update_live_profile_alert_state,
    verify_torben_live_profile,
)

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")
DEFAULT_REPO_SNAPSHOT_HOME = Path("/Users/ericfreeman/.hermes/hermes-agent/profiles/torben")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _write_ea_loop_snapshot(*, profile_home: Path, verification_payload: dict) -> dict:
    state_dir = profile_home / "state"
    paths = {
        "morning_brief": state_dir / "torben-morning-brief-inbox-context-latest.json",
        "calendar_audit": state_dir / "torben-calendar-alignment-audit-latest.json",
        "calendar_sync": state_dir / "torben-calendar-alignment-sync-latest.json",
        "meeting_prep": state_dir / "torben-meeting-prep-watch-latest.json",
        "gmail_pubsub": state_dir / "torben-gmail-pubsub-pull-latest.json",
        "live_profile_verify": state_dir / "torben-live-profile-verify-latest.json",
    }
    artifacts = {name: _load_json(path) for name, path in paths.items()}
    calendar_sync = ((artifacts["calendar_sync"].get("ea") or {}).get("calendar_alignment_sync") or {})
    calendar_policy = ((artifacts["calendar_sync"].get("ea") or {}).get("calendar_alignment_policy") or {})
    gmail_diag = artifacts["gmail_pubsub"].get("diagnostics") or {}
    meeting_google = ((artifacts["meeting_prep"].get("source_diagnostics") or {}).get("google") or {}).get("audit") or {}
    morning_diag = artifacts["morning_brief"].get("diagnostics") or {}

    calendar_events_created = len(calendar_sync.get("created") or [])
    calendar_events_deleted = len(calendar_sync.get("deleted") or [])
    calendar_events_updated = 0
    external_mutations = max(
        _int(calendar_sync.get("external_mutations")),
        _int(gmail_diag.get("external_mutations")),
        _int(meeting_google.get("external_mutations")),
        _int(morning_diag.get("external_mutations")),
    )
    automation_policy = load_torben_automation_policy(profile_home=profile_home)
    automation_decisions = ea_automation_decisions(policy=automation_policy)
    policy_mode = str(calendar_policy.get("mode") or "").strip()
    policy_covers_calendar = bool(policy_mode == "auto_private_busy_block")
    mutation_status = "read_only"
    approval_status = "not_required_no_mutation"
    if external_mutations:
        mutation_status = "policy_approved_synthetic_calendar_alignment" if policy_covers_calendar else "approval_required"
        approval_status = "existing_policy" if policy_covers_calendar else "approval_required"

    missing = [name for name, payload in artifacts.items() if not payload]
    degraded = []
    if missing:
        degraded.append("missing_artifacts:" + ",".join(missing))
    for name, payload in artifacts.items():
        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            degraded.append(f"{name}_error")

    snapshot = {
        "task": "torben_ea_loop_snapshot",
        "generated_at": _utc_now(),
        "status": "pass" if not degraded and verification_payload.get("status") == "pass" else "degraded",
        "source": "torben_live_profile_verify",
        "evidence": [str(path) for path in paths.values()],
        "artifacts": {name: str(path) for name, path in paths.items()},
        "proposed_action": "read_only_brief_watchdog_meeting_prep_and_gmail_pull",
        "approval_required": bool(external_mutations and not policy_covers_calendar),
        "approval_status": approval_status,
        "mutation_status": mutation_status,
        "failure_degraded_reason": "; ".join(degraded) if degraded else None,
        "emails_sent": 0,
        "emails_deleted": 0,
        "emails_archived": 0,
        "calendar_events_created": calendar_events_created,
        "calendar_events_updated": calendar_events_updated,
        "calendar_events_deleted": calendar_events_deleted,
        "external_mutations": external_mutations,
        "policy": {
            "calendar_alignment_mode": policy_mode or None,
            "calendar_alignment_policy_covers_synthetic_busy_blocks": policy_covers_calendar,
            "email_calendar_destructive_mutations_allowed": False,
        },
        "automation_policy": automation_decisions,
        "auto_invoke_allowed": bool((automation_decisions.get("recommendations") or {}).get("auto_invoke_allowed")),
        "recommendation_status": (
            "auto_surface_allowed"
            if (automation_decisions.get("recommendations") or {}).get("recommendation_allowed")
            else "auto_surface_blocked"
        ),
        "relationship_learning_status": (
            "question_auto_capture_allowed_answer_required_before_acting"
            if (automation_decisions.get("relationship_learning") or {}).get("recommendation_allowed")
            else "relationship_learning_blocked_by_policy"
        ),
        "source_summaries": {
            "gmail_pubsub_reason": artifacts["gmail_pubsub"].get("reason"),
            "meeting_prep_event_count": len(((artifacts["meeting_prep"].get("ea") or {}).get("calendar_events") or [])),
            "calendar_sync_created": calendar_events_created,
            "calendar_sync_deleted": calendar_events_deleted,
            "live_profile_status": verification_payload.get("status"),
        },
    }
    write_json(state_dir / "ea-loop-latest.json", snapshot)
    return snapshot


def main() -> int:
    profile_home = Path(os.getenv("TORBEN_PROFILE_HOME") or DEFAULT_TORBEN_HOME)
    repo_snapshot_home = Path(os.getenv("TORBEN_REPO_PROFILE_SNAPSHOT_HOME") or DEFAULT_REPO_SNAPSHOT_HOME)
    output_path = profile_home / "state" / "torben-live-profile-verify-latest.json"
    alert_state_path = profile_home / "state" / "torben-live-profile-verify-alert-state.json"
    investigation_request_path = profile_home / "state" / "torben-live-profile-investigation-request.json"
    payload = verify_torben_live_profile(
        profile_home=profile_home,
        repo_snapshot_home=repo_snapshot_home,
        check_snapshot_sync=not _truthy(os.getenv("TORBEN_VERIFY_SKIP_SNAPSHOT_SYNC")),
    )
    write_automation_policy_artifact(profile_home=profile_home)
    _write_ea_loop_snapshot(profile_home=profile_home, verification_payload=payload)
    if payload.get("status") == "pass":
        update_live_profile_alert_state(payload=payload, state_path=alert_state_path)
        clear_live_profile_investigation_request(
            payload=payload,
            request_path=investigation_request_path,
        )
        write_json(output_path, payload)
        print(json.dumps({"wakeAgent": False, "status": "pass", "task": "torben_live_profile_verify"}))
        return 0
    duplicate_suppressed = update_live_profile_alert_state(payload=payload, state_path=alert_state_path)
    if not duplicate_suppressed:
        stage_live_profile_investigation_request(
            payload=payload,
            request_path=investigation_request_path,
        )
    write_json(output_path, payload)
    if duplicate_suppressed:
        print(
            json.dumps(
                {
                    "wakeAgent": False,
                    "status": "duplicate_failure_suppressed",
                    "task": "torben_live_profile_verify",
                    "fingerprint": (payload.get("alert_dedupe") or {}).get("fingerprint"),
                }
            )
        )
        return 0
    print(render_verification_failure(payload), end="")
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-live-profile-verify", main))

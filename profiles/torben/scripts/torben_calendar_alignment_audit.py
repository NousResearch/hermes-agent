from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.calendar_audit import render_calendar_alignment_audit
from hermes_cli.signal_coo.calendar_sync import (
    calendar_alignment_sync_needs_attention,
    render_calendar_alignment_sync,
    sync_calendar_alignment_blocks,
)
from hermes_cli.signal_coo.google_evidence import collect_google_ea_evidence, write_json_artifact

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is available in Hermes uv, default remains safe otherwise.
    yaml = None


def _calendar_mutation_audit_path(home: Path) -> Path:
    raw_path = "state/torben-calendar-mutation-audit.jsonl"
    policy_path = home / "config" / "torben-automation-policy.yaml"
    if yaml is not None and policy_path.exists():
        try:
            policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
            calendar_policy = (((policy.get("ea") or {}).get("mutations") or {}).get("calendar_edit") or {})
            raw_path = str(calendar_policy.get("audit_log_path") or raw_path)
        except Exception:
            raw_path = "state/torben-calendar-mutation-audit.jsonl"
    audit_path = Path(raw_path)
    if not audit_path.is_absolute():
        audit_path = home / audit_path
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    return audit_path


def _append_mutation_audit(audit_path: Path, sync: dict, now: datetime) -> None:
    """Append an append-only record of synthetic busy-block mutations and breakers.

    Mode A is always-on auto private busy-block mirroring, so each live
    mutation and circuit-breaker trip must leave durable proof beyond the
    latest-state snapshot.
    """
    created = list(sync.get("created") or [])
    deleted = list(sync.get("deleted") or [])
    circuit_breakers = list(sync.get("circuit_breakers") or [])
    mutation_records = list(sync.get("mutation_records") or [])
    if not created and not deleted and not circuit_breakers and not mutation_records:
        return
    record = {
        "ts": now.isoformat().replace("+00:00", "Z"),
        "mode": "auto_private_busy_block",
        "dry_run": bool(sync.get("dry_run")),
        "audit_log_path": str(audit_path),
        "created": created,
        "deleted": deleted,
        "circuit_breakers": circuit_breakers,
        "mutation_records": mutation_records,
        "external_mutations": sync.get("external_mutations", 0),
        "mutation_cap": sync.get("mutation_cap", 0),
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _google_error_record(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _google_read_failure_payload(
    exc: Exception,
    *,
    now: datetime,
    lookahead_days: int,
    max_calendar_events: int,
) -> dict:
    error = _google_error_record(exc)
    generated_at = _iso(now)
    source_health = {
        "status": "failed",
        "component": "google_calendar_read",
        "provider": "google",
        "generated_at": generated_at,
        "error": error,
        "lookahead_days": lookahead_days,
        "max_calendar_events": max_calendar_events,
        "calendar_sync_skipped": True,
        "mutation_boundary": "No Google Calendar sync writes were attempted because source reads failed.",
        "google_write_api_calls": 0,
        "external_mutations": 0,
    }
    sync = {
        "status": "skipped_google_read_failure",
        "reason": "google_read_failure",
        "dry_run": False,
        "google_read_api_calls": 0,
        "google_write_api_calls": 0,
        "external_mutations": 0,
        "created": [],
        "would_create": [],
        "deleted": [],
        "would_delete": [],
        "already_exists": [],
        "skipped": [{"reason": "google_read_failure", "calendar_sync_writes_skipped": True}],
        "errors": [{"component": "google_calendar_read", **error}],
        "mutation_cap": 0,
    }
    return {
        "task": "torben_calendar_alignment_audit",
        "wakeAgent": True,
        "generated_at": generated_at,
        "status": "source_degraded",
        "error": error,
        "ea": {
            "calendar_events": [],
            "email_reply_candidates": [],
            "calendar_block_candidates": [],
            "open_loops": [],
            "calendar_alignment_sync": sync,
            "calendar_alignment_policy": {
                "lookahead_days": lookahead_days,
                "mode": "auto_private_busy_block",
                "read_failure_behavior": "fail_soft_skip_calendar_sync_writes",
                "mutation_boundary": "No Google Calendar sync writes are attempted unless source reads succeed.",
            },
        },
        "source_diagnostics": {
            "google": {
                "accounts": [],
                "secondary_calendar_collection": False,
                "calendar_events_collected": 0,
                "email_messages_collected": 0,
                "calendar_block_candidates": 0,
                "source_health": source_health,
                "audit": {
                    "generated_at": generated_at,
                    "accounts_checked": [],
                    "google_read_api_calls": 0,
                    "google_write_api_calls": 0,
                    "external_mutations": 0,
                    "warnings": [
                        f"Google calendar read failed; skipped calendar sync writes ({error['type']}: {error['message']})"
                    ],
                },
            }
        },
    }


def _render_google_read_failure_report(payload: dict) -> str:
    google = ((payload.get("source_diagnostics") or {}).get("google") or {})
    health = google.get("source_health") or {}
    error = health.get("error") or payload.get("error") or {}
    return (
        "Torben calendar alignment: skipped sync because Google reads failed.\n"
        f"Reason: {error.get('type') or 'Error'}: {error.get('message') or 'unknown'}\n"
        "Google writes: 0. External mutations: 0.\n"
        "No calendar busy-block create/delete was attempted.\n"
    )


def _write_google_read_failure_artifacts(state_dir, payload: dict) -> None:
    output_path = state_dir / "torben-calendar-alignment-audit-latest.json"
    report_path = state_dir / "torben-calendar-alignment-audit-latest.txt"
    sync_output_path = state_dir / "torben-calendar-alignment-sync-latest.json"
    sync_report_path = state_dir / "torben-calendar-alignment-sync-latest.txt"
    health_path = state_dir / "torben-calendar-alignment-source-health-latest.json"
    write_json_artifact(payload, output_path)
    report_path.write_text(render_calendar_alignment_audit(payload), encoding="utf-8")
    write_json_artifact(payload, sync_output_path)
    sync_report = _render_google_read_failure_report(payload)
    sync_report_path.write_text(sync_report, encoding="utf-8")
    source_health = (((payload.get("source_diagnostics") or {}).get("google") or {}).get("source_health") or {})
    write_json_artifact(source_health, health_path)


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    # Mode A (auto private busy-block mirroring) is always-on by design:
    # create busy blocks when calendar events are created, delete busy blocks
    # when calendar events are deleted. The watchdog therefore runs LIVE and
    # cannot be downgraded to stage/dry-run via profile, gateway, or shell env.
    # Tests/rehearsals should call sync_calendar_alignment_blocks(..., dry_run=True)
    # directly, not change the production watchdog's mutation mode.
    dry_run = False
    max_mutations = int(os.getenv("TORBEN_CALENDAR_ALIGNMENT_MAX_MUTATIONS", "20"))
    lookahead_days = int(os.getenv("TORBEN_CALENDAR_ALIGNMENT_LOOKAHEAD_DAYS", "21"))
    max_calendar_events = int(os.getenv("TORBEN_CALENDAR_ALIGNMENT_MAX_EVENTS", "2500"))
    now = datetime.now(timezone.utc)
    try:
        payload = collect_google_ea_evidence(
            config_path=home / "config" / "google_accounts.yaml",
            now=now,
            days=lookahead_days,
            max_calendar_events=max_calendar_events,
            max_email_messages=0,
            max_calendar_block_candidates=None,
            include_secondary_calendars=False,
        )
    except Exception as exc:  # noqa: BLE001 - fail-soft source-health artifact for cron reliability.
        payload = _google_read_failure_payload(
            exc,
            now=now,
            lookahead_days=lookahead_days,
            max_calendar_events=max_calendar_events,
        )
        _write_google_read_failure_artifacts(state_dir, payload)
        print(_render_google_read_failure_report(payload), end="")
        return 0
    output_path = state_dir / "torben-calendar-alignment-audit-latest.json"
    report_path = state_dir / "torben-calendar-alignment-audit-latest.txt"
    write_json_artifact(payload, output_path)
    report = render_calendar_alignment_audit(payload)
    report_path.write_text(report, encoding="utf-8")
    candidates = list(((payload.get("ea") or {}).get("calendar_block_candidates") or []))
    events = list(((payload.get("ea") or {}).get("calendar_events") or []))
    sync = sync_calendar_alignment_blocks(
        config_path=home / "config" / "google_accounts.yaml",
        candidates=candidates,
        source_events=events,
        cleanup_stale=True,
        cleanup_window_start=now.isoformat().replace("+00:00", "Z"),
        cleanup_window_end=(now + timedelta(days=lookahead_days)).isoformat().replace("+00:00", "Z"),
        dry_run=dry_run,
        max_mutations=max_mutations,
    )
    payload.setdefault("ea", {})["calendar_alignment_sync"] = sync
    sync_output_path = state_dir / "torben-calendar-alignment-sync-latest.json"
    sync_report_path = state_dir / "torben-calendar-alignment-sync-latest.txt"
    write_json_artifact(payload, sync_output_path)
    _append_mutation_audit(_calendar_mutation_audit_path(home), sync, now)
    sync_report = render_calendar_alignment_sync(payload)
    sync_report_path.write_text(sync_report, encoding="utf-8")
    if not calendar_alignment_sync_needs_attention(sync):
        return 0
    print(sync_report, end="")
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-calendar-alignment-watchdog", main))

"""Tests for the anticipation dry-run script helpers."""

from datetime import datetime, timedelta, timezone

from agent.anticipation import AnticipationPermission
from agent.anticipation_policy import AnticipationCandidate, AnticipationDecision
from scripts import anticipation_run
from scripts.anticipation_run import (
    format_dry_run_decisions,
    history_from_decision_logs,
    run_router_monitor_dry_run,
    _runtime_config_for_loop,
)

NOW = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)


def candidate(**overrides):
    values = {
        "loop_id": "stale_task_resurfacer",
        "title": "Stale thread: Hermes anticipation",
        "body": "Next I can add a dry-run command.",
        "confidence": 0.82,
        "proposed_permission": AnticipationPermission.SUGGEST,
        "dedupe_key": "dedupe",
        "created_at": NOW,
    }
    values.update(overrides)
    return AnticipationCandidate(**values)


def test_format_dry_run_decisions_reports_decision_without_delivery():
    decision = AnticipationDecision(action="suggest", reason="passed", candidate=candidate())

    output = format_dry_run_decisions([decision])

    assert "DRY RUN" in output
    assert "No messages were delivered" in output
    assert "Stale thread: Hermes anticipation" in output
    assert "suggest" in output
    assert "0.82" in output
    assert "Next I can add a dry-run command." in output


def test_format_dry_run_decisions_handles_empty_candidates():
    output = format_dry_run_decisions([])

    assert "No stale-task candidates found" in output
    assert "No messages were delivered" in output


def test_format_dry_run_decisions_hides_body_for_skipped_candidates():
    decision = AnticipationDecision(
        action="skip",
        reason="anticipation_disabled",
        candidate=candidate(body="private raw snippet should not display"),
    )

    output = format_dry_run_decisions([decision])

    assert "private raw snippet" not in output
    assert "hidden because decision was skipped" in output


def test_history_from_decision_logs_hydrates_matching_candidate_dedupe_and_budget():
    prior_ts = NOW - timedelta(minutes=10)
    logs = [
        {
            "ts": prior_ts.isoformat(),
            "action": "suggest",
            "dedupe_key_hash": "2cf2fb7f867feffafbfe66a5b8e8822da70738406dd3b85a28c83892da6e50a5",
        }
    ]

    history = history_from_decision_logs(logs, [candidate()], now=NOW)

    assert history.recent_dedupe_keys["dedupe"] == prior_ts
    assert history.notifications_today == 1
    assert history.last_notification_at == prior_ts


def test_runtime_config_for_loop_parses_quoted_false_global_booleans(monkeypatch):
    monkeypatch.setattr(
        anticipation_run,
        "load_config",
        lambda: {
            "anticipation": {
                "enabled": "false",
                "quiet_hours": {"enabled": "false", "start": "22:00", "end": "08:00"},
                "notification_budget": {"max_per_day": 3, "min_minutes_between": 0},
                "loops": {
                    "router_monitor": {
                        "enabled": "true",
                        "schedule": "manual",
                        "permission": "ask_to_execute",
                        "min_confidence": 0.70,
                        "lookback_days": 1,
                    }
                },
            }
        },
    )

    runtime = _runtime_config_for_loop("router_monitor")

    assert runtime.enabled is False
    assert runtime.loop_enabled is True
    assert runtime.quiet_hours_enabled is False


def test_run_router_monitor_dry_run_uses_policy_and_audit_log(monkeypatch):
    appended = []
    monkeypatch.setattr(
        anticipation_run,
        "load_config",
        lambda: {
            "anticipation": {
                "enabled": True,
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
                "notification_budget": {"max_per_day": 3, "min_minutes_between": 0},
                "loops": {
                    "router_monitor": {
                        "enabled": True,
                        "schedule": "manual",
                        "permission": "ask_to_execute",
                        "min_confidence": 0.70,
                        "lookback_days": 1,
                    }
                },
            }
        },
    )
    monkeypatch.setattr(anticipation_run, "read_recent_decision_logs", lambda limit=200: [])
    monkeypatch.setattr(anticipation_run, "append_decision_log", appended.append)
    snapshot = {
        "monitoring": {
            "quarantine_dir_exists": True,
            "cron_entries": [],
            "cron_log_last_at": "2026-05-05T17:00:00+00:00",
        },
        "unknown_devices": [],
    }

    decisions = run_router_monitor_dry_run(snapshot=snapshot, limit=3, now=NOW)

    assert len(decisions) == 1
    assert decisions[0].action == "ask_to_execute"
    assert decisions[0].candidate.loop_id == "router_monitor"
    assert appended == decisions


def test_router_monitor_cli_requires_dry_run():
    assert anticipation_run.main(["router_monitor"]) == 2


def test_router_monitor_cli_reports_bad_snapshot_path(capsys):
    assert anticipation_run.main(["router_monitor", "--dry-run", "--snapshot", "/no/such/snapshot.json"]) == 2
    captured = capsys.readouterr()
    assert "Unable to read router snapshot" in captured.err


def test_router_monitor_cli_json_outputs_decisions(monkeypatch, capsys):
    monkeypatch.setattr(
        anticipation_run,
        "run_router_monitor_dry_run",
        lambda **kwargs: [
            AnticipationDecision(
                action="silent_log",
                reason="passed",
                candidate=candidate(loop_id="router_monitor", title="Router one-time randomized unknown"),
            )
        ],
    )

    assert anticipation_run.main(["router_monitor", "--dry-run", "--snapshot", "dummy.json", "--json"]) == 0
    output = capsys.readouterr().out
    assert "Router one-time randomized unknown" in output
    assert "silent_log" in output

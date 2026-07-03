from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pytest


def test_standing_orders_markdown_is_loaded_and_parsed(tmp_path: Path):
    from proactive.standing_orders import load_standing_orders

    orders_path = tmp_path / "System" / "Standing Orders.md"
    orders_path.parent.mkdir(parents=True)
    orders_path.write_text(
        """# Standing Orders

## Daily Review
- scope: personal operations
- trigger: schedule: every 60 minutes
- allowed_actions: read_obsidian, status_check
- approval_gates: external_message, production_change
- escalation_rules: stop on secrets or money movement
- output_policy: silent unless anomaly
""",
        encoding="utf-8",
    )

    orders = load_standing_orders(orders_path)

    assert len(orders) == 1
    assert orders[0].name == "Daily Review"
    assert orders[0].scope == "personal operations"
    assert orders[0].trigger == "schedule: every 60 minutes"
    assert orders[0].allowed_actions == ["read_obsidian", "status_check"]
    assert orders[0].approval_gates == ["external_message", "production_change"]


def test_tool_policy_classifies_auto_confirm_and_deny(tmp_path: Path):
    from proactive.tool_policy import PolicyLevel, decide_action, load_tool_policy

    policy_path = tmp_path / "tool-policy.yaml"
    policy_path.write_text(
        """AUTO_ALLOW:
  - read_obsidian
  - status_check
CONFIRM_FIRST:
  - external_message
DENY:
  - leak_secrets
""",
        encoding="utf-8",
    )
    policy = load_tool_policy(policy_path)

    assert decide_action("read_obsidian", policy).level is PolicyLevel.AUTO_ALLOW
    assert decide_action("external_message", policy).level is PolicyLevel.CONFIRM_FIRST
    assert decide_action("leak_secrets", policy).level is PolicyLevel.DENY
    assert decide_action("unknown_tool", policy).level is PolicyLevel.CONFIRM_FIRST


def test_heartbeat_returns_silent_and_writes_audit_when_no_anomaly(tmp_path: Path):
    from proactive.heartbeat import HeartbeatConfig, run_heartbeat

    vault = tmp_path / "vault"
    vault.mkdir()
    result = run_heartbeat(HeartbeatConfig(obsidian_vault=vault))

    assert result.output == "[SILENT]"
    assert result.notification_payload is None
    audit_files = list((vault / "System" / "Agent Runs").glob("*.md"))
    assert len(audit_files) == 1
    assert "whether_user_was_notified: false" in audit_files[0].read_text(encoding="utf-8")


def test_heartbeat_returns_notification_payload_for_open_decision(tmp_path: Path):
    from proactive.heartbeat import HeartbeatConfig, run_heartbeat

    vault = tmp_path / "vault"
    decisions = vault / "System" / "Decision Inbox"
    decisions.mkdir(parents=True)
    (decisions / "pricing.md").write_text("- [ ] decide pricing change\n", encoding="utf-8")

    result = run_heartbeat(HeartbeatConfig(obsidian_vault=vault))

    assert result.output != "[SILENT]"
    assert result.notification_payload is not None
    assert result.notification_payload["trigger_type"] == "heartbeat"
    assert "decision" in result.notification_payload["summary"].lower()


def test_heartbeat_is_silent_for_ordinary_waiting_for_kj_input(tmp_path: Path):
    from proactive.heartbeat import HeartbeatConfig, run_heartbeat

    vault = tmp_path / "vault"
    commitments = vault / "System" / "Commitments"
    commitments.mkdir(parents=True)
    (commitments / "2026-06-29.md").write_text(
        """```yaml
id: commitment-waiting
source_message: 請 KJ 提供咖啡機照片
inferred_intent: missing_information_follow_up
due_at: null
condition: awaiting_kj_input
risk_level: low
next_action: ask KJ whether the photos are ready
status: waiting_for_kj
```
""",
        encoding="utf-8",
    )

    result = run_heartbeat(
        HeartbeatConfig(
            obsidian_vault=vault,
            now=datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc),
        )
    )

    assert result.output == "[SILENT]"
    assert result.notification_payload is None


def test_heartbeat_asks_when_waiting_for_kj_input_is_overdue(tmp_path: Path):
    from proactive.heartbeat import HeartbeatConfig, run_heartbeat

    vault = tmp_path / "vault"
    commitments = vault / "System" / "Commitments"
    commitments.mkdir(parents=True)
    (commitments / "2026-06-29.md").write_text(
        """```yaml
id: commitment-waiting
source_message: 請 KJ 提供咖啡機照片
inferred_intent: missing_information_follow_up
due_at: 2026-06-29T01:00:00+00:00
condition: awaiting_kj_input
risk_level: low
next_action: ask KJ whether the photos are ready
status: waiting_for_kj
```
""",
        encoding="utf-8",
    )

    result = run_heartbeat(
        HeartbeatConfig(
            obsidian_vault=vault,
            now=datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc),
        )
    )

    assert result.output != "[SILENT]"
    assert result.notification_payload is not None
    assert result.notification_payload["progress"]
    assert "waiting for KJ input" in result.notification_payload["summary"]
    assert "whether the requested information is ready" in result.notification_payload["recommended_next_action"]


def test_heartbeat_notification_renderer_is_human_readable():
    from proactive.heartbeat import render_notification_payload

    text = render_notification_payload(
        {
            "trigger_type": "heartbeat",
            "risk_level": "medium",
            "summary": "Proactive heartbeat found anomalies: cron failure: demo",
            "anomalies": ["cron failure: demo"],
            "progress": [],
            "recommended_next_action": "Hermes should ask KJ for a decision.",
        }
    )

    assert text.startswith("Proactive Hermes heartbeat")
    assert "Anomalies:" in text
    assert not text.lstrip().startswith("{")


def test_heartbeat_throttles_recent_waiting_for_kj_reminder(tmp_path: Path):
    from proactive.heartbeat import HeartbeatConfig, run_heartbeat

    vault = tmp_path / "vault"
    commitments = vault / "System" / "Commitments"
    commitments.mkdir(parents=True)
    (commitments / "2026-06-29.md").write_text(
        """```yaml
id: commitment-waiting
source_message: 請 KJ 提供咖啡機照片
condition: awaiting_kj_input
next_action: ask KJ whether the photos are ready
status: waiting_for_kj
```
""",
        encoding="utf-8",
    )
    state_dir = vault / "System" / "Proactive State"
    state_dir.mkdir(parents=True)
    (state_dir / "heartbeat-state.yaml").write_text(
        "last_waiting_input_reminder_at: 2026-06-29T01:30:00+00:00\n",
        encoding="utf-8",
    )

    result = run_heartbeat(
        HeartbeatConfig(
            obsidian_vault=vault,
            now=datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc),
        )
    )

    assert result.output == "[SILENT]"
    assert result.notification_payload is None


def test_inferred_commitment_writes_record(tmp_path: Path):
    from proactive.commitments import create_commitment_record, infer_commitment

    commitment = infer_commitment("明天提醒我檢查 Hermes cron 狀態")

    assert commitment is not None
    assert commitment.due_at is not None
    assert commitment.status == "open"
    path = create_commitment_record(commitment, tmp_path / "vault")
    text = path.read_text(encoding="utf-8")
    assert commitment.id in text
    assert "Hermes cron" in text


def test_waiting_for_kj_commitment_is_inferred_from_assistant_request():
    from proactive.commitments import infer_waiting_for_kj_commitment

    commitment = infer_waiting_for_kj_commitment(
        user_message="幫我把咖啡機拿去賣",
        assistant_response="請 KJ 提供咖啡機正面照片、側面照片與購買年份，我拿到後會繼續整理刊登文案。",
    )

    assert commitment is not None
    assert commitment.status == "waiting_for_kj"
    assert commitment.condition == "awaiting_kj_input"
    assert "提供咖啡機" in commitment.source_message
    assert "ask KJ" in commitment.next_action


def test_waiting_for_kj_commitment_ignores_general_questions():
    from proactive.commitments import infer_waiting_for_kj_commitment

    commitment = infer_waiting_for_kj_commitment(
        user_message="你在嗎",
        assistant_response="我在。你要我先看哪一個任務？",
    )

    assert commitment is None


def test_waiting_for_kj_response_writes_commitment_record(tmp_path: Path):
    from proactive.commitments import create_waiting_for_kj_record_from_response

    path = create_waiting_for_kj_record_from_response(
        user_message="幫我賣咖啡機",
        assistant_response="請提供咖啡機照片和購買年份；收到後我會繼續整理刊登文案。",
        obsidian_vault=tmp_path / "vault",
    )

    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "status: waiting_for_kj" in text
    assert "condition: awaiting_kj_input" in text
    assert "missing_information_follow_up" in text


def test_explicit_time_commitment_creates_cron_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from proactive.commitments import create_commitment_record, infer_commitment

    calls = []

    def fake_create_job(**kwargs):
        calls.append(kwargs)
        return {"id": "cron-123", **kwargs}

    monkeypatch.setattr("cron.jobs.create_job", fake_create_job)

    commitment = infer_commitment("明天提醒我檢查 OpenClaw delegated task")
    assert commitment is not None
    create_commitment_record(commitment, tmp_path / "vault", create_cron=True)

    assert len(calls) == 1
    assert calls[0]["name"].startswith("commitment:")
    assert calls[0]["deliver"] == "origin"

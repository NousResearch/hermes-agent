from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.gemini_daily_review_runtime import (
    _make_default_slack_sender,
    _hard_deadline,
    run_configured_review,
)
from agent.gemini_route_receipts import GeminiReceiptStore


def _config(db: Path) -> dict:
    return {
        "delegation": {
            "gemini_routing": {
                "enabled": True,
                "receipt_db": str(db),
                "retention": {"raw_days": 30, "aggregate_days": 180},
                "review": {
                    "enabled": True,
                    "sample_size": 5,
                    "timezone": "America/Los_Angeles",
                    "review_provider": "openai-codex",
                    "review_model": "gpt-5.6-sol",
                    "review_reasoning_effort": "medium",
                    "alert_target": "slack:C0A12345678",
                },
            }
        }
    }


def _seed_previous_day(db: Path) -> None:
    store = GeminiReceiptStore(db)
    receipt_id = store.prepare_attempt(
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        task_index=0,
        route_requested="auto",
        route_decision="gemini",
        route_reason="eligible",
        data_classification="standard",
        output_contract="text",
        goal_text="Summarize",
        context_text="Context",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
        started_at=datetime(2026, 9, 10, 19, tzinfo=timezone.utc),
    )
    store.mark_process_started(receipt_id, when=datetime(2026, 9, 10, 19, tzinfo=timezone.utc))
    store.complete_attempt(
        receipt_id,
        worker_status="completed",
        response_text="Summary",
        duration_ms=1,
        completed_at=datetime(2026, 9, 10, 19, 1, tzinfo=timezone.utc),
    )


def test_configured_review_runs_previous_local_day_and_stays_quiet_on_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db = tmp_path / "routing.sqlite3"
    _seed_previous_day(db)
    created = []
    alerts = []

    def reviewer_factory():
        created.append(object())
        return lambda prompt: {"verdict": "pass", "reason": "faithful"}

    result = run_configured_review(
        config=_config(db),
        now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
        reviewer_factory=reviewer_factory,
        alert_sender=alerts.append,
    )

    assert result["status"] == "passed"
    assert result["routing_day"] == "2026-09-10"
    assert len(created) == 1
    assert alerts == []
    assert capsys.readouterr() == ("", "")


def test_configured_review_alerts_only_the_exact_configured_channel_on_failure(tmp_path: Path):
    db = tmp_path / "routing.sqlite3"
    _seed_previous_day(db)
    alerts = []

    def sender(message: str) -> dict:
        alerts.append(message)
        return {
            "success": True,
            "chat_id": "C0A12345678",
            "message_id": "1720000000.000003",
        }

    result = run_configured_review(
        config=_config(db),
        now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
        reviewer_factory=lambda: (
            lambda prompt: {"verdict": "fail", "reason": "missed the requested evidence"}
        ),
        alert_sender=sender,
    )

    assert result["status"] == "failed"
    assert result["alert_status"] == "sent"
    assert len(alerts) == 1
    assert "missed the requested evidence" not in alerts[0]


def test_disabled_runtime_does_not_construct_reviewer_or_sender(tmp_path: Path):
    calls = []
    result = run_configured_review(
        config={"delegation": {"gemini_routing": {"enabled": False}}},
        reviewer_factory=lambda: calls.append("reviewer"),
        alert_sender=lambda message: calls.append(message),
    )
    assert result == {"status": "disabled"}
    assert calls == []


def test_configured_review_stays_idle_before_local_not_before(tmp_path: Path):
    config = _config(tmp_path / "routing.sqlite3")
    config["delegation"]["gemini_routing"]["review"]["not_before_local"] = "00:15"
    calls = []

    result = run_configured_review(
        config=config,
        now=datetime(2026, 9, 11, 7, 14, tzinfo=timezone.utc),
        reviewer_factory=lambda: calls.append("reviewer"),
        alert_sender=lambda message: calls.append(message),
    )

    assert result == {"status": "not_before"}
    assert calls == []
    assert not (tmp_path / "routing.sqlite3").exists()


def test_alert_target_must_be_one_exact_slack_channel(tmp_path: Path):
    config = _config(tmp_path / "routing.sqlite3")
    config["delegation"]["gemini_routing"]["review"]["alert_target"] = "all"
    with pytest.raises(ValueError, match="exact slack"):
        run_configured_review(
            config=config,
            reviewer_factory=lambda: lambda prompt: {"verdict": "pass", "reason": "ok"},
            alert_sender=lambda message: None,
        )


def test_no_agent_entrypoint_emits_empty_stdout_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("run_gemini_daily_review_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_configured_review", lambda: {"status": "passed"})

    assert module.main() == 0
    assert capsys.readouterr() == ("", "")
    source = script.read_text(encoding="utf-8")
    assert "AIAgent" not in source
    assert "from agent.gemini_daily_review import main" in source
    assert "print(" not in source.split("except Exception", 1)[0]


def test_no_agent_entrypoint_writes_fatal_errors_only_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("run_gemini_daily_review_error_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "run_configured_review", fail)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RuntimeError: boom" in captured.err


def test_no_agent_entrypoint_fails_when_required_alert_is_still_pending(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("gemini_daily_review_pending_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_configured_review",
        lambda: {"status": "failed", "alert_status": "pending"},
    )

    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "required Slack alert remains pending" in captured.err


def test_no_agent_entrypoint_subprocess_honors_custom_profile_home_and_is_silent(
    tmp_path: Path,
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text(
        """delegation:
  gemini_routing:
    enabled: false
""",
        encoding="utf-8",
    )
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "HERMES_HOME": str(home),
    }

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_default_slack_sender_verifies_workspace_membership_and_history():
    calls: list[tuple[str, dict]] = []

    class FakeClient:
        async def auth_test(self):
            calls.append(("auth_test", {}))
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **kwargs):
            calls.append(("conversations_info", kwargs))
            return {
                "ok": True,
                "channel": {
                    "id": "C0A12345678",
                    "is_member": True,
                    "context_team_id": "T_KIZUKI",
                },
            }

        async def conversations_history(self, **kwargs):
            calls.append(("conversations_history", kwargs))
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": "1720000000.000004",
                        "text": "Gemini daily review: batch grb_abcdef",
                    }
                ],
            }

    async def send(_config, channel, message):
        calls.append(("send", {"channel": channel, "message": message}))
        return {
            "success": True,
            "chat_id": channel,
            "message_id": "1720000000.000004",
        }

    result = _make_default_slack_sender(
        "C0A12345678",
        standalone_send=send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )("Gemini daily review: Receipt: ~/.hermes/routing.sqlite3 batch grb_abcdef")

    assert result == {
        "success": True,
        "platform": "slack",
        "team_id": "T_KIZUKI",
        "chat_id": "C0A12345678",
        "message_id": "1720000000.000004",
    }
    assert [name for name, _ in calls] == [
        "auth_test",
        "conversations_info",
        "send",
        "conversations_history",
    ]


def test_default_slack_sender_reconciles_unknown_send_from_channel_history():
    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

        async def conversations_history(self, **kwargs):
            assert "oldest" not in kwargs
            return {
                "ok": True,
                "messages": [
                    {"ts": "1720000000.000005", "text": "receipt batch grb_deadbeef"}
                ],
            }

    async def uncertain_send(*_args):
        raise TimeoutError("response lost after possible delivery")

    result = _make_default_slack_sender(
        "C0A12345678",
        standalone_send=uncertain_send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )("Gemini daily review: Receipt: x batch grb_deadbeef")

    assert result["message_id"] == "1720000000.000005"


def test_default_slack_sender_refuses_workspace_channel_identity_mismatch():
    sent: list[str] = []

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_WRONG"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

    async def send(*_args):
        sent.append("sent")
        return {}

    sender = _make_default_slack_sender(
        "C0A12345678",
        standalone_send=send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        sender("Gemini daily review: batch grb_deadbeef")
    assert sent == []


def test_internal_deadline_expires_before_cron_outer_timeout():
    with pytest.raises(TimeoutError, match="internal deadline"):
        with _hard_deadline(0.01):
            time.sleep(0.1)

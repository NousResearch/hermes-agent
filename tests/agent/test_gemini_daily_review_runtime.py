from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.gemini_daily_review_runtime import run_configured_review
from agent.gemini_route_receipts import GeminiReceiptStore


def _config(db: Path) -> dict:
    return {
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

    result = run_configured_review(
        config=_config(db),
        now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
        reviewer_factory=lambda: (
            lambda prompt: {"verdict": "fail", "reason": "missed the requested evidence"}
        ),
        alert_sender=alerts.append,
    )

    assert result["status"] == "failed"
    assert len(alerts) == 1
    assert "C0A12345678" in alerts[0]
    assert "missed the requested evidence" in alerts[0]


def test_disabled_runtime_does_not_construct_reviewer_or_sender(tmp_path: Path):
    calls = []
    result = run_configured_review(
        config={"gemini_routing": {"enabled": False}},
        reviewer_factory=lambda: calls.append("reviewer"),
        alert_sender=lambda message: calls.append(message),
    )
    assert result == {"status": "disabled"}
    assert calls == []


def test_alert_target_must_be_one_exact_slack_channel(tmp_path: Path):
    config = _config(tmp_path / "routing.sqlite3")
    config["gemini_routing"]["review"]["alert_target"] = "all"
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
    script = Path(__file__).parents[2] / "scripts" / "run_gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("run_gemini_daily_review_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_configured_review", lambda: {"status": "passed"})

    assert module.main() == 0
    assert capsys.readouterr() == ("", "")
    source = script.read_text(encoding="utf-8")
    assert "AIAgent" not in source
    assert "print(" not in source.split("except Exception", 1)[0]


def test_no_agent_entrypoint_writes_fatal_errors_only_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "run_gemini_daily_review.py"
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

from __future__ import annotations

import importlib
import json
from pathlib import Path


def _reload_self_review(monkeypatch, hermes_home: Path, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    import hermes_cli.self_review as self_review_mod

    return importlib.reload(self_review_mod)


def _reload_report(monkeypatch, hermes_home: Path, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    import hermes_cli.report as report_mod

    return importlib.reload(report_mod)


def _reload_ops(monkeypatch, hermes_home: Path, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    import hermes_cli.ops as ops_mod

    return importlib.reload(ops_mod)


def test_self_review_writes_latest_artifacts_under_hermes_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    self_review = _reload_self_review(monkeypatch, hermes_home, fake_home)

    findings = {
        "generated_at": "2026-03-29T14:00:00+00:00",
        "window_days": 7,
        "paths": {"hermes_root": str(hermes_home)},
        "sources": {"structured_events": 1},
        "current_config": {},
        "loop_patterns": {"total_loop_events": 3},
        "slow_tools": {"slow_event_count": 4},
        "task_health": {"completion_rate": 0.75},
        "model_efficiency": {},
        "scheduler_health": {},
    }
    proposals = {
        "proposed_at": "2026-03-29T14:00:01+00:00",
        "rationale": "test",
        "proposals": [
            {
                "key_path": "agent.max_turns",
                "current_value": 120,
                "proposed_value": 100,
                "rationale": "reduce",
                "confidence": "medium",
            }
        ],
        "scheduler_proposals": [
            {
                "task_id": "job-a",
                "name": "Job A",
                "current_enabled": True,
                "proposed_enabled": False,
                "rationale": "disable",
                "confidence": "high",
            }
        ],
        "advisories": [
            {
                "topic": "terminal loops",
                "finding": "lots",
                "recommendation": "review",
            }
        ],
    }

    monkeypatch.setattr(self_review, "analyze", lambda days=7: findings)
    monkeypatch.setattr(self_review, "propose", lambda _findings: proposals)
    monkeypatch.setattr(self_review, "render_report", lambda _findings, _proposals: "SELF REVIEW REPORT")

    report_text = self_review.run_full_review(days=7)

    reports_dir = hermes_home / "reports"
    proposal_dir = hermes_home / "proposals"
    latest_md = reports_dir / "self-review-latest.md"
    latest_json = reports_dir / "self-review-latest.json"

    assert report_text == "SELF REVIEW REPORT"
    assert latest_md.read_text(encoding="utf-8") == "SELF REVIEW REPORT"
    snapshot = json.loads(latest_json.read_text(encoding="utf-8"))
    assert snapshot["summary"]["config_change_count"] == 1
    assert snapshot["summary"]["scheduler_change_count"] == 1
    assert snapshot["summary"]["advisory_count"] == 1
    assert snapshot["artifacts"]["markdown"] == str(latest_md)
    assert snapshot["artifacts"]["json"] == str(latest_json)
    assert snapshot["artifacts"]["proposal"] is not None
    assert Path(snapshot["artifacts"]["proposal"]).exists()
    assert len(list(proposal_dir.glob("*-self-review.yaml"))) == 1
    assert not (fake_home / ".hermes").exists()


def test_report_writes_json_sidecar_and_exposes_summary(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    report_mod = _reload_report(monkeypatch, hermes_home, fake_home)

    events = [
        {"event": "tool_result", "tool_name": "terminal", "duration_ms": 100, "success": True},
        {"event": "tool_result", "tool_name": "terminal", "duration_ms": 300, "success": False},
        {"event": "model_call", "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 2000},
        {"event": "loop_detected", "tool_name": "terminal"},
        {"event": "slow_tool", "tool_name": "terminal"},
    ]
    tasks = [
        {"status": "completed", "error_info": None},
        {"status": "failed", "error_info": "timeout"},
    ]

    monkeypatch.setattr(report_mod, "_read_log_window", lambda days: events)
    monkeypatch.setattr(report_mod, "_read_tasks_window", lambda days: tasks)

    result = report_mod.generate_report(days=7)

    assert result["saved_to"] is not None
    assert result["saved_json_to"] is not None
    markdown_path = Path(result["saved_to"])
    json_path = Path(result["saved_json_to"])
    assert markdown_path.exists()
    assert json_path.exists()
    assert markdown_path.parent == hermes_home / "reports"
    assert json_path.parent == hermes_home / "reports"
    assert markdown_path.suffix == ".md"
    assert json_path.suffix == ".json"

    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["summary"]["total_tasks"] == 2
    assert sidecar["summary"]["failed_tasks"] == 1
    assert sidecar["summary"]["loop_detections"] == 1
    assert sidecar["summary"]["slow_tool_warnings"] == 1
    assert sidecar["summary"]["total_cost_usd"] > 0
    assert result["summary"]["total_tasks"] == 2
    assert result["summary"]["top_tools"][0]["tool"] == "terminal"
    assert result["summary"]["top_models"][0]["model"] == "gpt-4o"
    assert not (fake_home / ".hermes").exists()


def test_ops_respects_hermes_home_for_dashboard_data(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    logs_dir = hermes_home / "logs"
    logs_dir.mkdir(parents=True)

    import sqlite3

    db_path = hermes_home / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            session_id TEXT,
            status TEXT,
            model_used TEXT,
            current_step TEXT,
            started_at REAL,
            updated_at REAL,
            token_usage TEXT,
            checkpoint_data TEXT,
            error_info TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "task-123",
            "session-xyz",
            "completed",
            "gpt-4o-mini",
            "done",
            1.0,
            2.0,
            json.dumps({"input": 1, "output": 2}),
            json.dumps({}),
            None,
        ),
    )
    conn.commit()
    conn.close()
    (logs_dir / "structured.jsonl").write_text(
        json.dumps({"event": "tool_result", "tool_name": "terminal", "duration_ms": 42}) + "\n",
        encoding="utf-8",
    )

    ops_mod = _reload_ops(monkeypatch, hermes_home, fake_home)

    tasks = ops_mod.list_tasks(limit=5)
    events = ops_mod.recent_events(limit=5)
    summary = ops_mod.task_summary()

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "task-123"
    assert len(events) == 1
    assert events[0]["tool_name"] == "terminal"
    assert summary["total"] == 1
    assert summary["models_used"]["gpt-4o-mini"] == 1
    assert not (fake_home / ".hermes").exists()

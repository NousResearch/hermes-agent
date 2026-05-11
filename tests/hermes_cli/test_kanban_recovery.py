from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_recovery as kr


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "body": "",
        "result": "",
        "assignee": "forge",
        "status": "blocked",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _run(outcome="crashed", error="pid 123 not alive", run_id=1):
    return {"id": run_id, "outcome": outcome, "error": error, "summary": None}


def test_sanitize_child_stdout_removes_banner_and_tool_dump():
    raw = """
╔════════ Hermes Agent ════════╗
║ Provider: openrouter         ║
╚══════════════════════════════╝
Available tools: terminal, read_file, kanban_show, browser_click
- terminal: Execute shell commands
Real failure: pid 4242 not alive after spawn
Traceback (most recent call last): boom
"""
    cleaned = kr.sanitize_child_stdout(raw, max_chars=500)
    assert "Hermes Agent" not in cleaned
    assert "Available tools" not in cleaned
    assert "terminal: Execute" not in cleaned
    assert "pid 4242 not alive" in cleaned
    assert "Traceback" in cleaned


def test_classifies_crash_or_pid_dead_and_prepares_bounded_retry_spec():
    task = _task(
        status="ready",
        consecutive_failures=2,
        last_failure_error="last_failure_error='pid 999 not alive'",
    )
    category, confidence, evidence = kr.classify_recovery(task, [], [_run()])
    assert category == "worker_crash_or_pid_dead"
    assert confidence >= 0.8
    assert any("consecutive_failures=2" in item for item in evidence)

    item = kr.RecoveryItem(
        task_id="t_demo00",
        title="demo task",
        status="ready",
        assignee="forge",
        category=category,
        confidence=confidence,
        summary="pid 999 not alive",
        recommendation="inspect logs",
        retry_task_spec={"title": "Recover crashed Kanban task t_demo00"},
    )
    rendered = kr.render_recovery_report([item])
    assert "bounded_retry_task: available with --json" in rendered


def test_classifies_required_categories():
    cases = [
        ("auth_or_credential_issue", _task(last_failure_error="401 unauthorized token expired"), [], []),
        ("missing_artifact_or_path", _task(last_failure_error="No such file or directory: /tmp/out.stl"), [], []),
        ("stale_or_superseded", _task(body="Superseded by t_abc12345; no longer needed"), [], []),
        ("human_decision_required", _task(body="Need Matthew approval before continuing"), [], []),
        ("unknown_needs_review", _task(status="ready"), [], []),
    ]
    for expected, task, events, runs in cases:
        category, _, _ = kr.classify_recovery(task, events, runs)
        assert category == expected


def test_build_recovery_items_reads_board_without_mutating(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="crashy task",
            assignee="forge",
        )
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 1, last_failure_error = ? WHERE id = ?",
            ("pid 555 not alive", task_id),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, "forge", "crashed", "crashed", 1, 2, "pid 555 not alive"),
        )
        before = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"]

    items = kr.build_recovery_items(task_id=task_id)
    assert len(items) == 1
    assert items[0].category == "worker_crash_or_pid_dead"
    assert items[0].retry_task_spec is not None

    with kb.connect() as conn:
        after = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"]
    assert after == before


def test_json_report_includes_counts_and_no_auto_create_flag():
    report = kr.render_recovery_report([], json_output=True)
    data = json.loads(report)
    assert data["counts"]["worker_crash_or_pid_dead"] == 0
    assert data["items"] == []

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="terminal-state", assignee="test-worker")
        kb.claim_task(conn, tid)
        run = kb.latest_run(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    return tid


def test_complete_returns_terminal_stop_signal(worker_env):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_complete({"summary": "done"}))

    assert out["ok"] is True
    assert out["terminal"] is True
    assert out["terminal_status"] == "done"
    assert "stop" in out["instruction"].lower()


def test_worker_cannot_mutate_after_complete(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    assert json.loads(kt._handle_complete({"summary": "done"}))["ok"] is True

    late_comment = json.loads(
        kt._handle_comment({"task_id": worker_env, "body": "late mutation"})
    )
    assert late_comment.get("ok") is not True
    assert "terminal" in late_comment.get("error", "").lower()

    late_block = json.loads(kt._handle_block({"reason": "late block"}))
    assert late_block.get("ok") is not True
    assert "terminal" in late_block.get("error", "").lower()

    with kb.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "done"
        assert kb.list_comments(conn, worker_env) == []


def test_worker_cannot_mutate_foreign_task_after_own_complete(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        other = kb.create_task(conn, title="other", assignee="peer")

    assert json.loads(kt._handle_complete({"summary": "done"}))["ok"] is True

    late_foreign_comment = json.loads(
        kt._handle_comment({"task_id": other, "body": "late cross-task mutation"})
    )
    assert late_foreign_comment.get("ok") is not True
    assert "terminal" in late_foreign_comment.get("error", "").lower()

    with kb.connect() as conn:
        assert kb.list_comments(conn, other) == []


def test_block_returns_terminal_stop_signal_and_releases_active_session(
    worker_env,
    monkeypatch,
):
    from hermes_cli import active_sessions
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_SESSION_ID", "worker-session")
    lease, message = active_sessions.try_acquire_active_session(
        session_id="worker-session",
        surface="kanban-worker",
        config={"max_concurrent_sessions": 10},
    )
    assert message is None
    assert lease is not None
    assert active_sessions.active_session_registry_snapshot()

    out = json.loads(
        kt._handle_block({
            "reason": "needs human review",
            "kind": "needs_input",
        })
    )

    assert out["ok"] is True
    assert out["terminal"] is True
    assert out["terminal_status"] == "blocked"
    assert "stop" in out["instruction"].lower()
    assert active_sessions.active_session_registry_snapshot() == []


def test_worker_cannot_mutate_after_block(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    assert json.loads(
        kt._handle_block({
            "reason": "waiting on credentials",
            "kind": "capability",
        })
    )["ok"] is True

    late_comment = json.loads(
        kt._handle_comment({"task_id": worker_env, "body": "late mutation"})
    )
    assert late_comment.get("ok") is not True
    assert "terminal" in late_comment.get("error", "").lower()

    with kb.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "blocked"
        assert kb.list_comments(conn, worker_env) == []


def test_block_records_recovery_packet(worker_env, tmp_path):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    artifact = tmp_path / "partial-report.txt"
    artifact.write_text("PARTIAL\nuseful diagnostics\n", encoding="utf-8")

    out = json.loads(
        kt._handle_block({
            "reason": "pytest regression still failing",
            "kind": "transient",
            "attempted_files": [str(artifact)],
            "tests_run": ["pytest tests/example.py -q"],
            "remaining_failures": ["tests/example.py::test_regression"],
            "next_owner": "reviewer",
            "artifacts": [str(artifact)],
            "artifact_markers": {str(artifact): "PARTIAL"},
        })
    )

    assert out["ok"] is True
    with kb.connect() as conn:
        run = kb.latest_run(conn, worker_env)
        events = kb.list_events(conn, worker_env)

    recovery = run.metadata["block_recovery"]
    assert recovery["exact_blocker"] == "pytest regression still failing"
    assert recovery["kind"] == "transient"
    assert recovery["attempted_files"] == [str(artifact)]
    assert recovery["tests_run"] == ["pytest tests/example.py -q"]
    assert recovery["remaining_failures"] == ["tests/example.py::test_regression"]
    assert recovery["next_owner"] == "reviewer"
    assert recovery["artifacts_host_visible"] is True
    assert recovery["verified_artifacts"][0]["path"] == str(artifact)

    blocked_events = [event for event in events if event.kind == "blocked"]
    assert blocked_events
    assert blocked_events[-1].payload["block_recovery"]["artifacts_host_visible"] is True


def test_complete_rejects_missing_declared_artifact(worker_env, tmp_path):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    missing = tmp_path / "missing-report.txt"
    out = json.loads(
        kt._handle_complete({
            "summary": "done",
            "artifacts": [str(missing)],
        })
    )

    assert out.get("ok") is not True
    assert "artifact verification failed" in out.get("error", "")
    assert "does not exist" in out.get("error", "")
    with kb.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"


def test_complete_verifies_artifact_marker_and_records_evidence(
    worker_env,
    tmp_path,
):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    artifact = tmp_path / "report.txt"
    artifact.write_text("READY\nreal report\n", encoding="utf-8")

    out = json.loads(
        kt._handle_complete({
            "summary": "done",
            "artifacts": [str(artifact)],
            "artifact_markers": {str(artifact): "READY"},
        })
    )

    assert out["ok"] is True
    with kb.connect() as conn:
        run = kb.latest_run(conn, worker_env)
        assert run.metadata["artifacts"] == [str(artifact)]
        assert run.metadata["verified_artifacts"][0]["path"] == str(artifact)
        assert run.metadata["verified_artifacts"][0]["size"] > 0
        events = kb.list_events(conn, worker_env)
    completed = [event for event in events if event.kind == "completed"]
    assert completed
    assert completed[-1].payload["verified_artifacts"][0]["path"] == str(artifact)


def test_complete_rejects_artifact_marker_mismatch(worker_env, tmp_path):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    artifact = tmp_path / "report.txt"
    artifact.write_text("real report\n", encoding="utf-8")

    out = json.loads(
        kt._handle_complete({
            "summary": "done",
            "artifacts": [str(artifact)],
            "artifact_markers": {str(artifact): "READY"},
        })
    )

    assert out.get("ok") is not True
    assert "required marker was not found" in out.get("error", "")
    with kb.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"

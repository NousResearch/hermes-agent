from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from hermes_cli import ops


def _seed_dashboard_data(hermes_home: Path) -> None:
    logs_dir = hermes_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    db = SessionDB(db_path=hermes_home / "state.db")
    db.create_session(session_id="s1", source="cli")
    db.create_task(
        task_id="t1",
        session_id="s1",
        status="completed",
        model_used="gpt-4o-mini",
        current_step="done",
        checkpoint_data={"phase": "final", "message_count": 3},
        token_usage={"input": 10, "output": 20},
        error_info=None,
    )
    db.create_task(
        task_id="t2",
        session_id="s1",
        status="failed",
        model_used="gpt-5.4",
        current_step="tool_call",
        checkpoint_data={"phase": "tool"},
        token_usage={"input": 5, "output": 1},
        error_info="timeout",
    )
    db.close()

    (logs_dir / "structured.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "tool_result", "tool_name": "terminal", "duration_ms": 4200, "task_id": "t1"}),
                json.dumps({"event": "loop_detected", "tool_name": "terminal", "count": 2, "task_id": "t2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_ops_respects_hermes_home_for_all_dashboard_queries(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    _seed_dashboard_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    tasks = ops.list_tasks(limit=10)
    task = ops.get_task("t1")
    events = ops.recent_events(limit=10)
    slow = ops.slow_tools(limit=10)
    loops = ops.loop_events(limit=10)
    summary = ops.task_summary()

    assert len(tasks) == 2
    assert {t["task_id"] for t in tasks} == {"t1", "t2"}
    assert task is not None and task["task_id"] == "t1"
    assert len(events) == 2
    assert events[0]["tool_name"] == "terminal"
    assert len(slow) == 1 and slow[0]["duration_ms"] == 4200
    assert len(loops) == 1 and loops[0]["count"] == 2
    assert summary["total"] == 2
    assert summary["by_status"]["completed"] == 1
    assert summary["by_status"]["failed"] == 1
    assert summary["models_used"]["gpt-4o-mini"] == 1
    assert summary["models_used"]["gpt-5.4"] == 1
    assert not (fake_home / ".hermes").exists()


def test_list_tasks_and_get_task_parse_json_fields(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    _seed_dashboard_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    task = ops.get_task("t1")
    tasks = ops.list_tasks(limit=1)

    assert task is not None
    assert task["checkpoint_data"] == {"phase": "final", "message_count": 3}
    assert task["token_usage"] == {"input": 10, "output": 20}
    assert tasks[0]["task_id"] in {"t1", "t2"}
    assert tasks[0]["checkpoint_data"] in ({"phase": "final", "message_count": 3}, {"phase": "tool"})


def test_recent_chats_excludes_active_sessions_and_returns_metadata(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    db = SessionDB(db_path=hermes_home / "state.db")

    now = 1_900_000_000.0
    for idx in range(55):
        sid = f"active-{idx}"
        db.create_session(session_id=sid, source="cli")
        db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = NULL WHERE id = ?", (now - idx, sid))

    db.create_session(session_id="ended-new", source="cli")
    db.end_session("ended-new", end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, pinned_at = NULL WHERE id = ?",
        (now - 200, now - 190, "ended-new"),
    )

    db.create_session(session_id="ended-old", source="cli")
    db.end_session("ended-old", end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, pinned_at = ? WHERE id = ?",
        (now - 400, now - 390, now - 389, "ended-old"),
    )
    db._conn.commit()
    db.close()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    payload = ops.recent_chats(limit=2)

    assert payload["meta"]["total_historical"] == 2
    assert payload["meta"]["visible_count"] == 2
    assert payload["meta"]["pinned_count"] == 1
    assert [chat["id"] for chat in payload["items"]] == ["ended-old", "ended-new"]
    assert all(chat["ended_at"] is not None for chat in payload["items"])
    assert all(not chat["id"].startswith("active-") for chat in payload["items"])


def test_ops_eval_subcommand_lists_local_evals(capsys):
    ops.main(["eval", "list"])

    captured = capsys.readouterr()

    assert "Suites:" in captured.out
    assert "smoke" in captured.out
    assert "file-create-and-read" in captured.out


@patch("agent.failure_analysis.storage.FailureStore")
def test_ops_failures_subcommand_shows_top_fingerprints(mock_store_cls, capsys):
    mock_store = MagicMock()
    mock_store.top_fingerprints.return_value = [
        {
            "count": 2,
            "failure_type": "tool",
            "failure_subtype": "timeout",
            "fingerprint": "fp-test",
            "summary": "terminal timed out",
        }
    ]
    mock_store_cls.return_value = mock_store

    ops.main(["failures", "top"])

    captured = capsys.readouterr()
    assert "2x" in captured.out
    assert "tool.timeout" in captured.out
    assert "fp-test" in captured.out


def test_ops_reconciles_existing_state_db_without_tasks_table(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(parents=True)
    conn = sqlite3.connect(hermes_home / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL)")
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert ops.list_tasks(limit=5) == []

    conn = sqlite3.connect(hermes_home / "state.db")
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
        ).fetchone()
        assert row is not None
    finally:
        conn.close()

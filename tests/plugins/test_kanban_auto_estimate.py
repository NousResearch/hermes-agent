"""Kanban automatic effort estimation contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_router():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_auto_estimate_plugin", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


def test_board_auto_estimate_defaults_off_and_persists_toggle(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_router(), prefix="/api/plugins/kanban")
    client = TestClient(app)

    created = client.post("/api/plugins/kanban/boards", json={"slug": "widget"})
    assert created.status_code == 200
    assert created.json()["board"]["auto_estimate_enabled"] is False

    updated = client.patch(
        "/api/plugins/kanban/boards/widget", json={"auto_estimate_enabled": True}
    )
    assert updated.status_code == 200
    assert updated.json()["board"]["auto_estimate_enabled"] is True
    listed = client.get("/api/plugins/kanban/boards").json()["boards"]
    assert next(board for board in listed if board["slug"] == "widget")["auto_estimate_enabled"] is True


def test_create_task_schedules_only_when_board_setting_enabled(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    calls = []
    monkeypatch.setattr("hermes_cli.kanban_estimate.schedule_task_estimate", lambda *args, **kwargs: calls.append((args, kwargs)))

    with kb.connect_closing(board="default") as conn:
        first = kb.create_task(conn, title="off", board="default", initial_status="blocked")
    assert calls == []

    kb.write_board_metadata("default", auto_estimate_enabled=True)
    with kb.connect_closing(board="default") as conn:
        second = kb.create_task(conn, title="on", board="default", initial_status="running")
        third = kb.create_task(conn, title="blocked", board="default", initial_status="blocked")
    assert calls == [((second,), {"board": "default"})]
    with kb.connect_closing(board="default") as verify_conn:
        assert kb.get_task(verify_conn, first) is not None


def test_estimate_result_is_persisted_on_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.write_board_metadata("default", auto_estimate_enabled=True)
    with kb.connect_closing(board="default") as conn:
        task_id = kb.create_task(conn, title="persist me", board="default", initial_status="running")

    import hermes_cli.kanban_estimate as estimator

    monkeypatch.setattr(
        estimator,
        "estimate_text",
        lambda title, body, task_id=None: {
            "ok": True,
            "est_tokens": 1234,
            "complexity": "M",
            "rationale": "two files",
            "model": "test-model",
        },
    )
    estimator.estimate_and_persist(task_id, board="default")

    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.estimate == {
        "ok": True,
        "est_tokens": 1234,
        "complexity": "M",
        "rationale": "two files",
        "model": "test-model",
    }


def test_estimate_schedule_failure_does_not_fail_creation(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.write_board_metadata("default", auto_estimate_enabled=True)
    monkeypatch.setattr(
        "hermes_cli.kanban_estimate.schedule_task_estimate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("executor unavailable")),
    )
    with kb.connect_closing(board="default") as conn:
        task_id = kb.create_task(conn, title="still create", board="default", initial_status="blocked")
    with kb.connect_closing(board="default") as conn:
        assert kb.get_task(conn, task_id) is not None


def test_board_backfill_queues_only_unestimated_tasks_through_running(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    import hermes_cli.kanban_estimate as estimator

    with kb.connect_closing(board="default") as conn:
        queued_task = kb.create_task(conn, title="todo", board="default", initial_status="blocked")
        ready_task = kb.create_task(conn, title="ready", board="default", initial_status="blocked")
        done = kb.create_task(conn, title="done", board="default", initial_status="blocked")
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (queued_task,))
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (ready_task,))
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (done,))
        conn.execute("UPDATE tasks SET estimate = ? WHERE id = ?", ('{"ok":true}', queued_task))

    queued = []
    monkeypatch.setattr(estimator, "schedule_task_estimate", lambda task_id, *, board=None: queued.append((task_id, board)))
    assert estimator.schedule_board_estimates("default") == 1
    assert queued == [(ready_task, "default")]

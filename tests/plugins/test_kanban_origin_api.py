"""The shipped dashboard API carries origin both on cards and in the opened drawer."""
import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc


def test_real_router_preserves_origin_board_and_detail(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_origin_fixture_api", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    origin = {"profile": "default", "session_id": "fixture-session", "message_row_id": 42,
              "prompt_excerpt": "quoted fixture", "platform": "telegram", "chat_id": "fixture-chat"}
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="origin fixture", assignee="default", origin=origin)
    with TestClient(app) as client:
        board = client.get("/api/plugins/kanban/board")
        assert board.status_code == 200, board.text
        card = next(t for col in board.json()["columns"] for t in col["tasks"] if t["id"] == tid)
        detail = client.get(f"/api/plugins/kanban/tasks/{tid}")
        assert detail.status_code == 200, detail.text
        assert card["origin"] == detail.json()["task"]["origin"] == origin
        created = client.post("/api/plugins/kanban/tasks", json={"title": "dashboard fixture"})
        assert created.status_code == 200, created.text
        assert created.json()["task"]["origin"] is None

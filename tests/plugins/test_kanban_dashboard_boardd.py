"""HTTP boundary coverage for boardd-aware kanban dashboard routing."""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import boardd_shim
from hermes_cli import kanban_db as kb
from plugins.kanban.dashboard import plugin_api


def _fleet_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_BROKER", raising=False)
    monkeypatch.delenv("BOARDD_SOCK", raising=False)
    kb.create_board("fleet")
    kb.set_current_board("fleet")


def _client(mod) -> TestClient:
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    mod.register_boardd_exception_handlers(app)
    return TestClient(app)


def test_http_standalone_fleet_name_uses_local_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fleet_home(tmp_path, monkeypatch)
    mod = importlib.reload(plugin_api)
    client = _client(mod)

    response = client.post(
        "/api/plugins/kanban/tasks?board=fleet",
        json={"title": "standalone fleet task", "assignee": "worker"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["task"]["title"] == "standalone fleet task"


def test_http_custody_resolver_failure_returns_503_not_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fleet_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_BROKER", "1")
    mod = importlib.reload(plugin_api)

    def broken_resolver(*_args, **_kwargs):
        raise RuntimeError("canonical resolver unavailable")

    monkeypatch.setattr(boardd_shim, "routes_to_fleet", broken_resolver)
    client = _client(mod)
    response = client.get("/api/plugins/kanban/board?board=fleet")
    assert response.status_code == 503, response.text
    assert response.json() == {
        "detail": "boardd broker unavailable for fleet board"
    }

    # A failed custodied request must not create or mutate a local task row.
    conn = sqlite3.connect(str(kb.kanban_db_path(board="fleet")))
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()

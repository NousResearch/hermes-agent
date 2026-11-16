"""Backend contract for per-board Kanban pickup control."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _load_router():
    root = Path(__file__).resolve().parents[2]
    path = root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_pause_control_api_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="default")
    app = FastAPI()
    app.include_router(_load_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_get_is_read_only_and_post_returns_idempotent_readback(client):
    response = client.get("/api/plugins/kanban/control")
    assert response.status_code == 200
    initial = response.json()
    assert initial["state"] == "running"
    assert initial["boards"][0]["revision"] == 0
    with kbc.connect_closing(board="default") as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_dispatch_control").fetchone()[0] == 0

    paused = client.post(
        "/api/plugins/kanban/control/pause",
        json={"reason": "maintenance"},
    )
    assert paused.status_code == 200
    assert paused.json()["boards"][0]["state"] == "paused"
    assert paused.json()["boards"][0]["revision"] == 1

    repeated = client.post(
        "/api/plugins/kanban/control/pause",
        json={"reason": "retry"},
    )
    assert repeated.status_code == 200
    assert repeated.json()["boards"][0]["revision"] == 1
    assert repeated.json()["boards"][0]["changed_at_utc"].endswith("Z")
    assert repeated.json()["boards"][0]["reason"] == "maintenance"


def test_invalid_board_reason_and_unknown_state_have_stable_errors(client):
    missing = client.get("/api/plugins/kanban/control?board=missing")
    assert missing.status_code == 404
    missing_mutation = client.post(
        "/api/plugins/kanban/control/pause",
        json={"board": "missing"},
    )
    assert missing_mutation.status_code == 404

    malformed_board = client.get("/api/plugins/kanban/control?board=../escape")
    assert malformed_board.status_code == 400
    malformed_reason = client.post(
        "/api/plugins/kanban/control/pause",
        json={"reason": "x" * 2001},
    )
    assert malformed_reason.status_code == 400

    with kbc.connect_closing(board="default") as conn:
        conn.execute("DROP TABLE kanban_dispatch_control")
        conn.execute("CREATE TABLE kanban_dispatch_control (id INTEGER PRIMARY KEY, state TEXT)")
        conn.commit()
    unknown = client.get("/api/plugins/kanban/control")
    assert unknown.status_code == 503
    assert unknown.json()["detail"]["code"] == "control_state_unknown"


def test_partial_multi_board_mutation_is_structured_and_non_200(client, monkeypatch):
    kb.create_board("alpha")
    original = kbc.connect_closing

    @contextmanager
    def fail_alpha(*args, **kwargs):
        if kwargs.get("board") == "alpha":
            raise OSError("simulated alpha outage")
        with original(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(kbc, "connect_closing", fail_alpha)
    response = client.post("/api/plugins/kanban/control/pause", json={})
    assert response.status_code == 207
    payload = response.json()
    assert payload["state"] == "partial"
    assert [row["state"] for row in payload["boards"]] == ["paused", "unknown"]


def test_mutation_records_verified_request_principal(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="default")
    app = FastAPI()
    app.state.auth_required = True

    @app.middleware("http")
    async def verified_session(request, call_next):
        request.state.session = SimpleNamespace(user_id="operator-1", provider="portal")
        return await call_next(request)

    app.include_router(_load_router(), prefix="/api/plugins/kanban")
    verified = TestClient(app).post("/api/plugins/kanban/control/pause", json={})
    assert verified.status_code == 200
    assert verified.json()["boards"][0]["changed_by"] == "portal:operator-1"

    unauthenticated = FastAPI()
    unauthenticated.state.auth_required = True
    unauthenticated.include_router(_load_router(), prefix="/api/plugins/kanban")
    denied = TestClient(unauthenticated).post(
        "/api/plugins/kanban/control/resume", json={},
    )
    assert denied.status_code == 401

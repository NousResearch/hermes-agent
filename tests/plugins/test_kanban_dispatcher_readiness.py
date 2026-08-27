from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    module_name = "hermes_dashboard_plugin_kanban_dispatcher_readiness_test"
    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dispatcher_readiness_endpoint_surfaces_strict_state(monkeypatch):
    module = _load_plugin_module()
    expected = {
        "status": "offline",
        "ready": False,
        "gateway_pid": None,
        "message": "No gateway is running",
    }
    monkeypatch.setattr("hermes_cli.kanban._dispatcher_readiness", lambda **_kwargs: expected)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")

    response = TestClient(app).get("/api/plugins/kanban/dispatcher-readiness")

    assert response.status_code == 200
    assert response.json() == expected

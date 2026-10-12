"""Kanban dashboard plugin: task effort estimate.

The estimate endpoints call the auto-routed auxiliary model and parse a
compact JSON reply (tokens + complexity + rationale). Tests monkeypatch
``call_llm`` so no network is touched.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_est_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _fake_resp(content: str, model: str = "aux-mini"):
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], model=model)


def test_estimate_parses_model_json(client, monkeypatch):
    task_id = client.post("/api/plugins/kanban/tasks", json={"title": "big refactor"}).json()["task"]["id"]

    import agent.auxiliary_client as aux

    def fake_call_llm(**kwargs):
        assert kwargs.get("task") == "kanban_estimator"
        return _fake_resp('{"est_tokens": 42000, "complexity": "M", "rationale": "multi-file edit"}')

    monkeypatch.setattr(aux, "call_llm", fake_call_llm)

    body = client.post(f"/api/plugins/kanban/tasks/{task_id}/estimate").json()
    assert body["ok"] is True
    assert body["est_tokens"] == 42000
    assert body["complexity"] == "M"
    assert body["rationale"] == "multi-file edit"
    assert body["model"] == "aux-mini"


def test_estimate_tolerates_unparseable_reply(client, monkeypatch):
    task_id = client.post("/api/plugins/kanban/tasks", json={"title": "vague"}).json()["task"]["id"]

    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "call_llm", lambda **kw: _fake_resp("I cannot estimate this, sorry."))

    assert client.post(f"/api/plugins/kanban/tasks/{task_id}/estimate").json()["ok"] is False


def test_estimate_unknown_task_404(client):
    assert client.post("/api/plugins/kanban/tasks/t_missing/estimate").status_code == 404


def test_estimate_from_text_no_task(client, monkeypatch):
    """The create dialog estimates from typed title/body before a task exists."""
    import agent.auxiliary_client as aux
    monkeypatch.setattr(
        aux, "call_llm",
        lambda **kw: _fake_resp('{"est_tokens": 8000, "complexity": "S", "rationale": "localized"}'),
    )
    body = client.post(
        "/api/plugins/kanban/estimate", json={"title": "tweak a label", "body": "in settings"}
    ).json()
    assert body["ok"] is True
    assert body["est_tokens"] == 8000
    assert body["complexity"] == "S"


def test_estimate_from_text_requires_title(client):
    assert client.post("/api/plugins/kanban/estimate", json={"title": "  ", "body": "x"}).json()["ok"] is False

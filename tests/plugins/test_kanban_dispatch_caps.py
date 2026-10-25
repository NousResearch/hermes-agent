"""Manual dashboard dispatch must obey caps and refuse unresolved config."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("alpha", "beta", "gamma"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}", encoding="utf-8")
    path = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_dispatch_caps_test", path)
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, api)
    spec.loader.exec_module(api)
    app = FastAPI()
    app.include_router(api.router)
    spawned = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, **kw: spawned.append(task.id) or 4242)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for name in ("alpha", "alpha", "beta", "gamma"):
            kb.create_task(conn, title=name, assignee=name)
    return TestClient(app), home / "config.yaml", spawned


def _snapshot():
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        return [tuple(row) for row in conn.execute(
            "SELECT id, status, claim_lock, consecutive_failures FROM tasks ORDER BY id")]


@pytest.mark.parametrize("caps,query,expected", [
    ({"max_spawn": 1, "max_in_progress": 4, "max_in_progress_per_profile": 1}, "", 1),
    ({"max_spawn": 1, "max_in_progress": 2, "max_in_progress_per_profile": 1}, "?max=8", 2),
    ({"max_spawn": 8, "max_in_progress": 8, "max_in_progress_per_profile": 1}, "", 3),
    ({"max_spawn": 8, "max_in_progress": 8, "max_in_progress_per_profile": 1}, "?max=0", 0),
    ({"max_spawn": 8, "max_in_progress": 8, "max_in_progress_per_profile": 1}, "?max=-1", 0),
    ({"max_spawn": "1", "max_in_progress": "4", "max_in_progress_per_profile": "1"}, "", 1),
])
def test_dispatch_http_caps_bound_real_claims(dashboard, caps, query, expected):
    client, config_path, spawned = dashboard
    config_path.write_text(json.dumps({"kanban": caps}), encoding="utf-8")
    response = client.post("/dispatch" + query)
    assert response.status_code == 200, response.text
    assert len(response.json()["spawned"]) == len(spawned) == expected
    assert sum(row[1] == "running" for row in _snapshot()) == expected


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "1.5", "banana", [], {}])
@pytest.mark.parametrize("key", ["max_spawn", "max_in_progress", "max_in_progress_per_profile"])
def test_malformed_cap_refused_before_claim(dashboard, bad, key):
    client, config_path, spawned = dashboard
    config_path.write_text(json.dumps({"kanban": {key: bad}}), encoding="utf-8")
    before = _snapshot()
    response = client.post("/dispatch?max=8")
    assert response.status_code == 503, response.text
    assert "cap configuration" in response.json()["detail"]
    assert not spawned
    assert _snapshot() == before


@pytest.mark.parametrize("section", [False, [], "", 0, "invalid"])
def test_non_mapping_section_refused(dashboard, section):
    client, config_path, spawned = dashboard
    config_path.write_text(json.dumps({"kanban": section}), encoding="utf-8")
    before = _snapshot()
    assert client.post("/dispatch").status_code == 503
    assert not spawned
    assert _snapshot() == before


@pytest.mark.parametrize("last_good", [False, True])
def test_real_loader_parse_failure_never_dispatches_fallback(dashboard, last_good):
    client, config_path, spawned = dashboard
    from hermes_cli import config

    if last_good:
        config_path.write_text("kanban:\n  max_spawn: 1\n", encoding="utf-8")
        config.load_config()
    config_path.write_text("kanban: [broken", encoding="utf-8")
    before = _snapshot()
    for _ in range(2):  # includes the cached FailedConfigRead path
        assert client.post("/dispatch?max=8").status_code == 503
    assert not spawned
    assert _snapshot() == before


def test_load_exception_refused_before_claim(dashboard, monkeypatch):
    from hermes_cli import config

    client, _, spawned = dashboard

    def unreadable():
        raise OSError("synthetic config read failure")

    monkeypatch.setattr(config, "load_config", unreadable)
    before = _snapshot()
    assert client.post("/dispatch").status_code == 503
    assert not spawned
    assert _snapshot() == before


@pytest.mark.parametrize("caps", [None, {}, {"max_spawn": None, "max_in_progress": None,
                                       "max_in_progress_per_profile": None}])
def test_unset_caps_keep_memory_default(dashboard, monkeypatch, caps):
    from hermes_cli import kanban_db_dispatch as kbd

    client, config_path, spawned = dashboard
    config_path.write_text(json.dumps({"kanban": caps}), encoding="utf-8")
    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda: 1)
    response = client.post("/dispatch?max=8")
    assert response.status_code == 200, response.text
    assert len(response.json()["spawned"]) == len(spawned) == 1

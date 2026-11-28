"""Kanban dashboard plugin: API reads reuse the per-process DB initialisation.

``_conn`` used to call ``init_db`` before every request. ``init_db`` drops the
path from the "already initialised" cache on purpose, so every API call re-ran
the schema script and a full ``PRAGMA integrity_check`` — on a 12k-task board
about 1.75 s per request, paid by every board load and status-bar poll.
``connect`` alone initialises a path once per process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_connect_reuse_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_reads_do_not_rerun_the_integrity_probe(client, monkeypatch):
    assert client.post("/api/plugins/kanban/tasks", json={"title": "one"}).status_code == 200
    probes: list = []
    real = kbc._probe_for_corruption

    def counting(path):
        probes.append(path)
        return real(path)

    monkeypatch.setattr(kbc, "_probe_for_corruption", counting)
    for path in ("/board", "/board", "/stats", "/stats"):
        assert client.get(f"/api/plugins/kanban{path}").status_code == 200
    assert probes == []


def test_fresh_board_still_self_heals_on_first_request(client, tmp_path):
    # A missing DB file is created and initialised by the first API call.
    db = Path(kb.kanban_db_path())
    with kbc._INIT_LOCK:
        kbc._INITIALIZED_PATHS.discard(str(db.resolve()))
    db.unlink()
    for suffix in ("-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    r = client.post("/api/plugins/kanban/tasks", json={"title": "after reset"})
    assert r.status_code == 200, r.text
    titles = [t["title"] for c in client.get("/api/plugins/kanban/board").json()["columns"] for t in c["tasks"]]
    assert titles == ["after reset"]

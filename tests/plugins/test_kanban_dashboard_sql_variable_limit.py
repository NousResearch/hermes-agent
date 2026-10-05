"""The board view must survive boards larger than SQLite's bound-variable ceiling.

``_compute_task_diagnostics`` and the /board summary loaders bind every board id
into one ``IN (?,...)`` list; past ``SQLITE_MAX_VARIABLE_NUMBER`` (999 on builds
< 3.32, 32766 after) the board view died with ``sqlite3.OperationalError: too
many SQL variables``. The id lists now go through ``_id_chunks``; this pins the
legacy 999 ceiling so 1200 rows reproduce it deterministically.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_LEGACY_VARIABLE_CEILING = 999  # SQLITE_MAX_VARIABLE_NUMBER on builds < 3.32

def _load_plugin_module():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py (same harness as
    test_kanban_dashboard_plugin.py) so tests can call its internals directly."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_sql_limit_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home

def _seed_tasks(count: int) -> list[str]:
    conn = kbc.connect()
    try:
        with kb.write_txn(conn):
            return [kb.create_task(conn, title=f"bulk task {i}", assignee="bulk") for i in range(count)]
    finally:
        conn.close()

def _force_legacy_ceiling(monkeypatch):
    """Pin the legacy bound-variable ceiling on every Kanban connection opened from
    here on — the plugin opens its own connection per request, so the limit must
    ride the connect path rather than one connection object."""
    real_connect = kbc.connect

    def limited_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _LEGACY_VARIABLE_CEILING)
        return conn

    monkeypatch.setattr(kbc, "connect", limited_connect)

def test_diagnostics_stay_below_sqlite_variable_limit(kanban_home):
    """The diagnostics slurp (tasks/events/runs/graph) must chunk its IN-lists."""
    ids = _seed_tasks(1200)
    plugin = _load_plugin_module()
    conn = kbc.connect()
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _LEGACY_VARIABLE_CEILING)
        # All three raise "too many SQL variables" when unbounded (1200 > 999).
        assert isinstance(plugin._compute_task_diagnostics(conn, task_ids=None), dict)
        assert isinstance(kb.latest_summaries(conn, ids), dict)
        assert set(kb.task_graph_contexts(conn, ids)) == set(ids)
    finally:
        conn.close()

def test_board_endpoint_stays_below_sqlite_variable_limit(kanban_home, monkeypatch):
    """End-to-end: GET /board over a board past the ceiling returns cards, not a 500."""
    _force_legacy_ceiling(monkeypatch)
    _seed_tasks(1200)

    app = FastAPI()
    app.include_router(_load_plugin_module().router, prefix="/api/plugins/kanban")
    client = TestClient(app)

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200, r.text
    data = r.json()
    assert sum(len(c["tasks"]) for c in data["columns"]) == 1200
    assert data["latest_event_id"] > 0

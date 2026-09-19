"""Board reads must not force a whole-DB rescan per request.

``plugin_api._conn`` used to call ``kanban_db.init_db()`` on every request. That
clears ``_INITIALIZED_PATHS``, so the following ``connect()`` takes the slow
path: a full ``PRAGMA integrity_check`` over every page of the board DB, the
schema script and the optional-column migration pass — all under the
cross-process init flock. Polling the board then costs O(DB size) per request,
which is the read churn a dashboard under Kanban load showed.
"""
import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

OK = 200
BOARD_PATH = "/api/plugins/kanban/board"
TASKS_PATH = "/api/plugins/kanban/tasks"
READS = 3


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _load_plugin(monkeypatch):
    plugin_path = Path(os.environ.get(
        "HERMES_TEST_KANBAN_PLUGIN",
        str(Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"),
    ))
    spec = importlib.util.spec_from_file_location("kanban_board_read_test", plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/kanban")
    return app


def _count_whole_db_work(monkeypatch) -> dict:
    """Count whole-DB probes + schema/migration passes the connect path runs."""
    counts = {"probes": 0, "migrations": 0}
    real_probe = kbc._run_integrity_check
    real_migrate = kbc._migrate_add_optional_columns

    def counting_probe(conn):
        counts["probes"] += 1
        return real_probe(conn)

    def counting_migrate(conn):
        counts["migrations"] += 1
        return real_migrate(conn)

    monkeypatch.setattr(kbc, "_run_integrity_check", counting_probe)
    monkeypatch.setattr(kbc, "_migrate_add_optional_columns", counting_migrate)
    return counts


def _seed_board(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    for i in range(5):
        task_id = kb.create_task(kbc.connect(), title=f"seeded {i}", body="x" * 200)
        with kbc.connect() as conn:
            kb._append_event(conn, task_id, "progress", {"n": i}, run_id=None)


@pytest.mark.anyio
async def test_board_reads_do_not_rescan_the_whole_db(tmp_path, monkeypatch):
    _seed_board(tmp_path, monkeypatch)
    app = _load_plugin(monkeypatch)
    counts = _count_whole_db_work(monkeypatch)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(READS):
            response = await client.get(BOARD_PATH)
            assert response.status_code == OK

    assert counts["probes"] == 0, (
        f"{READS} board reads ran {counts['probes']} whole-DB integrity_check probe(s): "
        "every dashboard board read re-scans the entire board DB"
    )
    assert counts["migrations"] == 0, (
        f"{READS} board reads ran {counts['migrations']} schema/migration pass(es) under the init flock: "
        "reads must not write the schema on every poll"
    )


@pytest.mark.anyio
async def test_first_read_on_a_fresh_home_still_self_heals(tmp_path, monkeypatch):
    """The read path keeps creating the board schema it dropped init_db() for."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    app = _load_plugin(monkeypatch)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        listed = await client.get(BOARD_PATH)
        created = await client.post(TASKS_PATH, json={"title": "first task"})
        after = await client.get(BOARD_PATH)

    assert listed.status_code == OK
    assert listed.json()["columns"], "a fresh board still renders its columns"
    assert all(not c["tasks"] for c in listed.json()["columns"])
    assert created.status_code == OK, created.text
    assert [t["title"] for c in after.json()["columns"] for t in c["tasks"]] == ["first task"]

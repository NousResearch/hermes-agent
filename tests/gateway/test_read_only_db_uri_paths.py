"""Read-only SQLite opens address the real file whatever characters its path holds.

A raw ``f"file:{path}?mode=ro"`` URI ends the path at ``#`` or ``?`` and percent-decodes ``%xx``,
so a HERMES_HOME such as ``~/he#rmes`` opened some other path (dropping ``mode=ro`` with the rest,
which then CREATED an empty database there) and the probe reported the wrong answer.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_rooms as rooms
from gateway import readiness, run_idle_exit
from hermes_cli import approvals_suggest, kanban_db
from hermes_cli.observability import shared_metrics_snapshot

HOME_NAMES = [
    "he#rmes",
    "he%41rmes",
    pytest.param("he?rmes", marks=pytest.mark.platforms("posix")),  # '?' is not a Windows filename char
]


def _sqlite(path, *statements):
    with closing(sqlite3.connect(path)) as conn:
        for statement in statements:
            conn.execute(statement)
        conn.commit()


@pytest.fixture(params=HOME_NAMES)
def home(request, tmp_path):
    home = tmp_path / request.param
    home.mkdir()
    # Snapshot, not ``== [home]``: the suite's autouse isolation keeps its own HERMES_HOME here too.
    before = set(tmp_path.iterdir())
    yield home
    assert set(tmp_path.iterdir()) == before, "a read-only open created a file at a truncated path"


def test_readiness_probe_reads_the_state_db(home):
    _sqlite(home / "state.db", "CREATE TABLE sessions (id TEXT)")

    assert readiness._probe_state_db(home) == {"status": "ok"}


def test_idle_exit_sees_due_kanban_work(home, monkeypatch):
    db = home / "kanban.db"
    _sqlite(db, "CREATE TABLE tasks (status TEXT)", "INSERT INTO tasks VALUES ('ready')")
    monkeypatch.setattr(kanban_db, "list_boards", lambda include_archived: [{"slug": "default"}])
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda board=None: db)

    assert run_idle_exit._kanban_busy() == "kanban work on board default"


def test_hosted_rooms_import_reads_the_legacy_store(home):
    rooms.create_room(home / "state.db", room_id="room-1", name="Release room",
                      members=[{"profile": "ops", "handle": "ops"}], authority_gateway_id="gateway-a", now=10)

    assert [room["room_id"] for room in rooms.list_rooms(home / "shared-state.db")] == ["room-1"]


def test_cli_readers_open_the_state_db(home):
    _sqlite(home / "state.db", "CREATE TABLE sessions (started_at REAL)", "INSERT INTO sessions VALUES (42.0)")

    with closing(approvals_suggest._connect_readonly(home / "state.db")) as conn:
        assert conn.execute("SELECT started_at FROM sessions").fetchone() == (42.0,)
    assert shared_metrics_snapshot._first_session_started_at(home) == 42.0

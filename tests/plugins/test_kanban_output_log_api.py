"""The Kanban "Output log" dashboard route: resolution, states, paging, read-only proof.

The route under test is the backend half of the bundled ``kanban-output-log`` plugin
(``plugins/kanban-output-log/dashboard/plugin_api.py``). It resolves

    board -> task -> owner-admitted worker session -> assignee profile's state.db -> tail

and returns a tail of that session's messages as feed lines for the Desktop card drawer.

These tests drive the REAL route against a real scratch ``HERMES_HOME``: a real board
database created by ``kanban_db``, and a real profile store seeded through the storage
interface (``open_persistence`` + ``tests/fixtures/storage_seed``), so the fixture cannot
drift from the schema the product writes. They cover both halves of the contract — the
states the drawer renders, and the promise that a request writes nothing anywhere.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session

PROFILE = "worker"
SESSION_ID = "local-" + "a" * 64
PLUGIN_ID = "kanban-output-log"
PREFIX = f"/api/plugins/{PLUGIN_ID}"

#: A small but representative transcript: a user prompt, an assistant turn carrying BOTH a
#: reasoning block and a tool call, the tool's result, and a closing assistant turn.
MESSAGES = (
    {"role": "user", "content": "work kanban task t_demo", "timestamp": 1000.0},
    {
        "role": "assistant",
        "reasoning": "I should read the file first.",
        "content": "Reading it now.",
        "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}],
        "timestamp": 1001.0,
    },
    {"role": "tool", "tool_name": "read_file", "content": "line one\nline two", "timestamp": 1002.0},
    {"role": "assistant", "content": "All done.", "timestamp": 1003.0},
)

#: ``MESSAGES`` is 4 rows; one of them yields three feed lines (thinking, text, tool call).
EXPECTED_LINES = 6


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _load_plugin_module():
    """Load the plugin's router module by path, the way the dashboard's loader does."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / PLUGIN_ID / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        f"hermes_dashboard_plugin_{PLUGIN_ID}_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin_api():
    return _load_plugin_module()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch Hermes root with an initialised default board database."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return root


@pytest.fixture
def conn(home):
    c = kbc.connect()
    yield c
    c.close()


@pytest.fixture
def client(plugin_api, home):
    app = FastAPI()
    app.include_router(plugin_api.router, prefix=PREFIX)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _profile_state_db(home: Path) -> Path:
    return home / "profiles" / PROFILE / "state.db"


def _seed_worker_session(home: Path, *, session_id: str = SESSION_ID, messages=MESSAGES) -> list[int]:
    """Write a real profile store with one seeded session; returns the message row ids.

    The named profile home must exist before its store is opened — the storage layer refuses
    to create a profile implicitly (``assert_named_profile_home_live``), which is the same
    rule the product follows for a real profile.
    """
    (home / "profiles" / PROFILE).mkdir(parents=True, exist_ok=True)
    with open_persistence(profile_scope(profile=PROFILE)) as store:
        seed_session(store, session_id, "kanban")
        return seed_messages(store, session_id, *messages)


def _task(conn, *, status: str = "running", assignee: str = PROFILE) -> str:
    task_id = kb.create_task(conn, title="output log fixture", assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    return task_id


def _admit(conn, task_id: str, *, session_id: str = SESSION_ID, db: str | None = None) -> int:
    """Record an ``owner_admitted`` event the way the profile owner does.

    Mirrors ``gateway/session_kanban.py``: a ``task_runs`` row for the attempt, then the
    event whose payload carries both the live session id and the ``state.db`` the owner
    wrote it to.
    """
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) VALUES (?, ?, ?, ?)",
            (task_id, PROFILE, "running", int(time.time())),
        )
        run_id = conn.execute("SELECT id FROM task_runs ORDER BY id DESC LIMIT 1").fetchone()[0]
        kb._append_event(
            conn, task_id, "owner_admitted",
            {"db": db, "session_id": session_id, "request_id": "kanban:fixture", "pid": 1, "birth": 1.0},
            run_id=run_id,
        )
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    return run_id


def _get(client, task_id: str, **params):
    return client.get(f"{PREFIX}/tasks/{task_id}/output", params=params)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# A running card: the transcript the drawer renders
# ---------------------------------------------------------------------------

def test_running_card_returns_its_worker_transcript(client, conn, home):
    task_id = _task(conn, status="running")
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    body = _get(client, task_id).json()

    assert body["state"] == "ok"
    assert body["task_id"] == task_id
    assert body["task_status"] == "running"
    assert body["assignee"] == PROFILE
    assert body["profile"] == PROFILE
    assert body["session_id"] == SESSION_ID
    assert body["source"] == "owner_admitted"
    assert body["total"] == len(MESSAGES)
    assert body["has_earlier"] is False
    assert body["started_at"] == 1000.0
    assert body["last_at"] == 1003.0


def test_entries_expand_rows_into_ordered_feed_lines(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    entries = _get(client, task_id).json()["entries"]

    # One row can yield several lines: the assistant turn contributes thinking, text and
    # its tool call, in that order, disambiguated by `seq`.
    assert [(e["kind"], e["seq"]) for e in entries] == [
        ("user", 0),
        ("thinking", 0),
        ("text", 1),
        ("tool_call", 2),
        ("tool_result", 0),
        ("text", 1),
    ]
    assert entries[1]["text"] == "I should read the file first."
    assert entries[3]["tool_name"] == "read_file"
    assert entries[3]["text"] == '{"path": "a.txt"}'
    assert entries[4]["tool_name"] == "read_file"
    assert entries[4]["text"] == "line one line two"  # whitespace collapsed to one line


def test_row_ids_are_stable_and_ordered_for_client_keying(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    entries = _get(client, task_id).json()["entries"]

    keys = [(e["id"], e["seq"]) for e in entries]
    assert keys == sorted(keys), "feed lines must arrive chronologically"
    assert len(set(keys)) == len(keys), "ids must not collide for React keys"
    assert all(isinstance(e["id"], int) for e in entries)


def test_absolute_store_path_is_never_disclosed(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    raw = _get(client, task_id).text

    assert str(home) not in raw
    assert "state.db" not in raw
    assert str(_profile_state_db(home)) not in raw


# ---------------------------------------------------------------------------
# Finished, queued and degraded cards
# ---------------------------------------------------------------------------

def test_done_card_transcript_is_resolved_from_run_metadata(client, conn, home):
    """A card whose run recorded ``worker_session_id`` resolves without an admission event."""
    task_id = _task(conn, status="done")
    _seed_worker_session(home)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (task_id, PROFILE, "done", int(time.time()), json.dumps({"worker_session_id": SESSION_ID})),
        )
        run_id = conn.execute("SELECT id FROM task_runs ORDER BY id DESC LIMIT 1").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))

    body = _get(client, task_id).json()

    assert body["state"] == "ok"
    assert body["source"] == "run_metadata"
    assert body["session_id"] == SESSION_ID
    assert body["total"] == len(MESSAGES)
    assert body["returned"] == EXPECTED_LINES


def test_retry_uses_current_run_metadata_not_prior_admission(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    other_session = "local-" + "b" * 64
    _seed_worker_session(home, session_id=other_session,
                         messages=({"role": "assistant", "content": "second run", "timestamp": 2000.0},))
    _admit(conn, task_id, db=str(_profile_state_db(home)))
    with kb.write_txn(conn):
        run = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, metadata) "
            "VALUES (?, ?, ?, ?, ?) RETURNING id",
            (task_id, PROFILE, "running", int(time.time()),
             json.dumps({"worker_session_id": other_session})),
        ).fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run, task_id))

    body = _get(client, task_id).json()
    assert body["state"] == "ok"
    assert body["run_id"] == run
    assert body["session_id"] == other_session
    assert body["source"] == "run_metadata"
    assert [line["text"] for line in body["entries"]] == ["second run"]


def test_retry_without_admission_or_metadata_has_no_output(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))
    with kb.write_txn(conn):
        run = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, ?, ?, ?) RETURNING id",
            (task_id, PROFILE, "running", int(time.time())),
        ).fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run, task_id))
    # The per-task log is not run-scoped; it too can contain the previous attempt.
    log_path = kb.worker_log_path(task_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"Session: {SESSION_ID}\n", encoding="utf-8")

    body = _get(client, task_id).json()
    assert body["state"] == "no_session"
    assert body["session_id"] is None
    assert body["entries"] == []


def test_admitted_session_label_is_owning_run_not_reassigned_card(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    run = _admit(conn, task_id, db=str(_profile_state_db(home)))
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET current_run_id=?, assignee=? WHERE id=?", (run, "another", task_id))

    body = _get(client, task_id).json()
    assert body["state"] == "ok"
    assert body["assignee"] == "another"
    assert body["profile"] == PROFILE


def test_worker_log_session_line_is_the_last_resort(client, conn, home):
    """With no event and no run metadata, the per-task log's ``Session:`` line still resolves."""
    task_id = _task(conn, status="done")
    _seed_worker_session(home)
    log_path = kb.worker_log_path(task_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"Session: {SESSION_ID}\nsome other line\n", encoding="utf-8")

    body = _get(client, task_id).json()

    assert body["state"] == "ok"
    assert body["source"] == "worker_log"
    assert body["session_id"] == SESSION_ID


def test_queued_card_has_no_session(client, conn):
    task_id = _task(conn, status="todo")

    body = _get(client, task_id).json()

    assert body["state"] == "no_session"
    assert body["session_id"] is None
    assert body["entries"] == []
    assert body["total"] == 0
    assert body["task_status"] == "todo"


def test_pruned_session_is_reported_separately_from_an_empty_one(client, conn, home):
    """`session_missing` (row gone) and `no_messages` (row present, no rows) differ."""
    empty_session = "local-" + "c" * 64
    _seed_worker_session(home)  # the real session
    _seed_worker_session(home, session_id=empty_session, messages=())  # exists, no rows

    gone = _task(conn)
    _admit(conn, gone, session_id="local-" + "z" * 64, db=str(_profile_state_db(home)))
    empty = _task(conn)
    _admit(conn, empty, session_id=empty_session, db=str(_profile_state_db(home)))

    assert _get(client, gone).json()["state"] == "session_missing"
    assert _get(client, empty).json()["state"] == "no_messages"


def test_missing_profile_store_is_no_db_not_a_crash(client, conn, home):
    """The session resolves, but the owning profile has no store on disk."""
    task_id = _task(conn)
    _admit(conn, task_id, db=str(_profile_state_db(home)))  # never seeded -> no file

    body = _get(client, task_id).json()

    assert body["state"] == "no_db"
    assert body["session_id"] == SESSION_ID
    assert body["profile"] == PROFILE
    assert not _profile_state_db(home).exists(), "the route must not create the store"


def test_assignee_without_a_valid_profile_name_degrades_to_no_db(client, conn, home):
    """Board data names the profile, so a hostile name must not reach a path join."""
    task_id = _task(conn, assignee="../../etc")
    _admit(conn, task_id, session_id=SESSION_ID, db=str(_profile_state_db(home)))

    body = _get(client, task_id).json()

    # The admission event still resolves the session, but the profile names no real store.
    assert body["state"] == "no_db"


# ---------------------------------------------------------------------------
# Hostile input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slug", ["../../etc", "a/b", ".hidden", "-dash", "Upper Case", "a" * 65])
def test_malformed_board_slugs_are_400(client, slug):
    assert _get(client, "t_x", board=slug).status_code == 400


def test_unknown_task_is_404_and_leaves_the_board_intact(client, conn, home):
    task_id = _task(conn)
    before = _digest(kb.kanban_db_path())

    response = _get(client, "t_does_not_exist")

    assert response.status_code == 404
    assert _digest(kb.kanban_db_path()) == before
    with kb.write_txn(conn, allow_nested=True):
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == 1


def test_sql_metacharacters_in_a_task_id_are_not_executed(client, conn, home):
    """The task id is a bound parameter, never interpolated."""
    _task(conn)
    hostile = "t_'; DROP TABLE tasks; --"

    response = _get(client, hostile)

    assert response.status_code == 404
    assert client.get(f"{PREFIX}/tasks/t_nope/output").status_code == 404
    count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'").fetchone()[0]
    assert count == 1, "tasks table must survive a hostile id"


def test_a_hostile_payload_db_path_is_never_read(client, conn, home, tmp_path):
    """`task_events.payload.db` is board data: an absolute path there must be inert."""
    secret = tmp_path / "secret" / "state.db"
    secret.parent.mkdir()
    secret.write_text("not a database", encoding="utf-8")

    task_id = _task(conn)
    _admit(conn, task_id, db=str(secret))

    body = _get(client, task_id).json()

    # The untrusted path is ignored and resolution falls through to "no session".
    assert body["state"] == "no_session"
    assert body["session_id"] is None
    assert str(secret) not in _get(client, task_id).text


@pytest.mark.parametrize(
    "case,tail,before",
    [("tail_abc", "abc", None), ("tail_zero", 0, None), ("tail_too_big", 2001, None),
     ("before_zero", 200, 0), ("before_abc", 200, "abc")],
)
def test_out_of_range_paging_arguments_are_422(client, conn, case, tail, before):
    task_id = _task(conn)

    params = {"tail": tail}
    if before is not None:
        params["before"] = before

    assert _get(client, task_id, **params).status_code == 422


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------

def test_tail_bounds_the_window_and_has_earlier_drives_the_cursor(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    page = _get(client, task_id, tail=2).json()

    assert page["state"] == "ok"
    assert page["total"] == len(MESSAGES), "total counts the whole session, not the window"
    assert page["has_earlier"] is True
    assert page["last_at"] == 1003.0
    # The cursor addresses the oldest row in this window.
    assert page["next_before"] == page["entries"][0]["id"]


def test_before_pages_backwards_to_the_start_of_the_transcript(client, conn, home):
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    newest = _get(client, task_id, tail=2).json()
    older = _get(client, task_id, tail=2, before=newest["next_before"]).json()

    assert older["state"] == "ok"
    assert older["has_earlier"] is False
    older_ids = {e["id"] for e in older["entries"]}
    newest_ids = {e["id"] for e in newest["entries"]}
    assert older_ids.isdisjoint(newest_ids)
    assert max(older_ids) < min(newest_ids)


# ---------------------------------------------------------------------------
# Read-only proof
# ---------------------------------------------------------------------------

def test_the_board_connection_the_route_uses_refuses_every_write(plugin_api, conn, home):
    """`mode=ro` + `query_only`: the exact handle the route opens cannot write."""
    board = kb.kanban_db_path()
    readonly = plugin_api._connect_board_readonly(board)
    try:
        assert readonly.execute("PRAGMA query_only").fetchone()[0] == 1
        for statement in (
            "INSERT INTO tasks (id, title, status) VALUES ('t_x', 'x', 'todo')",
            "CREATE TABLE probe (id INTEGER)",
            "DROP TABLE tasks",
        ):
            with pytest.raises(sqlite3.OperationalError):
                readonly.execute(statement)
    finally:
        readonly.close()


def test_a_request_writes_nothing_anywhere(client, conn, home):
    """A 200 must leave both databases byte-identical and take no init lock.

    The claim under test is that the route never creates, migrates or writes a database. It
    is deliberately NOT asserted as "no file appears under the home": SQLite may materialise
    empty ``-wal``/``-shm`` sidecars for a WAL database that no process currently holds open,
    which is a property of a read-only open, not a write to any table.
    """
    task_id = _task(conn)
    _seed_worker_session(home)
    _admit(conn, task_id, db=str(_profile_state_db(home)))

    board = kb.kanban_db_path()
    store = _profile_state_db(home)
    board_before, store_before = _digest(board), _digest(store)
    databases_before = sorted(p.name for p in home.rglob("*.db"))
    # The fixture's own write connection already holds the board's init lock; what matters is
    # that the two GETs neither take a new one nor release/rewrite the existing one.
    locks_before = sorted(str(p.relative_to(home)) for p in home.rglob("*.init.lock"))

    assert _get(client, task_id).status_code == 200
    assert _get(client, task_id).status_code == 200

    assert _digest(board) == board_before, "the board database must be untouched"
    assert _digest(store) == store_before, "the profile store must be untouched"
    assert sorted(p.name for p in home.rglob("*.db")) == databases_before, "no new database"
    assert sorted(str(p.relative_to(home)) for p in home.rglob("*.init.lock")) == locks_before


def test_missing_board_is_404_and_is_not_created(client, home):
    """The default board's database is absent: report it, never create it."""
    (home / "kanban.db").unlink()
    files_before = sorted(p.name for p in home.rglob("*") if p.is_file())

    response = _get(client, "t_x", board="nosuchboard")

    assert response.status_code == 404
    assert sorted(p.name for p in home.rglob("*") if p.is_file()) == files_before


def test_uninitialised_board_is_409_and_stays_zero_bytes(client, home):
    (home / "kanban.db").unlink()
    (home / "kanban.db").write_bytes(b"")

    response = _get(client, "t_x")

    assert response.status_code == 409
    assert (home / "kanban.db").stat().st_size == 0


# ---------------------------------------------------------------------------
# The validation helpers, directly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", [".", "..", "../../etc", "a/b", "a\\b", "a b"])
def test_profile_state_db_rejects_anything_that_is_not_a_profile_name(plugin_api, profile):
    assert plugin_api._profile_state_db(profile) is None


def test_profile_state_db_treats_an_empty_assignee_as_the_default_profile(plugin_api, home):
    """An unassigned card names no profile; it reads the root store, never a joined path."""
    assert plugin_api._profile_state_db("") == home / "state.db"


def test_profile_state_db_maps_a_real_name_under_the_root(plugin_api, home):
    assert plugin_api._profile_state_db(PROFILE) == home / "profiles" / PROFILE / "state.db"


def test_profile_state_db_uses_the_root_store_for_the_default_profile(plugin_api, home):
    assert plugin_api._profile_state_db("default") == home / "state.db"
    assert plugin_api._profile_state_db(None) == home / "state.db"


@pytest.mark.parametrize(
    "value,accepted",
    [("/etc/passwd", False), ("/tmp/evil/state.db", False), (123, None), ("", False)],
)
def test_validated_state_db_refuses_paths_outside_the_root(plugin_api, home, value, accepted):
    assert (plugin_api._validated_state_db(value) is not None) is bool(accepted)


def test_validated_state_db_accepts_a_store_under_the_root(plugin_api, home):
    candidate = home / "profiles" / PROFILE / "state.db"
    assert plugin_api._validated_state_db(str(candidate)) == candidate.resolve()

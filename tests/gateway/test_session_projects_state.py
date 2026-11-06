"""``sessions.project_id`` storage: migration, lineage-wide binding, inheritance, one-time backfill,
and the server-owned project line in the system prompt."""

import sqlite3
from types import SimpleNamespace

import pytest

from hermes_cli import projects_db as pdb
from hermes_state import SessionDB
from hermes_state_session_projects import SESSION_PROJECT_BACKFILL_DONE, set_session_project


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _columns(conn, table="sessions"):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(conn):
    return {row[1] for row in conn.execute("PRAGMA index_list(sessions)")}


def test_fresh_store_has_nullable_project_id_and_its_index(db):
    assert "project_id" in _columns(db._conn)
    assert "idx_sessions_project_id" in _indexes(db._conn)
    db.create_session("s1", "api_server")
    assert db.get_session("s1")["project_id"] is None
    # A fresh store binds explicitly from birth: the legacy backfill never runs on it.
    assert db.get_meta(SESSION_PROJECT_BACKFILL_DONE) == "1"


def test_legacy_store_gains_the_column_on_open(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    # Simulate a pre-project_id store: drop the column (and the index that references it).
    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX IF EXISTS idx_sessions_project_id")
    conn.execute("ALTER TABLE sessions DROP COLUMN project_id")
    conn.commit()
    assert "project_id" not in _columns(conn)
    conn.close()
    reopened = SessionDB(path)
    try:
        assert "project_id" in _columns(reopened._conn)
        assert "idx_sessions_project_id" in _indexes(reopened._conn)
    finally:
        reopened.close()


def test_binding_covers_the_whole_compression_lineage(db):
    db.create_session("root", "api_server", cwd="/work/p")
    db.append_message("root", "user", "hello")
    assert db.try_acquire_compression_lock("root", "holder", ttl_seconds=60)
    db.publish_compression_child(parent_session_id="root", child_session_id="tip", source="api_server",
                                 system_prompt="p", messages=[{"role": "user", "content": "summary"}],
                                 compression_lock_holder="holder")
    assert set_session_project(db, "tip", "p_1")
    assert {db.get_session(s)["project_id"] for s in ("root", "tip")} == {"p_1"}
    set_session_project(db, "root", None)
    assert {db.get_session(s)["project_id"] for s in ("root", "tip")} == {None}


def test_compression_continuation_inherits_project(db):
    db.create_session("parent", "api_server", cwd="/work/p")
    set_session_project(db, "parent", "p_1")
    db.append_message("parent", "user", "original")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    db.publish_compression_child(parent_session_id="parent", child_session_id="child", source="api_server",
                                 system_prompt="p", messages=[{"role": "user", "content": "summary"}],
                                 compression_lock_holder="holder")
    child = db.get_session("child")
    assert (child["project_id"], child["cwd"]) == ("p_1", "/work/p")


def test_child_rows_inherit_project_only_in_the_parents_folder(db):
    db.create_session("parent", "api_server", cwd="/work/p")
    set_session_project(db, "parent", "p_1")
    db.create_session("same", "api_server", parent_session_id="parent")  # cwd inherited
    db.create_session("same-explicit", "api_server", parent_session_id="parent", cwd="/work/p")
    db.create_session("elsewhere", "api_server", parent_session_id="parent", cwd="/elsewhere")
    assert db.get_session("same")["project_id"] == "p_1"
    assert db.get_session("same-explicit")["project_id"] == "p_1"
    assert db.get_session("elsewhere")["project_id"] is None


def test_list_filters_on_project_id(db):
    for sid, pid in (("a", "p_1"), ("b", None), ("c", "p_2")):
        db.create_session(sid, "api_server")
        db.append_message(sid, "user", "hi")
        set_session_project(db, sid, pid)
    assert [s["id"] for s in db.list_sessions_rich(project_id="p_1")] == ["a"]
    assert {s["id"] for s in db.list_sessions_rich()} == {"a", "b", "c"}


def _legacy_store_with_projects(tmp_path):
    """A store whose backfill has not run yet, plus a projects.db beside it."""
    proj_dir, sub = tmp_path / "proj", tmp_path / "proj" / "sub"
    sub.mkdir(parents=True)
    archived_dir = tmp_path / "old"
    archived_dir.mkdir()
    with pdb.connect_closing(tmp_path / "projects.db") as conn:
        pid = pdb.create_project(conn, name="Proj", folders=[str(proj_dir)])
        old = pdb.create_project(conn, name="Old", folders=[str(archived_dir)])
        pdb.archive_project(conn, old)
    store = SessionDB(tmp_path / "state.db")
    rows = {
        "desktop": ("desktop", str(proj_dir)), "api": ("api_server", str(proj_dir) + "/"),
        "tui": ("tui", str(proj_dir)), "cli": ("cli", str(proj_dir)), "subdir": ("tui", str(sub)),
        "archived": ("desktop", str(archived_dir)), "explicit": ("desktop", str(proj_dir)),
    }
    for sid, (source, cwd) in rows.items():
        store.create_session(sid, source, cwd=cwd)
    set_session_project(store, "explicit", "p_other")
    store._conn.execute("DELETE FROM state_meta WHERE key = ?", (SESSION_PROJECT_BACKFILL_DONE,))
    store._conn.commit()
    store.close()
    return pid


def test_one_time_backfill_binds_client_sessions_in_a_primary_folder(tmp_path):
    pid = _legacy_store_with_projects(tmp_path)
    store = SessionDB(tmp_path / "state.db")
    try:
        bound = {sid: store.get_session(sid)["project_id"]
                 for sid in ("desktop", "api", "tui", "cli", "subdir", "archived", "explicit")}
        assert bound == {"desktop": pid, "api": pid, "tui": pid, "cli": None, "subdir": None,
                         "archived": None, "explicit": "p_other"}
        assert store.get_meta(SESSION_PROJECT_BACKFILL_DONE) == "1"
        # One-time: a later unbound row in the folder is not re-bound on the next open.
        store.create_session("later", "desktop", cwd=str(tmp_path / "proj"))
    finally:
        store.close()
    again = SessionDB(tmp_path / "state.db")
    try:
        assert again.get_session("later")["project_id"] is None
    finally:
        again.close()


def test_backfill_without_projects_db_is_a_marked_noop(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    store.create_session("s", "desktop", cwd=str(tmp_path))
    store._conn.execute("DELETE FROM state_meta WHERE key = ?", (SESSION_PROJECT_BACKFILL_DONE,))
    store._conn.commit()
    store.close()
    reopened = SessionDB(tmp_path / "state.db")
    try:
        assert reopened.get_session("s")["project_id"] is None
        assert reopened.get_meta(SESSION_PROJECT_BACKFILL_DONE) == "1"
        assert not (tmp_path / "projects.db").exists()  # never created by the backfill
    finally:
        reopened.close()


# ── A6: the server-owned project line ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("project", "line"), [
    ({"name": "Otto", "folder": "/Users/j/code/hermes-ios", "board_slug": "otto"},
     "Project: Otto — /Users/j/code/hermes-ios (kanban board: otto)."),
    ({"name": "Otto", "folder": "/x", "board_slug": None}, "Project: Otto — /x."),
    ({"name": "Otto", "folder": None, "board_slug": "otto"}, "Project: Otto (kanban board: otto)."),
    (None, ""),
])
def test_project_line_text(project, line):
    from agent.prompt_builder import build_session_project_line
    assert build_session_project_line(project) == line


def test_project_line_is_built_from_the_session_row(tmp_path):
    from agent.system_prompt import _session_project_part
    folder = tmp_path / "otto"
    folder.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="Otto", folders=[str(folder)], board_slug="otto")
    store = SessionDB(tmp_path / "state.db")
    try:
        store.create_session("bound", "api_server", cwd=str(folder))
        set_session_project(store, "bound", pid)
        store.create_session("plain", "api_server")
        agent = SimpleNamespace(_session_db=store, session_id="bound")
        # The agent's own profile home is the store's directory; point it at the test home's projects.db.
        from hermes_constants import get_hermes_home
        (tmp_path / "projects.db").write_bytes((get_hermes_home() / "projects.db").read_bytes())
        assert _session_project_part(agent) == [f"Project: Otto — {folder} (kanban board: otto)."]
        assert _session_project_part(SimpleNamespace(_session_db=store, session_id="plain")) == []
    finally:
        store.close()


def test_project_line_lives_in_the_context_tier(tmp_path):
    """Per-project text must stay out of the cross-session stable prefix and out of the per-turn
    volatile tail: it rides the context tier, built once per conversation from the session row."""
    import agent.system_prompt as sp
    folder = tmp_path / "otto"
    folder.mkdir()
    store = SessionDB(tmp_path / "state.db")
    try:
        with pdb.connect_closing(tmp_path / "projects.db") as conn:
            pid = pdb.create_project(conn, name="Otto", folders=[str(folder)], board_slug="otto")
        store.create_session("bound", "api_server", cwd=str(folder))
        set_session_project(store, "bound", pid)
        agent = SimpleNamespace(
            load_soul_identity=False, skip_context_files=True, valid_tool_names=[],
            _task_completion_guidance=False, _tool_use_enforcement=False, _environment_probe=False,
            _kanban_worker_guidance="", _memory_store=None, _memory_manager=None, model="", provider="",
            platform="", pass_session_id=False, session_id="bound", _session_db=store)
        parts = sp.build_system_prompt_parts(agent)
    finally:
        store.close()
    line = f"Project: Otto \u2014 {folder} (kanban board: otto)."
    assert line in parts["context"]
    assert "Project: Otto" not in parts["stable"] + parts["volatile"]

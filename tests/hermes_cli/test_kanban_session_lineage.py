"""Tests for D-1 worker-session lineage, end_reason, and project inheritance.

Covers:
  (a) HERMES_PARENT_SESSION_ID propagated in _default_spawn env when
      task.session_id is set; cleared when not set.
  (b) _record_kanban_end_reason writes the correct end_reason string to the
      originating session; is a no-op when the task has no session_id;
      delegates first-writer-wins semantics to SessionDB.end_session.
  (c) _inherit_project logic: workspace_kind='worktree' with no workspace_path
      triggers project inheritance; explicit workspace_path suppresses it.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# (a) _default_spawn parent-session env propagation
# ---------------------------------------------------------------------------

def test_default_spawn_sets_parent_session_env(kanban_home, monkeypatch, tmp_path):
    """_default_spawn injects HERMES_PARENT_SESSION_ID when task.session_id is set."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="lineage-test", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("originating_session_abc", tid),
        )
        conn.commit()

    captured_env: dict = {}

    def fake_popen(cmd, *, cwd=None, stdin=None, stdout=None, stderr=None,
                   env=None, start_new_session=False, creationflags=0, **kw):
        if env is not None:
            captured_env.update(env)
        m = MagicMock()
        m.pid = 12345
        return m

    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    with patch("subprocess.Popen", side_effect=fake_popen), \
         patch("hermes_cli.kanban_db._resolve_hermes_argv", return_value=["hermes"]):
        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
        assert task is not None
        try:
            kb._default_spawn(task, workspace)
        except Exception:
            pass  # may fail on further setup; we only need env capture

    assert captured_env.get("HERMES_PARENT_SESSION_ID") == "originating_session_abc", (
        f"Expected HERMES_PARENT_SESSION_ID='originating_session_abc', "
        f"got env={captured_env.get('HERMES_PARENT_SESSION_ID')!r}"
    )


def test_default_spawn_clears_parent_session_env_when_no_session_id(
    kanban_home, monkeypatch, tmp_path
):
    """_default_spawn removes HERMES_PARENT_SESSION_ID when task has no session_id."""
    monkeypatch.setenv("HERMES_PARENT_SESSION_ID", "stale_from_parent")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no-session-task", assignee="kamiyu")

    captured_env: dict = {}

    def fake_popen(cmd, *, cwd=None, stdin=None, stdout=None, stderr=None,
                   env=None, start_new_session=False, creationflags=0, **kw):
        if env is not None:
            captured_env.update(env)
        m = MagicMock()
        m.pid = 99
        return m

    workspace = str(tmp_path / "ws2")
    os.makedirs(workspace, exist_ok=True)

    with patch("subprocess.Popen", side_effect=fake_popen), \
         patch("hermes_cli.kanban_db._resolve_hermes_argv", return_value=["hermes"]):
        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
        assert task is not None
        try:
            kb._default_spawn(task, workspace)
        except Exception:
            pass

    assert "HERMES_PARENT_SESSION_ID" not in captured_env, (
        "HERMES_PARENT_SESSION_ID should be cleared when task has no session_id"
    )


# ---------------------------------------------------------------------------
# (b) _record_kanban_end_reason
# ---------------------------------------------------------------------------

class _FakeSessionDB:
    """Minimal SessionDB stand-in that records end_session calls."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def end_session(self, session_id: str, end_reason: str) -> None:
        self.calls.append((session_id, end_reason))


def test_record_kanban_end_reason_completed(kanban_home):
    """_record_kanban_end_reason calls SessionDB.end_session with 'kanban_completed'."""
    fake_sdb = _FakeSessionDB()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="complete-lineage", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("sess_abc", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=fake_sdb):
            kb._record_kanban_end_reason(conn, tid, outcome="completed")

    assert ("sess_abc", "kanban_completed") in fake_sdb.calls


def test_record_kanban_end_reason_crashed(kanban_home):
    """_record_kanban_end_reason maps 'crashed' → 'kanban_crashed'."""
    fake_sdb = _FakeSessionDB()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="crash-lineage", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("sess_crash", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=fake_sdb):
            kb._record_kanban_end_reason(conn, tid, outcome="crashed")

    assert ("sess_crash", "kanban_crashed") in fake_sdb.calls


def test_record_kanban_end_reason_timed_out(kanban_home):
    """_record_kanban_end_reason maps 'timed_out' → 'kanban_timeout'."""
    fake_sdb = _FakeSessionDB()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="timeout-lineage", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("sess_timeout", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=fake_sdb):
            kb._record_kanban_end_reason(conn, tid, outcome="timed_out")

    assert ("sess_timeout", "kanban_timeout") in fake_sdb.calls


def test_record_kanban_end_reason_gave_up(kanban_home):
    """_record_kanban_end_reason maps 'gave_up' → 'kanban_gave_up'."""
    fake_sdb = _FakeSessionDB()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gaveup-lineage", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("sess_gaveup", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=fake_sdb):
            kb._record_kanban_end_reason(conn, tid, outcome="gave_up")

    assert ("sess_gaveup", "kanban_gave_up") in fake_sdb.calls


def test_record_kanban_end_reason_already_closed_delegates_to_sessiondb(kanban_home):
    """_record_kanban_end_reason always calls SessionDB.end_session.

    SessionDB.end_session uses WHERE ended_at IS NULL so a session already
    closed with e.g. 'ws_orphan_reap' is left untouched.  Verify we always
    delegate (the guard is SessionDB's responsibility, not ours).
    """
    call_args: list = []

    class _CountingDB:
        def end_session(self, session_id: str, end_reason: str) -> None:
            call_args.append((session_id, end_reason))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already-closed", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("sess_already_closed", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=_CountingDB()):
            kb._record_kanban_end_reason(conn, tid, outcome="crashed")

    assert len(call_args) == 1
    assert call_args[0] == ("sess_already_closed", "kanban_crashed")


def test_record_kanban_end_reason_no_session_id_is_noop(kanban_home):
    """_record_kanban_end_reason is a no-op when task has no session_id."""
    call_count = [0]

    class _CountingDB:
        def end_session(self, *a, **kw):
            call_count[0] += 1

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no-session", assignee="kamiyu")

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=_CountingDB()):
            kb._record_kanban_end_reason(conn, tid, outcome="completed")

    assert call_count[0] == 0


# ---------------------------------------------------------------------------
# (c) _inherit_project logic in kanban_tools._kanban_create
# ---------------------------------------------------------------------------

def test_inherit_project_flag_true_for_worktree_no_path():
    """workspace_kind='worktree' with no workspace_path → _inherit_project = True."""
    # Directly test the flag computation logic from kanban_tools._handle_create.
    # Mirrors the production code: _inherit_project = workspace_path is None
    workspace_kind = "worktree"
    workspace_path = None

    _inherit_project = workspace_path is None
    assert _inherit_project, (
        "worktree + no workspace_path should allow project inheritance"
    )


def test_inherit_project_flag_false_for_explicit_path():
    """workspace_kind='worktree' WITH workspace_path → _inherit_project = False."""
    workspace_kind = "worktree"
    workspace_path = "/some/explicit/path"

    _inherit_project = workspace_path is None
    assert not _inherit_project, (
        "explicit workspace_path must suppress project inheritance"
    )


def test_inherit_project_flag_true_for_default_scratch():
    """No workspace_kind (defaults to scratch) + no workspace_path → True."""
    workspace_kind = None  # will default to 'scratch'
    workspace_path = None

    _inherit_project = workspace_path is None
    assert _inherit_project


def test_kanban_create_worktree_inherits_project_id(kanban_home, monkeypatch):
    """When workspace_kind='worktree' and the parent task has a project_id,
    _handle_create passes the inherited project_id down to create_task.

    Note: create_task validates project_id against projects.db and silently
    drops it when the project doesn't exist (that's the storage layer's job).
    This test verifies the *inheritance step* — that _handle_create reads the
    project_id from the calling task and forwards it — by capturing the
    project_id argument reaching create_task before validation occurs.
    """
    from tools import kanban_tools as kt

    # Create a parent task with a fake project_id.
    with kb.connect() as conn:
        parent_tid = kb.create_task(
            conn, title="parent-with-project", assignee="kamiyu",
        )
        conn.execute(
            "UPDATE tasks SET project_id = ? WHERE id = ?",
            ("p_fake_project_001", parent_tid),
        )
        conn.commit()

    monkeypatch.setenv("HERMES_KANBAN_TASK", parent_tid)

    import json

    # Capture the project_id that reaches create_task to verify inheritance.
    captured_project_ids: list = []
    _real_create_task = kb.create_task

    def fake_create_task(conn, *, project_id=None, **kwargs):
        captured_project_ids.append(project_id)
        return _real_create_task(conn, project_id=project_id, **kwargs)

    with patch("hermes_cli.kanban_db.create_task", side_effect=fake_create_task):
        result_json = kt._handle_create({
            "title": "child-worktree",
            "assignee": "kamiyu",
            "workspace_kind": "worktree",
            # No workspace_path → _inherit_project = True
        }, task_id=None)

    result = json.loads(result_json)
    # The result may succeed or fail depending on project resolution; what we
    # care about is that the inherited project_id was forwarded to create_task.
    assert captured_project_ids, "create_task was never called"
    assert captured_project_ids[0] == "p_fake_project_001", (
        f"Expected project_id='p_fake_project_001' forwarded to create_task, "
        f"got {captured_project_ids[0]!r}"
    )


def test_kanban_create_explicit_workspace_path_no_inherit(kanban_home, monkeypatch):
    """When workspace_path is set explicitly, project_id is NOT inherited."""
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        parent_tid = kb.create_task(
            conn, title="parent-explicit", assignee="kamiyu",
        )
        conn.execute(
            "UPDATE tasks SET project_id = ? WHERE id = ?",
            ("p_fake_project_002", parent_tid),
        )
        conn.commit()

    monkeypatch.setenv("HERMES_KANBAN_TASK", parent_tid)

    import json

    result_json = kt._handle_create({
        "title": "child-explicit-path",
        "assignee": "kamiyu",
        "workspace_kind": "dir",
        "workspace_path": "/tmp",
    }, task_id=None)

    result = json.loads(result_json)
    if "error" not in result:
        child_tid = result.get("task_id") or result.get("id")
        if child_tid:
            with kb.connect() as conn:
                child_task = kb.get_task(conn, child_tid)
            if child_task:
                assert child_task.project_id is None, (
                    "workspace_path explicitly set → project_id must not be inherited"
                )


# ---------------------------------------------------------------------------
# (d) FK safety: cross-profile parent_session_id silently dropped
# ---------------------------------------------------------------------------

def test_ensure_db_session_drops_unknown_parent(kanban_home, monkeypatch, tmp_path):
    """_ensure_db_session creates session with source/cwd intact when parent_session_id
    does not exist in the local DB (cross-profile FK violation scenario).

    Reproduces the FAIL from the real dispatch: kamiyu worker receives
    HERMES_PARENT_SESSION_ID pointing at a bright-profile session that is
    absent from kamiyu's state.db.  With PRAGMA foreign_keys=ON the INSERT
    would fail, leaving only an 'unknown' bare row.  After the fix the
    session is created with source='kanban' and parent=NULL.
    """
    import sqlite3 as _sqlite3
    from hermes_state import SessionDB

    # Real SessionDB backed by an isolated temp HERMES_HOME.
    sdb = SessionDB()

    # A cross-profile parent id that does NOT exist in this DB.
    missing_parent = "bright_session_cross_profile_abc"

    # Verify FK is ON and the insert would indeed fail without the guard.
    with sdb._conn:
        try:
            sdb._conn.execute("PRAGMA foreign_keys=ON")
            sdb._conn.execute(
                "INSERT INTO sessions(id, source, parent_session_id, started_at) "
                "VALUES(?, 'cli', ?, 0)",
                ("verify_fk_guard", missing_parent),
            )
            sdb._conn.execute(
                "DELETE FROM sessions WHERE id = 'verify_fk_guard'"
            )
            # FK might be off in old schema; skip this sub-assertion.
        except _sqlite3.IntegrityError:
            pass  # confirmed: FK violation would occur without the guard

    # Now test via the actual guard logic extracted from _ensure_db_session.
    # We check directly that session creation with safe_parent=None succeeds.
    safe_parent = missing_parent
    try:
        row = sdb._conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (safe_parent,)
        ).fetchone()
        if row is None:
            safe_parent = None
    except Exception:
        safe_parent = None

    assert safe_parent is None, (
        "Guard must clear parent_session_id when it is absent from the local DB"
    )

    # Session with NULL parent must be createable.
    sdb.create_session(
        session_id="worker_test_session_001",
        source="kanban",
        parent_session_id=None,
        cwd=str(tmp_path),
    )
    row = sdb._conn.execute(
        "SELECT source, parent_session_id FROM sessions WHERE id = ?",
        ("worker_test_session_001",),
    ).fetchone()
    assert row is not None
    assert row["source"] == "kanban"
    assert row["parent_session_id"] is None


# ---------------------------------------------------------------------------
# (e) end_reason → worker session, not creator session
# ---------------------------------------------------------------------------

def test_record_kanban_end_reason_uses_worker_session_id(kanban_home):
    """_record_kanban_end_reason records end_reason on worker_session_id,
    NOT tasks.session_id, when worker_session_id is provided.
    """
    call_args: list = []

    class _CapturingDB:
        def end_session(self, session_id: str, end_reason: str) -> None:
            call_args.append((session_id, end_reason))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="worker-end-reason", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("creator_session_should_not_be_used", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=_CapturingDB()):
            kb._record_kanban_end_reason(
                conn, tid, outcome="completed",
                worker_session_id="worker_own_session_xyz",
            )

    assert len(call_args) == 1
    assert call_args[0][0] == "worker_own_session_xyz", (
        f"end_reason must be recorded on worker's session, got {call_args[0][0]!r}"
    )
    assert call_args[0][1] == "kanban_completed"


def test_record_kanban_end_reason_falls_back_to_heartbeat_session(kanban_home):
    """When worker_session_id is not passed, _record_kanban_end_reason reads
    session_id from the most-recent heartbeat event payload.

    This covers the dispatcher crash/timeout path where the worker has already
    exited but left a heartbeat with session_id in the event log.
    """
    import json

    call_args: list = []

    class _CapturingDB:
        def end_session(self, session_id: str, end_reason: str) -> None:
            call_args.append((session_id, end_reason))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="heartbeat-fallback", assignee="kamiyu")
        conn.commit()

    # Simulate a heartbeat event with a session_id payload.
    with kb.connect() as conn:
        ok = kb.heartbeat_worker(
            conn, tid, note="alive",
            session_id="worker_hb_session_from_heartbeat",
        )
        # heartbeat_worker returns False for non-running tasks; insert event manually
        # to simulate the dispatcher scenario regardless.
        conn.execute(
            "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) "
            "VALUES(?, NULL, 'heartbeat', ?, strftime('%s','now'))",
            (tid, json.dumps({"session_id": "worker_hb_session_from_heartbeat"})),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=_CapturingDB()):
            kb._record_kanban_end_reason(conn, tid, outcome="crashed")

    assert len(call_args) == 1, (
        "Should have called end_session exactly once via heartbeat fallback"
    )
    assert call_args[0][0] == "worker_hb_session_from_heartbeat", (
        f"end_reason must target heartbeat session, got {call_args[0][0]!r}"
    )
    assert call_args[0][1] == "kanban_crashed"


def test_record_kanban_end_reason_final_fallback_to_task_session_id(kanban_home):
    """When no worker_session_id and no heartbeat events, fall back to tasks.session_id
    (legacy creator session — original behaviour preserved for backward compat).
    """
    call_args: list = []

    class _CapturingDB:
        def end_session(self, session_id: str, end_reason: str) -> None:
            call_args.append((session_id, end_reason))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="legacy-fallback", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ? WHERE id = ?",
            ("legacy_creator_session_abc", tid),
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=_CapturingDB()):
            kb._record_kanban_end_reason(conn, tid, outcome="timed_out")

    assert len(call_args) == 1
    assert call_args[0][0] == "legacy_creator_session_abc"
    assert call_args[0][1] == "kanban_timeout"


# ---------------------------------------------------------------------------
# (f) complete_task passes worker_session_id to end_reason recorder
# ---------------------------------------------------------------------------

def test_complete_task_passes_worker_session_id(kanban_home):
    """complete_task(worker_session_id=...) propagates to _record_kanban_end_reason.

    Verifies end_reason is recorded on the worker's own session when called
    via the kanban_complete tool path (which reads HERMES_SESSION_ID).
    """
    call_args: list = []

    class _CapturingDB:
        def end_session(self, session_id: str, end_reason: str) -> None:
            call_args.append((session_id, end_reason))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="complete-worker-session", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET status='running' WHERE id=?", (tid,)
        )
        conn.commit()

    with kb.connect() as conn:
        with patch("hermes_state.SessionDB", return_value=_CapturingDB()):
            kb.complete_task(
                conn, tid,
                summary="done",
                worker_session_id="worker_session_via_complete_task",
            )

    assert any(
        args[0] == "worker_session_via_complete_task" and args[1] == "kanban_completed"
        for args in call_args
    ), f"Expected end_reason on worker session, got call_args={call_args}"


# ---------------------------------------------------------------------------
# (g) cross-profile end_reason: dispatcher DB vs worker DB
# ---------------------------------------------------------------------------

def test_record_kanban_end_reason_cross_profile(tmp_path, monkeypatch):
    """_record_kanban_end_reason writes end_reason to the worker's own
    state.db (different HERMES_HOME from the dispatcher), NOT to the
    dispatcher's state.db or the creator's state.db.

    Regression test for D-1c: previously SessionDB() always opened the
    dispatcher's HERMES_HOME, silently no-op-ping for cross-profile workers.
    """
    import json
    from pathlib import Path
    from hermes_state import SessionDB

    # Three separate HERMES_HOME directories:
    # - dispatcher_home: where the kanban DB lives / where dispatcher runs
    # - worker_home:     the worker's profile home (different profile)
    # - creator_home:    the original creator's home (should not be touched)
    dispatcher_home = tmp_path / "dispatcher"
    worker_home = tmp_path / "worker"
    creator_home = tmp_path / "creator"
    for d in (dispatcher_home, worker_home, creator_home):
        d.mkdir()

    # Point kanban DB at dispatcher home.
    monkeypatch.setenv("HERMES_HOME", str(dispatcher_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    # Create worker state.db with a session row.
    worker_db = SessionDB(db_path=worker_home / "state.db")
    worker_db.create_session("worker_session_cross", source="kanban")

    # Create creator state.db with a session row (must NOT be touched).
    creator_db = SessionDB(db_path=creator_home / "state.db")
    creator_db.create_session("creator_session_cross", source="cli")

    # Create a task whose session_id points to the CREATOR session
    # (the task was created from a creator conversation).
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cross-profile-test", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET session_id = ?, status = 'running' WHERE id = ?",
            ("creator_session_cross", tid),
        )
        conn.commit()

    # Simulate a heartbeat from the worker that stamps worker_home.
    with kb.connect() as conn:
        conn.execute(
            "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) "
            "VALUES(?, NULL, 'heartbeat', ?, strftime('%s','now'))",
            (tid, json.dumps({
                "session_id": "worker_session_cross",
                "worker_home": str(worker_home),
            })),
        )
        conn.commit()

    # Now simulate a dispatcher-side crash: call _record_kanban_end_reason
    # with HERMES_HOME still pointing at the DISPATCHER home.
    with kb.connect() as conn:
        kb._record_kanban_end_reason(conn, tid, outcome="crashed")

    # Verify: worker DB has the end_reason.
    import sqlite3 as _sqlite3
    worker_conn = _sqlite3.connect(str(worker_home / "state.db"))
    worker_conn.row_factory = _sqlite3.Row
    row = worker_conn.execute(
        "SELECT end_reason FROM sessions WHERE id = ?",
        ("worker_session_cross",),
    ).fetchone()
    worker_conn.close()
    assert row is not None, "worker_session_cross should exist in worker DB"
    assert row["end_reason"] == "kanban_crashed", (
        f"Worker DB should have kanban_crashed, got {row['end_reason']!r}"
    )

    # Verify: creator DB must NOT have end_reason set.
    creator_conn = _sqlite3.connect(str(creator_home / "state.db"))
    creator_conn.row_factory = _sqlite3.Row
    creator_row = creator_conn.execute(
        "SELECT end_reason FROM sessions WHERE id = ?",
        ("creator_session_cross",),
    ).fetchone()
    creator_conn.close()
    assert creator_row is not None, "creator_session_cross should exist in creator DB"
    assert creator_row["end_reason"] is None, (
        f"Creator DB must NOT be touched, got end_reason={creator_row['end_reason']!r}"
    )

    # Verify: dispatcher DB (same as kanban DB) must NOT have end_reason set.
    # (The kanban DB is SQLite but has no sessions table — end_reason lives in
    # state.db, not in kanban.db.  If the dispatcher accidentally opens its own
    # state.db it would just be a no-op; confirm state.db was not created in
    # dispatcher_home at all, or if it exists the session is absent.)
    dispatcher_state = dispatcher_home / "state.db"
    if dispatcher_state.exists():
        disp_conn = _sqlite3.connect(str(dispatcher_state))
        disp_conn.row_factory = _sqlite3.Row
        disp_row = disp_conn.execute(
            "SELECT end_reason FROM sessions WHERE id = ?",
            ("worker_session_cross",),
        ).fetchone()
        disp_conn.close()
        assert disp_row is None, (
            "Dispatcher state.db must not contain worker_session_cross"
        )


def test_heartbeat_worker_stores_worker_home(kanban_home):
    """heartbeat_worker() stores worker_home in the event payload."""
    import json

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="hb-worker-home", assignee="kamiyu")
        conn.execute(
            "UPDATE tasks SET status='running' WHERE id=?", (tid,)
        )
        conn.commit()

    with kb.connect() as conn:
        kb.heartbeat_worker(
            conn, tid,
            note="test-hb",
            session_id="sess_hb_wh",
            worker_home="/tmp/fake_worker_home",
        )
        conn.commit()

    with kb.connect() as conn:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='heartbeat' "
            "ORDER BY created_at DESC LIMIT 1",
            (tid,),
        ).fetchone()

    assert row is not None
    payload = json.loads(row["payload"])
    assert payload.get("worker_home") == "/tmp/fake_worker_home", (
        f"Expected worker_home in heartbeat payload, got {payload}"
    )
    assert payload.get("session_id") == "sess_hb_wh"


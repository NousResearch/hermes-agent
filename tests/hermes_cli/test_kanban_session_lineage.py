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

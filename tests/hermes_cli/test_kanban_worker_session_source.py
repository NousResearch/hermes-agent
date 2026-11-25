"""Kanban worker runs must not surface as user conversations.

Workers spawn as `hermes chat -q "work kanban task <id>"`, which used to land in
state.db as an untitled `cli` row — the desktop sidebar then rendered one entry
per attempt, labeled with the worker's own prompt.
"""

import os

import pytest

from persistence.conversation.sessions import SessionQuery
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        yield store


def test_worker_spawn_tags_session_source_kanban(monkeypatch, tmp_path):
    """The dispatcher tags the worker's env so its session is a `kanban` row."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    captured = {}

    class _Proc:
        pid = 4321

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    task = kb.Task(
        id="t_b21733fb",
        title="ship it",
        body=None,
        assignee="default",
        status="in_progress",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    kbd._default_spawn(task, workspace)

    assert captured["env"]["HERMES_SESSION_SOURCE"] == "kanban"


def test_kanban_rows_stay_out_of_the_session_list(db):
    """A `kanban` row is filtered by the same exclude the sidebar sends."""
    seed_session(db, "chat", "desktop")
    seed_messages(db, "chat", {"role": "user", "content": "hey"})
    seed_session(db, "worker", "kanban")
    seed_messages(db, "worker", {"role": "user", "content": "work kanban task t_b21733fb"})

    listed = db.sessions.list_sessions(SessionQuery(exclude_sources=("cron", "kanban", "subagent", "tool")))

    assert [entry.session.session_id for entry in listed] == ["chat"]


def test_retag_reclaims_legacy_worker_rows(db, tmp_path):
    """Rows written before the tag existed are identified by workspace cwd.

    Two rows, not one: the count has to survive ``set_meta`` reusing the same
    cursor, which would otherwise report the meta write's rowcount instead.
    """
    workspaces = tmp_path / "kanban" / "workspaces"
    seed_session(db, "legacy", "cli", cwd=str(workspaces / "t_b21733fb"))
    seed_session(db, "legacy2", "cli", cwd=str(workspaces / "t_c0ffee"))
    seed_session(db, "mine", "cli", cwd=str(tmp_path / "www" / "repo"))

    assert db.sessions.retag_worker_sessions(str(workspaces)) == 2

    with raw_holder(db)._read_ctx() as conn:
        sources = {row[0]: row[1] for row in conn.execute("SELECT id, source FROM sessions")}
    assert sources == {"legacy": "kanban", "legacy2": "kanban", "mine": "cli"}


def test_retag_runs_once_per_workspaces_root(db, tmp_path):
    """The state_meta gate keeps the retag off every subsequent spawn."""
    workspaces = tmp_path / "kanban" / "workspaces"
    seed_session(db, "legacy", "cli", cwd=str(workspaces / "t_a"))
    db.sessions.retag_worker_sessions(str(workspaces))

    # A row that a *new* worker would never write as `cli`; if the gate leaked,
    # a later sweep would grab it too.
    seed_session(db, "later", "cli", cwd=str(workspaces / "t_b"))

    assert db.sessions.retag_worker_sessions(str(workspaces)) == 0
    assert db.sessions.get("later").source == "cli"


def test_retag_gate_is_per_board(db, tmp_path):
    """A second board's workspaces root still gets its own sweep.

    The gate is keyed on the root, so reclaiming board A must not convince the
    dispatcher that board B's legacy rows were already handled.
    """
    board_a = tmp_path / "kanban" / "boards" / "a" / "workspaces"
    board_b = tmp_path / "kanban" / "boards" / "b" / "workspaces"
    seed_session(db, "a1", "cli", cwd=str(board_a / "t_a"))
    seed_session(db, "b1", "cli", cwd=str(board_b / "t_b"))

    assert db.sessions.retag_worker_sessions(str(board_a)) == 1
    assert db.sessions.retag_worker_sessions(str(board_b)) == 1

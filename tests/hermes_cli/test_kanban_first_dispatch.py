"""PR handoffs are inputs to a task's first run, not proof of duplicate work."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    import hermes_cli.config as cfg
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(cfg, "load_config", lambda *a, **kw: {})
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


@pytest.mark.parametrize("with_parent", [False, True])
def test_first_dispatch_accepts_pr_input(board, with_parent):
    parents = []
    if with_parent:
        parent = kb.create_task(board, title="implementation", assignee="default")
        kb.claim_task(board, parent)
        kb.complete_task(board, parent, summary="implementation ready")
        parents.append(parent)
    task = kb.create_task(board, title="integrate", assignee="default", parents=parents)
    kb.add_comment(board, task, author="default", body="Integrate https://github.com/example/repo/pull/42")
    assert not kb.list_runs(board, task)
    assert kbd.check_respawn_guard(board, task) is None
    result = kbd.dispatch_once(board, dry_run=True)
    assert task in [row[0] for row in result.spawned]
    assert task not in dict(result.respawn_guarded)


@pytest.mark.parametrize("ended", [False, True])
def test_prior_implementer_still_guarded(board, ended):
    task = kb.create_task(board, title="implement", assignee="default")
    kb.claim_task(board, task)
    kb.add_comment(board, task, author="default", body="Opened https://github.com/example/repo/pull/42")
    if ended:
        with kb.write_txn(board):
            board.execute("UPDATE task_runs SET ended_at = started_at, outcome = 'crashed' WHERE task_id = ?", (task,))
    assert kbd.check_respawn_guard(board, task) == "active_pr"


def test_first_dispatch_does_not_bypass_auth_blocker(board):
    task = kb.create_task(board, title="integrate", assignee="default")
    kb.add_comment(board, task, author="default", body="Input https://github.com/example/repo/pull/42")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error = 'authentication failed' WHERE id = ?", (task,))
    assert kbd.check_respawn_guard(board, task) == "blocker_auth"

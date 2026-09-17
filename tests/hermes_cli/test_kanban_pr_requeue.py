"""PR evidence and explicit correction requests must not strand ready tasks."""
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    kb.init_db()


def test_explicit_unblock_supersedes_pr_comment_but_not_new_publication(board):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='Correct existing PR', assignee='worker')
        kb.add_comment(conn, tid, author='worker', body='Opened https://github.com/example/repo/pull/123')
        assert kbd.check_respawn_guard(conn, tid) == 'active_pr'
        assert kbd.respawn_guard_diagnostic(conn, tid, 'ready').data['reason'] == 'active_pr'
        kb.block_task(conn, tid, reason='Need the missing finding description')
        assert kb.unblock_task(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) is None
        assert kbd.respawn_guard_diagnostic(conn, tid, 'ready') is None
        kb.add_comment(conn, tid, author='operator', body='Please use the PR finding as context')
        assert kbd.check_respawn_guard(conn, tid) is None
        kb.add_comment(conn, tid, author='worker', body='Starting the requested correction')
        assert kbd.check_respawn_guard(conn, tid) is None
        # All operations may have the same second timestamp; event order matters.
        kb.add_comment(conn, tid, author='worker', body='Opened https://github.com/example/repo/pull/124')
        assert kbd.check_respawn_guard(conn, tid) == 'active_pr'
        conn.execute("UPDATE tasks SET last_failure_error = 'authentication failed' WHERE id = ?", (tid,))
        assert kbd.check_respawn_guard(conn, tid) == 'blocker_auth'


def test_operator_reference_link_is_not_worker_publication(board):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='Correct existing PR', assignee='impl-claude')
        kb.add_comment(conn, tid, author='operator', body='Finding source: https://github.com/example/repo/pull/123')
        assert kbd.check_respawn_guard(conn, tid) is None
        kb.add_comment(conn, tid, author='impl-claude', body='Opened https://github.com/example/repo/pull/124')
        assert kbd.check_respawn_guard(conn, tid) == 'active_pr'

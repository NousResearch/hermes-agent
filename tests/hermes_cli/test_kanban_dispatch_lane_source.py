from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch

@pytest.mark.parametrize('lane', ['ready', 'review'])
@pytest.mark.parametrize('accept_board', [True, False])
def test_dispatch_lane_reaches_spawn(tmp_path, monkeypatch, lane, accept_board):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.init_db()
    monkeypatch.setattr(dispatch, '_profile_exists_fn', lambda: lambda name: True)
    seen = []
    def spawn(task, workspace, *, source_status):
        seen.append((source_status, task.status, task.current_run_id))
    def spawn_with_board(task, workspace, *, source_status, board):
        return spawn(task, workspace, source_status=source_status)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='lane routing', assignee='worker', workspace_kind='dir', initial_status='blocked', workspace_path=str(tmp_path))
        assert kb.unblock_task(conn, tid)
        if lane == 'review':
            claimed = kb.claim_task(conn, tid)
            assert kb.request_review(conn, tid, summary='implemented', reviewer='worker', expected_run_id=claimed.current_run_id)
        row = conn.execute('SELECT * FROM tasks WHERE id = ?', (tid,)).fetchone()
        result = dispatch.DispatchResult()
        assert dispatch._dispatch_lane_task(conn, row, 'worker', result, lane=lane, dry_run=False,
            ttl_seconds=None, board=None, failure_limit=2, spawn_fn=spawn_with_board if accept_board else spawn,
            per_profile_cap=None, per_profile_running={})
        assert len(seen) == 1
        assert seen[0][0] == lane
        assert seen[0][1] == 'running'
        assert seen[0][2] == kb.get_task(conn, tid).current_run_id
        if lane == 'review':
            assert kb.complete_task(conn, tid, summary='approved', expected_run_id=seen[0][2])

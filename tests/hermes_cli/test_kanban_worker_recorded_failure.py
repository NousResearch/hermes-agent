"""An owner-recorded failure reaches the console line and ``task_runs.error``.

The 2026-11-02 boot failure surfaced as ``Kanban owner refused: Kanban execution unknown;
retained by owner`` because the submitter ignored what the run had already recorded. When the
owner recorded a failure for the run, that text — not "execution unknown" — must be what the
submitter prints, and the dead-worker sweep must fold it into the run's error.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import gateway_client
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client

RECORDED = "kanban_skills_unavailable: nowhere-skill missing on profile 'fabro'"


def test_unknown_receipt_surfaces_the_recorded_failure(monkeypatch, tmp_path):
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result',
                        lambda *_a, **_k: {'exit_code': 1, 'error': RECORDED})
    params = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}

    with pytest.raises(gateway_client.GatewayClientError) as failure:
        kanban_worker_client._exit_code({'status': 'unknown'}, params, str(tmp_path / 'kanban.db'))

    assert 'nowhere-skill' in str(failure.value) and "'fabro'" in str(failure.value)
    assert 'execution unknown' not in str(failure.value).lower()


def test_unknown_receipt_without_a_recorded_failure_keeps_the_honest_unknown(monkeypatch, tmp_path):
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result', lambda *_a, **_k: {'exit_code': 1})
    params = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}

    with pytest.raises(gateway_client.GatewayClientError, match='execution unknown'):
        kanban_worker_client._exit_code({'status': 'unknown'}, params, str(tmp_path / 'kanban.db'))


def test_the_console_line_carries_the_reason(monkeypatch, tmp_path, capsys):
    for key, value in {'HERMES_KANBAN_BOARD': 'b', 'HERMES_KANBAN_TASK': 't1',
                       'HERMES_KANBAN_RUN_ID': '1', 'HERMES_KANBAN_CLAIM_LOCK': 'lock',
                       'HERMES_KANBAN_DB': str(tmp_path / 'kanban.db')}.items():
        monkeypatch.setenv(key, value)

    async def refused(_params, _board):
        raise gateway_client.GatewayClientError(f'Kanban execution failed: {RECORDED}')

    monkeypatch.setattr(kanban_worker_client, 'run', refused)

    assert kanban_worker_client._submit() == 1
    stderr = capsys.readouterr().err
    assert 'nowhere-skill' in stderr and "'fabro'" in stderr, stderr


def _dead_pid() -> int:
    proc = subprocess.Popen(['true'])
    proc.wait()
    return proc.pid


def test_the_reason_lands_in_the_run_error(tmp_path, monkeypatch):
    """The dead-worker sweep folds the submitter's last line into ``task_runs.error``."""
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title='unrunnable', assignee='fabro', skills=['nowhere-skill'])
        task = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:owner")
        assert task is not None
        kbd._set_worker_pid(conn, task_id, _dead_pid())
        log = kb.worker_log_path(task_id)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f'Session: local-x\nKanban dispatch failed: Kanban execution failed: {RECORDED}\n',
                       encoding='utf-8')

        assert kbd.detect_crashed_workers(conn) == [task_id]
        error = conn.execute('SELECT error FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()[0]

    assert 'nowhere-skill' in error and "'fabro'" in error, error

"""Limite de cota usa SQLite real; nenhum modelo ou banco do operador.

Fixtures representam saidas de workers. A classificacao do exit code ja tem
cobertura propria; aqui medimos admissao, estacionamento e retomada.
"""
import time
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
    with kbc.connect() as conn:
        yield conn


def quota_exit(conn, tid, lane):
    now = int(time.time()) - 1000
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'default', 'rate_limited', 'rate_limited', ?, ?)",
            (tid, now - 10, now),
        )
        conn.execute(
            "UPDATE tasks SET status=?, last_failure_error='provider rate-limited (quota wall)' WHERE id=?",
            (lane, tid),
        )


@pytest.mark.parametrize('lane', ['ready', 'review'])
@pytest.mark.parametrize('cooldown', ['0', '300'])
def test_quota_budget_parks_without_losing_history_or_blocking_other_tasks(board, monkeypatch, lane, cooldown):
    monkeypatch.setenv('HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS', cooldown)
    tid = kb.create_task(board, title='quota esgotada', assignee='default')
    other = kb.create_task(board, title='outro trabalho', assignee='default')
    quota_exit(board, tid, lane)
    assert kbd.check_respawn_guard(board, tid, lane=lane) is None
    quota_exit(board, tid, lane)
    assert kbd.check_respawn_guard(board, tid, lane=lane) == 'rate_limit_exhausted'

    snapshot = [tuple(r) for r in board.execute('SELECT * FROM task_runs WHERE task_id=?', (tid,))]
    dry = kbd.dispatch_once(board, dry_run=True, max_in_progress=6)
    assert (tid, 'rate_limit_exhausted') in dry.respawn_guarded
    assert tid not in [s[0] for s in dry.spawned]
    assert other in [s[0] for s in dry.spawned]
    assert kb.get_task(board, tid).status == lane

    # Dispara apenas a guarda de admissao, nunca a inferencia.
    result = kbd.DispatchResult()
    def no_spawn(*args, **kwargs):
        pytest.fail('Guarda esgotada nao pode iniciar worker')

    row = board.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone()
    assert not kbd._dispatch_lane_task(
        board, row, 'default', result, lane=lane, dry_run=False,
        ttl_seconds=None, board=None, failure_limit=2, spawn_fn=no_spawn,
        per_profile_cap=None, per_profile_running={},
    )
    task = kb.get_task(board, tid)
    assert task.status == 'blocked'
    assert task.block_kind == 'capability'
    assert task.consecutive_failures == 0
    assert snapshot == [tuple(r) for r in board.execute('SELECT * FROM task_runs WHERE task_id=?', (tid,))]
    assert tid in result.auto_blocked
    assert kb.recompute_ready(board) == 0

    # Desbloqueio explicito autoriza uma sonda; tempo sozinho nao autoriza.
    assert kb.unblock_task(board, tid)
    assert kb.get_task(board, tid).status == lane
    assert kbd.check_respawn_guard(board, tid, lane=lane) is None
    quota_exit(board, tid, lane)
    assert kbd.check_respawn_guard(board, tid, lane=lane) == 'rate_limit_exhausted'


def test_real_progress_resets_quota_streak(board):
    tid = kb.create_task(board, title='cota recuperada', assignee='default')
    quota_exit(board, tid, 'ready')
    with kb.write_txn(board):
        now = int(time.time()) - 1100
        board.execute(
            "INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'default', 'review', 'review_requested', ?, ?)",
            (tid, now - 10, now),
        )
    quota_exit(board, tid, 'review')
    assert kbd.check_respawn_guard(board, tid, lane='review') is None

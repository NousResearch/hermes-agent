"""Atomic initial-hold release; regression for AgentOps #265."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / 'hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    for name in ('HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_WORKSPACES_ROOT',
                 'HERMES_KANBAN_TASK', 'HERMES_DELEGATED_CHILD_CONTEXT'):
        monkeypatch.delenv(name, raising=False)
    kb.init_db()
    return home


def snapshot(conn):
    return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY rowid')]
            for table in ('tasks', 'task_runs', 'task_events')}


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', *args],
        cwd=Path(__file__).parents[2], env=os.environ.copy(), capture_output=True,
        text=True, timeout=30)


def test_conditional_release_is_single_use_and_preserves_parent_gating(board):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title='parent', initial_status='blocked', idempotency_key='parent-key')
        child = kb.create_task(conn, title='child', parents=[parent], initial_status='blocked', idempotency_key='child-key')
    barrier = threading.Barrier(2)
    def release():
        with kbc.connect_closing() as conn:
            barrier.wait(timeout=10)
            return kb.unblock_task(conn, child, expected_initial_key='child-key')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: release(), range(2)))
    assert sorted(results) == [False, True]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, child).status == 'todo'
        assert kb.claim_task(conn, child) is None
    released = cli('unblock', parent, '--if-initial-key', 'parent-key')
    assert released.returncode == 0, released.stderr
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, parent).status == 'ready'
        before = snapshot(conn)
    assert cli('unblock', parent, '--if-initial-key', 'parent-key').returncode != 0
    with kbc.connect_closing() as conn:
        assert snapshot(conn) == before


@pytest.mark.parametrize('case', ['wrong_key', 'blank_key', 'ordinary_block', 'reblocked',
    'started_reblocked', 'scheduled', 'reason', 'bulk'])
def test_conditional_cli_refusal_preserves_entire_board_and_legacy_unblock(board, case):
    with kbc.connect_closing() as conn:
        task = kb.create_task(conn, title='held', idempotency_key='expected',
            initial_status='running' if case == 'ordinary_block' else 'blocked')
        if case in ('reblocked', 'started_reblocked'):
            assert kb.unblock_task(conn, task)
        if case == 'started_reblocked':
            assert kb.claim_task(conn, task)
        if case in ('ordinary_block', 'reblocked', 'started_reblocked'):
            assert kb.block_task(conn, task, kind='needs_input',
                reason='Human requested pause' if case == 'started_reblocked' else None)
        if case == 'scheduled':
            assert kb.schedule_task(conn, task)
        key = {'wrong_key': 'wrong', 'blank_key': ''}.get(case, 'expected')
        args = ['unblock', task, '--if-initial-key', key]
        if case == 'reason':
            args += ['--reason', 'must not add a comment']
        if case == 'bulk':
            args.insert(2, kb.create_task(conn, title='other', initial_status='blocked'))
        before = snapshot(conn)
    refused = cli(*args)
    assert refused.returncode != 0, refused.stdout
    with kbc.connect_closing() as conn:
        assert snapshot(conn) == before
    legacy = cli('unblock', task)
    assert legacy.returncode == 0, legacy.stderr
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task).status == 'ready'

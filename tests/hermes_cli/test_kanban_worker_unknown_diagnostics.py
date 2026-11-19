"""The execution-unknown refusal must name the board, request and owner it could not resolve.

``unknown`` is what a *replaced owner epoch* settles a live admission as, so the predecessor's
row usually still exists: the bare sentence sent every triage round back to the board to
reconstruct which DB, which request and which owner were involved.
"""

import json
import sqlite3

import pytest

from hermes_cli import kanban_worker_client as worker


PARAMS = {'board': 'default', 'task_id': 't_x', 'run_id': 7, 'claim_lock': 'm2studio:1'}


def _board(tmp_path):
    db = tmp_path / 'kanban.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,'
                 ' run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)')
    conn.execute('INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)',
                 ('t_x', 7, 'owner_admitted',
                  json.dumps({'db': '/p/engineering/state.db', 'session_id': 'local-abc', 'request_id':
                              'kanban:["default","t_x",7]', 'pid': 4242, 'birth': 1795100000.5}),
                  1795100000))
    conn.execute('INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)',
                 ('t_x', 8, 'owner_admitted', json.dumps({'pid': 9, 'birth': 1.0}), 1795100001))
    conn.commit()
    conn.close()
    return db


def test_unknown_refusal_names_board_request_and_owner(tmp_path):
    db = _board(tmp_path)
    detail = worker._unknown_detail({'status': 'unknown', 'owner_epoch': 5, 'generation': 1},
                                    PARAMS, str(db), 'local-abc')
    # The leading sentence stays verbatim: logs and operators match on it.
    assert detail.startswith('Kanban execution unknown; retained by owner')
    assert f'db={db}' in detail
    assert 'board=default' in detail
    assert 'task_id=t_x' in detail
    assert 'run_id=7' in detail
    assert 'request_id=kanban:["default","t_x",7]' in detail
    assert 'session_id=local-abc' in detail
    assert 'owner=pid:4242 birth:1795100000.5 db:/p/engineering/state.db' in detail
    assert 'receipt_epoch=5' in detail
    assert 'generation=1' in detail
    # Only this attempt's marker: run 8's owner must not be reported.
    assert 'pid:9' not in detail


def test_unknown_refusal_survives_an_unreadable_board(tmp_path):
    """A missing board must not turn the refusal into a different exception."""
    detail = worker._unknown_detail({'status': 'unknown'}, PARAMS, str(tmp_path / 'absent' / 'kanban.db'),
                                    'local-abc')
    assert detail.startswith('Kanban execution unknown; retained by owner')
    assert 'owner=pid:None' in detail


def test_exit_code_raises_the_named_refusal(tmp_path):
    from hermes_cli.gateway_client import GatewayClientError
    with pytest.raises(GatewayClientError) as raised:
        worker._exit_code({'status': 'unknown'}, PARAMS, str(_board(tmp_path)), 'local-abc')
    assert 'board=default' in str(raised.value)

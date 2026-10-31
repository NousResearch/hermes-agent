"""NULL-owner (legacy/process-owned) delegation rows stay deliverable.

Regression: rows with owner_execution_id IS NULL (local dispatch carries no
execution context) used to be unclaimable — ``complete`` passed (NULL matches
NULL) but every later read/claim/delivery with a real execution id hit
``permission_denied``, stranding the row as completed/pending with zero
delivery attempts and no user-visible error. The owning parent session must
still reach such rows; any other session must still be denied; and dispatch
without an execution context must fail loudly instead of writing a new
NULL-owner row.
"""
import json
import os
import sqlite3
import time
from contextlib import closing

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tools import async_delegation as ledger


def _handlers(execution_id):
    from gateway.session_worker_delegation import worker_delegation_handlers
    return worker_delegation_handlers(execution_id, os.getpid(), psutil.Process().create_time())


def _seed_null_owner_row(db_path, delegation_id='deleg_legacy', session_id='parent', state='completed'):
    """Simulate a pre-fix / local-dispatch row: no execution owner."""
    now = time.time()
    event = {'type': 'async_delegation', 'delegation_id': delegation_id,
             'parent_session_id': session_id, 'session_key': '',
             'origin_ui_session_id': '', 'origin_session_id': '',
             'status': 'completed', 'completed_at': now, 'summary': 'legacy'}
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """INSERT INTO async_delegations
                (delegation_id, origin_session, origin_ui_session_id, parent_session_id,
                 state, dispatched_at, updated_at, event_json, result_json,
                 delivery_state, delivery_attempts, owner_pid, task_json,
                 origin_session_id, owner_execution_id)
                VALUES (?, '', '', ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, '', NULL)""",
            (delegation_id, session_id, state, now - 10, now,
             json.dumps(event) if state != 'running' else None,
             json.dumps({'summary': 'legacy'}) if state != 'running' else None,
             os.getpid(), json.dumps({'goal': 'inert'})))
        conn.commit()
    return event


def _db(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    db.create_session('foreign', source='cli')
    with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
        ledger._initialize_schema(conn)
    return db


def test_null_owner_completed_row_claimable_by_owning_session_only(tmp_path):
    db = _db(tmp_path)
    try:
        _seed_null_owner_row(tmp_path / 'state.db')
        handlers = _handlers('worker-a')  # a real execution id, never NULL

        def apply(name, payload, sid='parent'):
            return db._execute_write(lambda conn: handlers[name](db, conn, sid, payload))

        assert apply('delegation.claim',
                     {'delegation_id': 'deleg_legacy', 'claim_id': 'c1'})['value'] is True
        row = apply('delegation.read', {'delegation_id': 'deleg_legacy'})['value']
        assert row['state'] == 'completed' and row['delivery_state'] == 'pending'
        assert row['result'] == {'summary': 'legacy'}
        assert apply('delegation.ack',
                     {'delegation_id': 'deleg_legacy', 'claim_id': 'c1'})['value'] is True

        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            apply('delegation.read', {'delegation_id': 'deleg_legacy'}, sid='foreign')
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            apply('delegation.claim',
                  {'delegation_id': 'deleg_legacy', 'claim_id': 'x'}, sid='foreign')
    finally:
        db.close()


def test_null_owner_running_row_completes_without_execution_id_then_delivers(tmp_path):
    from gateway.session_worker_delegation import _complete
    db = _db(tmp_path)
    try:
        _seed_null_owner_row(tmp_path / 'state.db', state='running')
        now = time.time()
        event = {'type': 'async_delegation', 'delegation_id': 'deleg_legacy',
                 'parent_session_id': 'parent', 'session_key': '',
                 'origin_ui_session_id': '', 'origin_session_id': '',
                 'status': 'completed', 'completed_at': now, 'summary': 'late'}
        # The completing child presents execution_id=None; NULL==NULL passes.
        db._execute_write(
            lambda conn: _complete(db, conn, 'parent',
                                   {'event': event, 'result': {'summary': 'late'}},
                                   execution_id=None))
        handlers = _handlers('worker-a')
        assert db._execute_write(
            lambda conn: handlers['delegation.claim'](
                db, conn, 'parent',
                {'delegation_id': 'deleg_legacy', 'claim_id': 'c2'}))['value'] is True
    finally:
        db.close()


def test_dispatch_without_execution_context_fails_loudly(tmp_path):
    from gateway.session_worker_delegation import _dispatch
    db = _db(tmp_path)
    try:
        payload = {'delegation_id': 'deleg_new', 'task': {'goal': 'inert'},
                   'dispatched_at': time.time(), 'parent_session_id': 'parent',
                   'session_key': '', 'origin_ui_session_id': '', 'origin_session_id': ''}
        # Both the handler factory and _dispatch itself must refuse: a NULL
        # owner row must never be written from this path again.
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            _handlers(None)
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            db._execute_write(
                lambda conn: _dispatch(db, conn, 'parent', payload, execution_id=None,
                                       worker_pid=os.getpid(),
                                       worker_birth=psutil.Process().create_time()))
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0] == 0
    finally:
        db.close()

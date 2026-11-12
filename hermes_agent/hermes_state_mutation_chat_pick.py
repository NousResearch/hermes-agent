"""Exact-retry receipts for a viewer's pick on a messaging chat (gateway/session_messaging_picks.py).

A messaging chat has no frozen policy for ``apply_action`` to write: its pick is the chat's own
``/model`` / ``/reasoning`` override. It still answers ``session.mutate`` like a LOCAL pick: the
same ``gateway.mutation.v1.*`` receipt key and digest (``mutation_receipt_key``), checked before the
override is applied so an exact retry never re-applies it, and committed with the revision bump in
one fenced transaction.
"""
import json

from hermes_state_runtime import (
    RuntimeStoreError, _epoch, _json, _text, mutation_receipt_key, replay_mutation_receipt,
)


def chat_pick_key(principal_id, session_id, params):
    """``(key, digest)`` of the pick *params* on *session_id* (the mount the fences name)."""
    for value in (principal_id, session_id, params['request_id']):
        _text(value)
    for fence in (params['expected_revision'], params.get('expected_generation')):
        if type(fence) is not int or fence < 0:
            raise RuntimeStoreError('invalid_params')
    payload = json.loads(_json(params['payload']))
    key, digest = mutation_receipt_key(
        principal_id, session_id, params['request_id'], operation=params['operation'], payload=payload,
        expected_revision=params['expected_revision'], expected_generation=params['expected_generation'])
    return key, digest


def _fenced_row(conn, session_id, params):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    if row['runtime_revision'] != params['expected_revision']:
        raise RuntimeStoreError('revision_conflict')
    if row['runtime_generation'] != params['expected_generation']:
        raise RuntimeStoreError('stale_generation')
    return row


def prepare_chat_pick(db, *, epoch, key, digest, session_id, params):
    """The stored receipt of an exact retry (``{'receipt': ...}``), else the fenced mount row
    (``{'row': ...}``); a reused request id with another payload refuses ``admission_conflict``."""
    with db._read_ctx() as conn:
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return {'receipt': replay}
        return {'row': dict(_fenced_row(conn, session_id, params))}


def commit_chat_pick(db, *, epoch, key, digest, session_id, params, result):
    """Bump the mount's revision and store *result* (plus the new revision) as the receipt, in one
    write that re-checks the fences. Returns the committed receipt result."""
    def write(conn):
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay
        _fenced_row(conn, session_id, params)
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        revision = conn.execute('SELECT runtime_revision FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
        committed = {**result, 'revision': revision}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (key, _json({'digest': digest, 'result': committed})))
        return committed
    return db._execute_write(write)

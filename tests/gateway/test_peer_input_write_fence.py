"""Every grant store fences spool mutations through commit; no runtime threads."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sqlite3

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_grant_state import grant_state_db_paths, revoke_grant_state
from gateway.platforms import api_server_room_attachments as spool_api
from tests.gateway.test_peer_input_custody_contract import _inputs, _request, receiver  # noqa: F401


@pytest.mark.parametrize('operation', ['prepare', 'put', 'discard'])
def test_profile_deny_cannot_commit_between_spool_authorization_and_commit(
        receiver, tmp_path, monkeypatch, operation):
    adapter, _, _, _ = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, claims, _, _, _, _, data = inputs
    request = _request(adapter, inputs)
    paths = grant_state_db_paths()
    shared, profile = paths
    assert Path(shared).resolve() != Path(profile).resolve()
    manifest = [{key: value for key, value in item.items() if key != 'path'}
                for item in spool.materialize(dispatch)]
    connect = hosted_rooms._connect

    def no_wait(path):
        conn = connect(path)
        conn.execute('PRAGMA busy_timeout=0')
        return conn

    monkeypatch.setattr(hosted_rooms, '_connect', no_wait)
    monkeypatch.setattr(hosted_rooms, '_transaction', partial(hosted_rooms.transaction, no_wait, immediate=False))
    revoke = hosted_rooms.revoke_room_grant_scope
    attempted = []

    def partial_revoke(path, **kwargs):
        attempted.append(Path(path).resolve())
        if Path(path).resolve() == Path(shared).resolve():
            raise sqlite3.OperationalError('injected shared-store failure')
        return revoke(path, **kwargs)

    monkeypatch.setattr(hosted_rooms, 'revoke_room_grant_scope', partial_revoke)
    transaction = spool._transaction
    denied_before_commit = []

    @contextmanager
    def before_commit(*, immediate=False):
        with transaction(immediate=immediate) as conn:
            yield conn
            if immediate:
                # Interleave real best-effort revocation on separate connections,
                # after authorization and mutation but before the spool commit.
                with pytest.raises(sqlite3.OperationalError, match='injected shared-store failure'):
                    revoke_grant_state(paths, claims=claims, expires_at=claims['status_expires_at'])
                denied_before_commit.append(hosted_rooms.room_grant_is_revoked(profile, claims=claims))

    monkeypatch.setattr(spool, 'prune', lambda **_: 0)  # No expired rows in this fixture.
    monkeypatch.setattr(spool, '_transaction', before_commit)
    guard = spool_api._write_guard(adapter, request, claims, 'attachment.stage')
    if operation == 'prepare':
        spool.prepare(dispatch, manifest, authorize_write=guard)
    elif operation == 'put':
        spool.put(claims=claims, task_id=dispatch.task_id, execution_generation=dispatch.execution_generation,
                  attachment_id=manifest[0]['attachment_id'], data=data[0], authorize_write=guard)
    else:
        spool.discard_attempt(claims=claims, task_id=dispatch.task_id,
                              execution_generation=dispatch.execution_generation, authorize_write=guard)
    assert attempted == [Path(path).resolve() for path in paths]
    assert denied_before_commit == [False]
    # Once the spool commit is over, the profile deny can become effective even
    # while the shared leg still fails: no permanent or process-local mutex.
    with pytest.raises(sqlite3.OperationalError, match='injected shared-store failure'):
        revoke_grant_state(paths, claims=claims, expires_at=claims['status_expires_at'])
    assert hosted_rooms.room_grant_is_revoked(profile, claims=claims)


@pytest.mark.parametrize('operation', ['prepare', 'put', 'discard'])
def test_unavailable_profile_write_fence_refuses_without_spool_mutation(
        receiver, tmp_path, monkeypatch, operation):
    adapter, _, _, _ = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, claims, _, _, _, _, data = inputs
    request = _request(adapter, inputs)
    _, profile = grant_state_db_paths()
    saved = spool.materialize(dispatch)
    manifest = [{key: value for key, value in item.items() if key != 'path'} for item in saved]
    connect = hosted_rooms._connect

    def no_wait(path):
        conn = connect(path)
        conn.execute('PRAGMA busy_timeout=0')
        return conn

    monkeypatch.setattr(hosted_rooms, '_connect', no_wait)
    lock = connect(profile)
    try:
        lock.execute('BEGIN IMMEDIATE')
        guard = spool_api._write_guard(adapter, request, claims, 'attachment.stage')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            if operation == 'prepare':
                spool.prepare(dispatch, manifest, authorize_write=guard)
            elif operation == 'put':
                spool.put(claims=claims, task_id=dispatch.task_id,
                          execution_generation=dispatch.execution_generation,
                          attachment_id=manifest[0]['attachment_id'], data=data[0], authorize_write=guard)
            else:
                spool.discard_attempt(claims=claims, task_id=dispatch.task_id,
                                      execution_generation=dispatch.execution_generation, authorize_write=guard)
    finally:
        lock.rollback()
        lock.close()
    assert spool.materialize(dispatch) == saved
    assert [Path(item['path']).read_bytes() for item in saved] == data

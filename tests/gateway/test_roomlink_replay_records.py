"""Route-owned receipt/identity invariants, without Files or selected Runtime."""
import asyncio
import hashlib
import json
import sqlite3

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite  # noqa: F401
from tests.gateway.test_canonical_peer_text_admission import dispatch, run_request


@pytest.fixture(autouse=True)
def target_home(target):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(target.home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('conflict', [None, 'title', 'source', 'route'])
async def test_prospective_normalization_never_binds_and_new_still_admits(target, monkeypatch, conflict):
    from gateway.platforms import api_server_runs
    from gateway.session_api import hosted_session_id
    from hermes_state_runtime import RuntimeStoreError
    issued = await invite(target)
    value = dispatch(issued)
    sid = 'room_' + hashlib.sha256('\0'.join((value.home_install_id, value.room_id,
        value.member_id, value.target_profile)).encode()).hexdigest()[:32]
    assert hosted_session_id(value) == sid
    if conflict == 'title':
        target.db._execute_write(lambda conn: conn.execute(
            "INSERT INTO sessions(id,source,title,hidden,started_at) VALUES('conflict','bot_room',?,1,1)",
            ('Group: ' + value.room_id,)))
    elif conflict:
        target.db._execute_write(lambda conn: conn.execute(
            'INSERT INTO sessions(id,source,title,hidden,started_at,session_key) VALUES(?,?,?,1,1,?)',
            (sid, 'local' if conflict == 'source' else 'bot_room', 'Group: ' + value.room_id,
             'foreign' if conflict == 'route' else None)))
    changes = target.db._conn.total_changes
    body, response = await target.adapter._normalize_room_dispatch(run_request(issued['grant'], value),
                                                                  {'hosted_room_dispatch': value.as_mapping()})
    assert target.db._conn.total_changes == changes
    assert not target.authority.sessions
    if conflict:
        assert response.status == 409
        return
    assert response is None and body == dict(input=value.prompt, session_id=sid,
        hosted_room_dispatch=value.as_mapping(), _room_execution_policy=issued['catalog']['execution_policy'])
    launched = []
    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    accepted = await target.adapter._handle_runs(run_request(issued['grant'], value))
    assert accepted.status == 202, accepted.text
    await asyncio.sleep(0)
    assert len(launched) == 1 and launched[0][1].session_id == sid
    assert target.db.get_session(sid)['hidden'] == 1


def test_nullable_receipt_migration_preserves_exact_legacy_fingerprint(tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    path = str(tmp_path / 'legacy.db')
    store = RunIdempotencyStore(path)
    store.reserve('scope', 'key', 'original-fingerprint', 'original-run', {'status': 'queued'})
    store.close()
    with sqlite3.connect(path) as conn:
        conn.execute('ALTER TABLE run_idempotency DROP COLUMN room_policy_json')
    store = RunIdempotencyStore(path)
    try:
        old = store.replay_record('scope', 'key')
        assert old['fingerprint'] == 'original-fingerprint' and old['room_policy'] is None
        assert store.lookup('scope', 'key', 'original-fingerprint')[0] == 'reused'
        assert store.lookup('scope', 'key', 'changed-fingerprint')[0] == 'conflict'
        assert store.replay_record('foreign', 'key') is None
        policy = execution_policy_mapping(target_profile='default', config={})
        store.reserve('scope', 'new', 'new-fingerprint', 'new-run', {'status': 'queued'}, room_policy=policy)
        assert store.replay_record('scope', 'new')['room_policy'] == policy
        assert 'room_policy' not in store.status_for_run('scope', 'new-run')['status']
    finally:
        store.close()

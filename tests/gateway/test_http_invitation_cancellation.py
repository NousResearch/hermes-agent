"""Cancel the actual HTTP awaiter while its real issuance worker is paused."""
import asyncio
import json
import threading

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invitation, request  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.test_peer_files_provider_readiness import runtime_boundary  # noqa: F401


def invitation_rows(t):
    import sqlite3
    result = {}
    for name in ('shared-state.db', 'state.db'):
        path = t.home / name
        if not path.exists():
            result[name] = []
            continue
        with sqlite3.connect(path) as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_room_peer_reservations'").fetchone()
            result[name] = list(conn.execute('SELECT * FROM hosted_room_peer_reservations')) if exists else []
    result['receipts'] = list(t.db._conn.execute("SELECT key,value FROM state_meta WHERE key LIKE 'gateway.peer.invite.%'"))
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['prepare', 'fence', 'committed'])
async def test_cancelled_http_invitation_worker_has_exact_commit_boundary(files_target, runtime_boundary, monkeypatch, boundary):
    from gateway import session_selected_route as sr, session_peer_target, hosted_room_peer
    from gateway.platforms import api_server_room_grants as grants
    t = files_target
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    bindings, minted, results, workers = [], [], [], []
    init = sr.PreparedSelectedRoute.__init__
    def record(self, *args, **kwargs):
        init(self, *args, **kwargs)
        bindings.append(self)
    monkeypatch.setattr(sr.PreparedSelectedRoute, '__init__', record)
    sign = hosted_room_peer.issue_room_grant
    def signing(*args, **kwargs):
        minted.append(True)
        return sign(*args, **kwargs)
    monkeypatch.setattr(hosted_room_peer, 'issue_room_grant', signing)
    def pause():
        entered.set()
        assert release.wait(10), 'test did not release actual worker'
    if boundary == 'prepare':
        prepare = t.runner._prepare_session_agent_runtime
        def preparing(**kwargs):
            pause()
            return prepare(**kwargs)
        monkeypatch.setattr(t.runner, '_prepare_session_agent_runtime', preparing)
    elif boundary == 'fence':
        policy = session_peer_target.target_policy
        paused = False
        def at_fence(*args, **kwargs):
            nonlocal paused
            conn = kwargs.get('connection')
            if conn is not None and conn.in_transaction and not paused:
                paused = True
                pause()
            return policy(*args, **kwargs)
        monkeypatch.setattr(session_peer_target, 'target_policy', at_fence)
    issue = grants._issue_http_invitation
    def worker(*args, **kwargs):
        workers.append(threading.get_ident())
        try:
            value = issue(*args, **kwargs)
            results.append(value)
            if boundary == 'committed':
                pause()
            return value
        finally:
            finished.set()
    monkeypatch.setattr(grants, '_issue_http_invitation', worker)
    before = invitation_rows(t)
    body = invitation(); body.pop('request_id')
    task = asyncio.create_task(t.adapter._handle_room_member_invitation(request(body)))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        await asyncio.sleep(0)  # collect the shielded worker completion callback
    assert workers and all(worker != threading.get_ident() for worker in workers)
    assert bindings
    for b in bindings:
        assert b._material is None and b._inputs is None
        assert b._secret_snapshot is None and b._terminal_snapshot is None
    if boundary != 'committed':
        assert minted == [], 'pre-linearization cancellation minted a grant'
        assert results == []
        assert invitation_rows(t) == before
    else:
        assert len(minted) == len(results) == 1
        catalog, claims, token = results[0]
        assert catalog['attachments'] is True
        assert hosted_room_peer.decode_room_grant(t.adapter._room_grant_secret(), token, permission='attachment.stage') == claims
        rows = invitation_rows(t)
        assert len(rows['shared-state.db']) == len(rows['state.db']) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['auth', 'owner'])
async def test_http_invitation_rechecks_authority_after_preparation(files_target, runtime_boundary, monkeypatch, drift):
    from gateway import hosted_room_peer
    t = files_target
    prepare = t.runner._prepare_session_agent_runtime
    minted = []
    sign = hosted_room_peer.issue_room_grant
    def signing(*args, **kwargs):
        minted.append(True)
        return sign(*args, **kwargs)
    monkeypatch.setattr(hosted_room_peer, 'issue_room_grant', signing)
    def changed(**kwargs):
        value = prepare(**kwargs)
        if drift == 'auth':
            monkeypatch.setattr(t.adapter, '_expected_api_key', lambda: 'rotated-test-key')
        else:
            t.runner.session_authority = None
        return value
    monkeypatch.setattr(t.runner, '_prepare_session_agent_runtime', changed)
    before = invitation_rows(t)
    body = invitation(); body.pop('request_id')
    response = await t.adapter._handle_room_member_invitation(request(body))
    assert response.status in (400, 401, 403, 409), response.text
    assert minted == []
    assert invitation_rows(t) == before

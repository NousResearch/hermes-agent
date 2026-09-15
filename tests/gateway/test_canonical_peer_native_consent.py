"""Native consent survives blocking work only while its exact binding is live."""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite, invitation, request  # noqa: F401


async def _dispatch(target, method='groups.peer.invite', params=None):
    return await target.connection.dispatch(dict(id=1, method=method,
        params=invitation() if params is None else params))


def _receipts(target):
    return [json.loads(row[0]) for row in target.db._conn.execute(
        "SELECT value FROM state_meta WHERE key LIKE 'gateway.peer.invite.v1.%'")]


async def _drift(target, kind):
    if kind == 'close':
        await target.connection.close()
    elif kind == 'native_binding':
        target.authority._native_legacy_transports.pop(target.connection.actor.transport_id)
    elif kind == 'transport':
        target.authority.events[target.connection.actor.transport_id] = object()
    elif kind == 'actor':
        target.connection.actor = replace(target.connection.actor, subject='another-native-user')
    elif kind == 'registry':
        registry = target.runner.session_authorities
        from gateway.session_authority import SessionAuthority
        registry._by_key[registry.launch_key] = SessionAuthority(target.runner,
            profile_id=str(target.home), instance_id='replacement', db=target.db,
            epoch=target.authority.epoch)
    elif kind in {'epoch', 'stored_epoch'}:
        from hermes_state_runtime import begin_runtime_epoch
        epoch = begin_runtime_epoch(target.db, instance_id='successor')
        # Updating the in-memory field as well must not renew frozen consent.
        if kind == 'epoch':
            target.authority.epoch = epoch


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['prepare', 'complete'])
@pytest.mark.parametrize('drift', ['close', 'native_binding', 'transport', 'actor', 'registry', 'epoch', 'stored_epoch'])
async def test_invitation_rechecks_frozen_native_consent(target, monkeypatch, boundary, drift):
    entered, resume = threading.Event(), threading.Event()
    original = target.db._execute_write

    def write(fn, *args, **kwargs):
        if fn.__name__ == boundary:
            entered.set()
            assert resume.wait(10), 'test failed to release writer'
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(target.db, '_execute_write', write)
    pending = asyncio.create_task(_dispatch(target))
    try:
        assert await asyncio.to_thread(entered.wait, 10), 'writer not reached'
        await _drift(target, drift)
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, 10)
    assert 'error' in result, result
    receipts = _receipts(target)
    if boundary == 'prepare':
        assert not receipts
    else:
        assert len(receipts) == 1 and receipts[0]['status'] == 'pending'


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('drift', ['close', 'registry', 'epoch', 'stored_epoch'])
async def test_revoke_rechecks_after_shared_lock(target, monkeypatch, exact, drift):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    first = await invite(target)
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='status')
    entered, resume = threading.Event(), threading.Event()
    original = hosted_rooms._transaction

    @contextmanager
    def transaction(path, **kwargs):
        with original(path, **kwargs) as conn:
            if Path(path) == target.home / 'shared-state.db' and kwargs.get('immediate'):
                entered.set()
                assert resume.wait(10), 'test failed to release shared fence'
            yield conn

    monkeypatch.setattr(hosted_rooms, '_transaction', transaction)
    method = 'groups.peer.revoke_exact' if exact else 'groups.peer.revoke'
    pending = asyncio.create_task(_dispatch(target, method, {'grant': first['grant']}))
    try:
        assert await asyncio.to_thread(entered.wait, 10), 'shared fence not reached'
        await _drift(target, drift)
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, 10)
    assert 'error' in result, result
    for path in (target.home / 'shared-state.db', target.home / 'state.db'):
        assert not hosted_rooms.room_grant_is_revoked(path, claims=claims)


_INVALID = [
    pytest.param('room_id', 'r' * 129, id='room-limit'),
    pytest.param('member_id', 'm' * 129, id='member-limit'),
    pytest.param('authority_gateway_id', 'a' * 129, id='gateway-limit'),
    pytest.param('authority_epoch', 2**63, id='epoch-overflow'),
    *[pytest.param(field, value, id=f'{field}-{label}')
      for field in ('ttl_seconds', 'status_ttl_seconds')
      for label, value in [('huge', 10**1000), ('nan', float('nan')),
                           ('inf', float('inf')), ('bool', True)]],
]


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', _INVALID)
async def test_preflight_never_signs_reserves_or_consumes_request_id(target, monkeypatch, field, value):
    from gateway import hosted_room_peer, hosted_rooms
    calls = []
    original = hosted_room_peer.issue_room_grant

    def sign(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(hosted_room_peer, 'issue_room_grant', sign)
    params = invitation() | {field: value}
    result = await _dispatch(target, params=params)
    assert result.get('error', {}).get('message') == 'invalid_params', result
    assert not _receipts(target)
    assert not calls
    http = dict(params)
    http.pop('request_id')
    response = await target.adapter._handle_room_member_invitation(request(http))
    assert response.status == 400, response.text
    assert json.loads(response.text)['error']['code'] == 'invalid_room_invitation'
    assert not calls
    for path in (target.home / 'shared-state.db', target.home / 'state.db'):
        assert not hosted_rooms.peer_room_is_reserved(path, room_id='room-one', target_profile='default')
    # Ordinary invalid input has no durable identity. Same ID, corrected fields works.
    first = await invite(target)
    assert await invite(target) == first


@pytest.mark.asyncio
async def test_preflight_preserves_protocol_sized_fields_and_sqlite_max(target):
    first = await invite(target, invitation() | {'room_id': 'r' * 128, 'member_id': 'm' * 128,
        'authority_gateway_id': 'a' * 128, 'home_install_id': 'h' * 256,
        'grant_id': 'g' * 256, 'authority_epoch': 2**63 - 1})
    assert first['grant']


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('failed_store', [None, 'shared-state.db', 'state.db'])
async def test_native_revoke_fences_both_stores_and_preserves_partial_deny(target, monkeypatch, exact, failed_store):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    first = await invite(target)
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='status')
    paths = (target.home / 'shared-state.db', target.home / 'state.db')
    table = 'hosted_room_revoked_grant_tokens' if exact else 'hosted_room_revoked_grants'
    checks = []

    def check_locks():
        for path in paths:
            with sqlite3.connect(path, timeout=0) as other:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    other.execute('BEGIN IMMEDIATE')
        checks.append(True)
        return 1

    original = hosted_rooms._transaction

    @contextmanager
    def transaction(path, **kwargs):
        with original(path, **kwargs) as conn:
            conn.create_function('check_native_fences', 0, check_locks)
            yield conn

    monkeypatch.setattr(hosted_rooms, '_transaction', transaction)
    target.db._conn.create_function('check_native_fences', 0, check_locks)
    for path in paths:
        with sqlite3.connect(path) as conn:
            conn.execute(f'CREATE TRIGGER check_fences BEFORE INSERT ON {table} '
                         'BEGIN SELECT check_native_fences(); END')
    if failed_store:
        with sqlite3.connect(target.home / failed_store) as conn:
            conn.execute(f"CREATE TRIGGER deny_revoke BEFORE INSERT ON {table} "
                         "BEGIN SELECT RAISE(ABORT, 'fixture store failure'); END")
    # Revocation needs owner consent, not execution-ready adapters or policy.
    target.runner.adapters.clear()
    (target.home / 'config.yaml').write_text('approvals:\n  mode: off\n')
    method = 'groups.peer.revoke_exact' if exact else 'groups.peer.revoke'
    result = await _dispatch(target, method, {'grant': first['grant']})
    assert ('error' in result) == bool(failed_store), result
    assert checks, 'real revocation INSERT never checked its fences'
    if failed_store is None:
        assert len(checks) == 2
    for path in paths:
        assert hosted_rooms.room_grant_is_revoked(path, claims=claims) == (path.name != failed_store)


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['prepare', 'complete'])
async def test_cancelled_awaiter_cannot_complete_after_real_connection_close(target, monkeypatch, boundary):
    from gateway import session_group_peers
    from gateway.session_controls import AuthorityConnection
    from unittest.mock import AsyncMock
    entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
    outcomes = []
    original_write = target.db._execute_write
    original_dispatch = session_group_peers.dispatch_group_peer

    def write(fn, *args, **kwargs):
        if fn.__name__ == boundary:
            entered.set()
            assert resume.wait(10)
        return original_write(fn, *args, **kwargs)

    def dispatch(*args, **kwargs):
        try:
            result = original_dispatch(*args, **kwargs)
            outcomes.append(result)
            return result
        except Exception as exc:
            outcomes.append(exc)
            raise
        finally:
            finished.set()

    monkeypatch.setattr(target.db, '_execute_write', write)
    monkeypatch.setattr(session_group_peers, 'dispatch_group_peer', dispatch)
    pending = asyncio.create_task(_dispatch(target))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        # This is the actual WS finally-path action, not a synthetic flag.
        await target.connection.close()
    finally:
        resume.set()
    assert await asyncio.to_thread(finished.wait, 10)
    assert len(outcomes) == 1 and isinstance(outcomes[0], Exception)
    before = _receipts(target)
    assert not before if boundary == 'prepare' else before[0]['status'] == 'pending'
    target.connection = AuthorityConnection(target.authority, AsyncMock(), target.identity, operator=True)
    if boundary == 'complete':
        from gateway import hosted_room_peer
        def no_mint(*args, **kwargs):
            pytest.fail('a genuinely pending receipt was reminted')
        monkeypatch.setattr(hosted_room_peer, 'issue_room_grant', no_mint)
        replay = await _dispatch(target)
        assert replay['error']['message'] == 'room_invitation_pending'
        assert _receipts(target) == before
    else:
        assert (await invite(target))['grant']


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
async def test_close_between_native_revoke_stores_preserves_only_prior_deny(target, monkeypatch, exact):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    first = await invite(target)
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='status')
    entered, resume = threading.Event(), threading.Event()
    name = 'revoke_room_grant_id' if exact else 'revoke_room_grant_scope'
    original = getattr(hosted_rooms, name)

    def revoke(path, **kwargs):
        if Path(path) == target.home / 'state.db':
            entered.set()
            assert resume.wait(10)
        return original(path, **kwargs)

    monkeypatch.setattr(hosted_rooms, name, revoke)
    method = 'groups.peer.revoke_exact' if exact else 'groups.peer.revoke'
    pending = asyncio.create_task(_dispatch(target, method, {'grant': first['grant']}))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        await target.connection.close()
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, 10)
    assert result['error']['message'] == 'permission_denied'
    assert hosted_rooms.room_grant_is_revoked(target.home / 'shared-state.db', claims=claims)
    assert not hosted_rooms.room_grant_is_revoked(target.home / 'state.db', claims=claims)

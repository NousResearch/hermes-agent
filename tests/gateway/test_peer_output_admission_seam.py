"""Lower-owner admission contract; the synthetic consumer is not Output readiness."""
import sqlite3
from types import MethodType, SimpleNamespace

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite  # noqa: F401
from tests.gateway.test_canonical_peer_text_admission import dispatch
from hermes_state_runtime import RuntimeStoreError


async def admission(target, issued):
    value = dispatch(issued)
    session_id = await target.adapter._ensure_hosted_member_session(value)
    return dict(user_message=value.prompt, conversation_history=[], session_id=session_id,
                request_id='seam-request', room_dispatch=value.as_mapping(),
                room_execution_policy=issued['catalog']['execution_policy'],
                _room_output_authorizer=getattr(target.adapter, '_room_output_admission', None),
                _room_grant_token=issued['grant'])


@pytest.mark.asyncio
@pytest.mark.parametrize('provider_state', ['absent', 'unbound', 'foreign', 'unavailable', 'no_receipt',
                                          'changed', 'read_only', 'ack_only'])
async def test_signed_output_cannot_fall_through_to_text(target, monkeypatch, provider_state):
    from gateway import session_api_turn
    rights = {'read_only': ('artifact.read',), 'ack_only': ('artifact.ack',)}.get(
        provider_state, ('artifact.read', 'artifact.ack'))
    target.adapter._room_output_invitation_permissions = MethodType(
        lambda self, **kwargs: rights, target.adapter)
    issued = await invite(target)
    calls = []

    def consumer(self, authority, shared, conn, token, value, policy, evidence):
        calls.append(conn.in_transaction)
        return {'unavailable': False, 'no_receipt': None}.get(provider_state, True)

    if provider_state == 'unbound':
        target.adapter._room_output_admission = consumer
    elif provider_state == 'foreign':
        target.adapter._room_output_admission = MethodType(consumer, SimpleNamespace())
    elif provider_state != 'absent':
        target.adapter._room_output_admission = MethodType(consumer, target.adapter)
    if provider_state == 'changed':
        original = session_api_turn.admit_session_input

        def interleave(*args, **kwargs):
            target.adapter._room_output_admission = MethodType(lambda *args: True, target.adapter)
            return original(*args, **kwargs)
        monkeypatch.setattr(session_api_turn, 'admit_session_input', interleave)
    with pytest.raises(RuntimeStoreError, match='room_output_unavailable'):
        session_api_turn.admit_api_turn(target.adapter, **await admission(target, issued))
    assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert calls == ([True] if provider_state in {'unavailable', 'no_receipt'} else [])


@pytest.mark.asyncio
@pytest.mark.parametrize('output', [False, True])
async def test_new_admission_holds_both_stores_and_replay_does_not_reauthorize(target, output):
    from gateway.session_api_turn import admit_api_turn
    seen = []
    if output:
        target.adapter._room_output_invitation_permissions = MethodType(
            lambda self, **kwargs: ('artifact.read', 'artifact.ack'), target.adapter)
    issued = await invite(target)
    kwargs = await admission(target, issued)

    def consumer(self, authority, shared, conn, token, value, policy, evidence):
        assert self is target.adapter and authority is target.authority
        assert conn is target.db._conn and conn.in_transaction and shared.in_transaction
        assert token == issued['grant'] and value.as_mapping() == kwargs['room_dispatch']
        assert policy == kwargs['room_execution_policy'] and evidence is None
        for path in (target.home / 'shared-state.db', target.home / 'state.db'):
            with sqlite3.connect(path, timeout=0) as other:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    other.execute('BEGIN IMMEDIATE')
        seen.append(True)
        return True

    if output:
        target.adapter._room_output_admission = MethodType(consumer, target.adapter)
        kwargs['_room_output_authorizer'] = target.adapter._room_output_admission
    first = admit_api_turn(target.adapter, **kwargs)
    assert seen == ([True] if output else [])
    target.adapter._room_output_admission = None
    replay = admit_api_turn(target.adapter, **kwargs)
    assert replay[2] == first[2]
    assert seen == ([True] if output else [])
    assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1

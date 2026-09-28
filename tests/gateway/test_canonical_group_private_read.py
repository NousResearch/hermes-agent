"""Canonical private Group read: stable references and exact consented handoff."""
import logging
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway import hosted_rooms as rooms
from gateway.config import Platform
from tests.gateway.test_canonical_group_messaging_list import CapturingReceiver, consumer
from tests.gateway.test_messaging_inventory_binding import bound, enrolled, snapshot
from tests.gateway.test_messaging_room_read_binding import (
    room_grant_params, room_revoke_params, room_rpc,
)


@pytest_asyncio.fixture
async def readable(consumer, monkeypatch):
    c = consumer
    # The original list fixture traps room reads; permit the real SQLite reads.
    monkeypatch.setattr(rooms, 'room_state', _ROOM_STATE)
    monkeypatch.setattr(rooms, 'read_events', _READ_EVENTS)
    c.service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    from gateway.hosted_room_driver import list_tasks
    assert list_tasks(c.db.db_path, room_id='alice-room') == []
    inventory = await enrolled(c)
    grant = (await room_rpc(c.alice, params=room_grant_params(inventory)))['result']
    rooms.append_event(
        c.db.db_path, room_id='alice-room', event_id='real-private-event',
        kind='message.user', actor={'kind': 'user', 'id': 'person-1'},
        payload={'text': 'VISIBLE PRIVATE MESSAGE @everyone MEDIA:/private/photo'},
        authority_gateway_id='inert-gateway', authority_epoch=1, now=22,
    )
    c.inventory_grant, c.room_grant = inventory, grant
    try:
        yield c
    finally:
        c.service.runtime._thread = None


_ROOM_STATE = rooms.room_state
_READ_EVENTS = rooms.read_events


@pytest.mark.asyncio
async def test_list_shows_provider_reference_not_position_and_detail_uses_exact_room(readable, monkeypatch):
    from gateway import group_chat_private_read as read
    from gateway import session_group_controls as controls
    c = readable
    revoked = (await room_rpc(c.alice, 'revoke', room_revoke_params(
        c.inventory_grant, c.room_grant)))['result']
    grant = (await room_rpc(c.alice, params=room_grant_params(
        c.inventory_grant, request_id='new-reference',
        expected_generation=revoked['generation'])))['result']
    assert grant['room_ref'] != c.room_grant['room_ref']
    c.room_grant = grant
    calls = []
    dispatch = controls.dispatch_group_control
    async def observe(context, method, params):
        calls.append((context, method, dict(params)))
        return await dispatch(context, method, params)
    monkeypatch.setattr(read, 'dispatch_group_control', observe)
    before = snapshot(c.db)
    assert await read.handle_private_group_read(c.runner, c.event('/group list')) == ''
    listed = c.adapter.sent.pop()[1]
    assert f"{grant['room_ref']}. Alice inventory" in listed
    assert f"/group {grant['room_ref']}" in listed
    assert await read.handle_private_group_read(c.runner, c.event(f"/group {grant['room_ref']}")) == ''
    target, body, reply_to, metadata = c.adapter.sent.pop()
    assert target == 'private-chat' and reply_to is None
    assert metadata == {'_interim_send': True}
    assert f"Group {grant['room_ref']} — Alice inventory" in body
    assert 'VISIBLE PRIVATE MESSAGE' in body and 'Status: idle' in body
    assert 'Refresh: /group ' + str(grant['room_ref']) in body
    assert 'MEDIA:' not in body and '/private/photo' not in body
    assert all(secret not in body for secret in ('bob-room', 'alice-room', 'inert-gateway'))
    assert [method for _, method, _ in calls] == ['groups.list', 'groups.state', 'groups.log', 'groups.log']
    assert all(params.get('room_id') == 'alice-room' for _, _, params in calls[1:])
    assert all(context is calls[1][0] for context, _, _ in calls[1:])
    assert snapshot(c.db) == before and not c.adapter.generic_sent


@pytest.mark.asyncio
async def test_wrong_or_withdrawn_room_consent_never_discloses_private_room(readable):
    from gateway.group_chat_private_read import handle_private_group_read
    c = readable
    before = snapshot(c.db)
    assert await handle_private_group_read(c.runner, c.event('/group 1')) == ''
    c.adapter.sent.clear()
    assert await handle_private_group_read(c.runner, c.event('/group 999'))
    assert not c.adapter.sent
    revoked = (await room_rpc(c.alice, 'revoke', room_revoke_params(
        c.inventory_grant, c.room_grant)))['result']
    assert revoked['active'] is False
    result = await handle_private_group_read(c.runner, c.event('/group 1'))
    assert result and not c.adapter.sent and not c.adapter.generic_sent
    assert snapshot(c.db) != before
    assert await handle_private_group_read(c.runner, c.event('/group list')) == ''
    assert 'Alice inventory' in c.adapter.sent[-1][1]
    assert '1. Alice inventory' not in c.adapter.sent[-1][1]


@pytest.mark.asyncio
async def test_receiver_replacement_during_read_suppresses_handoff(readable, monkeypatch, caplog):
    from gateway import group_chat_private_read as read
    c = readable
    replacement = CapturingReceiver(c.runner, c.adapter.config)
    dispatch = read.dispatch_group_control
    async def replace(context, method, params):
        value = await dispatch(context, method, params)
        if method == 'groups.state':
            c.runner.adapters[Platform.SIGNAL] = replacement
        return value
    monkeypatch.setattr(read, 'dispatch_group_control', replace)
    with caplog.at_level(logging.WARNING):
        result = await read.handle_private_group_read(c.runner, c.event('/group 1'))
    assert result and not c.adapter.sent and not replacement.sent
    assert not c.adapter.generic_sent and not replacement.generic_sent
    assert 'VISIBLE PRIVATE MESSAGE' not in caplog.text


@pytest.mark.asyncio
async def test_room_revoke_between_log_and_handoff_fences_private_body(readable, monkeypatch):
    from gateway import group_chat_private_read as read
    c = readable
    dispatch = read.dispatch_group_control
    async def revoke(context, method, params):
        result = await dispatch(context, method, params)
        if method == 'groups.log':
            await room_rpc(c.alice, 'revoke', room_revoke_params(
                c.inventory_grant, c.room_grant))
        return result
    monkeypatch.setattr(read, 'dispatch_group_control', revoke)
    result = await read.handle_private_group_read(c.runner, c.event('/group 1'))
    assert result and not c.adapter.sent and not c.adapter.generic_sent


@pytest.mark.asyncio
@pytest.mark.parametrize('args', ['/group 0', '/group -1', '/group 99999999999999999999', '/group 1.0', '/group １'])
async def test_bad_reference_never_reads_or_sends(readable, args, monkeypatch):
    from gateway import group_chat_private_read as read
    c = readable
    monkeypatch.setattr(read, 'dispatch_group_control', c.forbidden)
    assert await read.handle_private_group_read(c.runner, c.event(args))
    assert not c.adapter.sent


@pytest.mark.asyncio
async def test_source_retarget_after_read_never_sends_to_old_or_new_chat(readable, monkeypatch):
    from gateway import group_chat_private_read as read
    c = readable
    event = c.event('/group 1')
    original = read.read_group_detail
    async def retarget(context, prefix):
        body = await original(context, prefix)
        event.source.chat_id = 'another-private-chat'
        return body
    monkeypatch.setattr(read, 'read_group_detail', retarget)
    assert await read.handle_private_group_read(c.runner, event)
    assert not c.adapter.sent and not c.adapter.generic_sent


@pytest.mark.asyncio
async def test_transport_exception_never_logs_private_content_or_retries(readable, monkeypatch, caplog):
    from gateway import group_chat_private_read as read
    c = readable
    attempts = []
    async def failed(chat_id, content, **kwargs):
        attempts.append((chat_id, content, kwargs))
        raise RuntimeError('VISIBLE PRIVATE MESSAGE should not be logged')
    monkeypatch.setattr(c.adapter, 'send', failed)
    with caplog.at_level(logging.WARNING):
        assert await read.handle_private_group_read(c.runner, c.event('/group 1')) == ''
    assert len(attempts) == 1 and attempts[0][0] == 'private-chat'
    assert attempts[0][2]['metadata'] == {'_interim_send': True}
    assert 'VISIBLE PRIVATE MESSAGE' not in caplog.text
    assert not c.adapter.generic_sent

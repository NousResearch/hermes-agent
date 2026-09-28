"""Registered private Stop and exact approval via canonical Route; no legacy fallback."""
import asyncio
import time
import re
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms as rooms, hosted_room_driver as driver
from gateway.session_group_messaging_control import attest_room_control
from gateway.session_group_messaging_control import pending_room_approvals
from gateway.session_group_messaging_control import (
    prepare_native_control_binding, commit_native_control_binding,
)
from gateway.session_group_messaging_send import _stable_message_identity
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_canonical_group_messaging_send import authorized_send, send_event
from tests.gateway.test_canonical_group_messaging_list import consumer, route
from tests.gateway.test_messaging_inventory_binding import bound
from tests.gateway.test_messaging_room_control_binding import control_rpc, owner, params


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['idle', 'runner_busy', 'adapter_busy'])
async def test_registered_stop_needs_its_own_grant_and_targets_exact_room(authorized_send, monkeypatch, lane):
    c = authorized_send
    monkeypatch.delattr(c.service, 'stop_room')
    monkeypatch.setattr(c.runner, '_handle_legacy_rooms_command', c.forbidden)
    event = send_event(c, 'unused', message_id='stop-message')
    event.text = f'/group {c.room_grant["room_ref"]} stop'
    assert (await route(c, event, lane)) is not False
    assert not [e for e in rooms.read_events(c.db.db_path, room_id='send-room')['events']
                if e['kind'] == 'room.stop_requested']
    c.adapter.sent.clear()
    c.adapter.generic_sent.clear()
    grant = await control_rpc(owner(c), 'stop', 'grant',
        params(c.room_grant, 'stop', room_id='send-room'))
    assert 'result' in grant, grant
    await route(c, event, lane)
    fences = [e for e in rooms.read_events(c.db.db_path, room_id='send-room')['events']
              if e['kind'] == 'room.stop_requested']
    assert len(fences) == 1
    assert len(c.adapter.sent) == 1
    assert 'does not prove interruption' in c.adapter.sent[0][1]
    assert not c.adapter.generic_sent
    await route(c, event, lane)
    assert len([e for e in rooms.read_events(c.db.db_path, room_id='send-room')['events']
                if e['kind'] == 'room.stop_requested']) == 1
    revoked = await control_rpc(owner(c), 'stop', 'revoke',
        params(c.room_grant, 'stop', room_id='send-room', request_id='stop-native-revoke',
               expected_generation=grant['result']['generation'],
               binding_id=grant['result']['binding_id']))
    assert revoked['result']['active'] is False
    native = await c.alice.dispatch({'id': 72, 'method': 'groups.stop',
                                     'params': {'room_id': 'send-room',
                                                'cancel_id': 'native-stop-independent'}})
    assert 'result' in native, native
    assert len([e for e in rooms.read_events(c.db.db_path, room_id='send-room')['events']
                if e['kind'] == 'room.stop_requested']) == 2


@pytest.mark.asyncio
async def test_route_stop_writer_rejects_old_grant_across_revoke_regrant(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.delattr(c.service, 'stop_room')
    native = owner(c)
    first = (await control_rpc(native, 'stop', 'grant',
        params(c.room_grant, 'stop', room_id='send-room')))['result']
    event = send_event(c, 'unused', message_id='stop-race')
    event.text = f'/group {c.room_grant["room_ref"]} stop'
    from tests.gateway.test_messaging_inventory_binding import recipient
    digest = _stable_message_identity(event, recipient())[1]
    cancel_id = 'messaging-stop:' + digest
    context = attest_room_control(c.runner, event, c.room_grant['room_ref'], 'stop',
                                   {'room_id': 'send-room', 'cancel_id': cancel_id})
    delegated = context.delegated()
    revoked = (await control_rpc(native, 'stop', 'revoke',
        params(c.room_grant, 'stop', room_id='send-room', request_id='race-revoke',
               expected_generation=first['generation'], binding_id=first['binding_id'])))['result']
    assert not revoked['active']
    with pytest.raises(RuntimeStoreError):
        c.service.stop_room('send-room', cancel_id=cancel_id,
                            require_acknowledged=False, delegated_control=delegated)
    replacement = (await control_rpc(native, 'stop', 'grant',
        params(c.room_grant, 'stop', room_id='send-room', request_id='race-regrant',
               expected_generation=revoked['generation'])))['result']
    assert replacement['binding_id'] != first['binding_id']
    with pytest.raises(RuntimeStoreError):
        c.service.stop_room('send-room', cancel_id=cancel_id,
                            require_acknowledged=False, delegated_control=delegated)
    assert not [e for e in rooms.read_events(c.db.db_path, room_id='send-room')['events']
                if e['kind'] == 'room.stop_requested']


@pytest.mark.asyncio
async def test_read_or_send_does_not_grant_approval(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.setattr(c.runner, '_handle_legacy_rooms_command', c.forbidden)
    event = c.event(f'/group {c.room_grant["room_ref"]} approve pa-{"0" * 64} once')
    await route(c, event, 'idle')
    assert not c.adapter.sent
    assert len(c.adapter.generic_sent) == 1
    assert 'unavailable' in c.adapter.generic_sent[0]['content'].lower()


def pending_task(c):
    c.service.send(room_id='send-room', event_id='approval-intent',
                   payload={'text': '@reviewer work', 'thread_id': 'approval-thread'})
    task = driver.list_tasks(c.db.db_path, room_id='send-room', status='queued')[0]
    binding = next(b for b in c.service.bindings() if b.room_id == 'send-room')
    lease = driver.acquire_lease(c.db.db_path, room_id='send-room',
        gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
        process_generation=c.service.runtime.process_generation, ttl_seconds=30,
        clock=time.time)
    driver.start_task(c.db.db_path, task['identity'], lease, expected_cancel_generation=0,
                      clock=time.time)
    task = driver.get_task(c.db.db_path, task['identity'])
    c.service._set_pending_action('send-room', 'reviewer', {
        'kind': 'approval', 'task_id': task['identity'].task_id,
        'execution_generation': task['execution_generation'],
        'session_id': 'exact-session', 'request_id': 'approval-request',
        'approval': {'choices': ['once', 'deny']}})
    return {'room_id': 'send-room', 'member_id': 'reviewer',
            'task_id': task['identity'].task_id,
            'execution_generation': task['execution_generation'],
            'request_id': 'approval-request'}

def displayed_selector(c, detail):
    match = re.search(r'approve (pa-[0-9a-f]{64}) once\|deny', detail)
    assert match, detail
    return match.group(1)


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['idle', 'runner_busy', 'adapter_busy'])
@pytest.mark.parametrize('choice', ['once', 'deny'])
@pytest.mark.parametrize('uncertain', [False, True])
async def test_registered_approval_exact_task_once_or_deny(authorized_send, monkeypatch, lane, choice, uncertain):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    monkeypatch.setattr(c.runner, '_handle_legacy_rooms_command', c.forbidden)
    pending_task(c)
    calls = []
    def approve(**kw):
        calls.append(kw)
        if uncertain:
            raise OSError('injected lost response')
        return {'resolved': 1}
    c.service.rpc = SimpleNamespace(approve=approve)
    grant = await control_rpc(owner(c), 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room'))
    assert 'result' in grant, grant
    c.adapter.sent.clear()
    await route(c, c.event(f'/group {c.room_grant["room_ref"]}'), lane)
    detail = c.adapter.sent.pop()[1]
    selector = displayed_selector(c, detail)
    assert f'/group {c.room_grant["room_ref"]} approve {selector} once|deny' in detail
    await route(c, c.event(f'/group {c.room_grant["room_ref"]} approve 1 {choice}'), lane)
    assert not calls  # Old mutable ordinal is never a mutation selector.
    event = c.event(f'/group {c.room_grant["room_ref"]} approve {selector} {choice}')
    await route(c, event, lane)
    assert calls == [{'session_id': 'exact-session', 'request_id': 'approval-request',
                      'choice': choice}], (c.adapter.sent, c.adapter.generic_sent)
    assert len(c.adapter.sent) == 1
    assert ('uncertain' if uncertain else 'processed') in c.adapter.sent[0][1]
    c.adapter.sent.clear()
    await route(c, event, lane)
    assert len(calls) == 1  # Settled or uncertain Route replay never re-sends.


@pytest.mark.asyncio
async def test_route_approval_writer_rejects_old_grant_across_revoke_regrant(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    args = pending_task(c) | {'choice': 'once'}
    calls = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: calls.append(kw) or {'resolved': 1})
    native = owner(c)
    first = (await control_rpc(native, 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room')))['result']
    event = c.event(f'/group {c.room_grant["room_ref"]} approve 1 once')
    context = attest_room_control(c.runner, event, c.room_grant['room_ref'],
                                   'approval', args)
    delegated = context.delegated()
    revoked = (await control_rpc(native, 'approval', 'revoke',
        params(c.room_grant, 'approval', room_id='send-room',
               request_id='approval-race-revoke',
               expected_generation=first['generation'], binding_id=first['binding_id'])))['result']
    assert not revoked['active']
    with pytest.raises(RuntimeStoreError):
        c.service.approve_room_task(**args, delegated_control=delegated)
    replacement = (await control_rpc(native, 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room',
               request_id='approval-race-regrant',
               expected_generation=revoked['generation'])))['result']
    assert replacement['binding_id'] != first['binding_id']
    with pytest.raises(RuntimeStoreError):
        c.service.approve_room_task(**args, delegated_control=delegated)
    assert calls == []


@pytest.mark.asyncio
async def test_native_approval_remains_independent_of_private_control_consent(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    args = pending_task(c)
    calls = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: calls.append(kw) or {'resolved': 1})
    native = await owner(c).dispatch({'id': 73, 'method': 'groups.approve',
                                      'params': args | {'choice': 'deny'}})
    assert native['result']['approved'] is True, native
    assert calls == [{'session_id': 'exact-session', 'request_id': 'approval-request',
                      'choice': 'deny'}]


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['replace', 'remove', 'reorder'])
async def test_displayed_request_never_selects_successor(authorized_send, monkeypatch, change):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    pending_task(c)
    grant = await control_rpc(owner(c), 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room'))
    assert 'result' in grant
    ref = c.room_grant['room_ref']
    await route(c, c.event(f'/group {ref}'))
    selector = displayed_selector(c, c.adapter.sent.pop()[1])
    old = c.service._pending_actions[('send-room', 'reviewer')]
    if change == 'replace':
        c.service._set_pending_action('send-room', 'reviewer',
                                      {**old, 'request_id': 'replacement-never-displayed'})
    elif change == 'remove':
        c.service._set_pending_action('send-room', 'reviewer', None)
    else:
        c.service._pending_actions = {
            ('send-room', 'decoy'): {**old, 'request_id': 'decoy-request'},
            ('send-room', 'reviewer'): old,
        }
    calls = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: calls.append(kw) or {'resolved': 1})
    command = c.event(f'/group {ref} approve {selector} once')
    await route(c, command)
    if change == 'reorder':
        assert calls == [{'session_id': 'exact-session', 'request_id': 'approval-request',
                          'choice': 'once'}]
    else:
        assert calls == []
        assert not c.adapter.sent
        assert c.adapter.generic_sent
        assert 'unavailable' in c.adapter.generic_sent[-1]['content'].lower()


@pytest.mark.asyncio
async def test_same_message_edit_cannot_admit_different_request_after_context_recreation(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    pending_task(c)
    ref = c.room_grant['room_ref']
    grant = await control_rpc(owner(c), 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room'))
    assert 'result' in grant
    selector = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
    old = dict(c.service._pending_actions[('send-room', 'reviewer')])
    calls = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: calls.append(kw) or {'resolved': 1})
    first = c.event(f'/group {ref} approve {selector} once')
    assert first.message_id == first.source.message_id
    await route(c, first)
    assert len(calls) == 1
    # A new event/context uses the identical transport message identity.
    edited = c.event(f'/group {ref} approve {selector} deny')
    assert edited.message_id == edited.source.message_id == first.message_id
    await route(c, edited)
    assert len(calls) == 1
    c.service._set_pending_action('send-room', 'reviewer',
        {**old, 'request_id': 'new-request-for-same-message'})
    new_selector = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
    new_params = {'room_id': 'send-room', 'member_id': 'reviewer',
                  'task_id': old['task_id'], 'execution_generation': old['execution_generation'],
                  'request_id': 'new-request-for-same-message', 'choice': 'once'}
    # An independent connection observes the durable writer record, not a process cache.
    with sqlite3.connect(c.db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM state_meta WHERE key LIKE ?",
                            ('gateway.messaging.control.v1.source.%',)).fetchone()[0] == 1
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        attest_room_control(c.runner, c.event(f'/group {ref} approve {new_selector} once'),
                            ref, 'approval', new_params)
    await route(c, c.event(f'/group {ref} approve {new_selector} once'))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_displayed_selector_does_not_survive_control_regrant(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    pending_task(c)
    ref = c.room_grant['room_ref']
    native = owner(c)
    first = (await control_rpc(native, 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room')))['result']
    old_selector = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
    revoked = (await control_rpc(native, 'approval', 'revoke',
        params(c.room_grant, 'approval', room_id='send-room',
               request_id='selector-revoke', expected_generation=first['generation'],
               binding_id=first['binding_id'])))['result']
    replacement = (await control_rpc(native, 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room',
               request_id='selector-regrant', expected_generation=revoked['generation'])))['result']
    assert replacement['binding_id'] != first['binding_id']
    fresh_selector = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
    assert fresh_selector != old_selector
    calls = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: calls.append(kw) or {'resolved': 1})
    await route(c, c.event(f'/group {ref} approve {old_selector} once'))
    assert calls == []


@pytest.mark.asyncio
async def test_source_admission_rolls_back_with_route_writer(authorized_send, monkeypatch):
    c = authorized_send
    monkeypatch.delattr(c.service, 'approve_room_task')
    exact = pending_task(c) | {'choice': 'deny'}
    ref = c.room_grant['room_ref']
    assert 'result' in await control_rpc(owner(c), 'approval', 'grant',
        params(c.room_grant, 'approval', room_id='send-room'))
    selector = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
    event = c.event(f'/group {ref} approve {selector} deny')
    delegated = attest_room_control(c.runner, event, ref, 'approval', exact).delegated()
    def abort(conn):
        delegated.authorize_new(conn)
        delegated.authorize_commit(conn)
        raise RuntimeStoreError('abort-test-writer')
    with pytest.raises(RuntimeStoreError, match='abort-test-writer'):
        c.db._execute_write(abort)
    with sqlite3.connect(c.db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM state_meta WHERE key LIKE ?",
                            ('gateway.messaging.control.v1.source.%',)).fetchone()[0] == 0
    calls = []
    c.service.rpc = SimpleNamespace(approve=lambda **kw: calls.append(kw) or {'resolved': 1})
    await route(c, event)
    assert calls == [{'session_id': 'exact-session', 'request_id': 'approval-request',
                      'choice': 'deny'}]


def native_control_change(native, scope, verb, payload):
    prepared = prepare_native_control_binding(
        native, f'groups.messaging.room.{scope}.{verb}', payload)
    return commit_native_control_binding(prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['idle', 'runner_busy', 'adapter_busy'])
@pytest.mark.parametrize('change', ['revoke', 'regrant', 'recipient', 'receiver'])
async def test_detail_approval_selected_after_log_await(authorized_send, monkeypatch, lane, change):
    from gateway import group_chat_private_read as detail_module
    c = authorized_send
    pending_task(c)
    ref = c.room_grant['room_ref']
    native = owner(c)
    first = (await control_rpc(native, 'approval', 'grant',
             params(c.room_grant, 'approval', room_id='send-room')))['result']
    event = c.event(f'/group {ref}')
    assert event.message_id == event.source.message_id
    arrived, release = asyncio.Event(), asyncio.Event()
    original = detail_module.dispatch_group_control
    async def delayed(context, method, payload):
        result = await original(context, method, payload)
        if method == 'groups.log' and not arrived.is_set():
            arrived.set()
            await asyncio.wait_for(release.wait(), 10)
        return result
    monkeypatch.setattr(detail_module, 'dispatch_group_control', delayed)
    task = asyncio.create_task(route(c, event, lane))
    try:
        await asyncio.wait_for(arrived.wait(), 10)
        if change in {'revoke', 'regrant'}:
            revoked = native_control_change(native, 'approval', 'revoke',
                params(c.room_grant, 'approval', room_id='send-room',
                       request_id='during-log-revoke', expected_generation=first['generation'],
                       binding_id=first['binding_id']))
            assert revoked['active'] is False
            if change == 'regrant':
                replacement = native_control_change(native, 'approval', 'grant',
                    params(c.room_grant, 'approval', room_id='send-room',
                           request_id='during-log-regrant',
                           expected_generation=revoked['generation']))
                assert replacement['binding_id'] != first['binding_id']
        elif change == 'recipient':
            event.source.chat_id = 'other-private-chat'
        else:
            c.runner.adapters.clear()
    finally:
        release.set()
    await asyncio.wait_for(task, 15)
    if change == 'revoke':
        assert len(c.adapter.sent) == 1
        body = c.adapter.sent[0][1]
        assert f'Group {ref} — Send room' in body and 'Recent messages' in body
        assert 'Approval ' not in body and 'approve pa-' not in body
        assert 'reviewer' in body
    elif change == 'regrant':
        assert len(c.adapter.sent) == 1
        fresh = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
        assert f'approve {fresh} once|deny' in c.adapter.sent[0][1]
    else:
        assert not c.adapter.sent
        assert all('Approval ' not in row['content'] and 'approve pa-' not in row['content']
                   for row in c.adapter.generic_sent)
    if change in {'revoke', 'regrant'}:
        assert not c.adapter.generic_sent


@pytest.mark.asyncio
@pytest.mark.parametrize('scope', ['approval', 'stop'])
@pytest.mark.parametrize('change', ['revoke', 'regrant', 'recipient', 'receiver', 'valid'])
async def test_admitted_uncertain_reply_keeps_captured_control(authorized_send, monkeypatch, scope, change):
    c = authorized_send
    ref = c.room_grant['room_ref']
    native = owner(c)
    first = (await control_rpc(native, scope, 'grant',
             params(c.room_grant, scope, room_id='send-room')))['result']
    calls = []
    if scope == 'approval':
        pending_task(c)
        selector = pending_room_approvals(c.runner, c.event(f'/group {ref}'), ref)[1][0]['selector']
        event = c.event(f'/group {ref} approve {selector} deny')
        monkeypatch.delattr(c.service, 'approve_room_task')
        def effect(**kw):
            calls.append(kw)
            drift()
            raise OSError('injected lost RPC acknowledgement')
        c.service.rpc = SimpleNamespace(approve=effect)
    else:
        event = c.event(f'/group {ref} stop')
        monkeypatch.delattr(c.service, 'stop_room')
        def completion(*args, **kwargs):
            calls.append('completion')
            drift()
            raise OSError('injected Stop completion failure')
        monkeypatch.setattr(c.service, '_finish_room_stop', completion)
    assert event.message_id == event.source.message_id
    def drift():
        if change in {'revoke', 'regrant'}:
            revoked = native_control_change(native, scope, 'revoke',
                params(c.room_grant, scope, room_id='send-room',
                       request_id=f'{scope}-during-effect-revoke',
                       expected_generation=first['generation'], binding_id=first['binding_id']))
            if change == 'regrant':
                replacement = native_control_change(native, scope, 'grant',
                    params(c.room_grant, scope, room_id='send-room',
                           request_id=f'{scope}-during-effect-regrant',
                           expected_generation=revoked['generation']))
                assert replacement['binding_id'] != first['binding_id']
        elif change == 'recipient':
            event.source.chat_id = 'other-private-chat'
        elif change == 'receiver':
            c.runner.adapters.clear()
    await route(c, event, 'runner_busy')
    assert len(calls) == 1
    assert len(c.adapter.sent) == (1 if change == 'valid' else 0)
    if change == 'valid':
        assert 'uncertain' in c.adapter.sent[0][1].lower()
    assert not c.adapter.generic_sent
    if change == 'revoke':
        await route(c, event, 'runner_busy')
        assert len(calls) == 1  # Neither the RPC nor Stop completion is repeated.

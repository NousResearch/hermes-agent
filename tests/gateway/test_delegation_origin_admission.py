"""Worker completion borrows its initiating sender's proof, not the room's first/last speaker."""
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import claim_session_input, list_session_admissions, settle_session_input
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import async_delegation as delegation


async def setup_room(monkeypatch):
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', 'alice,bob,observer')
    config = GatewayConfig()
    config.group_sessions_per_user = False
    # Keep the import-time voice path inside this test's isolated home, including direct runs.
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', get_hermes_home() / 'gateway_voice_mode.json')
    runner = GatewayRunner(config)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    return runner, adapter, authority


async def human(runner, adapter, authority, user, message, outcome='completed'):
    source = adapter.build_source(chat_id='room', chat_type='group', user_id=user, scope_id='guild')
    event = MessageEvent(text=message, source=source, message_id=message)
    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    if outcome == 'cancelled':
        from hermes_state_runtime import cancel_session_input
        cancel_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id)
    else:
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id,
                             generation=claim['generation'], outcome=outcome)
    return source, receipt


def dispatch(runner, source, sid, name='worker', **routing):
    evt = {'type': 'async_delegation', 'delegation_id': name,
           'session_key': runner.session_store._generate_session_key(source),
           'parent_session_id': sid, 'user_id': source.user_id,
           'scope_id': source.scope_id, 'dispatched_at': time.time(),
           'status': 'completed', 'summary': 'worker finished'}
    evt.update(routing)
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {'status': 'completed', 'summary': 'worker finished'})
    return evt


@pytest.mark.asyncio
async def test_pinned_dispatch_does_not_decode_the_conversations_history(monkeypatch):
    import json
    from gateway.delegation_origin import delegation_route_envelope
    from gateway.session_automation import producer_identity
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    for i in range(8):
        await human(runner, adapter, authority, 'alice', f'later-{i}')
    loads, decoded = json.loads, []
    def measured_loads(value, *args, **kwargs):
        result = loads(value, *args, **kwargs)
        if isinstance(result, dict) and 'native_text_v1' in result:
            decoded.append(result)
        return result
    monkeypatch.setattr(json, 'loads', measured_loads)
    entry = runner.session_store.lookup_by_session_key(evt['session_key'])
    proof = await delegation_route_envelope(authority, entry, producer_identity(runner, evt))
    assert proof['event']['message_id'] == 'launch'
    assert len(decoded) == 1


def test_dispatch_captures_initiating_message(monkeypatch):
    from gateway import session_context
    values = {'HERMES_SESSION_USER_ID': 'alice', 'HERMES_SESSION_MESSAGE_ID': 'launch-message'}
    monkeypatch.setattr(session_context, 'get_session_env', lambda key, default='': values.get(key, default))
    assert delegation._capture_routing_origin()['message_id'] == 'launch-message'


@pytest.mark.asyncio
async def test_completion_uses_dispatch_sender_after_another_person_speaks(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    _, first = await human(runner, adapter, authority, 'observer', 'first')
    source, starter = await human(runner, adapter, authority, 'alice', 'start worker')
    assert first.ref == starter.ref
    evt = dispatch(runner, source, starter.ref.session_id)
    _, latest = await human(runner, adapter, authority, 'bob', 'later speaker')
    assert latest.ref == starter.ref
    assert await runner._deliver_completion_notification('worker finished', evt) is True
    rows = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)
    envelope = rows[-1]['payload']['native_text_v1']
    assert envelope['source']['user_id'] == 'alice'
    assert 'message_id' not in envelope['source']
    assert envelope['automation']['owner'] == starter.ref.session_id
    assert delegation.get_durable_delegation('worker')['delivery_state'] == 'delivered'
    assert await runner._deliver_completion_notification('worker finished', evt) is not False
    assert len(list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)) == len(rows)


@pytest.mark.asyncio
async def test_legacy_dispatch_does_not_borrow_a_later_automation_proof(tmp_path, monkeypatch):
    from gateway.delegation_origin import delegation_route_envelope
    from gateway.session_automation import producer_identity
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'start')
    evt = dispatch(runner, source, starter.ref.session_id)
    assert await runner._deliver_completion_notification('finished', evt) is True
    entry = runner.session_store.lookup_by_session_key(evt['session_key'])
    selected = await delegation_route_envelope(authority, entry, producer_identity(runner, evt))
    assert selected['event']['message_id'] == 'start'


@pytest.mark.asyncio
@pytest.mark.parametrize('missing_origin', [False, True])
async def test_exact_dispatch_message_survives_shared_speakers(tmp_path, monkeypatch, missing_origin):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    entry = runner.session_store.lookup_by_session_key(evt['session_key'])
    if missing_origin:
        from dataclasses import replace
        entry.origin = replace(entry.origin, user_id=None, user_name=None)
    await human(runner, adapter, authority, 'bob', 'later')
    assert await runner._deliver_completion_notification('done', evt) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    assert row['payload']['native_text_v1']['source']['user_id'] == 'alice'


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_field,bad_value', [('message_id', 'missing'), ('user_id', 'nobody'), ('scope_id', 'other-guild')])
async def test_missing_dispatch_proof_cannot_borrow_latest_speaker(tmp_path, monkeypatch, bad_field, bad_value):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, **{bad_field: bad_value})
    await human(runner, adapter, authority, 'bob', 'later')
    assert await runner._deliver_completion_notification('done', evt) is False
    assert delegation.get_durable_delegation('worker')['delivery_state'] == 'pending'


@pytest.mark.asyncio
async def test_original_sender_authorization_is_rechecked(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    await human(runner, adapter, authority, 'bob', 'later')
    monkeypatch.setattr(runner, '_is_user_authorized', lambda source, **kwargs: source.user_id == 'bob')
    assert await runner._deliver_completion_notification('done', evt) is False
    assert delegation.get_durable_delegation('worker')['delivery_state'] == 'pending'


@pytest.mark.asyncio
async def test_dispatch_completion_follows_compression_not_a_new_owner(tmp_path, monkeypatch):
    from gateway.session_automation import check_automation_route
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    sid = starter.ref.session_id
    evt = dispatch(runner, source, sid, message_id='launch')
    authority.db.publish_compression_child(parent_session_id=sid, child_session_id='child', source='discord',
        messages=[{'role': 'assistant', 'content': 'summary'}], require_compression_lease=False)
    runner.session_store.advance_compression_session(evt['session_key'], sid, 'child')
    assert await runner._deliver_completion_notification('done', evt) is True
    row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[-1]
    assert check_automation_route(runner, row['payload'], 'child', source, adapter)[1] == evt['session_key']


@pytest.mark.asyncio
async def test_nested_worker_uses_existing_delegate_ownership_proof(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    authority.db.create_session(session_id='execution', source='subagent',
        model_config={'_delegate_from': starter.ref.session_id})
    evt = dispatch(runner, source, 'execution', message_id='launch')
    await human(runner, adapter, authority, 'bob', 'later')
    assert await runner._deliver_completion_notification('done', evt) is True


@pytest.mark.asyncio
async def test_queued_completion_rechecks_sender_before_execution(tmp_path, monkeypatch):
    from gateway.session_envelope import check_native_route
    from hermes_state_runtime import RuntimeStoreError
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    assert await runner._deliver_completion_notification('done', evt) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    monkeypatch.setattr(runner, '_is_user_authorized', lambda source, **kwargs: False)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await check_native_route(runner, row['payload'], starter.ref.session_id, source, adapter)


@pytest.mark.asyncio
async def test_completion_batch_checks_each_workers_sender(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    alice, a = await human(runner, adapter, authority, 'alice', 'alice-launch')
    first = dispatch(runner, alice, a.ref.session_id, name='one', message_id='alice-launch')
    bob, b = await human(runner, adapter, authority, 'bob', 'bob-launch')
    second = dispatch(runner, bob, b.ref.session_id, name='two', message_id='bob-launch')
    monkeypatch.setattr(runner, '_is_user_authorized', lambda source, **kwargs: source.user_id == 'alice')
    assert await runner._deliver_async_delegation_group([first, second]) is False
    assert delegation.get_durable_delegation('one')['delivery_state'] == 'pending'
    assert delegation.get_durable_delegation('two')['delivery_state'] == 'pending'


@pytest.mark.asyncio
async def test_reset_never_receives_old_workers_result(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    runner.session_store.reset_session(evt['session_key'])
    _, fresh = await human(runner, adapter, authority, 'bob', 'new conversation')
    assert fresh.ref != starter.ref
    assert await runner._deliver_completion_notification('old result', evt) is not True
    rows = list_session_admissions(authority.db, session_id=fresh.ref.session_id, pending_only=False)
    assert len(rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['completed', 'failed', 'error', 'timeout', 'stalled', 'interrupted', 'unknown'])
async def test_all_settled_worker_states_deliver(tmp_path, monkeypatch, status):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch', status=status)
    await human(runner, adapter, authority, 'bob', 'later')
    assert await runner._deliver_completion_notification('worker settled', evt) is True
    assert delegation.get_durable_delegation('worker')['delivery_state'] == 'delivered'


@pytest.mark.asyncio
async def test_interim_batch_failure_uses_initiating_sender(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = {'type': 'async_delegation', 'delegation_id': 'batch', 'is_batch': True,
           'session_key': runner.session_store._generate_session_key(source),
           'parent_session_id': starter.ref.session_id, 'user_id': 'alice', 'scope_id': 'guild',
           'message_id': 'launch', 'dispatched_at': time.time(), 'status': 'running', 'goals': ['a', 'b']}
    delegation._persist_dispatch(evt)
    await human(runner, adapter, authority, 'bob', 'later')
    notice = dict(evt, task_failure_notice=True, n_tasks=2,
                  results=[{'task_index': 0, 'status': 'error', 'error': 'fixture failure', 'summary': None}],
                  completed_at=time.time())
    assert await runner._deliver_completion_notification('task 0 failed', notice) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    assert row['payload']['native_text_v1']['source']['user_id'] == 'alice'
    saved = delegation.get_durable_delegation('batch')
    assert saved is not None and saved['state'] == 'running'


@pytest.mark.asyncio
async def test_chained_worker_retains_sender_after_automation_resume(tmp_path, monkeypatch):
    from gateway.session_envelope import restore_native_source
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    first = dispatch(runner, source, starter.ref.session_id, name='first', message_id='launch')
    assert await runner._deliver_completion_notification('first done', first) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    resumed_source = restore_native_source(row['payload'], runner)
    assert not resumed_source.message_id
    claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=starter.ref.session_id)
    assert isinstance(claim, dict)
    second = dispatch(runner, resumed_source, starter.ref.session_id, name='second')
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                         generation=claim['generation'], outcome='completed')
    await human(runner, adapter, authority, 'bob', 'later')
    assert await runner._deliver_completion_notification('second done', second) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    assert row['payload']['native_text_v1']['source']['user_id'] == 'alice'


@pytest.mark.asyncio
async def test_watcher_separates_senders_in_same_drain(tmp_path, monkeypatch):
    import queue
    from unittest.mock import AsyncMock
    from tools.process_registry import process_registry
    runner, adapter, authority = await setup_room(monkeypatch)
    alice, a = await human(runner, adapter, authority, 'alice', 'alice-launch')
    first = dispatch(runner, alice, a.ref.session_id, name='one', message_id='alice-launch')
    bob, b = await human(runner, adapter, authority, 'bob', 'bob-launch')
    second = dispatch(runner, bob, b.ref.session_id, name='two', message_id='bob-launch')
    events = queue.Queue()
    events.put(first)
    events.put(second)
    monkeypatch.setattr(process_registry, 'completion_queue', events)
    monkeypatch.setattr(runner, '_sweep_orphaned_completion_ledgers', lambda: None)
    monkeypatch.setattr(runner, '_drain_watch_notifications', AsyncMock())
    groups = []
    async def deliver(group):
        groups.append(group)
        runner._running = False
        return True
    monkeypatch.setattr(runner, '_deliver_async_delegation_group', deliver)
    runner._running = True
    await runner._async_delegation_watcher(interval=0)
    assert [[evt['user_id'] for evt in group] for group in groups] == [['alice'], ['bob']]


@pytest.mark.asyncio
async def test_completion_restores_origin_proof_after_restart(tmp_path, monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    await human(runner, adapter, authority, 'bob', 'later')
    runner.session_store._save()
    authority.db.close()
    runner, adapter, authority = await setup_room(monkeypatch)
    assert await runner._deliver_completion_notification('done after restart', evt) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    assert row['payload']['native_text_v1']['source']['user_id'] == 'alice'
@pytest.mark.asyncio
@pytest.mark.parametrize('message_id', [None, 'launch'])
async def test_cancelled_input_cannot_authorize_worker(monkeypatch, message_id):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch', outcome='cancelled')
    evt = dispatch(runner, source, starter.ref.session_id, message_id=message_id)
    assert await runner._deliver_completion_notification('done', evt) is False
    assert delegation.get_durable_delegation('worker')['delivery_state'] == 'pending'


@pytest.mark.asyncio
async def test_admission_after_dispatch_is_not_origin_proof(monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, dispatched_at=1)
    assert await runner._deliver_completion_notification('done', evt) is False


@pytest.mark.asyncio
async def test_senderless_legacy_record_keeps_single_speaker_policy(monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, user_id=None)
    assert await runner._deliver_completion_notification('done', evt) is True


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('scope_id', 'other-guild'), ('profile', 'other'),
                                       ('thread_id', 'other-thread'), ('chat_id', 'other-room')])
async def test_dispatch_proof_never_changes_transport(monkeypatch, field, value):
    from dataclasses import replace
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    entry = runner.session_store.lookup_by_session_key(evt['session_key'])
    entry.origin = replace(entry.origin, **{field: value})
    assert await runner._deliver_completion_notification('done', evt) is False


@pytest.mark.asyncio
async def test_replaced_connector_cannot_borrow_dispatch_proof(monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    adapter.config.token = 'replacement-connector'
    assert await runner._deliver_completion_notification('done', evt) is False


@pytest.mark.asyncio
async def test_multiplex_completion_keeps_transport_binding(monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [('default', get_hermes_home())]
    source, starter = await human(runner, adapter, authority, 'alice', 'launch')
    evt = dispatch(runner, source, starter.ref.session_id, message_id='launch')
    await human(runner, adapter, authority, 'bob', 'later')
    assert await runner._deliver_completion_notification('done', evt) is True
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    assert row['payload']['native_text_v1']['provenance']['transport_home'] == str(get_hermes_home())


@pytest.mark.asyncio
async def test_role_authorized_batch_rechecks_once_then_again_before_execution(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.session_envelope import check_native_route
    from hermes_state_runtime import RuntimeStoreError
    runner, adapter, authority = await setup_room(monkeypatch)
    monkeypatch.setenv('DISCORD_ALLOWED_ROLES', '7')
    guild = SimpleNamespace(id=2)
    member = SimpleNamespace(id=1, guild=guild, roles=[SimpleNamespace(id=7)])
    guild.fetch_member = AsyncMock(return_value=member)
    adapter._client = SimpleNamespace(get_guild=lambda guild_id: guild)
    source = adapter.build_source(chat_id='room', chat_type='group', user_id='1', scope_id='2')
    source.role_authorized = True
    event = MessageEvent(text='launch', source=source, message_id='launch')
    with native_callback(runner, event, get_hermes_home()):
        starter = await authority.admit_native(event)
    claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=starter.ref.session_id)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=starter.admission_id,
                         generation=claim['generation'], outcome='completed')
    first = dispatch(runner, source, starter.ref.session_id, name='one', message_id='launch')
    second = dispatch(runner, source, starter.ref.session_id, name='two', message_id='launch')
    guild.fetch_member.reset_mock()
    assert await runner._deliver_async_delegation_group([first, second]) is True
    assert guild.fetch_member.await_count == 1
    row = list_session_admissions(authority.db, session_id=starter.ref.session_id, pending_only=False)[-1]
    assert row['payload']['native_text_v1']['reauthorize'] == 'roles'
    member.roles = []
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await check_native_route(runner, row['payload'], starter.ref.session_id, source, adapter)
    assert guild.fetch_member.await_count == 2


@pytest.mark.asyncio
async def test_mixed_sender_batch_is_refused_even_if_both_are_authorized(monkeypatch):
    runner, adapter, authority = await setup_room(monkeypatch)
    alice, a = await human(runner, adapter, authority, 'alice', 'alice-launch')
    first = dispatch(runner, alice, a.ref.session_id, name='one', message_id='alice-launch')
    bob, b = await human(runner, adapter, authority, 'bob', 'bob-launch')
    second = dispatch(runner, bob, b.ref.session_id, name='two', message_id='bob-launch')
    assert await runner._deliver_async_delegation_group([first, second]) is False

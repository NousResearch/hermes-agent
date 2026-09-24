"""Private list through real native consent, reader and registered dispatch.

Inert fixture extends #111939's accepted owner fixture. No service is started.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from hermes_cli.commands import resolve_command
from tests.gateway.test_messaging_inventory_binding import bound, enrolled, snapshot

__all__ = ['bound']


class CapturingReceiver(BasePlatformAdapter):
    def __init__(self, runner, config):
        super().__init__(config, Platform.SIGNAL)
        self.gateway_runner = runner
        self.sent = []
        self.generic_sent = []

    async def connect(self, *, is_reconnect=False):
        raise AssertionError('live connect')

    async def disconnect(self):
        raise AssertionError('live disconnect')

    async def get_chat_info(self, chat_id):
        raise AssertionError('live chat lookup')

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id='synthetic-out')

    async def _send_with_retry(self, **kwargs):
        # Observable negative boundary: sensitive lists must NEVER reach here.
        self.generic_sent.append(kwargs)
        return SendResult(success=True, message_id='synthetic-generic')


@pytest_asyncio.fixture
async def consumer(bound, monkeypatch):
    r = bound.runner
    r.config.multiplex_profiles = False
    r._external_drain_active = False
    r._busy_input_mode = 'queue'
    r._busy_text_mode = 'queue'
    r._sessions = {}
    a = CapturingReceiver(r, bound.adapter.config)
    bound.adapter = a
    r.adapters = {Platform.SIGNAL: a}
    a.set_message_handler(r._primary_message_handler())
    a._busy_session_handler = r._primary_busy_session_handler()
    for method in ('_handle_message_with_agent', '_queue_or_replace_pending_event',
                   '_claim_active_session_slot', '_hm_skill_slash_rewrite',
                   '_interrupt_running_agent_for_busy_event'):
        monkeypatch.setattr(r, method, bound.forbidden)
    from gateway import session_ingress
    monkeypatch.setattr(session_ingress, 'admit_message', bound.forbidden)
    # M73's existing numeric ledger is data to preserve, never to renumber.
    def classic_ledger(conn):
        conn.execute('CREATE TABLE hosted_room_messaging_refs ('
                     'room_ref INTEGER PRIMARY KEY AUTOINCREMENT, room_id TEXT NOT NULL UNIQUE)')
        conn.execute('INSERT INTO hosted_room_messaging_refs VALUES (41, ?)', ('classic-sentinel',))
    bound.db._execute_write(classic_ledger)

    def event(text='/group list', **changes):
        source = a.build_source(chat_id='private-chat', user_id='person-1',
                                scope_id='scope-1', message_id='message-1')
        source.is_one_to_one = True
        for key, value in changes.items():
            setattr(source, key, value)
        return MessageEvent(text=text, message_type=MessageType.COMMAND,
                            source=source, message_id='message-1', user_id=source.user_id)
    bound.event = event
    return bound


async def route(c, event, lane='idle'):
    r, a = c.runner, c.adapter
    key = r._session_key_for_source(event.source)
    if lane != 'idle':
        r._session_state(key).turn.agent = SimpleNamespace(interrupt=c.forbidden)
    if lane == 'adapter_busy':
        a._active_sessions[key] = asyncio.Event()
        await a._handle_message_while_active(event, key)
    else:
        await a._dispatch_inline_reply(event)
    assert not a._pending_messages
    assert not r._session_state(key).conversation.queued_events


@pytest.mark.asyncio
async def test_registered_idle_canonical_dispatch_is_handled(consumer):
    c = consumer
    await enrolled(c)
    event = c.event()
    handled, result = await c.runner._hm_dispatch_canonical_command(
        event, event.source, c.runner._session_key_for_source(event.source), 'group')
    assert handled, 'actual canonical idle dispatch does not handle group list'
    assert result == ''
    assert 'Alice inventory' in c.adapter.sent[0][1]


@pytest.mark.asyncio
@pytest.mark.parametrize('multiplex', [False, True])
async def test_real_adapter_idle_lifecycle_does_not_resend_private_body(consumer, multiplex):
    c = consumer
    c.runner.config.multiplex_profiles = multiplex
    c.adapter.set_message_handler(c.runner._primary_message_handler())
    c.adapter.config.typing_indicator = False
    await enrolled(c)
    before = snapshot(c.db)
    await c.adapter.handle_message(c.event())
    tasks = list(c.adapter._session_tasks.values())
    assert tasks
    await asyncio.wait_for(asyncio.gather(*tasks), 10)
    assert len(c.adapter.sent) == 1 and 'Alice inventory' in c.adapter.sent[0][1]
    assert not c.adapter.generic_sent and not c.adapter._pending_messages
    assert not c.adapter._active_sessions and not c.adapter._background_tasks
    assert snapshot(c.db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('multiplex', [False, True])
@pytest.mark.parametrize('lane', ['idle', 'runner_busy', 'adapter_busy'])
async def test_registered_list_discloses_only_native_owner_inventory(consumer, multiplex, lane):
    c = consumer
    c.runner.config.multiplex_profiles = multiplex
    c.adapter.set_message_handler(c.runner._primary_message_handler())
    c.adapter._busy_session_handler = c.runner._primary_busy_session_handler()
    await enrolled(c)
    command = resolve_command('group')
    assert command is not None, 'private canonical group command is not registered'
    assert command.gateway_only and command.busy_policy == 'dispatch'
    before = snapshot(c.db)
    await route(c, c.event(), lane)
    assert len(c.adapter.sent) == 1
    target, body, _, _ = c.adapter.sent[0]
    assert target == 'private-chat'
    assert 'Alice inventory' in body and '1 member' in body
    assert all(secret not in body for secret in ('BOB PRIVATE', 'alice-room', 'native-alice'))
    assert not c.adapter.generic_sent
    assert snapshot(c.db) == before
    print('PRIVATE_LIST', json.dumps({'lane': lane, 'multiplex': multiplex, 'body': body}))


def add_rooms(c, owner, count, *, start=10):
    from gateway import hosted_rooms as rooms
    for i in range(count):
        rid = f'synthetic-{owner.actor.subject}-{start + i}'
        c.service.authorize_room(owner.actor.subject, rid, create=True)
        rooms.create_room(c.db.db_path, room_id=rid, name=f'Visible {start + i}', now=start + i,
                          members=[dict(member_id='writer', profile='default', handle='writer')],
                          authority_gateway_id='inert-gateway')


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', [
    'no_grant', 'revoke', 'no_admin', 'generic_command', 'disabled_gating', 'home_only',
    'user', 'chat', 'thread', 'scope', 'profile', 'bot', 'edit', 'relay', 'shared',
    'not_private', 'unstamped', 'receiver', 'service', 'registry', 'registry_unready',
])
async def test_denial_never_reads_room_inventory(consumer, monkeypatch, denial):
    from gateway import hosted_rooms as rooms
    from gateway.config import HomeChannel
    from tests.gateway.test_messaging_inventory_binding import recipient, rpc
    c = consumer
    first = None if denial == 'no_grant' else await enrolled(c)
    event = c.event()
    if denial == 'revoke':
        assert 'result' in await rpc(c.alice, 'revoke', dict(
            request_id='revoke-one', recipient=recipient(), binding_id=first['binding_id'],
            expected_generation=first['generation']))
    elif denial in {'no_admin', 'generic_command', 'disabled_gating', 'home_only'}:
        c.adapter.config.extra['allow_admin_from'] = []
        if denial == 'generic_command':
            c.adapter.config.extra.update(allow_admin_from=['other'], user_allowed_commands=['group'])
        if denial == 'home_only':
            c.adapter.config.home_channel = HomeChannel(
                platform=Platform.SIGNAL, chat_id='private-chat', name='Synthetic Home', user_id='person-1')
    elif denial == 'user':
        event.source.user_id = 'person-2'
        c.adapter.config.extra.update(allow_from=['person-1', 'person-2'],
                                     allow_admin_from=['person-1', 'person-2'])
    elif denial in {'chat', 'thread', 'scope'}:
        setattr(event.source, denial + '_id', 'other')
    elif denial == 'profile':
        event.source.profile = 'unserved'
    elif denial in {'bot', 'edit', 'relay'}:
        setattr(event.source, {'bot': 'is_bot', 'edit': 'message_is_edit',
                              'relay': 'delivered_via_upstream_relay'}[denial], True)
    elif denial == 'shared':
        event.source.chat_type = 'group'
    elif denial == 'not_private':
        event.source.is_one_to_one = None
    elif denial == 'unstamped':
        del event.source._transport_adapter_ref
    elif denial == 'receiver':
        c.runner.adapters[Platform.SIGNAL] = CapturingReceiver(c.runner, c.adapter.config)
    elif denial == 'service':
        c.authority.hosted_room_service = None
    elif denial == 'registry':
        c.runner.session_authorities = None
    elif denial == 'registry_unready':
        from gateway.session_authorities import SessionAuthorities
        c.runner.session_authorities = SessionAuthorities(c.home)
        c.runner.session_authorities.add(c.home, None)
    monkeypatch.setattr(rooms, 'list_rooms', c.forbidden)
    before = snapshot(c.db)
    # Direct canonical dispatch retains the actual handler's own policy; avoids
    # unrelated unauthorized-DM pairing replies for deliberately invalid sources.
    handled, result = await c.runner._hm_dispatch_canonical_command(
        event, event.source, c.runner._session_key_for_source(event.source), 'group')
    assert handled and result and 'Alice' not in result
    assert not c.adapter.sent
    assert snapshot(c.db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['filtered', 'empty', 'page_two', 'out_of_range', 'cap'])
async def test_complete_bounded_enumeration_before_display_slicing(consumer, monkeypatch, mode):
    from gateway import hosted_room_messaging_runtime as runtime
    from tests.gateway.test_messaging_inventory_binding import rpc
    c = consumer
    if mode == 'empty':
        nobody = c.native('native-nobody')
        assert 'result' in await rpc(nobody)
    else:
        await enrolled(c)
    if mode in {'filtered', 'cap'}:
        add_rooms(c, c.bob, 8)
    if mode == 'page_two':
        add_rooms(c, c.alice, 8)
    if mode == 'cap':
        monkeypatch.setattr(runtime, 'MAX_INVENTORY_OFFSET', 8)
    calls = []
    original = runtime.dispatch_group_control
    async def observed(context, method, params):
        value = await original(context, method, params)
        calls.append((context, params.copy(), value))
        return value
    monkeypatch.setattr(runtime, 'dispatch_group_control', observed)
    before = snapshot(c.db)
    text = '/group list 2' if mode in {'page_two', 'out_of_range'} else '/group list'
    result = await c.runner._handle_group_command(c.event(text))
    assert all(call[0] is calls[0][0] for call in calls)
    assert calls[0][1] == {'limit': 8, 'offset': 0}
    if mode == 'cap':
        assert result == runtime.ENUMERATION_LIMIT and not c.adapter.sent
    elif mode == 'out_of_range':
        assert result == runtime.INVALID_PAGE and not c.adapter.sent
    else:
        body = c.adapter.sent[0][1]
        assert result == ''
        if mode == 'empty':
            assert body == 'No groups are available in your authorized inventory.'
        else:
            assert 'Alice inventory' in body
        if mode == 'filtered':
            assert calls[0][2] == {'rooms': [], 'next_offset': 8}
            assert calls[1][1]['offset'] == 8
        if mode == 'page_two':
            assert 'page 2 of 2' in body and 'Previous: /group list 1' in body
            assert 'Visible' not in body
    assert snapshot(c.db) == before


async def drift(c, event, first, kind):
    from tests.gateway.test_messaging_inventory_binding import grant_params, recipient, rpc
    if kind in {'revoke', 'aba'}:
        result = await rpc(c.alice, 'revoke', dict(request_id='revoke-one',
            recipient=recipient(), binding_id=first['binding_id'], expected_generation=1))
        assert result['result']['active'] is False
        if kind == 'aba':
            result = await rpc(c.alice, params=grant_params(request_id='grant-two', expected_generation=2))
            assert result['result']['binding_id'] != first['binding_id']
    elif kind == 'receiver':
        c.runner.adapters[Platform.SIGNAL] = CapturingReceiver(c.runner, c.adapter.config)
    elif kind == 'registry':
        from gateway.session_authorities import SessionAuthorities
        c.runner.session_authorities = SessionAuthorities(c.home)
        c.runner.session_authorities.add(c.home, c.authority)
    elif kind == 'runtime':
        c.authority.epoch += 1
    elif kind == 'admin':
        c.adapter.config.extra['allow_admin_from'] = []
    elif kind == 'thread':
        event.source.thread_id = 'replacement-thread'
    elif kind == 'service':
        c.authority.hosted_room_service = None


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['between_pages', 'before_send'])
@pytest.mark.parametrize('kind', ['revoke', 'aba', 'receiver', 'registry', 'runtime', 'admin', 'thread', 'service'])
async def test_original_continuation_fences_every_page_and_final_handoff(consumer, monkeypatch, stage, kind):
    from gateway import hosted_room_messaging_runtime as runtime
    c = consumer
    first = await enrolled(c)
    add_rooms(c, c.bob, 8)
    event = c.event()
    entered, resume = asyncio.Event(), asyncio.Event()
    calls = []
    original_dispatch = runtime.dispatch_group_control
    original_render = runtime.render_inventory_page

    async def observed(context, method, params):
        result = await original_dispatch(context, method, params)
        calls.append((context, params.copy()))
        if stage == 'between_pages' and len(calls) == 1:
            entered.set()
            await resume.wait()
        return result

    async def held(context, page, prefix):
        body = await original_render(context, page, prefix)
        if stage == 'before_send':
            assert 'Alice inventory' in body
            entered.set()
            await resume.wait()
        return body

    monkeypatch.setattr(runtime, 'dispatch_group_control', observed)
    monkeypatch.setattr(runtime, 'render_inventory_page', held)
    pending = asyncio.create_task(route(c, event, 'adapter_busy'))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await drift(c, event, first, kind)
        after_drift = snapshot(c.db)
    finally:
        resume.set()
        await asyncio.wait_for(pending, 10)
    assert not c.adapter.sent
    assert all('Alice inventory' not in item['content'] for item in c.adapter.generic_sent)
    assert all(call[0] is calls[0][0] for call in calls)
    if stage == 'between_pages':
        assert len(calls) == 1
    assert snapshot(c.db) == after_drift


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['raise', 'negative_result'])
async def test_handoff_is_not_retried_or_duplicated(consumer, monkeypatch, failure):
    c = consumer
    await enrolled(c)
    sends = []
    async def failed(chat_id, content, **kwargs):
        sends.append((chat_id, content))
        if failure == 'raise':
            raise RuntimeError('synthetic transport error with private body')
        return SendResult(success=False)
    monkeypatch.setattr(c.adapter, 'send', failed)
    await route(c, c.event(), 'adapter_busy')
    assert len(sends) == 1 and 'Alice inventory' in sends[0][1]
    assert not c.adapter.generic_sent


@pytest.mark.asyncio
@pytest.mark.parametrize('cursor', [True, '8', -1, 0, 4096])
async def test_invalid_or_exhausted_raw_cursor_is_never_complete_empty(consumer, monkeypatch, cursor):
    from gateway import hosted_room_messaging_runtime as runtime
    c = consumer
    await enrolled(c)
    original = runtime.dispatch_group_control
    async def corrupt(context, method, params):
        result = await original(context, method, params)
        return result | {'next_offset': cursor}
    monkeypatch.setattr(runtime, 'dispatch_group_control', corrupt)
    result = await c.runner._handle_group_command(c.event())
    assert result == (runtime.ENUMERATION_LIMIT if cursor == 4096 else runtime.UNAVAILABLE)
    assert not c.adapter.sent


@pytest.mark.asyncio
async def test_storage_unavailable_has_no_classic_fallback(consumer, monkeypatch):
    import sqlite3
    from gateway import hosted_rooms as rooms
    from gateway import hosted_room_messaging_runtime as runtime
    c = consumer
    await enrolled(c)
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError('synthetic private storage path')
    monkeypatch.setattr(rooms, 'list_rooms', unavailable)
    before = snapshot(c.db)
    assert await c.runner._handle_group_command(c.event()) == runtime.UNAVAILABLE
    assert not c.adapter.sent and snapshot(c.db) == before


@pytest.mark.asyncio
async def test_help_invalid_page_and_real_thread_are_read_only(consumer, monkeypatch):
    from gateway import hosted_rooms as rooms
    from tests.gateway.test_messaging_inventory_binding import grant_params, recipient, rpc
    c = consumer
    assert 'result' in await rpc(c.alice, params=grant_params(recipient=recipient(thread_id='real-thread')))
    c.adapter.typed_command_prefix = '!'
    before = snapshot(c.db)
    await route(c, c.event(thread_id='real-thread'), 'adapter_busy')
    assert c.adapter.sent[0][3] == {'thread_id': 'real-thread', '_interim_send': True}
    monkeypatch.setattr(rooms, 'list_rooms', c.forbidden)
    await route(c, c.event('/group help', thread_id='real-thread'))
    assert '!group list [page]' in c.adapter.sent[1][1]
    await route(c, c.event('/group send @everyone', thread_id='real-thread'))
    assert len(c.adapter.sent) == 2
    assert '@everyone' not in c.adapter.generic_sent[-1]['content']
    assert snapshot(c.db) == before


@pytest.mark.asyncio
async def test_current_standalone_wrapper_after_multi_profile_activation(consumer, monkeypatch):
    from agent import secret_scope
    from tui_gateway import launch_profile_policy
    c = consumer
    # Real startup ordering: install primary callback before canonical authority
    # is attached, then use that SAME callback once the inert authority is bound.
    c.runner.session_authority = None
    handler = c.runner._primary_message_handler()
    c.runner.session_authority = c.authority
    c.adapter.set_message_handler(handler)
    monkeypatch.setattr(launch_profile_policy, '_snapshot', None)
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', False)
    monkeypatch.setenv('SYNTHETIC_SCOPE_VALUE', 'launch-value')
    launch_profile_policy.activate_multi_profile_hosting()
    monkeypatch.setenv('SYNTHETIC_SCOPE_VALUE', 'ambient-poison')
    await enrolled(c)
    original = c.adapter.send
    async def scoped_send(*args, **kwargs):
        assert secret_scope.get_secret('SYNTHETIC_SCOPE_VALUE') == 'launch-value'
        return await original(*args, **kwargs)
    monkeypatch.setattr(c.adapter, 'send', scoped_send)
    await route(c, c.event(), 'adapter_busy')
    assert len(c.adapter.sent) == 1 and not c.adapter.generic_sent


@pytest.mark.asyncio
async def test_multiplex_default_wrapper_restores_a_after_b(consumer):
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home
    c = consumer
    c.runner.config.multiplex_profiles = True
    c.adapter.set_message_handler(c.runner._primary_message_handler())
    await enrolled(c)
    foreign = c.home / 'profiles' / 'secondary'
    foreign.mkdir(parents=True)
    (foreign / 'config.yaml').write_text('{}')
    for home in (c.home, foreign, c.home):
        with _profile_runtime_scope(home):
            await route(c, c.event())
            assert get_hermes_home() == home
    assert len(c.adapter.sent) == 3 and not c.adapter.generic_sent


@pytest.mark.asyncio
async def test_one_attestation_retained_until_exact_send(consumer, monkeypatch):
    from gateway import group_chat_slash as slash
    from gateway.session_group_messaging_read import _InventoryRead
    c = consumer
    await enrolled(c)
    add_rooms(c, c.bob, 8)
    attestations, checks = [], []
    original_attest, original_check = slash._attest_inventory, _InventoryRead.require_current
    def observed_attest(*args):
        context = original_attest(*args)
        attestations.append(context)
        return context
    def observed_check(self):
        value = original_check(self)
        checks.append(self)
        return value
    original_send = c.adapter.send
    async def observed_send(*args, **kwargs):
        assert len(attestations) == 1
        assert checks[-1] is attestations[0]
        return await original_send(*args, **kwargs)
    monkeypatch.setattr(slash, '_attest_inventory', observed_attest)
    monkeypatch.setattr(_InventoryRead, 'require_current', observed_check)
    monkeypatch.setattr(c.adapter, 'send', observed_send)
    await route(c, c.event())
    assert len(attestations) == len(c.adapter.sent) == 1


@pytest.mark.asyncio
async def test_launch_registry_cannot_be_removed_and_receiver_policy_is_used(consumer):
    from gateway.config import PlatformConfig
    c = consumer
    await enrolled(c)
    with pytest.raises(RuntimeError, match='launch session authority cannot be removed'):
        c.runner.session_authorities.remove(c.home)
    # Deliberately diverge the root config from the registered receiver's policy.
    c.runner.config.platforms[Platform.SIGNAL] = PlatformConfig(enabled=True,
        extra={'allow_from': ['person-1'], 'allow_admin_from': ['other']})
    assert c.runner._check_slash_access(c.event().source, 'group') is None
    assert c.runner._check_slash_access(c.event().source, 'model') is not None
    await route(c, c.event(), 'adapter_busy')
    assert len(c.adapter.sent) == 1 and not c.adapter.generic_sent

"""Adapter-owned journal references survive native admission without becoming authority."""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
import asyncio
import json
import threading

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import prepare_native, restore_native, snapshot_native
from gateway.session_results import execution_result
from hermes_state_runtime import RuntimeStoreError
from tests.waiting import async_wait_until


class JournalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token='owned-fixture'), Platform.TELEGRAM)
        self.identity = 'bot-a'
        self.deliveries = []
        self.validated = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        self.deliveries.append(content)
        return SendResult(success=True, message_id='sent')

    async def get_chat_info(self, chat_id):
        return {'id': chat_id}

    def validate_native_metadata(self, event):
        self.validated.append((event.source.chat_id, deepcopy(event.metadata), event.raw_message))
        return event.metadata == {'telegram_binding': {
            'identity': self.identity, 'chat': event.source.chat_id, 'message': event.message_id,
            'journal': 'journal-1', 'turn': 'turn-1'}}


def event_for(adapter):
    return MessageEvent(text='hello', message_id='m1',
                        source=adapter.build_source(chat_id='chat', user_id='sender', chat_type='dm'),
                        raw_message={'must_not_replay': True}, metadata={'telegram_binding': {
                            'identity': 'bot-a', 'chat': 'chat', 'message': 'm1',
                            'journal': 'journal-1', 'turn': 'turn-1'}})


@pytest_asyncio.fixture
async def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'sender')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = JournalAdapter()
    adapter.gateway_runner, runner.adapters = runner, {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='metadata')
    runner._wire_adapter_handlers(adapter)
    yield runner, adapter
    authority.db.close()


@pytest.mark.asyncio
async def test_bound_metadata_replays_and_real_admission_delivers(runtime, monkeypatch):
    runner, adapter = runtime
    event = event_for(adapter)
    from gateway.session_ingress_context import native_callback
    event.source._process_only_lock = threading.Lock()
    with native_callback(runner, event, runner._native_transport_homes[None]):
        payload = json.loads(json.dumps(snapshot_native(runner, event)))
    assert 'provenance' in payload['native_text_v1']
    restored = restore_native(payload, runner)
    assert restored.metadata == event.metadata
    assert restored.raw_message is None
    assert all(raw is None for _, _, raw in adapter.validated)
    turns, lifecycle = [], []
    bound = ContextVar('test_native_bound', default=None)

    @asynccontextmanager
    async def scope(self, actual):
        from hermes_constants import get_hermes_home
        assert get_hermes_home() == runner._native_transport_homes[None]
        assert self is adapter
        lifecycle.append(('enter', actual))
        actual._adapter_local = 'bound'
        token = bound.set(actual)
        try:
            yield
        finally:
            lifecycle.append(('exit', actual))
            bound.reset(token)

    monkeypatch.setattr(JournalAdapter, 'native_execution_scope', scope, raising=False)

    async def turn(actual, *_args, **_kwargs):
        assert bound.get() is actual and actual._adapter_local == 'bound'
        turns.append(actual)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    adapter._enqueue_text_event(event)
    await async_wait_until(lambda: adapter.deliveries and not adapter._background_tasks,
                           timeout=20, what='native metadata delivery')
    assert adapter.deliveries == ['ok']
    assert turns[0] is not event
    assert turns[0].metadata == event.metadata and turns[0].raw_message is None
    assert lifecycle == [('enter', turns[0]), ('exit', turns[0])]
    adapter.identity = 'bot-b'
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        restore_native(payload, runner)


@pytest.mark.asyncio
async def test_metadata_is_frozen_before_yield_and_hook_cannot_rewrite(runtime, monkeypatch):
    runner, adapter = runtime
    event = event_for(adapter)
    expected = deepcopy(event.metadata)

    async def yield_and_mutate(*_args):
        event.metadata['telegram_binding']['identity'] = 'attacker'
        return False

    monkeypatch.setattr('gateway.session_ingress_context.reauthorize_roles', yield_and_mutate)
    payload = await prepare_native(runner, event)
    assert payload['native_text_v1']['event']['metadata'] == expected

    def mutating_check(self, isolated):
        isolated.metadata.clear()
        isolated.source.chat_id = 'other'
        return True

    monkeypatch.setattr(JournalAdapter, 'validate_native_metadata', mutating_check)
    event = event_for(adapter)
    payload = snapshot_native(runner, event)
    assert payload['native_text_v1']['event']['metadata'] == expected
    assert payload['native_text_v1']['source']['chat_id'] == 'chat'


@pytest.mark.asyncio
async def test_pure_metadata_validation_binds_home_without_secret_hydration(runtime, monkeypatch):
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    runner, adapter = runtime
    event = event_for(adapter)
    home = runner._native_transport_homes[None]
    checked = []

    def forbidden_hydration(*_args, **_kwargs):
        pytest.fail('pure metadata validation must not run external secret providers')

    def check(self, isolated):
        checked.append(get_hermes_home())
        return isolated.metadata == event.metadata and self is adapter

    monkeypatch.setattr('hermes_cli.env_loader.hydrate_profile_secret_sources', forbidden_hydration)
    monkeypatch.setattr(JournalAdapter, 'validate_native_metadata', check)
    with native_callback(runner, event, home):
        payload = snapshot_native(runner, event)
    assert restore_native(payload, runner).metadata == event.metadata
    assert checked == [home, home]


@pytest.mark.asyncio
@pytest.mark.parametrize('with_runner', [True, False])
async def test_missing_adapter_restore_keeps_not_found_reason(runtime, with_runner):
    runner, adapter = runtime
    payload = snapshot_native(runner, event_for(adapter))
    assert 'provenance' not in payload['native_text_v1']
    runner.adapters.clear()
    with pytest.raises(RuntimeStoreError, match='not_found'):
        restore_native(payload, runner if with_runner else None)


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [[], False, '', {'gateway_session_id': 's'},
    {'telegram_': True}, {'telegram_binding': {1: 'coerced'}}, {'telegram_binding': float('nan')},
    {'telegram_binding': float('inf')}, {'telegram_binding': {'x'}},
    {'telegram_binding': 'x' * 16385}, {'telegram_binding': [0] * 1025},
    {'telegram_binding': [[[[[[[[[0]]]]]]]]]}, {'telegram_binding': {'identity': 'other'}}])
async def test_invalid_metadata_never_admits(runtime, metadata):
    runner, adapter = runtime
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        snapshot_native(runner, replace(event_for(adapter), metadata=metadata))


@pytest.mark.asyncio
async def test_missing_throwing_nonboolean_validator_and_cycles_fail_closed(runtime, monkeypatch):
    runner, adapter = runtime
    event = event_for(adapter)
    cycle = {}
    cycle['telegram_cycle'] = cycle
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        snapshot_native(runner, replace(event, metadata=cycle))
    for check in (None, lambda *_: 1, lambda *_: False):
        monkeypatch.setattr(JournalAdapter, 'validate_native_metadata', check)
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            snapshot_native(runner, event)

    def raises(*_):
        raise ValueError('stale journal')

    monkeypatch.setattr(JournalAdapter, 'validate_native_metadata', raises)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        snapshot_native(runner, event)
    monkeypatch.setattr(JournalAdapter, 'validate_native_metadata', lambda *_: True)
    event.source.user_id = 'unapproved'
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        snapshot_native(runner, event)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [ValueError('before worker'), asyncio.CancelledError()])
async def test_execution_scope_cannot_suppress_handler_failure(runtime, monkeypatch, failure):
    from gateway.session_adapter_execution import execute_native_handler
    runner, adapter = runtime
    event = event_for(adapter)
    payload = snapshot_native(runner, event)
    restored = restore_native(payload, runner)
    entered = []

    @asynccontextmanager
    async def suppressing(self, actual):
        entered.append(actual)
        try:
            yield
        except BaseException:
            pass  # Deliberately hostile scope: the gateway must preserve the failure.

    async def fails(actual):
        assert actual is restored and entered == [restored]
        raise failure

    monkeypatch.setattr(JournalAdapter, 'native_execution_scope', suppressing, raising=False)
    monkeypatch.setattr(runner, '_handle_message', fails)
    with pytest.raises(type(failure)) as caught:
        await execute_native_handler(runner, payload, restored)
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize('change,reason', [('closed', 'invalid_params'), ('detached', 'not_found')])
async def test_metadata_only_adapter_rechecks_after_profile_entry(runtime, monkeypatch, change, reason):
    from gateway.session_adapter_execution import execute_native_handler
    from gateway.session_ingress import _execution_scope
    runner, adapter = runtime
    payload = snapshot_native(runner, event_for(adapter))
    restored = restore_native(payload, runner)
    loading, release = asyncio.Event(), threading.Event()
    loop, handled = asyncio.get_running_loop(), []

    def load_secrets(_home):
        loop.call_soon_threadsafe(loading.set)
        assert release.wait(timeout=10), 'profile load was not released'
        return {}

    async def handler(actual):
        handled.append(actual)

    async def execute():
        with await _execution_scope(runner.session_authority, runner._native_transport_homes[None]):
            await execute_native_handler(runner, payload, restored)

    monkeypatch.setattr('gateway.run._load_profile_secret_scope', load_secrets)
    monkeypatch.setattr(runner, '_handle_message', handler)
    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(loading.wait(), timeout=10)
        if change == 'closed':
            monkeypatch.setattr(JournalAdapter, 'validate_native_metadata', lambda *_: False)
        else:
            runner.adapters.clear()
    finally:
        release.set()
    with pytest.raises(RuntimeStoreError, match=reason):
        await task
    assert handled == []


@pytest.mark.asyncio
async def test_execution_scope_refusal_and_rebinding_never_reach_handler(runtime, monkeypatch):
    from gateway.session_adapter_execution import execute_native_handler
    runner, adapter = runtime
    event = event_for(adapter)
    payload = snapshot_native(runner, event)
    calls = []

    async def handler(actual):
        calls.append(actual)
        return 'should not run'

    monkeypatch.setattr(runner, '_handle_message', handler)

    @asynccontextmanager
    async def refuse(self, actual):
        raise RuntimeStoreError('permission_denied')
        yield

    monkeypatch.setattr(JournalAdapter, 'native_execution_scope', refuse, raising=False)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await execute_native_handler(runner, payload, restore_native(payload, runner))

    @asynccontextmanager
    async def rebind(self, actual):
        actual.source.chat_id = 'other'
        yield

    monkeypatch.setattr(JournalAdapter, 'native_execution_scope', rebind)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await execute_native_handler(runner, payload, restore_native(payload, runner))
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['missing', 'automation', 'api', 'non-native'])
async def test_execution_scope_defaults_and_non_native_exclusions(runtime, monkeypatch, kind):
    from gateway.session_adapter_execution import execute_native_handler
    runner, adapter = runtime
    event = event_for(adapter)
    payload = snapshot_native(runner, event)

    @asynccontextmanager
    async def must_not_enter(self, actual):
        raise AssertionError('wrong lifecycle')
        yield

    async def handler(actual):
        return 'ordinary result'

    if kind != 'missing':
        monkeypatch.setattr(JournalAdapter, 'native_execution_scope', must_not_enter, raising=False)
    if kind == 'automation':
        payload['native_text_v1']['automation'] = {}
    elif kind == 'api':
        event.source.platform = Platform.API_SERVER
    elif kind == 'non-native':
        payload = {'text': 'ordinary'}
    monkeypatch.setattr(runner, '_handle_message', handler)
    assert await execute_native_handler(runner, payload, event) == 'ordinary result'


@pytest.mark.asyncio
@pytest.mark.parametrize('entrypoint', ['route', 'automation'])
async def test_closed_human_metadata_does_not_block_bound_automation_wake(runtime, monkeypatch, entrypoint):
    from gateway.session_automation import admit_route_wake
    from hermes_state_runtime import list_session_admissions
    runner, adapter = runtime
    authority = runner.session_authority
    original = event_for(adapter)
    texts = []

    async def turn(actual, *_args, **_kwargs):
        texts.append(actual.text)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    adapter._enqueue_text_event(original)
    await async_wait_until(lambda: adapter.deliveries and not adapter._background_tasks,
                           timeout=20, what='original human completion')
    route = runner.session_store._generate_session_key(original.source)
    sid = runner.session_store.peek_session_id(route)
    human = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
    adapter.identity = 'closed-journal'
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        restore_native(human['payload'], runner)

    if entrypoint == 'route':
        assert await admit_route_wake(runner, adapter, original.source, 'wake', 'test-wake')
    else:
        wake = MessageEvent(text='wake', source=original.source, internal=True, metadata={
            'gateway_session_key': route, 'gateway_session_id': sid})
        await authority.admit_automation(adapter, wake, 'test-wake')
    await async_wait_until(lambda: len(adapter.deliveries) == 2,
                           timeout=20, what='bound automation delivery')
    assert texts == ['hello', 'wake']
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 2 and 'automation' in rows[-1]['payload']['native_text_v1']
    # Borrowing a route still requires its live connector; journal closure is not revocation.
    adapter.config.token = 'different-connector'
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        await admit_route_wake(runner, adapter, original.source, 'stale', 'test-wake')


@pytest.mark.asyncio
async def test_prepared_producer_identity_uses_validated_source_not_journal_replay(runtime, monkeypatch):
    from gateway.platforms.webhook_ingress import _prepared, producer_scope
    from gateway.session_envelope import prepare_native as actual_prepare
    runner, adapter = runtime
    original = event_for(adapter)

    async def prepare_then_close(*args):
        payload = await actual_prepare(*args)
        adapter.identity = 'closed-after-preparation'
        return payload

    monkeypatch.setattr('gateway.session_envelope.prepare_native', prepare_then_close)
    async with producer_scope(adapter, original) as authority:
        payload, principal = await _prepared(authority, original)
    assert principal == 'messaging:' + json.dumps([None, 'telegram', 'chat', None, 'sender'], separators=(',', ':'))
    # Computing a principal is not executing or re-admitting the now-stale producer event.
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        restore_native(payload, runner)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        snapshot_native(runner, original)


@pytest.mark.asyncio
async def test_relay_route_borrows_only_sealed_source_from_closed_human_metadata(tmp_path, monkeypatch):
    from gateway.session import SessionSource
    from gateway.session_ingress_context import native_callback, register_transport_home, relay_callback
    from tests.gateway.test_automation_notification_category import _authority
    from tests.gateway.test_automation_wake_route import _relay_adapter, _relay_source
    from hermes_state_runtime import get_session_admission
    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter, transport = _relay_adapter()
    runner.adapters = {Platform.RELAY: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    live = True

    def validate(self, event):
        return live and event.text == 'human' and event.metadata == {'relay_journal': 'j1'}

    monkeypatch.setattr(type(adapter), 'validate_native_metadata', validate, raising=False)
    try:
        human = MessageEvent(text='human', source=_relay_source('human'), message_id='human',
                             metadata={'relay_journal': 'j1'})
        with native_callback(runner, human, tmp_path), relay_callback(transport, human):
            receipt = await authority.admit_native(human)
        payload = get_session_admission(authority.db, admission_id=receipt.admission_id)['payload']
        live = False
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            restore_native(payload, runner)
        persisted_source = SessionSource.from_dict(human.source.to_dict())
        assert runner._committed_relay_adapter(persisted_source) is adapter
        transport._upgrade_secret = 'changed-connector-secret'
        assert runner._committed_relay_adapter(persisted_source) is None
    finally:
        authority.db.close()

"""Real Buzz intake -> base busy lane -> GatewayRunner auth -> blocking approval.
No authorization, command, session, or pending-store mocks. Only CLI transport
is faked; the model's work is a real blocking gateway approval wait.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.event import MessageEvent
from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision

buzz = load_plugin_adapter('buzz')
SELF = '1' * 64
USER = '2' * 64
CHANNEL = 'synthetic-control-channel'
ROOT = '3' * 64
PROMPT = '4' * 64


@pytest_asyncio.fixture
async def journey(tmp_path, monkeypatch, request):
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
    home = tmp_path / 'control-home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('GATEWAY_ALLOWED_USERS', USER)
    (home / 'config.yaml').write_text('buzz:\n  require_mention: true\n  thread_require_mention: true\napprovals:\n  timeout: 10\n')
    # The primary profile's allowlist lives in its own .env; authorization runs inside that scope.
    (home / '.env').write_text(f'GATEWAY_ALLOWED_USERS={USER}\n')
    config = GatewayConfig(sessions_dir=home / 'sessions')
    if getattr(request, 'param', None) == 'served-other':
        from gateway.profile_routing import parse_profile_routes
        other_home = home / 'profiles' / 'other'
        other_home.mkdir(parents=True)
        other_home.joinpath('config.yaml').write_text('{}\n')
        other_home.joinpath('.env').write_text(f'GATEWAY_ALLOWED_USERS={USER}\n')
        config.multiplex_profiles = True
        config.multiplex_profile_allowlist = ['other']
        config.profile_routes = parse_profile_routes([
            {'name': 'original', 'platform': 'buzz', 'chat_id': CHANNEL, 'profile': 'default'}])
    runner = GatewayRunner(config=config)
    adapter = buzz.BuzzAdapter(PlatformConfig(enabled=True))
    adapter._self_pubkey = SELF
    adapter._display_name = 'TestAgent'
    adapter._channel_names[CHANNEL] = 'Synthetic shared channel'
    adapter._channel_meta[CHANNEL] = {'name': 'Synthetic shared channel', 'description': 'Not DM'}
    state = adapter._new_channel_state('group')
    adapter._channel_state[CHANNEL] = state
    runner.adapters[adapter.platform] = adapter
    adapter.gateway_runner = runner
    adapter.set_session_store(runner.session_store)
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    # Match startup: install the real central authorization callback so authorized
    # transport awaits (including the revocation/removal probes) actually run.
    adapter.set_authorization_check(runner._make_adapter_auth_check(adapter.platform))
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic user')
    sent = []

    async def cli(args, **kwargs):
        if args[:2] == ['messages', 'send']:
            sent.append((args, kwargs.get('input_text')))
            return 0, json.dumps({'accepted': True, 'event_id': PROMPT if len(sent) == 1 else f'{len(sent):064x}'}), ''
        return 0, '{}', ''

    adapter._run_cli = cli
    source = adapter.build_source(chat_id=CHANNEL, chat_type='group', user_id=USER, thread_id=ROOT)
    key = runner._session_key_for_source(source)
    assert adapter._event_session_key(MessageEvent(text='/approve session', source=source)) == key
    assert runner._is_user_authorized_for_source(source)
    runner._session_state(key).turn.agent = _AGENT_PENDING_SENTINEL  # model boundary only
    guard = asyncio.Event()
    adapter._active_sessions[key] = guard
    adapter._session_tasks[key] = asyncio.current_task()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    prompt_tasks = []

    async def prompt(data):
        runner._pending_approvals[key] = data
        result = await adapter.send(CHANNEL, 'Approval required. Reply /approve session or /deny.', metadata={'thread_id': ROOT})
        assert result.success and result.message_id == PROMPT
        ready.set()

    def notify(data):
        prompt_tasks.append(asyncio.run_coroutine_threadsafe(prompt(data), loop))

    approval.register_gateway_notify(key, notify)
    waiter = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, key, notify,
        {'command': 'synthetic guarded operation', 'pattern_key': 'synthetic-control', 'pattern_keys': ['synthetic-control']}))
    adapter._session_tasks[key] = waiter
    try:
        await asyncio.wait_for(ready.wait(), 5)
        assert approval.has_blocking_approval(key)
        assert not waiter.done()
        assert adapter._lookup_event_meta(state, PROMPT)[0] == SELF
        yield adapter, runner, state, key, waiter, sent, guard
    finally:
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(waiter, 5)
        for task in prompt_tasks:
            task.result(timeout=1)
        adapter._active_sessions.clear()
        adapter._session_tasks.clear()
        if runner._session_db is not None:
            await runner._session_db.close()


async def emit(journey, text, *, user=USER, root=ROOT, parent=PROMPT, event_id='5' * 64):
    adapter, runner, state, key, waiter, sent, guard = journey
    event = {'id': event_id, 'kind': 9, 'pubkey': user, 'content': text, 'created_at': 1001,
             'tags': [['h', CHANNEL], ['e', root, '', 'root'], ['e', parent, '', 'reply']]}
    before = set(adapter._session_tasks.values())
    await adapter._handle_event(CHANNEL, state, event)
    spawned = set(adapter._session_tasks.values()) - before
    if spawned:
        await asyncio.wait_for(asyncio.gather(*spawned), 5)


@pytest.mark.asyncio
async def test_approve_session_resolves_real_wait_no_echo(journey):
    adapter, runner, state, key, waiter, sent, guard = journey
    await emit(journey, '/approve session')
    assert not approval.has_blocking_approval(key), 'unmentioned control must reach real pending resolver'
    decision = await asyncio.wait_for(asyncio.shield(waiter), 2)
    assert decision['resolved'] is True and decision['choice'] == 'session'
    assert adapter._active_sessions[key] is guard
    assert key not in adapter._pending_messages
    assert len(sent) == 2  # prompt and actual central slash confirmation


@pytest.mark.asyncio
async def test_deny_resolves_real_wait_no_echo(journey):
    adapter, runner, state, key, waiter, sent, guard = journey
    await emit(journey, '/deny')
    assert not approval.has_blocking_approval(key), 'unmentioned deny must reach real pending resolver'
    decision = await asyncio.wait_for(asyncio.shield(waiter), 2)
    assert decision['resolved'] is True and decision['choice'] == 'deny'
    assert adapter._active_sessions[key] is guard
    assert key not in adapter._pending_messages
    assert len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_unauthorized_sender_cannot_resolve(journey, command, caplog):
    adapter, runner, state, key, waiter, sent, guard = journey
    assert not adapter._allowed_pubkeys  # prove central denial, not local filtering
    await emit(journey, command, user='6' * 64)
    assert 'Unauthorized user:' in caplog.text
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 1
    assert key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_unknown_parent_cannot_admit_control(journey, command):
    adapter, runner, state, key, waiter, sent, guard = journey
    await emit(journey, command, parent='7' * 64)
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 1 and key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_wrong_thread_cannot_resolve_other_pending(journey, command):
    adapter, runner, state, key, waiter, sent, guard = journey
    await emit(journey, command, root='8' * 64)
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 2
    assert 'no pending' in sent[-1][1].lower()
    assert adapter._active_sessions[key] is guard


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_unserved_profile_cannot_resolve(journey, command, caplog):
    from gateway.profile_routing import parse_profile_routes
    adapter, runner, state, key, waiter, sent, guard = journey
    runner.config.multiplex_profiles = True
    runner.config.profile_routes = parse_profile_routes([
        {'name': 'synthetic-rejected', 'platform': 'buzz', 'chat_id': CHANNEL, 'profile': 'unserved'}])
    await emit(journey, command)
    assert 'unserved profile' in caplog.text
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_slash_policy_still_controls_authorized_sender(journey, command):
    adapter, runner, state, key, waiter, sent, guard = journey
    runner.config.platforms[adapter.platform] = PlatformConfig(extra={'group_allow_admin_from': ['9' * 64]})
    await emit(journey, command)
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 2 and 'admin-only' in sent[-1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['sure go ahead', 'yes', '/restart', '/approvals', '/approve-all', '/denylist'])
async def test_ordinary_and_unrelated_slash_reply_stays_mention_gated(journey, text):
    adapter, runner, state, key, waiter, sent, guard = journey
    await emit(journey, text)
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 1 and key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve always', '/deny'])
async def test_seeded_stale_history_does_not_resurrect_pending(journey, command):
    adapter, runner, state, key, waiter, sent, guard = journey
    approval.unregister_gateway_notify(key)
    decision = await asyncio.wait_for(waiter, 2)
    assert decision['choice'] is None
    runner._pending_approvals.pop(key, None)
    cli = adapter._run_cli
    async def history_cli(args, **kwargs):
        if args[:2] == ['messages', 'get']:
            return 0, json.dumps([{'id': PROMPT, 'pubkey': SELF, 'content': 'Old approval prompt',
                'created_at': 500, 'kind': 9, 'tags': [['h', CHANNEL], ['e', ROOT, '', 'root']]}]), ''
        return await cli(args, **kwargs)
    adapter._run_cli = history_cli
    await adapter._seed_channel(CHANNEL, 'group')
    seeded = adapter._channel_state[CHANNEL]
    assert adapter._lookup_event_meta(seeded, PROMPT)[0] == SELF
    await emit((adapter, runner, seeded, key, waiter, sent, guard), command)
    assert not approval.has_blocking_approval(key)
    assert len(sent) == 2 and 'no pending' in sent[-1][1].lower()
    assert not approval.is_approved(key, 'synthetic-control')


@pytest.mark.asyncio
async def test_revocation_between_intake_and_central_dispatch(journey, monkeypatch, caplog):
    adapter, runner, state, key, waiter, sent, guard = journey
    # Name lookup is a transport await after mention admission and before central
    # authorization. A concurrent operator revokes access while it is in flight.
    async def revoked_name(_):
        monkeypatch.setenv('GATEWAY_ALLOWED_USERS', '9' * 64)
        return 'Synthetic user'
    adapter._resolve_user_name = revoked_name
    await emit(journey, '/approve session')
    assert 'Unauthorized user:' in caplog.text
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_pending_removed_during_transport_await_is_not_approved(journey):
    adapter, runner, state, key, waiter, sent, guard = journey
    async def expiring_name(_):
        approval.unregister_gateway_notify(key)
        return 'Synthetic user'
    adapter._resolve_user_name = expiring_name
    await emit(journey, '/approve session')
    decision = await asyncio.wait_for(waiter, 2)
    assert decision['choice'] is None
    assert not approval.has_blocking_approval(key)
    assert len(sent) == 2 and 'expir' in sent[-1][1].lower()
    assert not approval.is_approved(key, 'synthetic-control')


@pytest.mark.asyncio
async def test_attachment_only_reply_is_not_a_control(journey):
    adapter, runner, state, key, waiter, sent, guard = journey
    await adapter._handle_event(CHANNEL, state, {
        'id': 'a' * 64, 'kind': 9, 'pubkey': USER, 'content': '', 'created_at': 1001,
        'tags': [['h', CHANNEL], ['e', ROOT, '', 'root'], ['e', PROMPT, '', 'reply'],
                 ['imeta', 'url https://synthetic.invalid/malformed-attachment']]})
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert len(sent) == 1 and key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_allow_gateway_control_false_cannot_resolve(journey, command):
    adapter, runner, state, key, waiter, sent, guard = journey
    source = adapter.build_source(chat_id=CHANNEL, chat_type='group', user_id=USER, thread_id=ROOT)
    assert runner._is_user_authorized_for_source(source)
    # This flag is internal event metadata, not something a Buzz wire event can
    # supply. Enter at the real base adapter boundary used by injected events.
    event = MessageEvent(text=command, source=source, internal=True,
                         allow_gateway_control=False)
    assert adapter._event_session_key(event) == key
    pending = runner._pending_approvals[key]
    await adapter.handle_message(event)
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert runner._pending_approvals[key] is pending
    assert adapter._active_sessions[key] is guard
    assert adapter._pending_messages[key] is event
    assert adapter._pending_messages[key].allow_gateway_control is False
    assert len(sent) == 1
    # The same real waiter must still be resolvable by an authorized control.
    await emit(journey, command)
    decision = await asyncio.wait_for(asyncio.shield(waiter), 2)
    assert decision['resolved'] is True
    assert decision['choice'] == ('session' if command.startswith('/approve') else 'deny')


@pytest.mark.asyncio
@pytest.mark.parametrize('journey', ['served-other'], indirect=True)
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
async def test_served_other_profile_cannot_resolve_original_pending(journey, command, caplog):
    from gateway.profile_routing import parse_profile_routes
    from gateway.run import _multiplex_profile_homes
    adapter, runner, state, key, waiter, sent, guard = journey
    assert set(dict(_multiplex_profile_homes(runner.config))) == {'default', 'other'}
    original = adapter.build_source(chat_id=CHANNEL, chat_type='group', user_id=USER, thread_id=ROOT)
    assert original.profile == 'default' and runner._session_key_for_source(original) == key
    pending = runner._pending_approvals[key]
    original_routes = runner.config.profile_routes
    runner.config.profile_routes = parse_profile_routes([
        {'name': 'served-other', 'platform': 'buzz', 'chat_id': CHANNEL, 'profile': 'other'}])
    other = adapter.build_source(chat_id=CHANNEL, chat_type='group', user_id=USER, thread_id=ROOT)
    assert other.profile == 'other' and not getattr(other, 'profile_route_rejected', False)
    assert runner._is_user_authorized_for_source(other)
    other_key = runner._session_key_for_source(other)
    assert adapter._event_session_key(MessageEvent(text=command, source=other)) == other_key
    assert other_key != key and not approval.has_blocking_approval(other_key)
    await emit(journey, command)
    assert 'unserved profile' not in caplog.text
    assert approval.has_blocking_approval(key) and not waiter.done()
    assert runner._pending_approvals[key] is pending
    assert adapter._active_sessions[key] is guard
    assert key not in adapter._pending_messages
    assert len(sent) == 2 and 'no pending' in sent[-1][1].lower()
    # Route the otherwise identical reply back to its owning served profile.
    runner.config.profile_routes = original_routes
    await emit(journey, command, event_id='b' * 64)
    decision = await asyncio.wait_for(asyncio.shield(waiter), 2)
    assert decision['resolved'] is True
    assert decision['choice'] == ('session' if command.startswith('/approve') else 'deny')

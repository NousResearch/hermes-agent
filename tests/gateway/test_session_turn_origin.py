"""Where a gateway turn came from (``turn_origin`` on ``pre_llm_call`` / ``post_llm_call``).

A gateway agent's ``sender_id`` is the session principal for every admitted turn, so a memory
plugin cannot tell from it whether that principal really sent the turn. Only ``native`` says so: a
synthetic or relayed turn (a goal continuation committed as plain native text, a heartbeat, an
agent-to-agent forward) must never read as native. The turns run through the real admission
drain (``execute_admission``); the handler reads what the turn's agent receives.
"""
import json
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner
from tests.gateway.test_surface_admission import _authority


def _admission_origins(monkeypatch, runner):
    """Patch the turn handler to record ``(event text, turn_origin, sender)`` inside the drain."""
    from gateway.session_ingress import admission_origin, executing_admission
    from gateway.session_results import execution_result
    seen = []

    async def handle(event):
        assert executing_admission.get() is True
        seen.append((event.text, admission_origin.get(), event.source.user_id))
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message', handle, raising=False)
    return seen


@pytest.mark.asyncio
async def test_messaging_turns_native_only_for_the_real_sender(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import admission_origin, execute_admission
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_completion_admission import pending

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))
    sid = receipt.ref.session_id
    # The non-LOCAL goal continuation (run_goals ``_enqueue_fifo``): a non-internal synthetic event
    # that commits as plain ``native_text_v1`` with only ``inherit_channel_pin`` set.
    continuation = GatewayRunner._synthetic_prompt_event(source, 'keep going on the goal')
    continuation.message_id = 'goal-continuation'
    await _admit_human(runner, authority, continuation)
    # A delegation completion: a producer wake committed on the chat route (an ``automation`` envelope).
    key = runner.session_store._generate_session_key(source)
    assert await runner._deliver_async_delegation_group([dict(pending(key, 'ci-exec'), parent_session_id=sid)])
    seen = _admission_origins(monkeypatch, runner)
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    for row in rows:
        await execute_admission(authority, SessionRef(authority.profile_id, sid), row)

    kinds = ['automation' if 'automation' in row['payload']['native_text_v1'] else row['request_id'] for row in rows]
    assert dict(zip(kinds, [origin for _, origin, _ in seen])) == {
        'human': 'native', 'goal-continuation': 'automation', 'automation': 'automation'}
    assert {sender for _, _, sender in seen} == {'42'}  # one sender_id, three origins
    assert admission_origin.get() is None  # reset after each drain


@pytest.mark.asyncio
@pytest.mark.parametrize('orphan,origin', [('goal', 'automation'), ('bot', 'peer'), ('human', 'native')])
async def test_a_rescued_fifo_orphan_runs_with_its_own_origin(tmp_path, monkeypatch, orphan, origin):
    """The FIFO orphan rescue (#99882) runs the oldest orphaned overflow event as an admitted row's
    turn: that turn takes the orphan's origin, never the parked human row's ``native``."""
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))
    key = runner.session_store._generate_session_key(source)
    runner._session_state(key).conversation.queued_events.append({
        'goal': lambda: GatewayRunner._synthetic_prompt_event(source, 'orphan'),
        'bot': lambda: MessageEvent(text='orphan', source=replace(source, is_bot=True), message_id='o'),
        'human': lambda: MessageEvent(text='orphan', source=source, message_id='o')}[orphan]())
    seen = []

    async def with_agent(event, src, quick_key, run_generation):
        seen.append(event.text)
        _recording_turn(seen, turn_origin='native')  # at depth 0 the runner reads the admission's origin
        execution_result.get()['result'] = {'final_response': 'ok'}
        return {'final_response': 'ok'}
    monkeypatch.setattr(runner, '_handle_message_with_agent', with_agent, raising=False)
    sid = receipt.ref.session_id
    [row] = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    await execute_admission(authority, SessionRef(authority.profile_id, sid), row)
    assert seen == ['orphan', origin]


@pytest.mark.asyncio
@pytest.mark.parametrize('row,parked', [('native', 'native'), ('viewer', None), ('cli', None)])
async def test_the_rescue_keeps_a_parked_rows_input_off_native(tmp_path, monkeypatch, row, parked):
    """The row's own input waits behind the orphan and later runs on its own: a viewer's submit, which
    carries the session's source, must not then read as that source's platform message."""
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_ingress import admission_origin, executing_admission
    from gateway.session_turn_origin import direct_turn_origin

    runner, _, _, source = await _discord_owner(tmp_path, monkeypatch)
    key = runner.session_store._generate_session_key(source)
    runner._session_state(key).conversation.queued_events.append(
        GatewayRunner._synthetic_prompt_event(source, 'orphan'))
    incoming = MessageEvent(text='row input', source=source, message_id='row')
    tokens = executing_admission.set(True), admission_origin.set(row)
    try:
        ran, _, _ = runner._hm_rescue_orphaned_fifo(incoming, source, False, key)
        assert (ran.text, admission_origin.get()) == ('orphan', 'automation')
    finally:
        executing_admission.reset(tokens[0])
        admission_origin.reset(tokens[1])
    assert direct_turn_origin(incoming) == parked


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/queue', '/steer'])
async def test_a_busy_command_keeps_the_borrowed_source_mark(command):
    """A leftover steer or a /retry replay that holds a /queue or /steer is rebuilt by the busy handler."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    queued = []

    async def queue_turn(quick_key, event, turn, adapter, reply):
        queued.append(turn)
        return True
    runner._delivery_adapter_for = lambda source: object()
    runner._queue_command_turn = queue_turn
    runner._peek_session_state = lambda key: None
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', user_id='human')
    event = MessageEvent(text=f'{command} replayed text', source=source, borrowed_source=True)
    handler = runner._busy_queue_command if command == '/queue' else runner._busy_steer_command
    assert await handler(event, 'key', source) is None
    assert [(turn.text, turn.borrowed_source) for turn in queued] == [('replayed text', True)]


@pytest.mark.asyncio
async def test_a_platform_message_from_a_bot_is_peer(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from hermes_state_runtime import list_session_admissions

    runner, adapter, authority, _ = await _discord_owner(tmp_path, monkeypatch)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42', is_bot=True)
    receipt = await _admit_human(runner, authority, MessageEvent(text='from a bot', source=source, message_id='bot'))
    seen = _admission_origins(monkeypatch, runner)
    sid = receipt.ref.session_id
    [row] = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    await execute_admission(authority, SessionRef(authority.profile_id, sid), row)
    assert seen == [('from a bot', 'peer', '42')]


@pytest.mark.parametrize('platform,user_id,is_bot,origin', [
    ('discord', '42', False, 'native'), ('discord', '42', True, 'peer'), ('a2a', 'peer-agent', False, 'peer'),
    ('webhook', 'webhook:ci', False, None), ('ntfy', 'alerts', False, None), ('telegram', None, False, None),
], ids=['account', 'bot-account', 'a2a-platform', 'webhook-route', 'ntfy-topic', 'no-sender'])
def test_a_platform_message_is_native_only_when_it_names_a_human_sender(platform, user_id, is_bot, origin):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_turn_origin import direct_turn_origin, row_turn_origin
    source = SessionSource(platform=Platform(platform), chat_id='chat', user_id=user_id, is_bot=is_bot)
    event = MessageEvent(text='hi', source=source)
    row = {'request_id': 'r', 'payload': {'text': 'hi', 'native_text_v1': {'event': {}}}}
    assert row_turn_origin(row, SimpleNamespace(source=source), None, event) == origin
    assert direct_turn_origin(event) == origin


@pytest.mark.asyncio
async def test_text_rebuilt_on_a_borrowed_source_is_never_native(tmp_path, monkeypatch):
    """A /retry replay or a leftover /steer re-sends text on a source that did not write it; the
    marker survives native admission."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from hermes_state_runtime import list_session_admissions

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='an earlier turn', source=source,
                                                                 borrowed_source=True))
    seen = _admission_origins(monkeypatch, runner)
    sid = receipt.ref.session_id
    [row] = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert row['payload']['native_text_v1']['event']['borrowed_source'] is True
    await execute_admission(authority, SessionRef(authority.profile_id, sid), row)
    assert seen == [('an earlier turn', None, '42')]


@pytest.mark.asyncio
async def test_a_leftover_steer_is_rebuilt_on_a_borrowed_source():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, build_session_key
    from tests.gateway.test_pending_steer_drain import Adapter

    runner = object.__new__(GatewayRunner)
    runner._draining = False
    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', user_id='human', chat_type='dm')
    key = build_session_key(source)
    runner._merge_leftover_steer({'pending_steer': 'late steer'}, adapter, source, key, None, 'queued text')
    late = adapter._pending_messages.pop(key)
    queued = MessageEvent(text='queued event', source=source, message_id='q')
    runner._defer_past_exhaustion_reset({'pending_steer': 'steer before reset'}, adapter, source, key, queued)
    events = [late, adapter._pending_messages[key], *runner._session_state(key).conversation.queued_events]
    assert [(event.text, event.borrowed_source) for event in events] == [
        ('late steer', True), ('steer before reset', True), ('queued event', False)]


@pytest.mark.asyncio
async def test_viewer_cli_hidden_and_bot_submits(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    authority = await _authority(tmp_path, monkeypatch, None)
    seen = _admission_origins(monkeypatch, authority.runner)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    bot = {'id': 'bot:peer', 'name': 'Peer', 'is_bot': True}
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        for rid, extra in (('viewer', {}), ('cli', {'clarify_batch': True}),
                           ('hidden', {'display_kind': 'hidden'}), ('relayed', {'turn_author': bot})):
            reply = await connection.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
                'session_id': 's', 'submission_id': rid, 'text': rid, **extra}})
            assert reply.get('result', {}).get('status') == 'queued', reply
            await authority.sessions['s'].task
    finally:
        await connection.close()

    assert [(text, origin) for text, origin, _ in seen] == [
        ('viewer', 'viewer'), ('cli', 'cli'), ('hidden', 'automation'), ('relayed', 'peer')]
    # The viewer's turn runs as the session's own principal, like a platform message would.
    assert {sender for _, _, sender in seen} == {authority.sessions['s'].source.user_id}


@pytest.mark.asyncio
async def test_api_turns_are_api_unless_a_bot_wrote_them(api, owner, monkeypatch):
    from gateway.session_api_turn import admit_api_turn
    from gateway.session_ingress import execute_admission

    seen = _admission_origins(monkeypatch, owner.runner)
    for sid, author in (('client', None), ('peer-run', {'id': 'bot:peer', 'name': 'Peer', 'is_bot': True})):
        _, ref, row = admit_api_turn(api, session_id=sid, user_message=sid, conversation_history=[],
                                     **({'turn_author': author} if author else {}))
        await execute_admission(owner, ref, row)
    assert [(text, origin) for text, origin, _ in seen] == [('client', 'api'), ('peer-run', 'peer')]


@pytest.mark.asyncio
async def test_local_producer_admissions_are_automation(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_automation import admit_route_wake
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state import SessionDB
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner', db=db)
    seen = _admission_origins(monkeypatch, runner)
    try:
        viewer = Principal('test-owner', 'default', frozenset({'session:create', 'session:submit'}), 'viewer')
        ref = create_local_session(authority, viewer, dict(request_id='cli', source='cli', model='fixture', toolsets=[]))
        live = authority.sessions[ref.session_id]
        entry = runner.session_store.lookup_by_session_key(live.route)
        adapter = runner._adapter_for_source(live.source)
        assert await runner._admit_plugin_injection(
            adapter, live.source, session_key=live.route, session_id=entry.session_id,
            content='plugin injection', plugin_id='desk') is True
        assert await admit_route_wake(runner, adapter, live.source, 'loop tick', 'loop-tick') is True
        await async_wait_until(lambda: len(seen) == 2, timeout=10, what='both producer turns')
    finally:
        db.close()
    assert [(text, origin) for text, origin, _ in seen] == [
        ('plugin injection', 'automation'), ('loop tick', 'automation')]


def _a2a_policy():
    request = {'source': 'a2a', 'a2a_identity': ['https://peer.example', 'tenant', 'peer-agent', 'key']}
    return SimpleNamespace(source='a2a', platform='a2a', request_json=json.dumps(request))


@pytest.mark.parametrize('request_id,payload,policy,origin', [
    ('fwd', {'text': 'hi'}, _a2a_policy(), 'peer'),
    ('bot:dm', {'text': 'hi', 'local_automation_v1': {'identity': 'bot:dm', 'turn_author': {'id': 'bot:x'}}},
     None, 'peer'),
    ('wake', {'text': 'hi', 'local_automation_v1': {'identity': 'wake'}}, None, 'automation'),
    ('card', {'text': 'hi', 'kanban_card_v1': {'task': 't1'}}, None, 'automation'),
    ('onboarding-opening:r', {'text': 'hi', 'display_v1': {'kind': 'hidden'}}, None, 'automation'),
    ('hosted:[1]', {'text': 'room task'}, None, None),
], ids=['a2a-forward', 'bot-dm', 'local-automation', 'kanban-card', 'onboarding-opening', 'hosted-room'])
def test_row_turn_origin_for_rows_without_a_cheap_live_route(request_id, payload, policy, origin):
    from gateway.config import Platform
    from gateway.session_turn_origin import row_turn_origin
    live = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL))
    event = SimpleNamespace(internal='local_automation_v1' in payload)
    assert row_turn_origin({'request_id': request_id, 'payload': payload}, live, policy, event) == origin


def _recording_turn(agent_kwargs, **ctx_fields):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    def run(message, *, turn_origin='unset', **kwargs):
        agent_kwargs.append(turn_origin)
        return {'final_response': 'ok'}
    turn = TurnRunner(SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: []),
                      TurnContext(message='hi', session_id='s', session_key='k',
                                  source=SimpleNamespace(user_id='42', user_name='Owner', is_bot=False), **ctx_fields))
    turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run), [], None, None, None)


def test_the_runner_passes_the_admitted_origin_else_the_direct_one():
    from gateway.session_ingress import admission_origin, executing_admission
    seen = []
    _recording_turn(seen, turn_origin='native')  # a direct turn: TurnContext's own origin
    tokens = executing_admission.set(True), admission_origin.set('viewer')
    try:
        _recording_turn(seen, turn_origin='native')  # the admitted row wins
        _recording_turn(seen, _interrupt_depth=1)  # an in-band follow-up is not that row
    finally:
        executing_admission.reset(tokens[0])
        admission_origin.reset(tokens[1])
    assert seen == ['native', 'viewer', None]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,origin', [
    ('moa', 'native'), ('internal', 'automation'), ('heartbeat', 'automation'), ('goal', 'automation'),
    ('bot', 'peer'),
])
async def test_direct_turns_reach_the_agent_with_their_origin(tmp_path, monkeypatch, kind, origin):
    """No admission (``/moa``, ``/model --once`` or a gateway without a session authority): the real
    turn preparation and ``_run_agent_inner`` carry the origin to the runner."""
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource
    from gateway.turn_context import TurnContext

    monkeypatch.setattr('gateway.run._load_gateway_config', dict)
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', dict)
    source = SessionSource(platform=Platform.DISCORD, chat_id='channel', user_id='user', user_name='Example')
    event = {'moa': lambda: MessageEvent(text='a /moa turn', source=source, message_id='m1'),
             'internal': lambda: MessageEvent(text='a process notice', source=source, internal=True),
             'heartbeat': lambda: GatewayRunner._synthetic_prompt_event(source, 'heartbeat due'),
             'goal': lambda: GatewayRunner._synthetic_prompt_event(source, 'keep going'),
             'bot': lambda: MessageEvent(text='from a bot', source=replace(source, is_bot=True), message_id='m2')}[kind]()
    if kind == 'heartbeat':
        event._heartbeat_session_id = 'origin-test'
    entry = SessionEntry(session_id='origin-test', session_key='discord:origin-test',
                         created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config, runner.adapters, runner.session_store = GatewayConfig(), {}, object()
    runner._get_proxy_url = lambda: None
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, entry, entry.session_key))
    runner._hmwa_open_session = AsyncMock(return_value=(False, True))
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ''
    runner._rehydrate_prompt_pins = AsyncMock()
    runner._persist_prompt_pins = AsyncMock()
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, load_transcript=AsyncMock(return_value=[]))
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._consume_pending_native_image_paths = lambda key: []
    runner._adapter_for_source = lambda source: None
    runner._bind_adapter_run_generation = lambda *args: None
    runner._run_agent_display_settings = lambda source: GatewayRunner._RunAgentDisplay(
        platform_key='discord', resolve_display_setting=lambda *args: False,
        **{name: getattr(TurnContext(), name) for name in runner._DISPLAY_TO_TURN_CTX if name != 'resolve_display_setting'})
    seen = []

    class _Observed(BaseException):
        pass

    def run(message, *, turn_origin='unset', **kwargs):
        seen.append(turn_origin)
        raise _Observed

    def run_without_delivery(ctx, worker, *args):
        worker._native_image_run_message = lambda: ctx.message
        worker._run_conversation_with_approval(SimpleNamespace(run_conversation=run), [], None, None, None)

    runner._run_agent_bind_turn_wiring = run_without_delivery
    runner._run_agent = runner._run_agent_inner
    with pytest.raises(_Observed):
        await runner._handle_message_with_agent(event, source, entry.session_key, 1)
    assert seen == [origin]

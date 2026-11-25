"""``turn_origin()`` names a Hermes producer only where that producer said so.

Execution trust (``internal``) is not an origin: a Bot Chat delivery, a restart's resume of a
person's interrupted turn and a late /steer drained after a wake all run ``internal`` or under a
wake's event, and read ``None``. A process notification names itself at its producer, and the name
is read back from the committed admission's identity at restore; any other row restores as ``None``.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import build_session_context
from gateway.session_contract import SessionActor
from tests.gateway.test_queued_followup_processing_hooks import HookRecordingAdapter, _install_fake_agent, _make_runner
from tests.waiting import async_wait_until

PERSON = {'id': 'uid:1000', 'name': 'Mark', 'is_bot': False}
BOT = {'id': 'bot:cto', 'name': 'cto', 'is_bot': True}
PEER_DM_TRANSPORT = 'api-peer-dm'  # the actor ``hermes peer dm`` is admitted as (``api_server_bot_chat``)


@pytest_asyncio.fixture
async def bot_chat(tmp_path, monkeypatch):
    """A real runner's local Bot Chat; its FIFO drain (``execute_admission``) restores each committed
    row and hands it to ``_handle_message``, which binds the turn like the turn runner and reads it."""
    from gateway import run
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    from hermes_cli.plugin_session import turn_origin

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_a: {'platform_toolsets': {'cli': []}})
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    owner = SessionActor('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'bot', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Bot Chat')
    state = SimpleNamespace(seen=[], authority=authority, chat=chat, home=tmp_path, owner=owner)

    async def answer(event):
        entry = runner.session_store.lookup_by_session_key(authority.sessions[chat.session_id].route)
        tokens = runner._set_session_env(build_session_context(event.source, runner.config, entry, event))
        try:
            state.seen.append((event.message_id, event.internal, turn_origin()))
        finally:
            runner._clear_session_env(tokens)
        return 'pong'

    monkeypatch.setattr(runner, '_handle_message', answer)
    try:
        yield state
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_every_bot_chat_delivery_reads_no_origin(bot_chat):
    """A peer DM, an authored DM and an unauthored producer report all run ``internal`` for trust; none
    is a process notification or a delegation result."""
    from gateway.session_bot import deliver
    cases = {'a' * 32: (PEER_DM_TRANSPORT, None), 'b' * 32: (PEER_DM_TRANSPORT, PERSON),
             'c' * 32: ('native', PERSON), 'd' * 32: ('native', BOT), 'e' * 32: ('native', None)}
    for key, (transport, author) in cases.items():
        actor = SessionActor('uid:1000', 'default', bot_chat.owner.capabilities, transport)
        params = dict(id=key, profile='default', message='msg ' + key)
        if author is not None:
            params['author'] = author
        await deliver(SimpleNamespace(authority=bot_chat.authority, actor=actor), params)
    await async_wait_until(lambda: len(bot_chat.seen) == len(cases), timeout=10, what='every delivery to run')
    assert sorted(bot_chat.seen) == [('bot:' + key, True, None) for key in sorted(cases)]


def test_a_row_restores_the_origin_its_identity_names():
    """The origin is read from the identity every committed row already carries, an older build's
    included: no descriptor field, so no retry fingerprint changes."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_automation import restore_origin
    source = SessionSource(platform=Platform.LOCAL, chat_id='c')

    def read(descriptor):
        return restore_origin(MessageEvent(text='x', source=source, internal=True), descriptor).turn_origin()
    assert read({'identity': '["async_delegation","d1",""]'}) == 'delegation_result'
    assert read({'identity': '["async_delegation","d1","task_failure:0"]'}) == 'delegation_result'
    for kind in ('completion', 'watch_match', 'watch_disabled'):
        assert read({'identity': f'["{kind}","proc_1",1.0]'}) == 'process_notification', kind
    assert read({'identity': '["heartbeat","proc_1","e1"]', 'tick': {'output': '', 'omitted': 0}}) == 'process_notification'
    for other in ({}, {'identity': '["heartbeat","s1",1.0,2]', 'heartbeat': 's1'}, {'identity': '["heartbeat","s","e"]'},
                  {'identity': '["cron","job",1]'}, {'identity': '["loop-tick","k","u"]'},
                  {'identity': '["plugin","p","k","u"]'}, {'identity': 'bot:' + 'a' * 32}, {'identity': 'notice'},
                  {'identity': '{"completion": 1}'}, {'turn_origin': 'process_notification'}, {'identity': 7}):
        assert read(other) is None, other


@pytest.mark.asyncio
async def test_a_pre_upgrade_watch_notification_retry_is_acknowledged(tmp_path, monkeypatch):
    """A watch notification an older build admitted (its producer named no origin) and retried after
    the upgrade with the identical event matches the committed row: one row, unchanged, and it
    restores as the process notification it is."""
    from gateway.session_authority import admission_row
    from gateway.session_envelope import restore_native
    from tests.gateway.test_automation_notification_category import _native_owner
    runner, authority, _adapter, _, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        evt = {'type': 'watch_match', 'event_id': 'w1', 'session_id': 'proc_x', 'session_key': metadata['gateway_session_key'],
               'platform': 'discord', 'chat_id': '42', 'chat_type': 'dm', 'exit_code': 0, 'output': 'done',
               'started_at': 1.0, 'parent_session_id': metadata['gateway_session_id']}
        with monkeypatch.context() as old_build:
            old_build.setattr(MessageEvent, 'set_turn_origin', lambda self, origin: None)
            assert await runner._inject_watch_notification('[done]', dict(evt), raise_not_accepted=True) is True
        owner = authority.logical_owner(metadata['gateway_session_id'])

        def rows():
            return [r for r in authority.persistence.admissions.list(session_id=owner, pending_only=False)
                    if (r.payload.get('native_text_v1') or {}).get('automation')]
        [first] = rows()
        assert await runner._inject_watch_notification('[done]', dict(evt), raise_not_accepted=True) is True
        [again] = rows()
        assert (again.admission_id, again.payload) == (first.admission_id, first.payload)
        assert 'turn_origin' not in again.payload['native_text_v1']['automation']
        row = admission_row(authority.persistence.admissions.get(admission_id=again.admission_id))
        assert restore_native(row['payload'], runner).turn_origin() == 'process_notification'
    finally:
        runner.session_store.close_all_db_handles()


KEY = "agent:main:telegram:group:-100"


class _SteeredAgent:
    """The wake's turn ends with a /steer that arrived after its last tool batch."""
    calls: list = []
    first: dict = {}

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kw):
        from hermes_cli.plugin_session import turn_origin
        type(self).calls.append((message.rsplit("] ", 1)[-1], turn_origin()))
        result = {"final_response": "ok", "messages": [], "api_calls": 1}
        if len(type(self).calls) == 1:
            result.update(type(self).first)
        return result


def _source():
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="111", user_name="Alice")


def _wake():
    event = MessageEvent(text="[process done]", source=_source(), message_id="wake", message_type=MessageType.TEXT,
                         internal=True)
    event.set_turn_origin("process_notification")
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [None, "person", "wake"])
async def test_a_follow_up_after_a_wake_reads_its_own_origin(monkeypatch, tmp_path, queued):
    """The late /steer runs as the next turn: the person's text, not the wake's. A message queued
    behind the wake (/queue) reads its own event either way."""
    from hermes_cli.plugin_session import turn_origin
    _SteeredAgent.calls, _SteeredAgent.first = [], {"pending_steer": "typed after last tool"}
    _install_fake_agent(monkeypatch, tmp_path, _SteeredAgent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    wake = _wake()
    if queued is not None:
        follow = MessageEvent(text="queued " + queued, source=wake.source, message_id="queued",
                              message_type=MessageType.TEXT, internal=queued == "wake")
        if queued == "wake":
            follow.set_turn_origin("delegation_result")
        adapter._pending_messages[KEY] = follow
    tokens = runner._set_session_env(build_session_context(wake.source, GatewayConfig(), event=wake))
    try:
        await runner._run_agent(message="[process done]", context_prompt="", history=[], source=wake.source,
                                session_id="s1", session_key=KEY)
        after = turn_origin()
    finally:
        runner._clear_session_env(tokens)
    # The steer runs first; the queued message keeps its place right behind it (_merge_leftover_steer).
    expected = [("[process done]", "process_notification"), ("typed after last tool", None)]
    if queued is not None:
        expected.append(("queued " + queued, "delegation_result" if queued == "wake" else None))
    assert _SteeredAgent.calls == expected
    assert after == "process_notification"  # the wake's own binding is back once its follow-ups ran


@pytest.mark.asyncio
async def test_interrupt_text_run_as_the_next_turn_after_a_wake_is_not_the_wake(monkeypatch, tmp_path):
    _SteeredAgent.calls = []
    _SteeredAgent.first = {"interrupted": True, "interrupt_message": "do this instead"}
    _install_fake_agent(monkeypatch, tmp_path, _SteeredAgent)
    runner = _make_runner(HookRecordingAdapter())
    wake = _wake()
    tokens = runner._set_session_env(build_session_context(wake.source, GatewayConfig(), event=wake))
    try:
        await runner._run_agent(message="[process done]", context_prompt="", history=[], source=wake.source,
                                session_id="s1", session_key=KEY)
    finally:
        runner._clear_session_env(tokens)
    assert _SteeredAgent.calls == [("[process done]", "process_notification"), ("do this instead", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [None, "process_notification"])
async def test_a_restart_resume_of_an_interrupted_turn_reads_no_origin(named):
    """The restart resume (``_schedule_startup_auto_resume``) re-runs a person's interrupted turn as an
    empty ``internal`` event through ``_run_startup_resume_event``: no producer named it, so the turn
    the adapter hands on reads ``None``. The ``named`` case shows the same path keeps an origin a
    producer did set."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_origin
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    seen = []

    async def handler(event):
        tokens = set_session_vars(platform="telegram", chat_id="-100", session_key=KEY, session_id="s1",
                                  turn_event=event)
        try:
            seen.append((event.internal, turn_origin()))
        finally:
            clear_session_vars(tokens)

    adapter.set_message_handler(handler)
    event = MessageEvent(text="", message_type=MessageType.TEXT, source=_source(), internal=True)
    event.set_turn_origin(named)
    await runner._run_startup_resume_event(adapter, event, KEY)
    await async_wait_until(lambda: seen, timeout=10, what="the resumed turn")
    assert seen == [(True, named)]

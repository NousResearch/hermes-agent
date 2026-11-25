"""``turn_internal()`` reads the origin of a turn's input, not its execution trust.

A Bot Chat delivery is admitted ``internal`` for trust, yet a ``hermes peer dm`` or an authored
DM is a message someone sent; only an unauthored producer report (cron job output) is Hermes'
own. The origin is recorded by the producer and survives the committed admission's restore and
its one retry. A late /steer drained as the next turn after a wake runs under the wake's event
but is the person's text.
"""
import asyncio
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
TRANSIENT = 'Error code: 429 - rate limit exceeded'
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
    from gateway.session_results import execution_result
    from hermes_cli.plugin_session import turn_internal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_a: {'platform_toolsets': {'cli': []}})
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    owner = SessionActor('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'bot', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Bot Chat')
    state = SimpleNamespace(seen=[], fail=[], gate=None, authority=authority, chat=chat, home=tmp_path, owner=owner)

    async def answer(event):
        entry = runner.session_store.lookup_by_session_key(authority.sessions[chat.session_id].route)
        tokens = runner._set_session_env(build_session_context(event.source, runner.config, entry, event))
        try:
            state.seen.append((event.message_id, turn_internal()))
        finally:
            runner._clear_session_env(tokens)
        if state.gate is not None:
            await state.gate.wait()
        if state.fail:
            execution_result.get()['result'] = {'final_response': '', 'messages': [], 'failed': True,
                                                'error': state.fail.pop(0)}
            return ''
        return 'pong'

    monkeypatch.setattr(runner, '_handle_message', answer)
    try:
        yield state
    finally:
        runner.session_store.close_all_db_handles()


async def _deliver(state, key, transport='native', author=None):
    from gateway.session_bot import deliver
    actor = SessionActor('uid:1000', 'default', state.owner.capabilities, transport)
    params = dict(id=key, profile='default', message='msg ' + key)
    if author is not None:
        params['author'] = author
    return await deliver(SimpleNamespace(authority=state.authority, actor=actor), params)


@pytest.mark.asyncio
async def test_a_peer_dm_and_an_authored_delivery_read_false_and_a_producer_report_true(bot_chat):
    cases = {'a' * 32: (PEER_DM_TRANSPORT, None), 'b' * 32: (PEER_DM_TRANSPORT, PERSON),
             'c' * 32: ('native', PERSON), 'd' * 32: ('native', BOT), 'e' * 32: ('native', None)}
    for key, (transport, author) in cases.items():
        await _deliver(bot_chat, key, transport, author)
    await async_wait_until(lambda: len(bot_chat.seen) == len(cases), timeout=10, what='every delivery to run')
    assert dict(bot_chat.seen) == {'bot:' + 'a' * 32: False, 'bot:' + 'b' * 32: False, 'bot:' + 'c' * 32: False,
                                   'bot:' + 'd' * 32: False, 'bot:' + 'e' * 32: True}


async def _retried(state, key, transport, strip_origin=False):
    """The delivery fails transiently once; its one retry is admitted from the receipt."""
    from gateway.session_bot_mailbox import mailbox_lock
    from tools.bot_live_delivery import _locked, _read, _write
    state.fail[:] = [TRANSIENT]
    state.gate = asyncio.Event()
    await _deliver(state, key, transport)
    if strip_origin:  # a receipt written before the field existed (``write_receipt`` merges, so rewrite)
        async with mailbox_lock(state.authority):
            with _locked(state.home) as root:
                record = _read(root / f'{key}.json')
                assert record is not None and record.pop('started_by_hermes') is False
                _write(root / f'{key}.json', record)
    state.gate.set()
    await async_wait_until(lambda: len(state.seen) == 2, timeout=10, what='the delivery and its retry')
    return state.seen


@pytest.mark.asyncio
async def test_a_peer_dm_s_retry_keeps_its_origin(bot_chat):
    key = 'f' * 32
    assert await _retried(bot_chat, key, PEER_DM_TRANSPORT) == [('bot:' + key, False), ('bot:' + key + ':retry', False)]


@pytest.mark.asyncio
async def test_a_producer_report_s_retry_stays_hermes_own(bot_chat):
    key = '1' * 32
    assert await _retried(bot_chat, key, 'native') == [('bot:' + key, True), ('bot:' + key + ':retry', True)]


@pytest.mark.asyncio
async def test_a_receipt_that_recorded_no_origin_retries_as_unknown_never_true(bot_chat):
    key = '2' * 32
    assert await _retried(bot_chat, key, PEER_DM_TRANSPORT, strip_origin=True) == [
        ('bot:' + key, False), ('bot:' + key + ':retry', None)]


@pytest.mark.asyncio
async def test_a_bot_chat_row_queued_before_the_origin_field_reads_unknown(bot_chat, monkeypatch):
    """A row the pre-field writer committed carries neither ``started_by_hermes`` nor a
    ``turn_author``: restored and drained on this code, it cannot tell a peer DM from a producer
    report, so it reads unknown, never True. It still runs once."""
    from gateway import session_automation
    monkeypatch.setattr(session_automation, '_origin_descriptor', lambda *_a, **_k: {})  # the old writer
    bot_chat.gate = asyncio.Event()
    await _deliver(bot_chat, '3' * 32, PEER_DM_TRANSPORT)
    await _deliver(bot_chat, '4' * 32, 'native')
    bot_chat.gate.set()
    await async_wait_until(lambda: len(bot_chat.seen) == 2, timeout=10, what='both old rows to run')
    assert bot_chat.seen == [('bot:' + '3' * 32, None), ('bot:' + '4' * 32, None)]


def test_an_old_row_of_another_producer_still_reads_hermes_own():
    """Only a Bot Chat row is read as unknown without the field: a cron, completion, delegation,
    heartbeat or route wake row never recorded it and stays Hermes' own (True)."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_automation import restore_origin
    from gateway.session import SessionSource
    from gateway.config import Platform
    source = SessionSource(platform=Platform.LOCAL, chat_id='c')
    for identity in ('["cron","job",1]', '["heartbeat","s",0,1]'):
        event = restore_origin(MessageEvent(text='x', source=source, internal=True), {'identity': identity})
        assert event.started_by_hermes() is True
    event = restore_origin(MessageEvent(text='x', source=source, internal=True), {'identity': 'bot:' + '5' * 32})
    assert event.started_by_hermes() is None


KEY = "agent:main:telegram:group:-100"


class _SteeredAgent:
    """The wake's turn ends with a /steer that arrived after its last tool batch."""
    calls: list = []
    first: dict = {}

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kw):
        from hermes_cli.plugin_session import turn_internal
        type(self).calls.append((message.rsplit("] ", 1)[-1], turn_internal()))
        result = {"final_response": "ok", "messages": [], "api_calls": 1}
        if len(type(self).calls) == 1:
            result.update(type(self).first)
        return result


def _wake():
    from gateway.config import Platform
    from gateway.session import SessionSource
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="111", user_name="Alice")
    return MessageEvent(text="[process done]", source=source, message_id="wake", message_type=MessageType.TEXT,
                        internal=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [None, "person", "wake"])
async def test_a_follow_up_after_a_wake_reads_its_own_origin(monkeypatch, tmp_path, queued):
    """The late /steer runs as the next turn: the person's text, not the wake's. A message queued
    behind the wake (/queue) reads its own event either way."""
    from hermes_cli.plugin_session import turn_internal
    _SteeredAgent.calls, _SteeredAgent.first = [], {"pending_steer": "typed after last tool"}
    _install_fake_agent(monkeypatch, tmp_path, _SteeredAgent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    wake = _wake()
    if queued is not None:
        follow = MessageEvent(text="queued " + queued, source=wake.source, message_id="queued",
                              message_type=MessageType.TEXT, internal=queued == "wake")
        adapter._pending_messages[KEY] = follow
    tokens = runner._set_session_env(build_session_context(wake.source, GatewayConfig(), event=wake))
    try:
        await runner._run_agent(message="[process done]", context_prompt="", history=[], source=wake.source,
                                session_id="s1", session_key=KEY)
        after = turn_internal()
    finally:
        runner._clear_session_env(tokens)
    # The steer runs first; the queued message keeps its place right behind it (_merge_leftover_steer).
    expected = [("[process done]", True), ("typed after last tool", False)]
    if queued is not None:
        expected.append(("queued " + queued, queued == "wake"))
    assert _SteeredAgent.calls == expected
    assert after is True  # the wake's own binding is back once its follow-ups ran


@pytest.mark.asyncio
async def test_interrupt_text_run_as_the_next_turn_after_a_wake_is_not_the_wake(monkeypatch, tmp_path):
    """The drain keeps only the interrupting text, not who sent it: unknown, never the wake's True."""
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
    assert _SteeredAgent.calls == [("[process done]", True), ("do this instead", None)]

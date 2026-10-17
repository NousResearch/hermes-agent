"""A slash command that becomes a turn (``/queue``, ``/steer``, ``/plan``, a skill) joins the chat's
durable FIFO like text, so it runs after the turns already admitted, never inside the running one.

Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite. The turn body
(``_handle_message_with_agent``) is a stand-in; the running turn calls the real post-run drain
step ``_run_agent_drain_pending``, so anything left in the in-process slot shows up "inside A".
"""
import asyncio
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


def _owner(monkeypatch, mode=None):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    if mode is not None:
        runner._busy_input_mode = mode
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id=f's{len(sent)}')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    monkeypatch.setattr(runner, '_read_user_config', lambda: {'approvals': {'destructive_slash_confirm': False}})
    return runner, adapter, sent


class _Agent:
    interrupted = False

    def interrupt(self, *args, **kwargs):
        self.interrupted = True


def _skill(monkeypatch):
    """One installed skill, ``/demo``, without a skills directory."""
    import agent.skill_bundles as bundles
    import agent.skill_commands as skills
    from agent import skill_utils

    monkeypatch.setattr(skills, 'get_skill_commands', lambda: {'/demo': {'name': 'demo'}})
    monkeypatch.setattr(skills, 'resolve_skill_command_key', lambda command, **_: '/demo' if command == 'demo' else None)
    monkeypatch.setattr(skills, 'split_stacked_skill_commands', lambda rest, **_: ([], rest))
    monkeypatch.setattr(skills, 'build_skill_invocation_message', lambda key, text='', **_: f'[demo skill] {text}')
    monkeypatch.setattr(bundles, 'resolve_bundle_command_key', lambda command: None)
    monkeypatch.setattr(skill_utils, 'get_disabled_skill_names', lambda **_: set())


async def _start_a(monkeypatch, mode=None, *, agent=True):
    """A runs (executed by the drain) and waits for ``finish``. ``order`` records each turn as
    ``(message_id, where)``: True when the drain executed it as an admission, 'inside A' when it
    ran from the in-process slot inside the running turn. Without ``agent`` A stays on the
    pending sentinel (an agent still being built)."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result

    runner, adapter, sent = _owner(monkeypatch, mode)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    rig = SimpleNamespace(runner=runner, adapter=adapter, sent=sent, authority=authority, source=source,
                          key=runner.session_store._generate_session_key(source), order=[],
                          running=asyncio.Event(), finish=asyncio.Event())

    async def turn(event, source, quick_key, run_generation):
        rig.order.append((event.message_id, executing_admission.get()))
        result = {'final_response': 'reply ' + str(event.message_id), 'messages': []}
        if event.message_id == 'A':
            if agent:
                runner._session_state(quick_key).turn.agent = _Agent()
            rig.running.set()
            await rig.finish.wait()
            pending_event, pending = await runner._run_agent_drain_pending(
                result, runner._delivery_adapter_for(source), source, quick_key)
            if pending_event is not None or pending:
                rig.order.append((getattr(pending_event, 'message_id', pending), 'inside A'))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = result
        return result['final_response']
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(rig.running.wait(), 10)
    return rig


def _rows(rig):
    from hermes_state_runtime import list_session_admissions
    sid = rig.runner.session_store.peek_session_id(rig.key)
    return [row['request_id'] for row in list_session_admissions(rig.authority.db, session_id=sid, pending_only=False)]


async def _run_a_then(monkeypatch, mode, command, *, before_finish=None, settled=lambda: False):
    """A runs; B (plain text) is admitted behind it; then *command* arrives. Returns the turns."""
    from gateway.platforms.event import MessageEvent, MessageType

    rig = await _start_a(monkeypatch, mode)
    await rig.adapter.handle_message(MessageEvent(text='B', source=rig.source, message_id='B'))
    await async_wait_until(lambda: len(_rows(rig)) >= 2, what='B admitted behind A')
    await rig.adapter.handle_message(MessageEvent(text=command, message_type=MessageType.COMMAND,
                                                  source=rig.source, message_id='C'))
    await async_wait_until(lambda: rig.sent or len(_rows(rig)) >= 3 or rig.adapter._pending_messages,
                           what='the command to be queued')
    if before_finish is not None:
        await before_finish(rig)
    rig.finish.set()
    order = rig.order
    await async_wait_until(lambda: [m for m, _ in order].count('C') or settled(), timeout=15, what='C to run')
    await async_wait_until(lambda: ('B', True) in order or settled(), timeout=15, what='B to run')
    return order


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,command', [('queue', '/queue C'), ('interrupt', '/queue C'),
                                          ('queue', '/steer C')])  # _Agent cannot steer: the queue fallback
async def test_busy_queue_runs_after_the_turn_already_admitted(monkeypatch, mode, command):
    order = await _run_a_then(monkeypatch, mode, command)
    assert order == [('A', True), ('B', True), ('C', True)], order


@pytest.mark.asyncio
async def test_busy_skill_slash_runs_after_the_turn_already_admitted(monkeypatch):
    _skill(monkeypatch)
    order = await _run_a_then(monkeypatch, 'queue', '/demo C')
    assert order == [('A', True), ('B', True), ('C', True)], order


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/queue C', '/steer C', '/plan C'])
async def test_idle_rewrite_waits_behind_a_queued_row(monkeypatch, command):
    """Between turns (slot released, the drain has not claimed B yet), a slash command rewritten
    into a turn is admitted behind B instead of running in-process ahead of it."""
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import admit_message, executing_admission
    from gateway.session_ingress_context import native_callback
    from gateway.session_results import execution_result
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions

    runner, adapter, _sent = _owner(monkeypatch)
    real_schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    order = []

    async def turn(event, source, quick_key, run_generation):
        order.append((event.message_id, executing_admission.get()))
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    follow_up = MessageEvent(text='B', source=source, message_id='B')
    with native_callback(runner, follow_up, get_hermes_home()):
        waiter = asyncio.create_task(admit_message(authority, follow_up))
        await async_wait_until(lambda: authority.waiters)
    owner = runner.session_store.peek_session_id(runner.session_store._generate_session_key(source))
    cmd = MessageEvent(text=command, message_type=MessageType.COMMAND, source=source, message_id='C')
    with native_callback(runner, cmd, get_hermes_home()):
        task = asyncio.create_task(runner._handle_message(cmd))
        await async_wait_until(lambda: order or len(list_session_admissions(authority.db, session_id=owner)) >= 2,
                               what=f'{command} to run or be admitted')
    monkeypatch.setattr(SessionAuthority, '_schedule', real_schedule)
    authority._schedule(SessionRef(authority.profile_id, owner))
    await asyncio.wait_for(task, 15)
    waiter.cancel()
    assert order == [('B', True), ('C', True)], f'{command} did not wait behind the queued B: {order}'


@pytest.mark.asyncio
@pytest.mark.parametrize('prompt', ['slash-confirm', 'update', 'clarify'])
async def test_an_admitted_turn_never_answers_a_pending_prompt(monkeypatch, prompt):
    """A prompt that opened after ``/queue yes`` was admitted (a destructive slash-confirm, an
    /update prompt, a clarify) never takes the executing turn as its answer: the turn runs. A
    restart replays the row as a plain restored event, which is checked directly."""
    from tools import clarify_gateway, slash_confirm
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress import executing_admission

    answered = []

    async def confirm(choice):
        answered.append(choice)
        return 'confirmed'
    opened = []
    monkeypatch.setattr(clarify_gateway, 'get_pending_for_session',
                        lambda key, **_: SimpleNamespace(clarify_id='q', response=None) if opened else None)
    monkeypatch.setattr(clarify_gateway, 'attempt_text_response_for_session',
                        lambda key, text: answered.append(text) or clarify_gateway.TEXT_RESOLVED)

    async def open_prompt(rig):
        opened.append(prompt)
        if prompt == 'slash-confirm':
            slash_confirm.register(rig.key, 'confirm-1', 'undo', confirm)
        elif prompt == 'update':
            rig.runner._session_state(rig.key).persistent.update_prompt_pending = True
            monkeypatch.setattr(rig.runner, '_hm_write_update_response', lambda text: answered.append(text))
        token = executing_admission.set(True)  # restart recovery: a restored row, no in-memory marker
        try:
            replayed = MessageEvent(text='yes', source=rig.source, message_id='R')
            assert await rig.runner._hm_pending_reply_intercepts(replayed, rig.source, rig.key) is None
        finally:
            executing_admission.reset(token)
    order = await _run_a_then(monkeypatch, 'queue', '/queue yes', before_finish=open_prompt,
                              settled=lambda: answered)
    assert answered == [] and 'C' in [m for m, _ in order], (answered, order)


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/queue C', '/steer C'])  # no agent can steer: the queue fallback
async def test_queued_turn_keeps_the_relay_callback(monkeypatch, command):
    """The rebuilt turn carries the command's authenticated relay callback, so a relay-fronted chat
    can queue; the strict identity check in ``capture_provenance`` still holds."""
    from pathlib import Path
    from unittest.mock import AsyncMock
    from gateway import session_ingress_context as ctx
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run_busy import GatewayBusySessionMixin
    from gateway.session import SessionSource

    accepted, transport, home = [], object(), Path.cwd()

    class Runner(GatewayBusySessionMixin):
        session_authority = object()

        def _peek_session_state(self, key):
            return None

        def _delivery_adapter_for(self, source):
            return adapter

        def _queue_depth(self, *args, **kwargs):
            return 0
    runner = Runner()

    async def handler(event):  # the observer's message handler: native callback, then provenance
        with ctx.native_callback(runner, event, home):
            ctx.capture_provenance(runner, event)
        accepted.append(event.text)
    adapter = SimpleNamespace(_transport=transport, _background_tasks=set(), gateway_runner=None, name='relay',
                              _message_handler=handler, _release_turn_marker=AsyncMock(),
                              _unwrap_ephemeral=lambda response: (response, 0))
    monkeypatch.setattr(ctx, '_binding', lambda *a: (adapter, home, home, {'relay': 'seal'}))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='42', user_id='42', chat_type='dm',
                           delivered_via_upstream_relay=True)
    original = MessageEvent(text=command, message_type=MessageType.COMMAND, source=source, message_id='C')
    method = runner._busy_queue_command if command.startswith('/queue') else runner._busy_steer_command
    with ctx.relay_callback(transport, original), ctx.native_callback(runner, original, home):
        await method(original, 'chat', source)
    assert accepted == ['C']


@pytest.mark.asyncio
@pytest.mark.parametrize('caller', ['adapter', 'runner'])
@pytest.mark.parametrize('kind', ['skill', 'bundle'])
async def test_stop_during_a_busy_skill_rewrite_drops_it(monkeypatch, caller, kind):
    """A /stop that lands while a mid-run skill or bundle command is still being rewritten (a slow
    cold skill scan) drops it, as it clears the in-process slot: it never commits afterwards.
    ``adapter``: the adapter's busy handler; ``runner``: the runner's busy path (no adapter guard)."""
    import threading
    import agent.skill_bundles as bundles
    from gateway.platforms.event import MessageEvent, MessageType

    _skill(monkeypatch)
    started, gate = threading.Event(), threading.Event()

    def slow(text):
        started.set()
        gate.wait(10)
        return f'[{kind}] {text}'
    if kind == 'skill':
        import agent.skill_commands as skills
        monkeypatch.setattr(skills, 'build_skill_invocation_message', lambda key, text='', **_: slow(text))
    else:
        monkeypatch.setattr(bundles, 'resolve_bundle_command_key', lambda command: '/demo' if command == 'demo' else None)
        monkeypatch.setattr(bundles, 'build_bundle_invocation_message', lambda key, args, **_: (slow(args), [], []))
    rig = await _start_a(monkeypatch, 'queue')
    command = MessageEvent(text='/demo C', message_type=MessageType.COMMAND, source=rig.source, message_id='C')
    task = asyncio.create_task(rig.adapter.handle_message(command) if caller == 'adapter'
                               else rig.runner._handle_message(command))
    await async_wait_until(lambda: started.is_set() or task.done(), what='the rewrite to start')
    await rig.adapter.handle_message(MessageEvent(text='/stop', message_type=MessageType.COMMAND,
                                                  source=rig.source, message_id='S'))
    gate.set()
    await asyncio.wait_for(task, 15)
    rig.finish.set()
    assert 'C' not in _rows(rig) and 'C' not in [m for m, _ in rig.order], (_rows(rig), rig.order)


@pytest.mark.asyncio
@pytest.mark.parametrize('command,agent,ack', [('/queue C', True, 'gateway.queue.queued'),
                                               ('/steer C', True, 'gateway.steer.queued_no_agent'),
                                               ('/steer C', False, 'gateway.steer.queued_starting')])
async def test_queued_ack_follows_the_commit(monkeypatch, command, agent, ack):
    """The busy sender may be released before the row commits (#140454 releases it once the input
    waits behind an earlier one of its chat). The "Queued" reply follows the commit, not that
    release: none before it, one once it lands."""
    from agent.i18n import t
    import gateway.session_ingress as ingress
    from gateway.platforms.event import MessageEvent, MessageType

    real, held = ingress.admit_message, asyncio.Event()

    async def released_early(authority, event):
        accepted = ingress.busy_acceptance.get()
        if event.message_id == 'C' and accepted is not None:
            accepted.set_result(None)
            await held.wait()
        return await real(authority, event)
    monkeypatch.setattr(ingress, 'admit_message', released_early)
    rig = await _start_a(monkeypatch, 'queue', agent=agent)
    await rig.adapter.handle_message(MessageEvent(text=command, message_type=MessageType.COMMAND,
                                                  source=rig.source, message_id='C'))
    assert t(ack) not in rig.sent and 'C' not in _rows(rig), rig.sent
    held.set()
    await async_wait_until(lambda: t(ack) in rig.sent, what='the ack after the commit')
    assert 'C' in _rows(rig) and rig.sent.count(t(ack)) == 1, rig.sent
    rig.finish.set()


@pytest.mark.asyncio
async def test_queued_approval_word_is_a_turn_never_an_approval(monkeypatch):
    """Keep working: ``/queue yes`` queues a user turn; it never answers a blocking dangerous-command
    approval the way a typed "yes" does."""
    from tools import approval

    approved = []
    monkeypatch.setattr(approval, 'has_blocking_approval', lambda key: True)

    async def approve(self, event):
        approved.append(event.text)
        return 'approved'
    from gateway.run import GatewayRunner
    monkeypatch.setattr(GatewayRunner, '_handle_approve_command', approve)
    order = await _run_a_then(monkeypatch, 'queue', '/queue yes')
    assert approved == [] and 'C' in [m for m, _ in order], (approved, order)

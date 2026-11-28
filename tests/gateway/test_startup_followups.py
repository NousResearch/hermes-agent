"""An admitted startup follow-up is settled with the turn that actually takes it."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.startup_followups import StartupFollowups
from gateway import session_finite
from gateway.startup_followups.coordination import Candidate, transfer
from tests.gateway.test_prompt_queued_controls import _ledger
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["sent", "no_request", "no_agent"])
async def test_startup_admission_is_consumed_or_returned_to_its_fifo(tmp_path, monkeypatch, outcome):
    db, authority = _authority(tmp_path, monkeypatch)
    runner = authority.runner
    runner._is_session_run_current = lambda _key, generation: generation == 17
    runner._prepend_inbound_reply_context = lambda event, source, text: text
    runner._steer_text_with_origin = lambda text, event: text
    monkeypatch.setattr("agent.agent_init_session.agent_interface", lambda _agent: None)
    claimed, release = asyncio.Event(), asyncio.Event()
    ran, requests = [], []

    async def execute(authority, ref, row):
        ran.append(row['request_id'])
        if row['request_id'] != 'first':
            return 'done'
        claimed.set()
        await asyncio.wait_for(release.wait(), 10)
        if outcome == 'no_agent':
            raise RuntimeError('synthetic agent construction failure')
        startup = StartupFollowups(lambda _entries: None)
        agent = SimpleNamespace(
            _gateway_startup_followups=startup,
            _session_db=SimpleNamespace(set_message_api_content=MagicMock(return_value=True)),
            session_id='s', _persist_user_message_override=None,
        )
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        user = {'role': 'user', 'content': 'first', '_row_id': 1}
        wire = {'role': 'user', 'content': 'first'}
        assert startup.stage([user], [wire], 0)
        if outcome == 'sent':
            startup.commit_request(agent, {'messages': [wire]})
            requests.append(wire['content'])
        startup.finish()
        return 'done'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    event = SimpleNamespace(text='second', source=SimpleNamespace(), message_id='message-2')
    with db:
        await _submit(authority, 'first')
        await asyncio.wait_for(claimed.wait(), 10)
        second = await _submit(authority, 'second')
        await transfer(authority, second, event, Candidate('route', 17, 'interrupt'))
        pending = authority.persistence.admissions.get(admission_id=second.admission_id)
        assert (pending.status, pending.outcome) == ('terminal', 'steering')
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        if outcome != 'sent':
            from tests.waiting import async_wait_until
            await async_wait_until(lambda: ran == ['first', 'second'], timeout=10,
                                   what='the unread follow-up runs from its original FIFO position')
            await asyncio.wait_for(authority.sessions['s'].task, 10)

    if outcome == 'sent':
        assert requests == ['first\n\nsecond']
        assert ran == ['first']
        assert _ledger(authority.persistence) == [('first', 'terminal', 'completed'),
                                                 ('second', 'terminal', 'steered')]
    else:
        assert ran == ['first', 'second']
        assert _ledger(authority.persistence)[-1] == ('second', 'terminal', 'completed')


@pytest.mark.asyncio
async def test_concurrent_startup_transfers_preserve_fifo_order(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    runner = authority.runner
    runner._is_session_run_current = lambda _key, generation: generation == 17
    runner._prepend_inbound_reply_context = lambda event, source, text: text
    runner._steer_text_with_origin = lambda text, event: text
    monkeypatch.setattr('agent.agent_init_session.agent_interface', lambda _agent: None)
    claimed, release = asyncio.Event(), asyncio.Event()
    entered_write, release_write = threading.Event(), threading.Event()
    requests = []

    async def execute(authority, ref, row):
        claimed.set()
        await asyncio.wait_for(release.wait(), 10)
        startup = StartupFollowups(lambda _entries: None)
        agent = SimpleNamespace(
            _gateway_startup_followups=startup,
            _session_db=SimpleNamespace(set_message_api_content=MagicMock(return_value=True)),
            session_id='s', _persist_user_message_override=None,
        )
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        user = {'role': 'user', 'content': 'first', '_row_id': 1}
        wire = {'role': 'user', 'content': 'first'}
        assert startup.stage([user], [wire], 0)
        startup.commit_request(agent, {'messages': [wire]})
        requests.append(wire['content'])
        return 'done'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'first')
        await asyncio.wait_for(claimed.wait(), 10)
        second = await _submit(authority, 'second')
        third = await _submit(authority, 'third')
        admissions = authority.persistence.admissions
        steer = admissions.steer

        def block_first(*, admission_id, **kwargs):
            if admission_id == second.admission_id:
                entered_write.set()
                assert release_write.wait(10)
            return steer(admission_id=admission_id, **kwargs)

        monkeypatch.setattr(admissions, 'steer', block_first)
        candidate = Candidate('route', 17, 'interrupt')
        second_event = SimpleNamespace(text='second', source=SimpleNamespace(), message_id='m2')
        third_event = SimpleNamespace(text='third', source=SimpleNamespace(), message_id='m3')
        second_transfer = asyncio.create_task(transfer(authority, second, second_event, candidate))
        try:
            assert await asyncio.to_thread(entered_write.wait, 10)
            third_transfer = asyncio.create_task(transfer(authority, third, third_event, candidate))
        finally:
            release_write.set()
        await asyncio.gather(second_transfer, third_transfer)
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert requests == ['first\n\nsecond\n\nthird']
        assert [row[2] for row in _ledger(authority.persistence)] == ['completed', 'steered', 'steered']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode, expected_anchor', [('interrupt', 'm2'), ('steer', 'm1')])
async def test_gateway_first_request_takes_a_quoted_startup_followup(
        tmp_path, monkeypatch, mode, expected_anchor):
    from unittest.mock import AsyncMock, patch

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from gateway.startup_followups import note_sent
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from tests.agent.test_run_agent import _mock_response
    from tests.agent.test_steer import TestSteerSurvivesRedirectRebuild as SteerFixture
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.FEISHU: adapter}
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: mode)
    monkeypatch.setattr(runner, '_effective_busy_text_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_hm_in_telegram_grace', lambda *args: False)
    monkeypatch.setattr(runner, '_steer_text_with_origin', lambda text, event: text)
    monkeypatch.setattr('gateway.run_inbound.pending_setup_busy_notice', lambda *args: None)
    monkeypatch.setattr('agent.agent_init_session.agent_interface', lambda _agent: None)
    sends, requests, stream_anchors, ran = [], [], [], []

    async def send(chat_id, content, reply_to=None, metadata=None, **kwargs):
        sends.append((content, reply_to))
        return SendResult(success=True, message_id='sent')

    adapter.send = send
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='startup-test')
    runner._wire_adapter_handlers(adapter, authorization_check=lambda *args, **kwargs: True)
    source = adapter.build_source(chat_id='synthetic-chat', chat_type='dm', user_id='synthetic-user')
    first = MessageEvent(text='first question', source=source, message_id='m1')
    second = MessageEvent(text='also address this', source=source, message_id='m2',
                          reply_to_message_id='old', reply_to_text='quoted context')
    claimed, release = asyncio.Event(), asyncio.Event()

    async def fake_turn(event, source, key, run_generation):
        ran.append(event.message_id)
        if event.message_id != 'm1':
            return 'unexpected second turn'
        claimed.set()
        await asyncio.wait_for(release.wait(), 10)
        session_id = next(sid for sid, live in authority.sessions.items() if live.route == key)
        generation = authority.executing[session_id]
        agent = SteerFixture()._loop_agent()
        agent.session_id = session_id
        agent._session_db = SimpleNamespace(set_message_api_content=MagicMock(return_value=True))
        stream = SimpleNamespace(_initial_reply_to_id='m1')
        ctx = SimpleNamespace(session_key=key, persist_user_display_kind=None, reply_expected=True,
                              event_message_id='m1', inbound_message_id='m1', stream_consumer_holder=[stream])
        startup = StartupFollowups(lambda entries: note_sent(
            runner, ctx, agent, authority, session_id, generation, entries))
        agent._gateway_startup_followups = startup
        authority.adopt_agent(session_id, generation, agent)

        def persist(messages, *_args):
            for row in reversed(messages):
                if row.get('role') == 'user':
                    row['_row_id'] = 101
                    break

        def model(kwargs):
            requests.append(kwargs['messages'])
            stream_anchors.append(stream._initial_reply_to_id)
            return _mock_response(content='one combined answer', finish_reason='stop')

        agent._interruptible_api_call = model
        with (patch.object(agent, '_ensure_db_session'), patch.object(agent, '_flush_messages_to_session_db'),
              patch.object(agent, '_persist_session', side_effect=persist), patch.object(agent, '_save_trajectory'),
              patch.object(agent, '_cleanup_task_resources')):
            result = await asyncio.to_thread(agent.run_conversation, event.text)
        execution_result.get()['result'] = result
        return result['final_response']

    monkeypatch.setattr(runner, '_handle_message_with_agent', fake_turn)
    try:
        await adapter.handle_message(first)
        await asyncio.wait_for(claimed.wait(), 10)
        await adapter.handle_message(second)
        sid = next(iter(authority.sessions))
        await async_wait_until(
            lambda: any(row.outcome == 'steering' for row in authority.persistence.admissions.list(
                session_id=sid, pending_only=False)),
            timeout=10, what='the follow-up is transferred to the running admission')
        release.set()
        await async_wait_until(lambda: bool(sends), timeout=10, what='the combined answer is sent')
        await async_wait_until(lambda: all(not live.task or live.task.done() for live in authority.sessions.values()),
                               timeout=10, what='the authority settles the turn')
        assert ran == ['m1']
        assert len(requests) == 1
        assert 'first question' in requests[0][-1]['content']
        assert 'also address this' in requests[0][-1]['content']
        assert 'quoted context' in requests[0][-1]['content']
        assert sends == [('one combined answer', expected_anchor)]
        assert stream_anchors == [expected_anchor]
    finally:
        runner.session_store.close_all_db_handles()
        adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['pending', 'adopted', 'transferring'])
async def test_stop_discards_a_startup_followup_before_any_request(tmp_path, monkeypatch, phase):
    from unittest.mock import AsyncMock

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.FEISHU: adapter}
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_effective_busy_text_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_hm_in_telegram_grace', lambda *args: False)
    monkeypatch.setattr('gateway.run_inbound.pending_setup_busy_notice', lambda *args: None)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id='sent'))
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='stop-test')
    runner._wire_adapter_handlers(adapter, authorization_check=lambda *args, **kwargs: True)
    source = adapter.build_source(chat_id='chat', chat_type='dm', user_id='user')
    first = MessageEvent(text='first', source=source, message_id='m1')
    second = MessageEvent(text='second', source=source, message_id='m2')
    key = adapter._event_session_key(first)
    claimed, release = asyncio.Event(), asyncio.Event()
    entered_write, release_write = threading.Event(), threading.Event()
    ran = []

    async def fake_turn(event, *_args):
        ran.append(event.message_id)
        if event.message_id == 'm1':
            claimed.set()
            await asyncio.wait_for(release.wait(), 30)

    monkeypatch.setattr(runner, '_handle_message_with_agent', fake_turn)
    try:
        await adapter.handle_message(first)
        await asyncio.wait_for(claimed.wait(), 10)
        sid = next(iter(authority.sessions))
        if phase == 'adopted':
            startup = StartupFollowups(lambda _entries: None)
            agent = SimpleNamespace(_gateway_startup_followups=startup, interrupt=MagicMock())
            authority.adopt_agent(sid, authority.executing[sid], agent)
        if phase == 'transferring':
            admissions = authority.persistence.admissions
            steer = admissions.steer

            def delayed_steer(**kwargs):
                entered_write.set()
                assert release_write.wait(30)
                return steer(**kwargs)

            monkeypatch.setattr(admissions, 'steer', delayed_steer)
        second_dispatch = asyncio.create_task(adapter.handle_message(second))
        if phase == 'transferring':
            assert await asyncio.to_thread(entered_write.wait, 20)
            generation = authority.executing[sid]
            stop = asyncio.create_task(runner._hm_stop_owner_turn(key))
            await async_wait_until(lambda: authority.route_epoch(key) > 0,
                                   timeout=20, what='stop fences the route')
            await async_wait_until(lambda: authority.pending_stops.get(sid) == generation,
                                   timeout=10, what='stop reaches the running turn while the write waits')
            release_write.set()
            await asyncio.wait_for(stop, 20)
        else:
            await async_wait_until(
                lambda: any(row.outcome == 'steering' for row in authority.persistence.admissions.list(
                    session_id=sid, pending_only=False)),
                timeout=20, what='the follow-up is transferred to the running admission')
            await runner._hm_stop_owner_turn(key)
        await asyncio.wait_for(second_dispatch, 20)
        release.set()
        await async_wait_until(lambda: all(not live.task or live.task.done() for live in authority.sessions.values()),
                               timeout=10, what='the stopped session settles')
        assert ran == ['m1']
        assert authority.persistence.admissions.list(session_id=sid, pending_only=False)[-1].outcome in (
            'cancelled', 'interrupted')
    finally:
        release_write.set()
        release.set()
        runner.session_store.close_all_db_handles()
        adapter._shutdown_sdk_executor()

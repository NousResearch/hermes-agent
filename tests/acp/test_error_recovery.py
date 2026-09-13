"""Regression tests for failed ACP request cleanup, using the real adapter."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


@pytest.fixture
def session():
    calls = []
    def run(**kwargs):
        calls.append(kwargs['user_message'])
        return {'final_response': 'done', 'messages': [{'role': 'assistant', 'content': 'done'}]}
    runtime = SimpleNamespace(session_id='head', run_conversation=run)
    state = SessionState('test', runtime, cancel_event=threading.Event())
    manager = SessionManager()
    manager._sessions['test'] = state
    manager.save_session = lambda *_: None
    return HermesACPAgent(manager), state, calls


async def send(adapter, text):
    return await adapter.prompt([TextContentBlock(type='text', text=text)], 'test')


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['callbacks', 'executor', 'persist', 'output', 'usage'])
async def test_failed_request_releases_session_and_drains_existing_queue(session, monkeypatch, boundary):
    adapter, state, calls = session
    state.queued_prompts.append('already queued')
    if boundary == 'output':
        async def update(*_):
            return None
        adapter._conn = SimpleNamespace(session_update=update, request_permission=update)
    owner, name = {
        'callbacks': (adapter, '_wire_turn_callbacks'),
        'executor': (adapter, '_run_agent_turn'),
        'persist': (adapter.session_manager, 'save_session'),
        'output': (adapter._conn, 'session_update'),
        'usage': (adapter, '_send_usage_update'),
    }[boundary]
    original = getattr(owner, name)
    count = 0
    def fail_once(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError('injected request failure')
        return original(*args, **kwargs)
    if boundary in {'output', 'usage'}:
        async def async_fail_once(*args, **kwargs):
            return await fail_once(*args, **kwargs)
        monkeypatch.setattr(owner, name, async_fail_once)
    else:
        monkeypatch.setattr(owner, name, fail_once)
    with pytest.raises(RuntimeError, match='injected request failure'):
        await send(adapter, 'original')
    assert not state.is_running
    assert state.current_prompt_text == ''
    assert state.queued_prompts == []
    assert calls.count('already queued') == 1
    assert calls.count('original') <= 1
    assert (await send(adapter, 'next')).stop_reason == 'end_turn'
    assert calls[-1] == 'next'


@pytest.mark.asyncio
async def test_duplicate_cancel_does_not_repeat_hard_interrupt_or_cancel_idle_session(session, monkeypatch):
    adapter, state, _ = session
    interrupts = []
    monkeypatch.setattr('acp_adapter.server.request_hard_interrupt', lambda agent: interrupts.append(agent))
    await adapter.cancel('test')
    assert not state.cancel_event.is_set()
    assert interrupts == []
    state.is_running = True
    await adapter.cancel('test')
    await adapter.cancel('test')
    assert len(interrupts) == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_worker_ownership_until_queue_can_run(session, monkeypatch):
    adapter, state, calls = session
    loop = asyncio.get_running_loop()
    entered, interrupted = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    stopped = threading.Event()
    def run(**kwargs):
        text = kwargs['user_message']
        calls.append(text)
        if text == 'original':
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            stopped.set()
            return {'messages': [], 'interrupted': True}
        assert stopped.is_set(), 'queued work overlapped the cancelled worker'
        return {'messages': []}
    state.agent.run_conversation = run
    monkeypatch.setattr('acp_adapter.server.request_hard_interrupt', lambda _: interrupted.set())
    task = asyncio.create_task(send(adapter, 'original'))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await send(adapter, 'queued')
        task.cancel()
        await asyncio.wait_for(interrupted.wait(), 5)
        assert state.is_running
        assert len(calls) == 1
        task.cancel()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stopped.is_set()
    assert not state.is_running
    assert not state.queued_prompts
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_internal_error_keeps_original_details_and_connection_accepts_next_request(session, monkeypatch):
    from acp.agent.router import build_agent_router
    from acp.connection import Connection
    from acp.exceptions import RequestError

    adapter, state, calls = session
    def broken_save(*_):
        raise OSError('injected persistence failure')
    monkeypatch.setattr(adapter.session_manager, 'save_session', broken_save)
    replies = []
    async def capture(payload):
        replies.append(payload)
    connection = object.__new__(Connection)
    connection._handler = build_agent_router(adapter)
    connection._sender = SimpleNamespace(send=capture)
    connection._observers = []
    def request(ident, text):
        return {'jsonrpc': '2.0', 'id': ident, 'method': 'session/prompt',
                'params': {'sessionId': 'test', 'prompt': [{'type': 'text', 'text': text}]}}
    with pytest.raises(RequestError):
        await connection._run_request(request(1, 'first'))
    assert replies[0]['error'] == {'code': -32603, 'message': 'Internal error',
                                   'data': {'details': 'injected persistence failure'}}
    assert not state.is_running
    monkeypatch.setattr(adapter.session_manager, 'save_session', lambda *_: None)
    await connection._run_request(request(2, 'next'))
    assert replies[1]['result']['stopReason'] == 'end_turn'
    assert calls == ['first', 'next']
    assert not state.queued_prompts


@pytest.mark.asyncio
@pytest.mark.parametrize('original_failed', [False, True])
async def test_cancellation_during_queue_echo_preserves_work_and_original_error(session, monkeypatch, original_failed):
    adapter, state, calls = session
    echo_entered = asyncio.Event()
    async def update(_session_id, event):
        if event.session_update == 'user_message_chunk':
            echo_entered.set()
            await asyncio.Event().wait()
    adapter._conn = SimpleNamespace(session_update=update, request_permission=update)
    state.queued_prompts.append('accepted queued')
    if original_failed:
        original_save = adapter.session_manager.save_session
        def fail_first(*args):
            monkeypatch.setattr(adapter.session_manager, 'save_session', original_save)
            raise OSError('original persistence failure')
        monkeypatch.setattr(adapter.session_manager, 'save_session', fail_first)
    task = asyncio.create_task(send(adapter, 'original'))
    await asyncio.wait_for(echo_entered.wait(), 5)
    task.cancel()
    error = OSError if original_failed else asyncio.CancelledError
    with pytest.raises(error):
        await task
    assert calls == ['original', 'accepted queued']
    assert not state.queued_prompts
    assert not state.is_running

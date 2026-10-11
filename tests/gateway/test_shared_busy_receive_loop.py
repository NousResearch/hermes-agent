"""A busy-session follow-up releases the adapter's serial receive loop once it is durably queued.

IRC and Signal await ``handle_message`` before reading their next frame. With the shared runtime
a busy follow-up is admitted into the session FIFO; the adapter must return as soon as that row
is committed, not when the queued turn finally answers, while still delivering exactly one reply.
"""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token='owned-fixture'), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=f'sent-{len(self.sent)}')

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {'id': chat_id}


async def _busy_owner(monkeypatch, turns):
    """Real runner + authority over SQLite; turn production is a gated stand-in (no model)."""
    import gateway.session_finite as finite
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')

    async def produce(authority, ref, row):
        return await turns[row['payload']['text']]()
    monkeypatch.setattr(finite, 'execute_finite_admission', produce)
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='busy-owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    return authority, adapter, source


def _statuses(authority):
    from hermes_state_runtime import list_session_admissions
    return {row['payload']['text']: row['status'] for session_id in authority.sessions
            for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)}


async def _wait_for(predicate, timeout=5):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, 'condition not reached'
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_busy_follow_up_returns_after_commit_and_is_answered_once(monkeypatch):
    first_gate, second_gate = asyncio.Event(), asyncio.Event()

    async def first():
        await first_gate.wait()
        return 'FIRST_REPLY'

    async def second():
        await second_gate.wait()
        return 'SECOND_REPLY'
    authority, adapter, source = await _busy_owner(monkeypatch, {'one': first, 'two': second})
    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    # The receive loop hands over the follow-up while the first turn is still running.
    receive = asyncio.create_task(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')))
    done, _ = await asyncio.wait({receive}, timeout=3)
    assert receive in done, 'receive loop parked until the queued turn completes'
    receive.result()
    assert _statuses(authority) == {'one': 'started', 'two': 'queued'}
    assert adapter.sent == []
    first_gate.set()
    await _wait_for(lambda: _statuses(authority).get('two') == 'started')
    second_gate.set()
    await _wait_for(lambda: 'SECOND_REPLY' in adapter.sent and not adapter._background_tasks)
    assert adapter.sent == ['FIRST_REPLY', 'SECOND_REPLY']
    assert _statuses(authority) == {'one': 'terminal', 'two': 'terminal'}


@pytest.mark.asyncio
async def test_detached_busy_completion_failure_still_tells_the_user(monkeypatch):
    first_gate = asyncio.Event()
    unhandled = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unhandled.append(ctx))

    async def first():
        await first_gate.wait()
        return 'FIRST_REPLY'

    async def second():
        return 'SECOND_REPLY'
    authority, adapter, source = await _busy_owner(monkeypatch, {'one': first, 'two': second})
    extract = adapter._extract_response_content

    async def broken_extract(text, *args, **kwargs):
        if text == 'SECOND_REPLY':
            raise RuntimeError('render exploded')
        return await extract(text, *args, **kwargs)
    monkeypatch.setattr(adapter, '_extract_response_content', broken_extract)
    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    # The receive loop must neither park nor see the later delivery failure.
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 3)
    first_gate.set()
    await _wait_for(lambda: len(adapter.sent) == 2 and not adapter._background_tasks)
    # Exactly one reply per input; the error notice may race the first turn's own delivery.
    assert 'FIRST_REPLY' in adapter.sent
    assert sum('render exploded' in text for text in adapter.sent) == 1, adapter.sent
    assert _statuses(authority) == {'one': 'terminal', 'two': 'terminal'}
    assert unhandled == []

"""Serial transports regain their receive loop once input is durable."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, SessionSource
from gateway.session_ingress import admit_message, dispatch_shared_busy


@pytest.mark.asyncio
async def test_busy_receive_returns_after_acceptance_but_delivers_once_after_completion(monkeypatch):
    import gateway.session_ingress as ingress
    accepted = asyncio.Event()
    release = asyncio.Event()

    async def admit(event):
        accepted.set()
        await release.wait()
        return SimpleNamespace(status='queued', admission_id='queued')

    authority = SimpleNamespace(admit_native=admit, native_waiters=set(), waiters={}, db=None)
    adapter = SimpleNamespace(_background_tasks=set(), _message_handler=lambda event: admit_message(authority, event))
    delivery = AsyncMock()
    monkeypatch.setattr(ingress, 'deliver_response', delivery)
    # admit_message re-reads the committed row (a drain may already have settled it); still queued.
    monkeypatch.setattr('hermes_state_runtime.get_session_admission', lambda db, admission_id: {'status': 'queued'})
    event = MessageEvent(text='next', source=SessionSource(platform=Platform.TELEGRAM, chat_id='chat'))
    receive = asyncio.create_task(dispatch_shared_busy(adapter, event, 'route'))
    await asyncio.wait_for(accepted.wait(), 5)
    assert not receive.done()  # The admission has not committed yet.
    release.set()
    await asyncio.wait_for(receive, 5)
    assert not authority.waiters['queued'].done()
    delivery.assert_not_awaited()
    authority.waiters['queued'].set_result('completed answer')
    await asyncio.gather(*adapter._background_tasks)
    delivery.assert_awaited_once()
    assert delivery.await_args.args[-1] == 'completed answer'


@pytest.mark.asyncio
async def test_pre_admission_refusal_reaches_receive_loop():
    from hermes_state_runtime import RuntimeStoreError

    async def refused(event):
        raise RuntimeStoreError('permission_denied')

    adapter = SimpleNamespace(_background_tasks=set(), _message_handler=refused)
    event = MessageEvent(text='refused', source=SessionSource(platform=Platform.TELEGRAM, chat_id='chat'))
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await dispatch_shared_busy(adapter, event, 'route')
    assert not adapter._background_tasks

"""Terminal notices survive failures at the real Telegram send/edit/callback boundaries."""
import asyncio
import concurrent.futures
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.run import _approval_send_outcome
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval


@pytest.fixture
def surface(monkeypatch):
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '1')
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='test-token'))
    bot = AsyncMock()
    bot.send_message.return_value = SimpleNamespace(message_id=321)
    adapter._bot = bot
    monkeypatch.setattr(adapter, "send", AsyncMock(return_value=SimpleNamespace(success=True)))
    hooks = []
    monkeypatch.setattr(approval, 'register_gateway_settle', lambda sk, rid, fn: hooks.append(fn) or True)
    return adapter, hooks


async def publish(adapter):
    return await adapter.send_exec_approval(chat_id='123', command='synthetic command',
        session_key='agent:main:telegram:group:123:7', request_id='review-request',
        metadata={'thread_id': '7'})


async def drain_settle(hook, reason):
    loop = asyncio.get_running_loop()
    before = asyncio.all_tasks()
    hook(reason)
    barrier = loop.create_future()
    loop.call_soon(barrier.set_result, None)
    await barrier
    spawned = asyncio.all_tasks() - before
    if spawned:
        await asyncio.wait_for(asyncio.gather(*spawned, return_exceptions=True), 5)


@pytest.mark.asyncio
async def test_timeout_edit_failure_still_delivers_terminal_notice(surface, monkeypatch):
    adapter, hooks = surface
    assert (await publish(adapter)).success
    edit = AsyncMock(wraps=adapter.edit_message)
    monkeypatch.setattr(adapter, 'edit_message', edit)
    adapter._bot.edit_message_text.side_effect = OSError('synthetic edit failure')
    await drain_settle(hooks[0], 'timeout')
    # Terminal retirement must not be treated as a skippable streaming preview.
    assert edit.call_args.kwargs['finalize'] is True
    assert not adapter._approval_state
    assert edit.await_count == 1
    # A final edit tries Markdown then plain text before the single fallback notice.
    assert adapter._bot.edit_message_text.await_count == 2
    assert adapter.send.await_count == 1, 'Failed retirement edit silently loses the terminal notice'
    assert adapter.send.call_args.args[0] == '123'
    assert adapter.send.call_args.kwargs['metadata']['thread_id'] == '7'
    assert adapter.send.call_args.kwargs['metadata']['_interim_send'] is True


@pytest.mark.asyncio
async def test_failed_callback_answer_does_not_skip_retirement(surface, monkeypatch):
    adapter, hooks = surface
    assert (await publish(adapter)).success
    resolve = Mock(return_value=1)
    monkeypatch.setattr(approval, 'resolve_gateway_approval', resolve)
    query = SimpleNamespace(data='ea:once:1',
        message=SimpleNamespace(message_id=321, chat_id=123,
            chat=SimpleNamespace(type='group'), message_thread_id=7),
        from_user=SimpleNamespace(id=1, first_name='Tester'),
        answer=AsyncMock(side_effect=OSError('synthetic answerCallbackQuery failure')),
        edit_message_text=AsyncMock())
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
    await drain_settle(hooks[0], 'resolved')
    resolve.assert_called_once_with('agent:main:telegram:group:123:7', 'once', request_id='review-request')
    assert not adapter._approval_state
    assert query.edit_message_text.await_count + adapter._bot.edit_message_text.await_count >= 1, \
        'A failed toast leaves live-looking buttons with no remaining retirement owner'


@pytest.mark.asyncio
async def test_ack_after_transport_deadline_is_observed_and_card_retired(surface, monkeypatch):
    adapter, hooks = surface
    monkeypatch.setattr('plugins.platforms.telegram.adapter._TEXT_SEND_DEADLINE', 0.05)
    release = asyncio.Event()
    cancelled = asyncio.Event()
    child_tasks = []

    async def cancellation_resistant_send(**kwargs):
        child_tasks.append(asyncio.current_task())
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return SimpleNamespace(message_id=321)

    adapter._bot.send_message.side_effect = cancellation_resistant_send
    try:
        result = await asyncio.wait_for(publish(adapter), 5)
        await asyncio.wait_for(cancelled.wait(), 5)
        future = concurrent.futures.Future()
        future.set_result(result)
        assert _approval_send_outcome(future, timeout=0) == 'ambiguous'
        assert result.raw_response['exec_approval_settlement'] is True
        await drain_settle(hooks[0], 'timeout')
        release.set()
        await asyncio.wait_for(asyncio.gather(*child_tasks), 5)
        # A barrier lets any production completion observer schedule its card edit.
        await drain_settle(lambda reason: None, 'noop')
        assert adapter._bot.edit_message_text.await_count == 1, \
            'The abandoned SDK send returns a real message ID that never reaches on_sent'
    finally:
        release.set()
        if child_tasks:
            await asyncio.wait_for(asyncio.gather(*child_tasks, return_exceptions=True), 5)


@pytest.mark.asyncio
async def test_already_settled_before_publication_sends_nothing(surface, monkeypatch):
    adapter, _ = surface
    monkeypatch.setattr(approval, 'register_gateway_settle', lambda *_: False)
    result = await publish(adapter)
    assert result.success and result.raw_response['exec_approval_settlement'] is True
    assert not adapter._approval_state
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_immediate_send_failure_drops_adapter_state(surface):
    adapter, hooks = surface
    adapter._bot.send_message.side_effect = OSError('synthetic immediate failure')
    result = await publish(adapter)
    assert not result.success
    assert not adapter._approval_state
    await drain_settle(hooks[0], 'notify_failed')
    adapter._bot.edit_message_text.assert_not_called()

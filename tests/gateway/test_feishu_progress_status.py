"""Semantic header colours, independent of progress-detail prose."""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_error_history_remains_amber_after_eviction_not_green():
    from gateway.progress_cards import TurnProgress
    from gateway.platforms.base import SendResult
    adapter = SimpleNamespace(send_progress_card=AsyncMock(return_value=SendResult(success=True, message_id='card')), send=AsyncMock())
    progress = TurnProgress(adapter, 'chat', session_id='test')
    progress.record({'type': 'tool.started', 'tool_call_id': 'bad', 'tool_name': 'terminal'})
    progress.record({'type': 'tool.completed', 'tool_call_id': 'bad', 'tool_name': 'terminal', 'is_error': True})
    for n in range(30):
        progress.record({'type': 'commentary', 'text': str(n)})
    await progress.finish('completed')
    assert progress.snapshot()['status'] == 'completed_with_warnings'
    assert '异常' in '\n'.join(progress.snapshot()['details'])


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['failed', 'interrupted'])
async def test_error_history_does_not_override_failure_or_cancellation(outcome):
    from gateway.progress_cards import TurnProgress
    from gateway.platforms.base import SendResult
    adapter = SimpleNamespace(send_progress_card=AsyncMock(return_value=SendResult(success=True, message_id='card')), send=AsyncMock())
    progress = TurnProgress(adapter, 'chat', session_id='test')
    progress.record({'type': 'tool.started', 'tool_call_id': 'bad', 'tool_name': 'terminal'})
    progress.record({'type': 'tool.completed', 'tool_call_id': 'bad', 'tool_name': 'terminal', 'is_error': True})
    await progress.finish(outcome)
    assert progress.snapshot()['status'] == outcome

from tests.gateway.test_feishu_progress_cards import WiredFeishuAdapter


@pytest.mark.asyncio
async def test_confirmation_changes_header_without_copying_private_prompt():
    import asyncio
    import queue
    from gateway.progress_cards import run_progress_card
    adapter = WiredFeishuAdapter()
    ctx = SimpleNamespace(source=SimpleNamespace(chat_id='chat'), session_id='test',
        _progress_reply_to=None, _progress_metadata=None, progress_queue=queue.Queue(),
        _run_still_current=lambda: True, _progress_outcome=None, agent_holder=[None], _progress_cards=True)
    from gateway.run_turn_runner import TurnRunner
    runner = TurnRunner(None, ctx)
    runner.native_tool_start_callback('confirm', 'clarify', {'question': 'PRIVATE QUESTION'})
    task = asyncio.create_task(run_progress_card(ctx, adapter))
    try:
        for _ in range(30):
            if adapter.cards:
                break
            await asyncio.sleep(0.01)
        assert adapter.cards
        assert adapter.cards[0][1]['header']['template'] == 'orange'
        assert 'PRIVATE QUESTION' not in str(adapter.cards)
        runner.native_tool_complete_callback('confirm', 'clarify', {}, '{"user_response": "approved"}')
        ctx._progress_outcome = 'completed'
        await asyncio.wait_for(task, timeout=2)
        assert adapter.cards[-1][1]['header']['template'] == 'green'
        assert adapter.cards[-1][0] == 'om-card'
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('status,colour,label', [
    ('running', 'blue', '⏳ 处理中'),
    ('completed', 'green', '✓ 本轮已结束'),
    ('failed', 'red', '✕ 执行失败'),
    ('interrupted', 'grey', '■ 已停止'),
    ('cancelled', 'grey', '■ 已取消'),
    ('blocked', 'orange', '⚠ 遇到阻塞'),
    ('awaiting_confirmation', 'orange', '⚠ 需要你确认'),
    ('completed_with_warnings', 'orange', '⚠ 本轮结束 · 有异常记录'),
    ('unknown-state', 'grey', '○ 状态未确认'),
])
async def test_header_state_is_coloured_and_labelled_on_create_and_patch(status, colour, label):
    adapter = WiredFeishuAdapter()
    snapshot = {'status': status, 'details': ['source inspection'], 'session_id': 'status-test'}
    created = await adapter.send_progress_card('chat', snapshot)
    await adapter.send_progress_card('chat', snapshot, message_id=created.message_id)
    assert [item[0] for item in adapter.cards] == [None, 'om-card']
    for _, card, _ in adapter.cards:
        assert card['header']['template'] == colour
        assert card['header']['title']['content'] == f'Hermes · {label}'
        assert card['body']['elements'][0]['expanded'] is False
        assert 'source inspection' in str(card['body'])

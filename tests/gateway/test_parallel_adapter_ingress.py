"""Exercise real adapter-specific batching and thread normalization without transport IO."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_feishu_custom_batcher_keeps_parallel_requests_separate(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    async def produce(*args):
        pytest.fail('this test inspects batches before dispatch')

    runner, authority, _ = await setup_owner(monkeypatch, produce, platform=Platform.FEISHU)
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters[Platform.FEISHU] = adapter
    runner._wire_adapter_handlers(adapter)
    monkeypatch.setattr(adapter, '_schedule_text_batch_flush', lambda key: None)
    try:
        await adapter._enqueue_text_event(event(adapter, 'f1', 'first'))
        await adapter._enqueue_text_event(event(adapter, 'f2', 'second'))
        assert {(e.message_id, e.text) for e in adapter._pending_text_batches.values()} == {
            ('f1', 'first'), ('f2', 'second'),
        }
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_slack_channel_response_and_root_thread_reply_share_only_their_work(monkeypatch):
    from plugins.platforms.slack.adapter import SlackAdapter
    gate = asyncio.Event()

    async def produce(*args):
        await gate.wait()
        return 'DONE'

    runner, authority, delivery = await setup_owner(monkeypatch, produce, platform=Platform.SLACK)
    builder = SlackAdapter(PlatformConfig(enabled=True, token='xoxb-fixture', extra={'reply_in_thread': False}))
    builder.gateway_runner = runner
    runner.adapters[Platform.SLACK] = builder
    runner._wire_adapter_handlers(builder)
    monkeypatch.setattr(builder, 'send', delivery.send)
    monkeypatch.setattr(builder, 'send_typing', delivery.send_typing)
    monkeypatch.setattr(builder, 'get_chat_info', delivery.get_chat_info)
    monkeypatch.setattr(builder, 'edit_message', delivery.edit_message)
    monkeypatch.setattr(builder, '_resolve_user_name', AsyncMock(return_value='Alice'))
    monkeypatch.setattr(builder, '_resolve_channel_name', AsyncMock(return_value='Room'))
    monkeypatch.setattr(builder, '_humanize_user_mentions', AsyncMock(side_effect=lambda text, **kwargs: text))
    monkeypatch.setattr(builder, '_channel_prompt_with_identity', lambda *args: None)

    async def incoming(mid, thread=None):
        raw = {'ts': mid, 'channel_type': 'channel'}
        if thread:
            raw['thread_ts'] = thread
        thread_id = builder._session_thread_ts(raw, mid, False, {})
        return await builder._build_message_event(
            raw, text='request', original_text='request', command_probe_text='request',
            is_command_text=False, channel_id='room', team_id='team', ts=mid, user_id='alice',
            thread_ts=thread_id, is_dm=False, media_urls=[], media_types=[],
            media_text_inlined=[], channel_context=None,
        )

    try:
        root = await incoming('root')
        assert root.source.thread_id is None, 'reply_in_thread=false must keep channel delivery'
        await builder.handle_message(root)
        await async_wait_until(lambda: len(admissions(authority)) == 1, what='Slack root admitted')
        wrong = await incoming('wrong', 'foreign-thread')
        wrong.reply_to_message_id = 'root'
        await builder.handle_message(wrong)
        assert len(admissions(authority)) == 1
        await builder.handle_message(await incoming('follow', 'root'))
        await async_wait_until(lambda: len(admissions(authority)) == 2, what='root thread continues channel work')
        assert len({r['target_session_id'] for r in admissions(authority)}) == 1
        gate.set()
        await async_wait_until(lambda: not builder._background_tasks, what='Slack output receipts confirmed')
        output = next(mid for _, text, _, mid in delivery.deliveries if text == 'DONE')
        await builder.handle_message(await incoming('output-thread-follow', output))
        assert len(admissions(authority)) == 2, 'an unrelated root-thread scope must not select another work'
        assert 'No work was changed' in delivery.sent[-1]
    finally:
        gate.set()
        await async_wait_until(lambda: not builder._background_tasks, what='Slack cleanup')
        authority.db.close()

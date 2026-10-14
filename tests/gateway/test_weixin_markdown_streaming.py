"""Weixin Markdown output is independent of transport delta partitioning."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.weixin_markdown import StreamingMarkdownFilter, filter_markdown
from gateway.stream_consumer import StreamConsumerConfig
from gateway.stream_consumer_factory import create_stream_consumer


@pytest.mark.parametrize("raw, expected", [
    ("*中文* _韩文한글_ ***混合English*** ___漢字___", "中文 韩文한글 混合English 漢字"),
    ("*English* **中文粗体** ___English___", "*English* **中文粗体** ___English___"),
    ("##### 标题\n###### 标题\n#### 保留\n####### 保留", "标题\n标题\n#### 保留\n####### 保留"),
    ("***\n_ _ _\n---\n> *中文*\n|**中文**|", "***\n_ _ _\n---\n> 中文\n|**中文**|"),
    ("a ![图片](https://example.test/image.png) b", "a  b"),
    ("![incomplete](url", "![incomplete](url"),
    ("*未闭合\n后文 ![普通]文本", "*未闭合\n后文 ![普通]文本"),
    ("```python\n*中文*\n![literal](url)\n```\n*中文*", "```python\n*中文*\n![literal](url)\n```\n中文"),
    ("`*中文* ![literal](url)` 和 ``_中文_``", "`*中文* ![literal](url)` 和 ``_中文_``"),
])
def test_filter_is_partition_independent(raw, expected):
    assert filter_markdown(raw) == expected
    for cut in range(len(raw) + 1):
        parser = StreamingMarkdownFilter()
        assert parser.feed(raw[:cut]) + parser.feed(raw[cut:]) + parser.flush() == expected
    parser = StreamingMarkdownFilter()
    assert "".join(parser.feed(char) for char in raw) + parser.flush() == expected


def consumer(*, buffer_only=False):
    sends = []
    changed = asyncio.Event()

    async def send(chat_id, content, **kwargs):
        sends.append((content, kwargs))
        changed.set()
        return SendResult(success=True, message_id=str(len(sends)))

    adapter = SimpleNamespace(
        SUPPORTS_BLOCK_STREAMING=True, SUPPORTS_MESSAGE_EDITING=False, SUPPORTS_NATIVE_STREAMING=False,
        config=PlatformConfig(extra={"block_streaming": {"min_chars": 1, "idle_ms": 50}}),
        send=send, edit_message=AsyncMock(side_effect=AssertionError("Weixin cannot edit")),
    )
    stream = create_stream_consumer(adapter=adapter, chat_id="peer", config=StreamConsumerConfig(buffer_only=buffer_only))
    return stream, sends, changed


@pytest.mark.asyncio
async def test_blocks_send_before_completion_and_final_footer_only_once():
    stream, sends, changed = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta("*中文*第一段\n")
    await asyncio.wait_for(changed.wait(), 5)
    assert sends[0][0] == "中文第一段\n"
    changed.clear()
    stream.on_delta("第二段")
    await asyncio.wait_for(changed.wait(), 5)
    stream.finish("*中文*第一段\n第二段\n最终说明")
    await asyncio.wait_for(task, 5)
    assert "".join(text for text, _ in sends) == "中文第一段\n第二段\n最终说明"
    assert all("▉" not in text for text, _ in sends)
    assert stream.delivered_final_matches("*中文*第一段\n第二段\n最终说明") is True
    assert stream.final_content_delivered


@pytest.mark.asyncio
async def test_code_media_think_and_silence_remain_hidden_until_resolved():
    stream, sends, changed = consumer(buffer_only=True)
    stream.on_delta("<think>private reasoning</think>\n```python\nprint('*中文*')\n```\nMEDIA:/tmp/image.png")
    stream.finish("```python\nprint('*中文*')\n```\nMEDIA:/tmp/image.png")
    await stream.run()
    assert len(sends) == 1
    assert "private reasoning" not in sends[0][0] and "MEDIA:" not in sends[0][0]
    assert "print('*中文*')" in sends[0][0]
    silent, deliveries, _ = consumer()
    silent.on_delta("NO_REPLY")
    silent.finish("NO_REPLY")
    await silent.run()
    assert deliveries == []


@pytest.mark.asyncio
async def test_segment_boundary_and_stop_preserve_delivery_ownership():
    stream, sends, _ = consumer()
    stream.on_delta("准备工作")
    stream.on_delta(None)
    stream.on_delta("最终回答")
    stream.finish("最终回答")
    await stream.run()
    assert [text for text, _ in sends] == ["准备工作", "最终回答"]
    assert stream.delivered_final_matches("最终回答") is True
    stopped, deliveries, _ = consumer()
    stopped._run_still_current = lambda: False
    stopped.on_delta("stale reply")
    stopped.finish("stale reply")
    await stopped.run()
    assert deliveries == []


@pytest.mark.asyncio
async def test_approval_flush_and_failed_final_fallback_do_not_repeat_acknowledged_blocks():
    from gateway.platforms.weixin_streaming import unsent_block_tail
    stream, sends, _ = consumer()
    stream.on_delta("审批前的说明")
    boundary = stream.close_for_approval_prompt()
    task = asyncio.create_task(stream.run())
    assert await asyncio.wait_for(boundary, 5) is True
    assert sends[0][0] == "审批前的说明"
    assert sends[0][1]["metadata"]["_interim_send"] is True
    stream.on_delta("审批后回答")
    stream.finish("审批后回答")
    await task
    assert [text for text, _ in sends] == ["审批前的说明", "审批后回答"]

    failed, deliveries, changed = consumer()
    task = asyncio.create_task(failed.run())
    failed.on_delta("已经送达\n")
    await asyncio.wait_for(changed.wait(), 5)
    failed.adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
    failed.finish("已经送达\n待补发\nMEDIA:/tmp/image.png")
    await task
    assert failed.final_content_delivered is False
    tail = unsent_block_tail("已经送达\n待补发\nMEDIA:/tmp/image.png", failed.acknowledged_block_prefix())
    assert "已经送达" not in tail and "待补发" in tail and "MEDIA:" in tail


@pytest.mark.asyncio
async def test_real_turn_callbacks_and_proxy_select_blocks_and_honor_opt_outs(monkeypatch):
    from gateway.config import GatewayConfig, Platform, StreamingConfig
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.display_config import resolve_display_setting
    seed, deliveries, _ = consumer()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(streaming=StreamingConfig(enabled=True))
    runner.adapters = {Platform.WEIXIN: seed.adapter}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    source = SessionSource(Platform.WEIXIN, "peer")
    ctx = SimpleNamespace(
        mute_notification_reply=False, streaming_tts_consumer_holder=[None], user_config={}, source=source,
        scheduled_heartbeat=False, interim_assistant_messages_enabled=False,
        resolve_display_setting=resolve_display_setting, _status_thread_metadata=None, progress_queue=None,
        event_message_id=None, _run_still_current=lambda: True, stream_consumer_holder=[None],
    )
    turn = TurnRunner(runner, ctx)
    stream, delta, interim, _ = turn._setup_stream_consumer("weixin")
    assert stream is not None and stream.stream_deltas_enabled
    delta("准备说明")
    interim("准备说明", already_streamed=True)
    delta("*中文最终回答*")
    stream.finish("*中文最终回答*")
    await stream.run()
    assert [text for text, _ in deliveries] == ["准备说明", "中文最终回答"]
    result = {"final_response": "*中文最终回答*"}
    runner._block_stream_delivery_state(result, stream, seal=True)
    assert result["already_sent"] is True
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    assert type(runner._proxy_stream_consumer(source, None, None, lambda: True)) is type(stream)
    ctx.resolve_display_setting = lambda *_: False
    assert turn._setup_stream_consumer("weixin")[0] is None
    runner.config.streaming.enabled = False
    assert runner._proxy_stream_consumer(source, None, None, lambda: True) is None

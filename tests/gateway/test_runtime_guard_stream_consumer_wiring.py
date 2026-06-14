from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.runtime_guard import GuardContext, GuardDecision, register_runtime_guard_provider
from gateway.run import _stream_response_previewed
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _adapter(extra: dict | None = None):
    adapter = MagicMock()
    adapter.config = PlatformConfig(enabled=True, extra=extra or {})
    adapter.platform = Platform.DISCORD
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))
    return adapter


def test_stream_consumer_runtime_guard_disabled_does_not_disable_streaming():
    consumer = GatewayStreamConsumer(_adapter(), "chat-1")
    context = GuardContext(surface="assistant_stream", platform=Platform.DISCORD, chat_id="chat-1")

    assert consumer.runtime_guard_disables_streaming(context) is False


def test_stream_consumer_ignores_unrelated_platform_extra_enabled_key():
    consumer = GatewayStreamConsumer(_adapter({"enabled": True}), "chat-1")
    context = GuardContext(surface="assistant_stream", platform=Platform.DISCORD, chat_id="chat-1")

    assert consumer.runtime_guard_disables_streaming(context) is False


@pytest.mark.asyncio
async def test_stream_consumer_runtime_guard_enabled_disable_policy_blocks_visible_stream():
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "dry_run": False,
                "scope": {"platforms": ["discord"], "chat_ids": ["chat-1"]},
                "streaming": {"policy": "disable"},
            }
        }
    )
    consumer = GatewayStreamConsumer(adapter, "chat-1")
    context = GuardContext(surface="assistant_stream", platform=Platform.DISCORD, chat_id="chat-1")

    assert consumer.runtime_guard_disables_streaming(context) is True
    assert await consumer._send_or_edit("visible streaming text") is False
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_stream_consumer_guard_first_visible_provider_deny_blocks_before_send():
    seen = []

    class DenyingGuard:
        def check(self, context):
            seen.append(context)
            return GuardDecision.block(reason="lease_conflict", status="denied")

    register_runtime_guard_provider("deny_first_visible_stream_test", DenyingGuard())
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "provider": "deny_first_visible_stream_test",
                "dry_run": False,
                "scope": {"platforms": ["discord"], "chat_ids": ["chat-1"]},
                "streaming": {"policy": "guard_first_visible"},
            }
        }
    )
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    assert await consumer._send_or_edit("visible streaming text") is False
    adapter.send.assert_not_called()
    assert seen
    assert seen[0].surface == "assistant_stream"


@pytest.mark.asyncio
async def test_stream_consumer_guard_first_visible_provider_error_blocks_before_send():
    class ExplodingGuard:
        def check(self, context):
            raise RuntimeError("lease provider unavailable")

    register_runtime_guard_provider("explode_first_visible_stream_test", ExplodingGuard())
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "provider": "explode_first_visible_stream_test",
                "dry_run": False,
                "fail_closed": True,
                "scope": {"platforms": ["discord"], "chat_ids": ["chat-1"]},
                "streaming": {"policy": "guard_first_visible"},
            }
        }
    )
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    assert await consumer._send_or_edit("visible streaming text") is False
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_stream_consumer_disable_policy_blocks_direct_new_chunk_send():
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "dry_run": False,
                "scope": {"platforms": ["discord"], "chat_ids": ["chat-1"]},
                "streaming": {"policy": "disable"},
            }
        }
    )
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    assert await consumer._send_new_chunk("visible chunk", "reply-1") == "reply-1"
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_stream_consumer_disable_policy_blocks_oversized_first_visible_send():
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "dry_run": False,
                "scope": {"platforms": ["discord"], "chat_ids": ["chat-1"]},
                "streaming": {"policy": "disable"},
            }
        }
    )
    adapter.MAX_MESSAGE_LENGTH = 550
    adapter.truncate_message.side_effect = (
        lambda text, limit, len_fn=None: [text[:limit], text[limit:]]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(buffer_threshold=1),
    )
    consumer.on_delta("x" * 700)
    consumer.finish()

    await consumer.run()

    adapter.send.assert_not_called()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
    assert _stream_response_previewed(consumer, "x" * 700) is False


@pytest.mark.asyncio
async def test_stream_response_previewed_requires_actual_delivery_confirmation():
    adapter = _adapter()
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    assert _stream_response_previewed(consumer, "model text") is False

    consumer._final_content_delivered = True
    assert _stream_response_previewed(consumer, "model text") is True


@pytest.mark.asyncio
async def test_stream_consumer_runtime_guard_can_scope_on_session_and_parent_metadata():
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "dry_run": False,
                "scope": {
                    "platforms": ["discord"],
                    "chat_ids": ["chat-1"],
                    "thread_ids": ["thread-1"],
                    "parent_chat_ids": ["parent-1"],
                    "guild_ids": ["guild-1"],
                    "user_ids": [],
                    "session_keys": ["session-1"],
                },
                "streaming": {"policy": "disable"},
            }
        }
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        metadata={
            "thread_id": "thread-1",
            "parent_chat_id": "parent-1",
            "guild_id": "guild-1",
            "user_id": "user-1",
            "session_key": "session-1",
        },
    )

    assert await consumer._send_or_edit("visible streaming text") is False
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_stream_consumer_runtime_guard_dry_run_disable_policy_does_not_block_streaming():
    adapter = _adapter(
        {
            "runtime_guard": {
                "enabled": True,
                "dry_run": True,
                "scope": {"platforms": ["discord"], "chat_ids": ["chat-1"]},
                "streaming": {"policy": "disable"},
            }
        }
    )
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    assert await consumer._send_or_edit("visible streaming text") is True
    adapter.send.assert_awaited_once()

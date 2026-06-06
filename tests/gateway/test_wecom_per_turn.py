"""Tests for per-turn stream isolation and concurrent consumer scenarios."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig


class TestPerTurnStreamIsolation:
    """Verify that concurrent consumers with different turn_ids don't interfere."""

    @pytest.mark.asyncio
    async def test_concurrent_turns_same_chat_isolated(self):
        """Two concurrent consumers in same chat maintain independent streams."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        # Consumer 1 starts streaming
        await adapter.send_stream_frame("consumer1 frame1", chat_id="chat-1", turn_id="turn-1")
        assert "chat-1:turn-1" in adapter._stream_turns

        # Consumer 2 starts streaming (concurrent)
        await adapter.send_stream_frame("consumer2 frame1", chat_id="chat-1", turn_id="turn-2")
        assert "chat-1:turn-2" in adapter._stream_turns

        # Both turns coexist
        assert len([k for k in adapter._stream_turns if k.startswith("chat-1:")]) == 2

        # Consumer 1 finalizes
        ok1 = await adapter.send_stream_frame(
            "consumer1 final", chat_id="chat-1", finalize=True, turn_id="turn-1"
        )
        assert ok1 is True
        assert "chat-1:turn-1" not in adapter._stream_turns
        # Consumer 2's turn still exists
        assert "chat-1:turn-2" in adapter._stream_turns

        # Consumer 2 finalizes
        ok2 = await adapter.send_stream_frame(
            "consumer2 final", chat_id="chat-1", finalize=True, turn_id="turn-2"
        )
        assert ok2 is True
        assert "chat-1:turn-2" not in adapter._stream_turns

    @pytest.mark.asyncio
    async def test_one_turn_expired_other_continues(self):
        """When one turn hits stream expired, other concurrent turns can continue."""
        from gateway.platforms.wecom import STREAM_EXPIRED_ERRCODE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)

        # Consumer 1 and 2 both start
        await adapter.send_stream_frame("c1 frame", chat_id="chat-1", turn_id="turn-1")
        await adapter.send_stream_frame("c2 frame", chat_id="chat-1", turn_id="turn-2")
        assert "chat-1:turn-1" in adapter._stream_turns
        assert "chat-1:turn-2" in adapter._stream_turns

        # Consumer 1 hits expired error on finalize
        adapter._send_reply_request = AsyncMock(
            return_value={"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "stream expired"}
        )
        ok1 = await adapter.send_stream_frame(
            "c1 final", chat_id="chat-1", finalize=True, turn_id="turn-1"
        )
        assert ok1 is False
        assert "chat-1" in adapter._stream_expired_chats
        assert "chat-1:turn-1" not in adapter._stream_turns  # turn-1 cleaned up

        # Consumer 2's existing turn can still finalize
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})
        ok2 = await adapter.send_stream_frame(
            "c2 final", chat_id="chat-1", finalize=True, turn_id="turn-2"
        )
        assert ok2 is True  # ✅ turn-2 not blocked by chat-level expired
        assert "chat-1:turn-2" not in adapter._stream_turns

    @pytest.mark.asyncio
    async def test_expired_chat_blocks_new_turn_creation(self):
        """After one turn expired, new turn creation is blocked."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._stream_expired_chats.add("chat-1")
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        # Try to create a new turn after chat is expired
        ok = await adapter.send_stream_frame("new frame", chat_id="chat-1", turn_id="new-turn")
        assert ok is False
        assert "chat-1:new-turn" not in adapter._stream_turns


class TestNativeFallbackStreamClose:
    """Verify that native streaming fallback closes open streams."""

    @pytest.mark.asyncio
    async def test_native_fallback_closes_stream_on_success(self):
        """When native fails mid-stream, best-effort finalize succeeds."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
        from gateway.platforms.base import BasePlatformAdapter

        class MockAdapter(BasePlatformAdapter):
            MAX_MESSAGE_LENGTH = 4096
            SUPPORTS_MESSAGE_EDITING = False
            SUPPORTS_NATIVE_STREAMING = True

            def __init__(self):
                self._typing_paused = set()
                self.frames = []
                self.frame_count = 0

            def supports_native_streaming(self, chat_type=None, metadata=None):
                return True

            async def send_stream_frame(
                self, text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
            ):
                self.frame_count += 1
                self.frames.append({"text": text, "finalize": finalize})
                # First 2 frames succeed, 3rd fails (non-expired error)
                if self.frame_count == 3:
                    raise RuntimeError("network error")
                # 4th frame (finalize in fallback) succeeds
                return True

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.frames.append({"send": content})
                return type("SendResult", (), {"success": True, "message_id": "msg-1"})()

        MockAdapter.__abstractmethods__ = frozenset()
        adapter = MockAdapter()
        cfg = StreamConsumerConfig(chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        # Send enough to trigger frames
        consumer.on_delta("First frame content that exceeds the threshold.")
        consumer.on_delta(" Second frame content also exceeds threshold.")
        consumer.on_delta(" Third will fail.")

        import asyncio
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)
        consumer.finish()
        await task

        # Should have: seed, frame1, frame2, (frame3 fails), finalize in fallback
        # finalize succeeds → no send() fallback needed
        assert len([f for f in adapter.frames if f.get("finalize")]) >= 1
        # Verify no send() fallback happened (stream was closed successfully)
        assert len([f for f in adapter.frames if "send" in f]) == 0

    @pytest.mark.asyncio
    async def test_native_fallback_falls_to_send_on_finalize_fail(self):
        """When native fails and finalize also fails, falls through to send()."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
        from gateway.platforms.base import BasePlatformAdapter

        class MockAdapter(BasePlatformAdapter):
            MAX_MESSAGE_LENGTH = 4096
            SUPPORTS_MESSAGE_EDITING = False
            SUPPORTS_NATIVE_STREAMING = True

            def __init__(self):
                self._typing_paused = set()
                self.frames = []
                self.frame_count = 0

            def supports_native_streaming(self, chat_type=None, metadata=None):
                return True

            async def send_stream_frame(
                self, text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
            ):
                self.frame_count += 1
                self.frames.append({"text": text, "finalize": finalize})
                # All frames fail (simulating complete stream failure)
                if self.frame_count >= 2:
                    raise RuntimeError("stream dead")
                return True

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.frames.append({"send": content})
                return type("SendResult", (), {"success": True, "message_id": "msg-1"})()

        MockAdapter.__abstractmethods__ = frozenset()
        adapter = MockAdapter()
        cfg = StreamConsumerConfig(chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        consumer.on_delta("Content that will cause stream to fail.")

        import asyncio
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)
        consumer.finish()
        await task

        # Finalize failed → should fall through to send()
        assert len([f for f in adapter.frames if "send" in f]) == 1

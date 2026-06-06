"""Tests for per-turn stream isolation and concurrent consumer scenarios."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig


class TestPerTurnStreamIsolation:
    """Verify that concurrent consumers with different turn_ids don't interfere."""

    @pytest.mark.asyncio
    async def test_multiple_users_concurrent_streaming(self):
        """Multiple users (different chats) streaming concurrently don't interfere."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            # Setup 3 different users/chats
            adapter._last_chat_req_ids["user-1"] = "req-1"
            adapter._last_chat_req_ids["user-2"] = "req-2"
            adapter._last_chat_req_ids["user-3"] = "req-3"
            adapter._send_json = AsyncMock()
            adapter._ws = AsyncMock(closed=False)
            adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

            # User 1, 2, 3 all start streaming simultaneously
            await adapter.send_stream_frame("user1 content", chat_id="user-1", turn_id="turn-1")
            await adapter.send_stream_frame("user2 content", chat_id="user-2", turn_id="turn-2")
            await adapter.send_stream_frame("user3 content", chat_id="user-3", turn_id="turn-3")

            # All 3 turns active
            assert "user-1:turn-1" in adapter._stream_turns
            assert "user-2:turn-2" in adapter._stream_turns
            assert "user-3:turn-3" in adapter._stream_turns

            # User 2 finishes first
            ok2 = await adapter.send_stream_frame(
                "user2 final", chat_id="user-2", finalize=True, turn_id="turn-2"
            )
            assert ok2 is True
            assert "user-2:turn-2" not in adapter._stream_turns
            # User 1 and 3 still active
            assert "user-1:turn-1" in adapter._stream_turns
            assert "user-3:turn-3" in adapter._stream_turns

            # User 1 finishes
            ok1 = await adapter.send_stream_frame(
                "user1 final", chat_id="user-1", finalize=True, turn_id="turn-1"
            )
            assert ok1 is True
            assert "user-1:turn-1" not in adapter._stream_turns
            # User 3 still active
            assert "user-3:turn-3" in adapter._stream_turns

            # User 3 finishes
            ok3 = await adapter.send_stream_frame(
                "user3 final", chat_id="user-3", finalize=True, turn_id="turn-3"
            )
            assert ok3 is True
            assert "user-3:turn-3" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_turns_same_chat_isolated(self):
        """Two concurrent consumers in same chat maintain independent streams."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["chat-1"] = "req-1"
            adapter._send_json = AsyncMock()
            # Mock _ws with closed=False and async close()
            adapter._ws = AsyncMock(closed=False)
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
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_one_user_expired_others_unaffected(self):
        """User A hits stream expired; Users B and C continue normally."""
        from gateway.platforms.wecom import STREAM_EXPIRED_ERRCODE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["user-A"] = "req-A"
            adapter._last_chat_req_ids["user-B"] = "req-B"
            adapter._last_chat_req_ids["user-C"] = "req-C"
            adapter._send_json = AsyncMock()
            adapter._ws = AsyncMock(closed=False)

            # All 3 users start streaming
            await adapter.send_stream_frame("A content", chat_id="user-A", turn_id="turn-A")
            await adapter.send_stream_frame("B content", chat_id="user-B", turn_id="turn-B")
            await adapter.send_stream_frame("C content", chat_id="user-C", turn_id="turn-C")

            # User A hits stream expired
            adapter._send_reply_request = AsyncMock(
                return_value={"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "expired"}
            )
            okA = await adapter.send_stream_frame(
                "A final", chat_id="user-A", finalize=True, turn_id="turn-A"
            )
            assert okA is False
            assert "user-A" in adapter._stream_expired_chats
            assert "user-A:turn-A" not in adapter._stream_turns

            # Users B and C should NOT be affected (different chats)
            adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})
            okB = await adapter.send_stream_frame(
                "B final", chat_id="user-B", finalize=True, turn_id="turn-B"
            )
            okC = await adapter.send_stream_frame(
                "C final", chat_id="user-C", finalize=True, turn_id="turn-C"
            )
            assert okB is True  # ✅ User B unaffected
            assert okC is True  # ✅ User C unaffected
            assert "user-B:turn-B" not in adapter._stream_turns
            assert "user-C:turn-C" not in adapter._stream_turns

            # Only user-A is in expired list
            assert "user-A" in adapter._stream_expired_chats
            assert "user-B" not in adapter._stream_expired_chats
            assert "user-C" not in adapter._stream_expired_chats
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_one_turn_expired_other_continues(self):
        """When one turn hits stream expired, other concurrent turns can continue."""
        from gateway.platforms.wecom import STREAM_EXPIRED_ERRCODE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["chat-1"] = "req-1"
            adapter._send_json = AsyncMock()
            adapter._ws = AsyncMock(closed=False)

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
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_expired_chat_blocks_new_turn_creation(self):
        """After one turn expired, new turn creation is blocked."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["chat-1"] = "req-1"
            adapter._stream_expired_chats.add("chat-1")
            adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

            # Try to create a new turn after chat is expired
            ok = await adapter.send_stream_frame("new frame", chat_id="chat-1", turn_id="new-turn")
            assert ok is False
            assert "chat-1:new-turn" not in adapter._stream_turns
        finally:
            await adapter.disconnect()


class TestNativeFallbackStreamClose:
    """Verify that native streaming fallback closes open streams."""

    @pytest.mark.asyncio
    async def test_seed_success_first_frame_fails_still_finalizes(self):
        """Seed frame opens stream bubble, first content frame fails → finalize called.

        This is the critical edge case: seed frame has length 0 but opens the
        WeCom typing bubble. If the first content frame fails, we must still
        finalize based on _native_stream_opened, not _native_last_pushed_len.
        """
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
        from gateway.platforms.base import BasePlatformAdapter

        class MockAdapter(BasePlatformAdapter):
            MAX_MESSAGE_LENGTH = 4096
            SUPPORTS_MESSAGE_EDITING = False
            SUPPORTS_NATIVE_STREAMING = True

            def __init__(self):
                self._typing_paused = set()
                self.send_stream_frame_calls = []
                self.send_calls = []
                self.should_fail_first_content = True

            def supports_native_streaming(self, chat_type=None, metadata=None):
                return True

            async def send_stream_frame(
                self, text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
            ):
                call_info = {"text_len": len(text), "finalize": finalize, "text_preview": text[:20]}
                self.send_stream_frame_calls.append(call_info)

                # Seed frame (empty) always succeeds
                if len(text) == 0 and not finalize:
                    return True

                # First non-seed, non-finalize frame fails
                if self.should_fail_first_content and len(text) > 0 and not finalize:
                    self.should_fail_first_content = False
                    raise RuntimeError("first content frame failed")

                # Finalize frames and subsequent content frames succeed
                return True

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.send_calls.append({"content_preview": content[:20]})
                return type("SendResult", (), {"success": True, "message_id": "msg-1"})()

        MockAdapter.__abstractmethods__ = frozenset()
        adapter = MockAdapter()
        cfg = StreamConsumerConfig(chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        # Send short content to minimize frame count
        consumer.on_delta("X")

        import asyncio
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Verify: seed succeeded, then finalize was attempted (not skipped)
        assert len(adapter.send_stream_frame_calls) >= 2
        # First call: seed (length 0)
        assert adapter.send_stream_frame_calls[0]["text_len"] == 0
        assert not adapter.send_stream_frame_calls[0]["finalize"]

        # At least one finalize call should have been made
        finalize_calls = [c for c in adapter.send_stream_frame_calls if c["finalize"]]
        assert len(finalize_calls) >= 1, "Finalize should be called even though seed had length 0"

        # No send() fallback (finalize succeeded)
        assert len(adapter.send_calls) == 0

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

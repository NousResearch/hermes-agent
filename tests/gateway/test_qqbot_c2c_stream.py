"""C2C native streaming on the in-tree QQ adapter.

QQ updates one private-chat bubble via stream_messages. Group and guild chats
stay on send(). A frame that fails before QQ has shown anything falls back to
one send(); a later failure must not send the whole answer again.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _adapter():
    from plugins.platforms.qqbot.adapter import QQAdapter

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    adapter._chat_type_map["openid"] = "c2c"
    adapter._last_msg_id["openid"] = "in-1"
    return adapter


def _record(responses):
    posts = []

    async def _api(method, path, body=None, timeout=30.0):
        posts.append({"method": method, "path": path, "body": body})
        response = responses[len(posts) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    return posts, _api


class TestC2CStreamFrames:
    @pytest.mark.asyncio
    async def test_empty_seed_does_not_call_qq(self):
        adapter = _adapter()
        adapter._api_request = mock.AsyncMock()
        assert await adapter.send_stream_frame("", chat_id="openid", turn_id="t1") is True
        adapter._api_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cumulative_frames_share_seq_and_close_on_finish(self):
        adapter = _adapter()
        posts, api = _record([{"id": "stream-9"}, {"id": "stream-9"}, {"id": "stream-9"}])
        adapter._api_request = api

        assert await adapter.send_stream_frame("Hel", chat_id="openid", reply_to="in-9", turn_id="t1")
        assert await adapter.send_stream_frame("Hello", chat_id="openid", reply_to="in-9", turn_id="t1")
        assert await adapter.send_stream_frame(
            "Hello!", chat_id="openid", reply_to="in-9", turn_id="t1", finalize=True)
        assert await adapter.send_stream_frame(
            "Hello!", chat_id="openid", reply_to="in-9", turn_id="t1", finalize=True)

        assert [item["path"] for item in posts] == ["/v2/users/openid/stream_messages"] * 3
        first, second, final = (item["body"] for item in posts)
        assert "stream_msg_id" not in first
        assert first["index"] == 0 and first["input_state"] == 1
        assert first["input_mode"] == "replace" and first["content_type"] == "markdown"
        assert first["msg_id"] == first["event_id"] == "in-9"
        assert second["stream_msg_id"] == final["stream_msg_id"] == "stream-9"
        assert second["index"] == 1 and final["index"] == 2
        assert final["input_state"] == 10
        assert first["msg_seq"] == second["msg_seq"] == final["msg_seq"]
        assert final["content_raw"] == "Hello!"

    @pytest.mark.asyncio
    async def test_turns_do_not_share_a_stream(self):
        adapter = _adapter()
        posts, api = _record([{"id": "one"}, {"id": "two"}])
        adapter._api_request = api
        await adapter.send_stream_frame("A", chat_id="openid", turn_id="t1")
        await adapter.send_stream_frame("B", chat_id="openid", turn_id="t2")
        assert "stream_msg_id" not in posts[0]["body"]
        assert "stream_msg_id" not in posts[1]["body"]
        assert posts[0]["body"]["index"] == posts[1]["body"]["index"] == 0

    @pytest.mark.asyncio
    async def test_non_c2c_and_missing_passive_id_do_not_post(self):
        adapter = _adapter()
        adapter._api_request = mock.AsyncMock()
        adapter._chat_type_map["group"] = "group"
        assert await adapter.send_stream_frame("hi", chat_id="group", turn_id="t1") is False
        adapter._last_msg_id.clear()
        assert await adapter.send_stream_frame("hi", chat_id="openid", turn_id="t1") is False
        adapter._api_request.assert_not_awaited()
        assert adapter.supports_native_streaming("group") is False
        assert adapter.supports_native_streaming("dm") is True

    @pytest.mark.asyncio
    async def test_prefix_regression_replays_the_accepted_text(self):
        adapter = _adapter()
        posts, api = _record([{"id": "stream-9"}, {"id": "stream-9"}])
        adapter._api_request = api
        await adapter.send_stream_frame("Hello", chat_id="openid", turn_id="t1")
        await adapter.send_stream_frame("nope", chat_id="openid", turn_id="t1")
        await adapter.send_stream_frame("Hello world", chat_id="openid", turn_id="t1")
        assert len(posts) == 2
        assert posts[1]["body"]["content_raw"] == "Hello world"
        assert posts[1]["body"]["index"] == 1

    @pytest.mark.asyncio
    async def test_open_failure_blocks_the_turn_and_later_failure_does_not(self):
        adapter = _adapter()
        posts, api = _record([RuntimeError("down")])
        adapter._api_request = api
        assert await adapter.send_stream_frame("hi", chat_id="openid", turn_id="dead") is False
        assert await adapter.send_stream_frame("hi", chat_id="openid", turn_id="dead", finalize=True) is False
        assert len(posts) == 1

        live = _adapter()
        calls = {"n": 0}

        async def flaky(method, path, body=None, timeout=30.0):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("blip")
            return {"id": "stream-9"}

        live._api_request = flaky
        assert await live.send_stream_frame("Hel", chat_id="openid", turn_id="live") is True
        assert await live.send_stream_frame("Hello", chat_id="openid", turn_id="live") is True
        assert await live.send_stream_frame("Hello!", chat_id="openid", turn_id="live", finalize=True) is True
        assert calls["n"] == 3


class TestNativeConsumerContract:
    @pytest.mark.asyncio
    async def test_successful_native_frames_do_not_send(self):
        adapter = _adapter()
        posts, api = _record([{"id": "stream-9"}, {"id": "stream-9"}])
        adapter._api_request = api
        adapter.send = mock.AsyncMock()
        consumer = GatewayStreamConsumer(
            adapter, "openid", StreamConsumerConfig(chat_type="dm", cursor=""),
            initial_reply_to_id="in-1")
        consumer._use_native_streaming = True
        consumer._native_stream_opened = True

        assert await consumer._send_or_edit("Hello", finalize=False)
        assert await consumer._send_or_edit("Hello world", finalize=True)
        adapter.send.assert_not_awaited()
        assert posts[-1]["body"]["input_state"] == 10
        assert posts[-1]["body"]["content_raw"] == "Hello world"
        assert consumer._final_content_delivered is True

    @pytest.mark.asyncio
    async def test_open_failure_falls_back_to_one_send(self):
        adapter = _adapter()
        posts, api = _record([RuntimeError("down")])
        adapter._api_request = api
        adapter.send = mock.AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="sent"))
        consumer = GatewayStreamConsumer(
            adapter, "openid", StreamConsumerConfig(chat_type="dm", cursor=""),
            initial_reply_to_id="in-1")
        consumer._use_native_streaming = True
        consumer._native_stream_opened = True

        assert await consumer._send_or_edit("Hello", finalize=False)
        assert adapter.send.await_count == 1
        assert len(posts) == 1


def test_qq_stream_cursor_is_blank_even_when_editing_is_enabled():
    from gateway.run_turn import GatewayTurnMixin

    source = mock.Mock(platform=Platform.QQBOT, chat_id="openid", chat_type="dm")
    scfg = mock.Mock(
        cursor="▉", edit_interval=0.3, buffer_threshold=20,
        fresh_final_after_seconds=5, transport="edit")
    adapter = mock.Mock(SUPPORTS_MESSAGE_EDITING=True, SUPPORTS_NATIVE_STREAMING=True)
    config, _pause = GatewayTurnMixin._build_stream_consumer_config(
        object(), source, scfg, adapter, on_missing_cursor="fallback")
    assert config.cursor == ""

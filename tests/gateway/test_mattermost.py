"""Tests for Mattermost platform adapter."""
import json
import os
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.run import (
    _resolve_gateway_display_bool,
    _resolve_progress_thread_id,
)


class TestMattermostProgressThreadRouting:
    def test_top_level_mattermost_progress_uses_event_message_id(self):
        assert _resolve_progress_thread_id(
            Platform.MATTERMOST,
            source_thread_id=None,
            event_message_id="top_post_123",
        ) == "top_post_123"


class TestMattermostDisplayHygiene:

    def test_mattermost_platform_opt_in_can_enable_interim_assistant_messages(self):
        """Mattermost can still opt into commentary explicitly per platform."""
        user_config = {
            "display": {
                "interim_assistant_messages": False,
                "platforms": {
                    "mattermost": {"interim_assistant_messages": True},
                },
            }
        }

        assert _resolve_gateway_display_bool(
            user_config,
            "mattermost",
            "interim_assistant_messages",
            default=True,
            platform=Platform.MATTERMOST,
            require_platform_override_for={Platform.MATTERMOST},
        ) is True


    def test_global_thinking_progress_still_applies_to_other_platforms(self):
        """The Mattermost guard must not silently neuter Telegram/other chats."""
        user_config = {"display": {"thinking_progress": True}}

        assert _resolve_gateway_display_bool(
            user_config,
            "telegram",
            "thinking_progress",
            default=False,
            platform=Platform.TELEGRAM,
            require_platform_override_for={Platform.MATTERMOST},
        ) is True


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestMattermostConfigLoading:
    @pytest.mark.parametrize("route", ["direct", "cron"])
    def test_yaml_limit_is_profile_local_across_delivery(self, tmp_path, monkeypatch, route):
        import asyncio
        from types import SimpleNamespace
        from gateway.config import load_gateway_config
        from gateway.platform_registry import platform_registry
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        from tools.send_message_tool import _send_to_platform, prepare_send_message_platforms
        from cron.scheduler_delivery import _standalone_send

        monkeypatch.setenv("MATTERMOST_TOKEN", "test-mattermost-token")
        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
        monkeypatch.setenv("MATTERMOST_MAX_POST_LENGTH", "9000")
        prepare_send_message_platforms()
        registry_limit = platform_registry.get("mattermost").max_message_length
        adapters = []
        for name, limit in [("small", 500), ("large", 8000), ("default", 4000)]:
            home = tmp_path / name
            home.mkdir()
            (home / "config.yaml").write_text(
                f"mattermost:\n  max_post_length: {limit}\n" if name != "default" else "{}\n"
            )
            monkeypatch.setenv("HERMES_HOME", str(home))
            config = load_gateway_config().platforms[Platform.MATTERMOST]
            adapter = MattermostAdapter(config)
            adapters.append(adapter)
            session = MagicMock()
            response = AsyncMock()
            response.status = 201
            response.json.return_value = {"id": "ack"}
            response.__aenter__.return_value = response
            session.post.return_value = response
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            with patch("aiohttp.ClientSession", return_value=session), patch(
                "tools.send_message_tool._live_adapter", return_value=(None, None)
            ):
                if route == "direct":
                    result = asyncio.run(_send_to_platform(Platform.MATTERMOST, config, "channel", "x" * 6000))
                else:
                    target = SimpleNamespace(job={"id": "job"}, where="mattermost:channel",
                                             platform=Platform.MATTERMOST, pconfig=config,
                                             chat_id="channel", thread_id=None)
                    result, error = _standalone_send(target, "x" * 6000, [])
                    assert error is None
            assert result["success"]
            posts = [call.kwargs["json"]["message"] for call in session.post.call_args_list]
            assert all(len(post) <= limit for post in posts)
            assert (len(posts) == 1) == (limit > 6000)
            assert adapter.MAX_MESSAGE_LENGTH == limit
        assert [adapter.MAX_MESSAGE_LENGTH for adapter in adapters] == [500, 8000, 4000]
        assert platform_registry.get("mattermost").max_message_length == registry_limit



    def test_mattermost_home_channel(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "mm-tok-abc123")
        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
        monkeypatch.setenv("MATTERMOST_HOME_CHANNEL", "ch_abc123")
        monkeypatch.setenv("MATTERMOST_HOME_CHANNEL_NAME", "General")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.get_home_channel(Platform.MATTERMOST)
        assert home is not None
        assert home.chat_id == "ch_abc123"
        assert home.name == "General"


# ---------------------------------------------------------------------------
# Adapter format / truncate
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a MattermostAdapter with mocked config."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com"},
    )
    adapter = MattermostAdapter(config)
    return adapter


class TestMattermostFormatMessage:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_image_markdown_to_url(self):
        """![alt](url) should be converted to just the URL."""
        result = self.adapter.format_message("![cat](https://img.example.com/cat.png)")
        assert result == "https://img.example.com/cat.png"


    def test_regular_markdown_preserved(self):
        """Regular markdown (bold, italic, code) should be kept as-is."""
        content = "**bold** and *italic* and `code`"
        assert self.adapter.format_message(content) == content


class TestMattermostTruncateMessage:
    @pytest.mark.parametrize("raw, expected", [
        (None, 4000), ("bad", 4000), (True, 4000), (499, 4000),
        (500, 500), (8000, 8000), (16383, 16383), (20000, 16383),
        (500.5, 4000), ("8000", 8000),
    ])
    @pytest.mark.asyncio
    async def test_configured_limit_bounds_gateway_posts(self, raw, expected):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": raw}))
        adapter._api_post = AsyncMock(return_value={"id": "post"})
        text = "a" * (expected + 20)
        await adapter.send("channel", text)
        posts = [call.args[1]["message"] for call in adapter._api_post.await_args_list]
        assert adapter.MAX_MESSAGE_LENGTH == expected
        assert len(posts) >= 2
        assert all(len(post) <= expected for post in posts)

    def setup_method(self):
        self.adapter = _make_adapter()


    def test_long_message_splits(self):
        msg = "a " * 2500  # 5000 chars
        chunks = self.adapter.truncate_message(msg, 4000)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4000


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

class TestMattermostSend:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._session = MagicMock()

    @pytest.mark.asyncio
    async def test_send_calls_api_post(self):
        """send() should POST to /api/v4/posts with channel_id and message."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"id": "post123"})
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        self.adapter._session.post = MagicMock(return_value=mock_resp)

        result = await self.adapter.send("channel_1", "Hello!")

        assert result.success is True
        assert result.message_id == "post123"

        # Verify post was called with correct URL
        call_args = self.adapter._session.post.call_args
        assert "/api/v4/posts" in call_args[0][0]
        # Verify payload
        payload = call_args[1]["json"]
        assert payload["channel_id"] == "channel_1"
        assert payload["message"] == "Hello!"


    @pytest.mark.asyncio
    async def test_send_with_thread_reply(self):
        """When reply_mode is 'thread', reply_to should become root_id."""
        self.adapter._reply_mode = "thread"

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"id": "post456"})
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        # send() now calls _resolve_root_id → _api_get("posts/<id>") first
        # to make sure root_id points to a thread root, so we need to mock
        # the GET too.  Return an empty dict (no root_id) so the resolver
        # falls back to the original reply_to as the root.
        mock_get_resp = AsyncMock()
        mock_get_resp.status = 200
        mock_get_resp.json = AsyncMock(return_value={"id": "root_post", "root_id": ""})
        mock_get_resp.text = AsyncMock(return_value="")
        mock_get_resp.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_get_resp.__aexit__ = AsyncMock(return_value=False)

        self.adapter._session.post = MagicMock(return_value=mock_resp)
        self.adapter._session.get = MagicMock(return_value=mock_get_resp)

        result = await self.adapter.send("channel_1", "Reply!", reply_to="root_post")

        assert result.success is True
        payload = self.adapter._session.post.call_args[1]["json"]
        assert payload["root_id"] == "root_post"


    @pytest.mark.asyncio
    async def test_progress_send_with_invalid_thread_root_never_falls_back_flat(self):
        """Tool/status/progress bubbles must stay quiet when the thread is broken."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "bad_root", "root_id": ""})
        self.adapter._last_post_status = 400
        self.adapter._last_post_error = "api.context.invalid_param.app_error: invalid root_id"
        self.adapter._api_post = AsyncMock(return_value={})

        result = await self.adapter.send(
            "channel_1",
            "⚙️ terminal...",
            metadata={"thread_id": "bad_root"},
        )

        assert result.success is False
        assert self.adapter._api_post.call_count == 1
        payload = self.adapter._api_post.call_args_list[0][0][1]
        assert payload["root_id"] == "bad_root"

    @pytest.mark.parametrize("limit, body", [
        (4000, "Final answer body"), (500, "x" * 500), (16383, "x" * 16383),
    ], ids=["short", "minimum", "maximum"])
    @pytest.mark.asyncio
    async def test_notify_send_with_invalid_thread_root_falls_back_flat_with_warning(self, limit, body):
        """Fallback decoration fits the limit without consuming answer text."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        self.adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": limit}))
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "bad_root", "root_id": ""})
        self.adapter._last_post_status = 400
        self.adapter._last_post_error = "api.context.invalid_param.app_error: invalid root_id"
        self.adapter._api_post = AsyncMock(
            side_effect=lambda path, payload: {} if "root_id" in payload else {"id": "flat_final"})

        result = await self.adapter.send(
            "channel_1", body, reply_to="bad_root", metadata={"notify": True})

        assert result.success is True
        assert result.message_id == "flat_final"
        payloads = [call.args[1] for call in self.adapter._api_post.await_args_list]
        assert payloads[0]["root_id"] == "bad_root"
        flat = [p for p in payloads if "root_id" not in p]
        assert all(p["channel_id"] == "channel_1" for p in flat)
        assert all(len(p["message"]) <= limit for p in payloads)
        warning = "⚠️ Mattermost thread delivery failed; posting final reply in channel.\n\n"
        assert "Mattermost thread delivery failed" in flat[0]["message"]
        assert "".join(p["message"].removeprefix(warning).removeprefix(warning.rstrip()) for p in flat) == body


    @pytest.mark.parametrize("outcome", ["ok", "unacknowledged", "timeout", "notice_unacknowledged", "cancelled"])
    @pytest.mark.asyncio
    async def test_separate_warning_keeps_acknowledgements(self, tmp_path, outcome):
        import asyncio
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        self.adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500}))
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "bad_root", "root_id": ""})
        self.adapter._last_post_status = 400
        self.adapter._last_post_error = "invalid root_id"
        self.adapter._upload_file = AsyncMock(return_value="file")
        final = {"ok": {"id": "content"}, "unacknowledged": {}, "timeout": TimeoutError(),
                 "notice_unacknowledged": {}, "cancelled": asyncio.CancelledError()}[outcome]
        notice = {} if outcome == "notice_unacknowledged" else {"id": "notice"}
        self.adapter._api_post = AsyncMock(side_effect=[{}, notice, final])
        path = tmp_path / "image.png"
        path.write_bytes(b"image")
        call = self.adapter.send_image_file(
            "channel", str(path), caption="x" * 500,
            metadata={"notify": True, "thread_id": "bad_root"})
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await call
            assert self.adapter._api_post.await_count == 3
            return
        result = await call
        expected = ["notice", "content"] if outcome == "ok" else ([] if outcome == "notice_unacknowledged" else ["notice"])
        assert result.success is (outcome == "ok")
        assert result.message_id == (expected[-1] if expected else None)
        assert result.continuation_message_ids == tuple(expected[:-1])
        assert bool(result.error) is (outcome != "ok")
        payloads = [c.args[1] for c in self.adapter._api_post.await_args_list]
        assert len(payloads) == (2 if outcome == "notice_unacknowledged" else 3)
        assert "file_ids" not in payloads[1]
        assert all(len(p["message"]) <= 500 for p in payloads)
        if len(payloads) == 3:
            assert payloads[2]["message"] == "x" * 500
            assert payloads[2]["file_ids"] == ["file"]

    @pytest.mark.parametrize("outcome", ["disconnect", "timeout", "cancelled", "ok", "rejected"])
    @pytest.mark.asyncio
    async def test_gateway_does_not_replay_after_split_warning(self, outcome):
        import asyncio
        import aiohttp
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        self.adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500}))
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "bad_root", "root_id": ""})
        payloads = []

        def post(url, **kwargs):
            payloads.append(kwargs["json"])
            response = AsyncMock()
            response.__aenter__.return_value = response
            response.status = 201
            response.json.return_value = {"id": "content"}
            if len(payloads) == 1:
                response.status = 400
                response.text.return_value = "invalid root_id"
            elif len(payloads) == 2:
                response.json.return_value = {"id": "notice"}
            elif len(payloads) == 3:
                if outcome == "rejected":
                    response.status = 400
                    response.text.return_value = "invalid message"
                elif outcome != "ok":
                    response.json.side_effect = {
                        "disconnect": aiohttp.ServerDisconnectedError("connection reset by peer"),
                        "timeout": TimeoutError(), "cancelled": asyncio.CancelledError(),
                    }[outcome]
            return response

        self.adapter._session = MagicMock()
        self.adapter._session.post.side_effect = post
        body = "x" * 500
        call = self.adapter._send_with_retry(
            "channel", body, reply_to="bad_root", metadata={"notify": True})
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await call
        else:
            result = await call
            # An acknowledged warning is part of the logical receipt too: a
            # rejected content POST must not replay it via formatting fallback.
            assert result.success is (outcome == "ok")
            assert result.message_id == ("content" if outcome == "ok" else "notice")
            assert result.continuation_message_ids == (("notice",) if outcome == "ok" else ())
            if outcome != "ok":
                assert self.adapter._is_timeout_error(result.error)
                assert not result.retryable
        assert len(payloads) == 3
        assert payloads[0]["root_id"] == "bad_root"
        assert "root_id" not in payloads[1] and "root_id" not in payloads[2]
        assert payloads[2]["message"] == body
        assert all(len(payload["message"]) <= 500 for payload in payloads)

    @pytest.mark.asyncio
    async def test_gateway_retries_connector_failure_before_send(self):
        import aiohttp
        from types import SimpleNamespace

        payloads = []
        def post(url, **kwargs):
            payloads.append(kwargs["json"])
            if len(payloads) == 1:
                key = SimpleNamespace(host="mattermost.example", port=443, ssl=True)
                raise aiohttp.ClientConnectorError(key, OSError("connection refused"))
            response = AsyncMock()
            response.status = 201
            response.json.return_value = {"id": "delivered"}
            response.__aenter__.return_value = response
            return response
        self.adapter._session = MagicMock()
        self.adapter._session.post.side_effect = post
        result = await self.adapter._send_with_retry("channel", "hello", base_delay=0)
        assert result.success and result.message_id == "delivered"
        assert len(payloads) == 2 and "hello" in payloads[-1]["message"]

    @pytest.mark.parametrize("batch", [False, True])
    @pytest.mark.parametrize("outcome", ["ok", "timeout", "disconnect", "rejected_body_timeout"])
    @pytest.mark.asyncio
    async def test_media_caption_budget_preserves_text_files_and_receipts(self, tmp_path, batch, outcome):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500}))
        adapter._reply_mode = "thread"
        adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        payloads = []
        def post(url, **kwargs):
            payloads.append(kwargs["json"])
            response = AsyncMock()
            response.status = 201
            response.json.return_value = {"id": f"post-{len(payloads)}"}
            if len(payloads) == 2 and outcome == "rejected_body_timeout":
                response.status = 400
                response.text.side_effect = TimeoutError()
            if len(payloads) == 2 and outcome != "ok":
                import aiohttp
                response.json.side_effect = TimeoutError() if outcome == "timeout" else aiohttp.ServerDisconnectedError()
            response.__aenter__.return_value = response
            return response
        adapter._session = MagicMock()
        adapter._session.post.side_effect = post
        caption = "  @here\n![keep](https://example.com/image)\t" + "x" * 700 + "  "
        path = tmp_path / "image.png"
        path.write_bytes(b"image")
        adapter._upload_file = AsyncMock(side_effect=["file-a", "file-b"])
        if batch:
            result = await adapter.send_multiple_images(
                "channel", [(path.as_uri(), caption), (path.as_uri(), "")], metadata={"thread_id": "root"})
        else:
            result = await adapter.send_image_file("channel", str(path), caption=caption, reply_to="root")
        assert result.success is (outcome == "ok")
        assert len(payloads) == 2
        assert all(len(p["message"]) <= adapter.MAX_MESSAGE_LENGTH for p in payloads)
        assert "".join(p["message"] for p in payloads) == caption
        assert [fid for p in payloads for fid in p.get("file_ids", [])] == (["file-a", "file-b"] if batch else ["file-a"])
        assert all(p["root_id"] == "root" for p in payloads)
        expected_ids = ("post-1", "post-2") if outcome == "ok" else ("post-1",)
        assert (*result.continuation_message_ids, result.message_id) == expected_ids

    @pytest.mark.asyncio
    async def test_progress_send_with_broken_thread_and_no_recorded_error_stays_quiet(self):
        """Same rule when no post error was recorded: still no flat fallback."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "bad_root", "root_id": ""})
        self.adapter._api_post = AsyncMock(return_value={})

        result = await self.adapter.send(
            "channel_1",
            "⚙️ terminal...",
            metadata={"thread_id": "bad_root"},
        )

        assert result.success is False
        assert self.adapter._api_post.call_count == 1
        payload = self.adapter._api_post.call_args_list[0][0][1]
        assert payload["root_id"] == "bad_root"


class TestMattermostTextReceipts:
    @staticmethod
    def transport(adapter, outcome):
        import asyncio
        import aiohttp
        calls = []

        def request(method, url, **kwargs):
            calls.append((method, kwargs["json"]))
            response = AsyncMock()
            response.__aenter__.return_value = response
            response.status = 201
            response.json.return_value = {"id": f"post-{len(calls)}"}
            if len(calls) == 3 and outcome != "ok":
                if outcome == "rejected":
                    response.status = 400
                    response.text.return_value = "invalid message"
                elif outcome == "rejected_body_timeout":
                    response.status = 400
                    response.text.side_effect = TimeoutError()
                else:
                    response.json.side_effect = {
                        "timeout": TimeoutError(),
                        "disconnect": aiohttp.ServerDisconnectedError(),
                        "exception": ValueError("malformed acknowledgement"),
                        "cancelled": asyncio.CancelledError(),
                    }[outcome]
            return response

        adapter._session = MagicMock()
        adapter._session.post.side_effect = lambda url, **kw: request("POST", url, **kw)
        adapter._session.put.side_effect = lambda url, **kw: request("PUT", url, **kw)
        adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        return calls

    @pytest.mark.parametrize("outcome", ["ok", "rejected", "timeout", "disconnect", "exception", "cancelled"])
    @pytest.mark.asyncio
    async def test_text_retains_all_acknowledgements_without_replay(self, outcome):
        import asyncio
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500}))
        calls = self.transport(adapter, outcome)
        send = adapter._send_with_retry("channel", "x" * 1300, base_delay=0)
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await send
        else:
            result = await send
            assert result.success is (outcome == "ok")
            expected = ("post-1", "post-2", "post-3") if outcome == "ok" else ("post-1", "post-2")
            assert (*result.continuation_message_ids, result.message_id) == expected
        assert len(calls) == 3
        assert all(len(p["message"]) <= 500 for _, p in calls)

    @pytest.mark.parametrize("outcome", ["ok", "rejected", "rejected_body_timeout", "timeout", "exception", "cancelled"])
    @pytest.mark.asyncio
    async def test_edit_chunks_losslessly_and_surfaces_every_continuation(self, outcome):
        import asyncio
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "thread"}))
        calls = self.transport(adapter, outcome)
        content = "  ```python\n" + "print('x (1/2)'); \t\n" * 70 + "```  "
        edit = adapter.edit_message("channel", "original", content)
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await edit
            return
        result = await edit
        assert all(len(p["message"]) <= 500 for _, p in calls)
        assert calls[0][0] == "PUT" and all(method == "POST" for method, _ in calls[1:])
        assert all(p["props"]["disable_mentions"] for _, p in calls)
        assert all(p["root_id"] == "root" for _, p in calls[1:])
        assert result.success is (outcome == "ok")
        acked = len(calls) if outcome == "ok" else 2
        assert (*result.continuation_message_ids, result.message_id) == tuple(f"post-{i}" for i in range(1, acked + 1))
        prefix = "".join(p["message"] for _, p in calls[:acked])
        assert content.startswith(prefix)
        if outcome == "ok":
            assert prefix == content
        else:
            assert result.raw_response["partial_overflow"]
            assert result.raw_response["delivered_prefix"] == content[:len(prefix)]
            assert result.raw_response.get("_delivery_uncertain", False) is (outcome in {"timeout", "exception"})

    @pytest.mark.parametrize("outcome", ["rejected", "timeout"])
    @pytest.mark.asyncio
    async def test_stream_first_send_retains_failed_receipts(self, outcome):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500}))
        calls = self.transport(adapter, outcome)
        consumer = GatewayStreamConsumer(adapter=adapter, chat_id="channel", config=StreamConsumerConfig(cursor=""))
        consumer._accumulated = "x" * 1300
        assert not await consumer._send_or_edit(consumer._accumulated, finalize=True)
        assert consumer._preview_message_ids == {"post-1", "post-2"}
        assert consumer._message_id == "post-2"
        await consumer._finalize_edit_path(_Tick(got_done=True))
        assert len(calls) == 3

    @pytest.mark.parametrize("outcome", ["rejected", "timeout"])
    @pytest.mark.asyncio
    async def test_stream_partial_edit_only_retries_known_unsent_tail(self, outcome):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "thread"}))
        calls = self.transport(adapter, outcome)
        consumer = GatewayStreamConsumer(adapter=adapter, chat_id="channel", metadata={"thread_id": "root"},
                                         config=StreamConsumerConfig(cursor=""))
        content = "a" * 490 + " (1/2)  \n" + "b" * 490 + " (2/2)  \n" + "remaining tail"
        consumer._message_id = "original"
        consumer._last_sent_text = "preview"
        consumer._accumulated = content
        assert not await consumer._send_or_edit(content, finalize=True)
        assert consumer._preview_message_ids.issuperset({"post-1", "post-2"})
        await consumer._finalize_edit_path(_Tick(got_done=True))
        if outcome == "timeout":
            assert len(calls) == 3
            assert consumer._delivery_ambiguous
        else:
            assert len(calls) == 4
            assert calls[-1][1]["message"] == content[1000:].lstrip()
            assert consumer.final_content_delivered


class TestMattermostFinalizePostAcknowledgements:
    @pytest.mark.parametrize("outcome", [
        "empty", "null", "array", "string", "missing_id", "null_id", "empty_id",
        "blank_id", "numeric_id", "bool_id", "array_id", "object_id", "invalid_json",
        "500", "502", "503", "504", "503_body_timeout", "503_body_disconnect",
        "ok", "rejected", "rejected_body_timeout", "connector", "cancelled",
    ])
    @pytest.mark.asyncio
    async def test_native_finalize_stops_after_unacknowledged_continuation(self, outcome):
        """An acknowledged preview edit + continuation must never replay an unknown POST."""
        import asyncio
        import aiohttp
        from aiohttp.client_reqrep import ConnectionKey
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "thread"}))
        posts, edits = [], []
        bodies = {
            "empty": {}, "null": None, "array": [], "string": "accepted",
            "missing_id": {"message": "accepted", "_post_rejection_status": 400},
            "null_id": {"id": None}, "empty_id": {"id": ""}, "blank_id": {"id": " \t"},
            "numeric_id": {"id": 7}, "bool_id": {"id": True},
            "array_id": {"id": ["unknown"]}, "object_id": {"id": {"unknown": True}},
        }

        def post(url, **kwargs):
            assert url.endswith("/api/v4/posts")
            posts.append(kwargs["json"])
            response = AsyncMock()
            response.__aenter__.return_value = response
            response.status = 201
            response.json.return_value = {"id": f"post-{len(posts)}"}
            if len(posts) == 3:
                if outcome in bodies:
                    response.json.return_value = bodies[outcome]
                elif outcome == "invalid_json":
                    response.json.side_effect = json.JSONDecodeError("invalid", "{", 1)
                elif outcome in {"500", "502", "503", "504", "503_body_timeout", "503_body_disconnect"}:
                    response.status = int(outcome[:3])
                    response.text.return_value = "server failed after accepting the request"
                    if outcome == "503_body_timeout":
                        response.text.side_effect = TimeoutError()
                    elif outcome == "503_body_disconnect":
                        response.text.side_effect = aiohttp.ServerDisconnectedError()
                elif outcome.startswith("rejected"):
                    response.status = 400
                    response.text.return_value = "invalid message"
                    if outcome == "rejected_body_timeout":
                        response.text.side_effect = TimeoutError()
                elif outcome == "connector":
                    key = ConnectionKey("example.invalid", 443, True, True, None, None, None)
                    response.__aenter__.side_effect = aiohttp.ClientConnectorError(key, OSError("connection refused"))
                elif outcome == "cancelled":
                    response.json.side_effect = asyncio.CancelledError()
            return response

        def put(url, **kwargs):
            assert url.endswith("/api/v4/posts/post-1/patch")
            edits.append(kwargs["json"])
            response = AsyncMock()
            response.__aenter__.return_value = response
            response.status = 200
            response.json.return_value = {"id": "post-1"}
            return response

        adapter._session = MagicMock()
        adapter._session.post.side_effect = post
        adapter._session.put.side_effect = put
        adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        consumer = GatewayStreamConsumer(adapter, "channel", metadata={"thread_id": "root"},
                                         config=StreamConsumerConfig(cursor=""))
        assert await consumer._send_or_edit("preview")
        content = "a" * 490 + " (1/2)  \n" + "b" * 490 + " (2/2)  \n" + "remaining tail"
        consumer._accumulated = content
        finalize = consumer._finalize_edit_path(_Tick(got_done=True))
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await finalize
            assert len(posts) == 3
            return
        await finalize

        assert len(edits) == 1
        assert edits[0]["message"] + posts[1]["message"] == content[:1000]
        assert all(len(payload["message"]) <= 500 and payload["props"]["disable_mentions"]
                   for payload in posts + edits)
        assert all(payload["root_id"] == "root" for payload in posts)
        if outcome in {"rejected", "rejected_body_timeout", "connector"}:
            assert len(posts) == 4  # authoritative rejection / before-send failure permits known-tail recovery
            assert posts[-1]["message"] == content[1000:].lstrip()
            assert consumer._preview_message_ids | {consumer._message_id} == {"post-1", "post-2", "post-4"}
            assert consumer.final_content_delivered
        elif outcome == "ok":
            assert len(posts) == 3
            assert consumer._message_id == "post-3"
            assert consumer._preview_message_ids == {"post-1", "post-2", "post-3"}
            assert consumer.final_content_delivered
        else:
            assert len(posts) == 3, "native finalize replayed an unacknowledged continuation"
            assert consumer._preview_message_ids == {"post-1", "post-2"}
            assert consumer._message_id == "post-2"
            assert consumer._delivery_ambiguous
            await consumer._finalize_edit_path(_Tick(got_done=True))
            assert not await consumer._send_or_edit(content, finalize=True)
            assert len(posts) == 3


# ---------------------------------------------------------------------------
# WebSocket event parsing
# ---------------------------------------------------------------------------

class TestMattermostWebSocketParsing:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._bot_user_id = "bot_user_id"
        self.adapter._bot_username = "hermes-bot"
        # Mock handle_message to capture the MessageEvent without processing
        self.adapter.handle_message = AsyncMock()

    @pytest.mark.asyncio
    async def test_parse_posted_event(self):
        """'posted' events should extract message from double-encoded post JSON."""
        post_data = {
            "id": "post_abc",
            "user_id": "user_123",
            "channel_id": "chan_456",
            "message": "@bot_user_id Hello from Matrix!",
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),  # double-encoded JSON string
                "channel_type": "O",
                "sender_name": "@alice",
            },
        }

        await self.adapter._handle_ws_event(event)
        assert self.adapter.handle_message.called
        msg_event = self.adapter.handle_message.call_args[0][0]
        # @mention is stripped from the message text
        assert msg_event.text == "Hello from Matrix!"
        assert msg_event.message_id == "post_abc"


    @pytest.mark.asyncio
    async def test_ignore_system_posts(self):
        """Posts with a 'type' field (system messages) should be ignored."""
        post_data = {
            "id": "sys_post",
            "user_id": "user_123",
            "channel_id": "chan_456",
            "message": "user joined",
            "type": "system_join_channel",
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": "O",
            },
        }

        await self.adapter._handle_ws_event(event)
        assert not self.adapter.handle_message.called


    @pytest.mark.asyncio
    async def test_leading_space_slash_command_is_command(self):
        """Mattermost mobile suggests leading-space slash commands."""
        post_data = {
            "id": "post_cmd",
            "user_id": "user_123",
            "channel_id": "chan_dm",
            "message": " /new",
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": "D",
                "sender_name": "@bob",
            },
        }

        await self.adapter._handle_ws_event(event)
        assert self.adapter.handle_message.called
        msg_event = self.adapter.handle_message.call_args[0][0]
        assert msg_event.text == "/new"
        assert msg_event.message_type is MessageType.COMMAND
        assert msg_event.get_command() == "new"


# ---------------------------------------------------------------------------
# Mention behavior (require_mention + free_response_channels)
# ---------------------------------------------------------------------------

class TestMattermostMentionBehavior:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._bot_user_id = "bot_user_id"
        self.adapter._bot_username = "hermes-bot"
        self.adapter.handle_message = AsyncMock()

    def _make_event(self, message, channel_type="O", channel_id="chan_456"):
        post_data = {
            "id": "post_mention",
            "user_id": "user_123",
            "channel_id": channel_id,
            "message": message,
        }
        return {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": channel_type,
                "sender_name": "@alice",
            },
        }

    @pytest.mark.asyncio
    async def test_require_mention_true_skips_without_mention(self):
        """Default: messages without @mention in channels are skipped."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            os.environ.pop("MATTERMOST_FREE_RESPONSE_CHANNELS", None)
            await self.adapter._handle_ws_event(self._make_event("hello"))
            assert not self.adapter.handle_message.called


    @pytest.mark.asyncio
    async def test_free_response_channel_responds_without_mention(self):
        """Messages in free-response channels don't need @mention."""
        with patch.dict(os.environ, {"MATTERMOST_FREE_RESPONSE_CHANNELS": "chan_456,chan_789"}):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            await self.adapter._handle_ws_event(self._make_event("hello", channel_id="chan_456"))
            assert self.adapter.handle_message.called


# ---------------------------------------------------------------------------
# File upload (send_image)
# ---------------------------------------------------------------------------

class TestMattermostAuthoritativeFallback:
    def transport(self, outcomes):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500}))
        adapter._session = MagicMock()
        adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        adapter._upload_file = AsyncMock(side_effect=[f"file-{i}" for i in range(20)])
        posts, events = [], []

        def post(url, **kwargs):
            import aiohttp
            payload = kwargs["json"]
            posts.append(payload)
            events.append(("post", payload.get("file_ids")))
            response = AsyncMock()
            response.__aenter__.return_value = response
            response.status = 201
            response.text.return_value = "invalid message"
            outcome = outcomes[len(posts) - 1] if len(posts) <= len(outcomes) else "ok"
            response.json.return_value = {"id": f"post-{len(posts)}"}
            if isinstance(outcome, int):
                response.status = outcome
            elif outcome == "empty":
                response.json.return_value = {}
            elif outcome == "invalid":
                response.json.side_effect = ValueError("invalid JSON")
            elif outcome == "invalid_body":
                response.json.return_value = []
            elif outcome == "body_claims_rejection":
                response.json.return_value = {"_post_rejection_status": 413}
            elif outcome == "timeout":
                response.json.side_effect = TimeoutError()
            elif outcome == "disconnect":
                response.json.side_effect = aiohttp.ServerDisconnectedError()
            elif outcome == "cancelled":
                import asyncio
                response.json.side_effect = asyncio.CancelledError()
            elif outcome == "broken_root":
                response.status = 400
                response.text.return_value = "invalid root_id"
            elif outcome == "rejection_body_timeout":
                response.status = 413
                response.text.side_effect = TimeoutError()
            return response

        adapter._session.post.side_effect = post
        return adapter, posts, events

    def images(self, tmp_path, count=2):
        paths = [tmp_path / f"image-{i}.png" for i in range(count)]
        for path in paths:
            path.write_bytes(b"image")
        return [(str(path), "") for path in paths]

    @pytest.mark.parametrize("status", [400, 403, 404, 413, 422])
    @pytest.mark.asyncio
    async def test_batch_fallback_requires_authoritative_rejection(self, tmp_path, status):
        adapter, posts, events = self.transport([status])
        result = await adapter.send_multiple_images(
            "channel", self.images(tmp_path), caption="  caption @here\n",
            metadata={"thread_id": "root", "mattermost_explicit_thread": True})
        assert result.success
        assert result.raw_response["delivered_media"] == 2
        assert result.raw_response["failed_media"] == 0
        assert result.raw_response["message_ids"] == ["post-2", "post-3"]
        assert [p.get("file_ids") for p in posts] == [["file-0", "file-1"], ["file-0"], ["file-1"]]
        assert sum(p["message"] == "  caption @here\n" for p in posts[1:]) == 1
        assert all(p["root_id"] == "root" and p["props"]["disable_mentions"] for p in posts)
        assert adapter._upload_file.await_count == 2

    @pytest.mark.parametrize("outcome", [401, 429, 500, 503, "empty", "timeout", "disconnect",
                                          "invalid", "invalid_body", "rejection_body_timeout", "body_claims_rejection"])
    @pytest.mark.asyncio
    async def test_batch_never_replays_without_authoritative_rejection(self, tmp_path, outcome):
        adapter, posts, events = self.transport([outcome])
        adapter._last_post_status = 413
        result = await adapter.send_multiple_images("channel", self.images(tmp_path), caption="caption")
        assert not result.success
        assert result.raw_response["delivered_media"] == 0
        assert result.raw_response["message_ids"] == []
        assert len(posts) == 1

    @pytest.mark.asyncio
    async def test_empty_result_cannot_borrow_stale_rejection(self, tmp_path):
        adapter, posts, events = self.transport([])
        adapter._last_post_status = 413
        adapter._api_post = AsyncMock(return_value={})
        result = await adapter.send_multiple_images("channel", self.images(tmp_path), caption="caption")
        assert not result.success
        assert adapter._api_post.await_count == 1

    @pytest.mark.asyncio
    async def test_batch_fallback_waits_before_each_single_file_send(self, tmp_path, monkeypatch):
        adapter, posts, events = self.transport([413])
        async def delay(seconds):
            events.append(("delay", seconds))
        monkeypatch.setattr("plugins.platforms.mattermost.adapter.asyncio.sleep", delay)
        result = await adapter.send_multiple_images("channel", self.images(tmp_path), human_delay=0.25)
        assert result.success
        assert events == [("post", ["file-0", "file-1"]), ("delay", 0.25),
                          ("post", ["file-0"]), ("delay", 0.25), ("post", ["file-1"])]

    @pytest.mark.parametrize("outcome", [413, "empty", "timeout", "disconnect"])
    @pytest.mark.asyncio
    async def test_fallback_stops_at_first_failed_single_with_prior_receipts(self, tmp_path, outcome):
        adapter, posts, events = self.transport([413, "ok", outcome])
        result = await adapter.send_multiple_images("channel", self.images(tmp_path, 6), caption="caption")
        assert result.success  # Gateway means any-media success, public receipt means all requested.
        assert not result.raw_response["success"]
        assert result.raw_response["delivered_media"] == 1
        assert result.raw_response["failed_media"] == 5
        assert result.raw_response["message_ids"] == ["post-2"]
        assert len(posts) == 3
        assert adapter._upload_file.await_count == 5

    @pytest.mark.parametrize("warning", [False, True])
    @pytest.mark.asyncio
    async def test_acknowledged_caption_or_warning_prevents_single_replay(self, tmp_path, warning):
        adapter, posts, events = self.transport(["broken_root", "ok", 413] if warning else ["ok", 413])
        result = await adapter.send_multiple_images(
            "channel", self.images(tmp_path), caption="x" * (500 if warning else 700),
            metadata={"notify": warning, "thread_id": "root", "mattermost_explicit_thread": True})
        assert not result.success
        assert result.raw_response["delivered_media"] == 0
        assert result.raw_response["message_ids"] == ["post-2" if warning else "post-1"]
        assert len(posts) == (3 if warning else 2)
        assert all(len(p["message"]) <= 500 for p in posts)

    @pytest.mark.asyncio
    async def test_shared_caption_survives_unposted_single_fallbacks(self, tmp_path):
        adapter, posts, events = self.transport([413, 413])
        caption = "  explicit @here\n"
        result = await adapter.send_multiple_images("channel", self.images(tmp_path), caption=caption)
        assert not result.success
        assert result.raw_response["delivered_media"] == 0
        assert result.raw_response["message_ids"] == ["post-3"]
        assert len(posts) == 3
        assert posts[-1]["message"] == caption and "file_ids" not in posts[-1]

    @pytest.mark.parametrize("all_missing", [False, True])
    @pytest.mark.asyncio
    async def test_tuple_caption_survives_an_entire_missing_batch(self, tmp_path, all_missing):
        adapter, posts, events = self.transport([])
        images = [(str(tmp_path / f"missing-{i}.png"), "  tuple caption\n" if i == 0 else "") for i in range(5)]
        if not all_missing:
            images += self.images(tmp_path, 1)
        result = await adapter.send_multiple_images("channel", images)
        assert result.success is (not all_missing)
        assert posts[0]["message"] == "  tuple caption\n"
        assert result.raw_response["delivered_media"] == (0 if all_missing else 1)

    @pytest.mark.asyncio
    async def test_fallback_cancellation_propagates_without_another_post(self, tmp_path):
        import asyncio
        adapter, posts, events = self.transport([413, "cancelled"])
        with pytest.raises(asyncio.CancelledError):
            await adapter.send_multiple_images("channel", self.images(tmp_path), caption="caption")
        assert len(posts) == 2


class TestMattermostFileUpload:
    @pytest.mark.parametrize("content_ok", [False, True])
    @pytest.mark.parametrize("missing", [False, True])
    @pytest.mark.asyncio
    async def test_batch_retains_prior_post_acknowledgements(self, tmp_path, content_ok, missing):
        path = tmp_path / "image.png"
        if not missing:
            path.write_bytes(b"image")
        self.adapter._upload_file = AsyncMock(return_value="file")
        # The optional length feature can emit an acknowledged warning before this post.
        ids = ["notice", "content"] if content_ok else ["notice"]
        data = {"message_ids": ids, **({"id": "content"} if content_ok else {})}
        self.adapter._post_message = AsyncMock(return_value=data)
        result = await self.adapter.send_multiple_images(
            "channel", [(path.as_uri(), "")], caption="caption")
        assert result.raw_response["message_ids"] == ids
        assert result.message_id == ids[-1]
        assert result.continuation_message_ids == tuple(ids[:-1])
        delivered = int(content_ok and not missing)
        assert result.raw_response["delivered_media"] == delivered
        assert result.raw_response["media_delivered"] is bool(delivered)
        assert result.raw_response["partial_failure"] is (not bool(delivered))
        assert result.success is bool(delivered)
        self.adapter._post_message.assert_awaited_once()

    @pytest.mark.parametrize("caption", ["  caption\n", " \t "])
    @pytest.mark.asyncio
    async def test_missing_first_image_keeps_tuple_caption(self, tmp_path, caption):
        missing = tmp_path / "missing.png"
        valid = tmp_path / "valid.png"
        valid.write_bytes(b"image")
        self.adapter._upload_file = AsyncMock(return_value="file")
        self.adapter._api_post = AsyncMock(return_value={"id": "ack"})
        result = await self.adapter.send_multiple_images(
            "channel", [(missing.as_uri(), caption), (valid.as_uri(), "")])
        assert result.success
        assert result.raw_response["delivered_media"] == 1
        assert self.adapter._api_post.await_args.args[1]["message"] == caption

    @pytest.mark.parametrize("caption", [None, "  explicit @here\n", " \t "])
    @pytest.mark.asyncio
    async def test_file_caption_preserved_or_safe_filename(self, tmp_path, caption):
        path = tmp_path / ("  @channel\n" + "a" * 180 + ".png")
        path.write_bytes(b"image")
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        self.adapter._upload_file = AsyncMock(return_value="file")
        self.adapter._api_post = AsyncMock(return_value={"id": "ack"})
        result = await self.adapter.send_image_file("channel", str(path), caption=caption,
                                                    metadata={"thread_id": "reply"})
        payload = self.adapter._api_post.await_args.args[1]
        assert result.success
        assert payload["root_id"] == "root"
        assert payload["file_ids"] == ["file"]
        assert payload["props"]["disable_mentions"] is True
        if caption is not None:
            assert payload["message"] == caption
        else:
            assert payload["message"].startswith("📎 @\u200bchannel ")
            assert len(payload["message"]) <= 162
            assert "\n" not in payload["message"]

    @pytest.mark.parametrize("failure", ["missing", "post_response_lost", "later_exception"])
    @pytest.mark.asyncio
    async def test_image_batch_partial_receipts_without_retry(self, tmp_path, failure):
        count = 6 if failure == "later_exception" else 1
        paths = [tmp_path / f"image-{i}.png" for i in range(count)]
        for path in paths:
            if failure != "missing":
                path.write_bytes(b"image")
        self.adapter._upload_file = AsyncMock(return_value="file")
        self.adapter._api_post = AsyncMock(side_effect=(
            [{"id": "ack"}, RuntimeError("lost response")] if failure == "later_exception" else [{}]
        ))
        self.adapter.send_image = AsyncMock(side_effect=AssertionError("must not retry a possibly accepted post"))
        result = await self.adapter.send_multiple_images("channel", [(path.as_uri(), "") for path in paths])
        assert result.success is (failure == "later_exception")
        assert result.error
        assert result.message_id == ("ack" if failure == "later_exception" else None)
        assert result.raw_response["delivered_media"] == (5 if failure == "later_exception" else 0)
        assert result.raw_response["total_media"] == count
        assert result.raw_response["message_ids"] == (["ack"] if failure == "later_exception" else [])
        assert self.adapter._api_post.await_count == {"missing": 0, "post_response_lost": 1, "later_exception": 2}[failure]
        self.adapter.send_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_local_file_is_failure(self, tmp_path):
        self.adapter._api_post = AsyncMock()
        result = await self.adapter.send_document("channel", str(tmp_path / "missing.pdf"))
        assert result.success is False
        assert result.error
        self.adapter._api_post.assert_not_awaited()

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._session = MagicMock()

    @pytest.mark.asyncio
    @patch("tools.url_safety.is_safe_url", return_value=True)
    async def test_send_image_downloads_and_uploads(self, _mock_safe):
        """send_image should download the URL, upload via /api/v4/files, then post."""
        # Mock the download (GET)
        mock_dl_resp = AsyncMock()
        mock_dl_resp.status = 200
        mock_dl_resp.read = AsyncMock(return_value=b"\x89PNG\x00fake-image-data")
        mock_dl_resp.content_type = "image/png"
        mock_dl_resp.__aenter__ = AsyncMock(return_value=mock_dl_resp)
        mock_dl_resp.__aexit__ = AsyncMock(return_value=False)

        # Mock the upload (POST to /files)
        mock_upload_resp = AsyncMock()
        mock_upload_resp.status = 200
        mock_upload_resp.json = AsyncMock(return_value={
            "file_infos": [{"id": "file_abc123"}]
        })
        mock_upload_resp.text = AsyncMock(return_value="")
        mock_upload_resp.__aenter__ = AsyncMock(return_value=mock_upload_resp)
        mock_upload_resp.__aexit__ = AsyncMock(return_value=False)

        # Mock the post (POST to /posts)
        mock_post_resp = AsyncMock()
        mock_post_resp.status = 200
        mock_post_resp.json = AsyncMock(return_value={"id": "post_with_file"})
        mock_post_resp.text = AsyncMock(return_value="")
        mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_post_resp.__aexit__ = AsyncMock(return_value=False)

        # Route calls: first GET (download), then POST (upload), then POST (create post)
        self.adapter._session.get = MagicMock(return_value=mock_dl_resp)
        post_call_count = 0
        original_post_returns = [mock_upload_resp, mock_post_resp]

        def post_side_effect(*args, **kwargs):
            nonlocal post_call_count
            resp = original_post_returns[min(post_call_count, len(original_post_returns) - 1)]
            post_call_count += 1
            return resp

        self.adapter._session.post = MagicMock(side_effect=post_side_effect)

        result = await self.adapter.send_image(
            "channel_1", "https://img.example.com/cat.png", caption="A cat"
        )

        assert result.success is True
        assert result.message_id == "post_with_file"


# ---------------------------------------------------------------------------
# Dedup cache
# ---------------------------------------------------------------------------

class TestMattermostDedup:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._bot_user_id = "bot_user_id"
        # Mock handle_message to capture calls without processing
        self.adapter.handle_message = AsyncMock()


    def test_prune_seen_clears_expired(self):
        """Dedup cache should remove entries older than TTL on overflow."""
        now = time.time()
        dedup = self.adapter._dedup
        # Fill with enough expired entries to trigger pruning
        for i in range(dedup._max_size + 10):
            dedup._seen[f"old_{i}"] = now - 600  # 10 min ago (older than default TTL)

        # Add a fresh one
        dedup._seen["fresh"] = now

        # Trigger pruning by calling is_duplicate with a new entry (over max_size)
        dedup.is_duplicate("trigger_prune")

        # Old entries should be pruned, fresh one kept
        assert "fresh" in dedup._seen
        assert len(dedup._seen) < dedup._max_size + 10


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestMattermostRequirements:
    def test_check_requirements_with_token_and_url(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "test-token")
        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
        from plugins.platforms.mattermost.adapter import check_mattermost_requirements
        assert check_mattermost_requirements() is True


    def test_validate_config_accepts_platform_values(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_TOKEN", raising=False)
        monkeypatch.delenv("MATTERMOST_URL", raising=False)
        from plugins.platforms.mattermost.adapter import validate_mattermost_config

        config = PlatformConfig(
            enabled=True,
            token="cfg-token",
            extra={"url": "https://mm.example.com"},
        )
        assert validate_mattermost_config(config) is True


# ---------------------------------------------------------------------------
# Media type propagation (MIME types, not bare strings)
# ---------------------------------------------------------------------------

class TestMattermostMediaTypes:
    """Verify that media_types contains actual MIME types (e.g. 'image/png')
    rather than bare category strings ('image'), so downstream
    ``mtype.startswith("image/")`` checks in run.py work correctly."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._bot_user_id = "bot_user_id"
        self.adapter.handle_message = AsyncMock()

    def _make_event(self, file_ids):
        post_data = {
            "id": "post_media",
            "user_id": "user_123",
            "channel_id": "chan_456",
            "message": "@bot_user_id file attached",
            "file_ids": file_ids,
        }
        return {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": "O",
                "sender_name": "@alice",
            },
        }

    @pytest.mark.asyncio
    async def test_image_media_type_is_full_mime(self):
        """An image attachment should produce 'image/png', not 'image'."""
        file_info = {"name": "photo.png", "mime_type": "image/png"}
        self.adapter._api_get = AsyncMock(return_value=file_info)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"\x89PNG fake")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        self.adapter._session = MagicMock()
        self.adapter._session.get = MagicMock(return_value=mock_resp)

        with patch("gateway.platforms.base.cache_image_from_bytes", return_value="/tmp/photo.png"):
            await self.adapter._handle_ws_event(self._make_event(["file1"]))

        msg = self.adapter.handle_message.call_args[0][0]
        assert msg.media_types == ["image/png"]
        assert msg.media_types[0].startswith("image/")


@pytest.mark.asyncio
async def test_mattermost_top_level_channel_post_is_thread_root():
    adapter = _make_adapter()
    adapter._reply_mode = "thread"
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter.handle_message = AsyncMock()
    post_data = {
        "id": "top_post_123",
        "user_id": "user_123",
        "channel_id": "chan_456",
        "message": "@hermes-bot start work",
        "root_id": "",
    }
    event = {
        "event": "posted",
        "data": {
            "post": json.dumps(post_data),
            "channel_type": "O",
            "sender_name": "@alice",
        },
    }

    await adapter._handle_ws_event(event)

    msg_event = adapter.handle_message.call_args[0][0]
    assert msg_event.source.thread_id == "top_post_123"
    assert msg_event.source.message_id == "top_post_123"
    assert msg_event.message_id == "top_post_123"


# ---------------------------------------------------------------------------
# Multiplex secondary-profile scope
# ---------------------------------------------------------------------------
#
# __init__'s url/reply_mode, validate_mattermost_config's url,
# _standalone_send's url, and _handle_ws_event's require_mention/
# free_response_channels/allowed_channels, all previously read raw
# os.getenv unconditionally (only MATTERMOST_TOKEN was already scoped).
# _apply_yaml_config also wrote MATTERMOST_REQUIRE_MENTION/
# MATTERMOST_FREE_RESPONSE_CHANNELS/MATTERMOST_ALLOWED_CHANNELS into the
# process-global os.environ unconditionally. Under multiplex, os.environ
# holds the DEFAULT profile's YAML-to-env bridge output -- a secondary
# profile with its own (different or absent) Mattermost config would
# silently connect to the default profile's server, or have its
# mention-gating/channel-allowlist decisions driven by the default
# profile's settings. Mirrors the LINE/DingTalk/IRC fix for #98738.

@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("MATTERMOST_URL", "https://default.example.com")
    monkeypatch.setenv("MATTERMOST_REPLY_MODE", "thread")
    monkeypatch.setenv("MATTERMOST_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATTERMOST_FREE_RESPONSE_CHANNELS", "chan_default")
    monkeypatch.setenv("MATTERMOST_ALLOWED_CHANNELS", "chan_default")


class TestMultiplexProfileScope:

    @pytest.mark.asyncio
    async def test_ws_event_gating_uses_scoped_settings_not_default(
        self, monkeypatch
    ):
        """A secondary profile's own require_mention/free_response_channels/
        allowed_channels (installed via the scope) must gate its messages --
        not the default profile's bridged settings."""
        from agent.secret_scope import (
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        monkeypatch.setenv("MATTERMOST_REQUIRE_MENTION", "true")
        monkeypatch.delenv("MATTERMOST_FREE_RESPONSE_CHANNELS", raising=False)

        adapter = _make_adapter()
        adapter._bot_user_id = "bot_user_id"
        adapter._bot_username = "hermes-bot"
        adapter.handle_message = AsyncMock()

        post_data = {
            "id": "post_scoped",
            "user_id": "user_123",
            "channel_id": "chan_456",
            "message": "hello with no mention",
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": "O",
                "sender_name": "@alice",
            },
        }

        set_multiplex_active(True)
        token = set_secret_scope({"MATTERMOST_REQUIRE_MENTION": "false"})
        try:
            await adapter._handle_ws_event(event)
        finally:
            reset_secret_scope(token)
            set_multiplex_active(False)

        # The profile's own scope disables require_mention -- the message
        # must be dispatched even without an @mention, despite the default
        # profile's env bridge saying require_mention=true.
        assert adapter.handle_message.called

    def test_apply_yaml_config_scoped_skips_env_write_and_seeds_extra(
        self, multiplex_scope
    ):
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        multiplex_scope()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            seeded = _apply_yaml_config({}, {"require_mention": False, "allowed_channels": ["c1"]})
            assert seeded == {"require_mention": False, "allowed_channels": ["c1"]}
            # Under a secondary profile's scope the env bridge must be
            # skipped -- writing here would leak into every other profile's
            # os.environ.
            assert "MATTERMOST_REQUIRE_MENTION" not in os.environ


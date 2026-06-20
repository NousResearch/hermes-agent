"""Tests for Mattermost platform adapter."""
import json
import os
import time
import aiohttp
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig
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

    def test_threaded_mattermost_progress_prefers_existing_thread_root(self):
        assert _resolve_progress_thread_id(
            Platform.MATTERMOST,
            source_thread_id="root_post_123",
            event_message_id="reply_post_456",
        ) == "root_post_123"

    def test_telegram_progress_does_not_use_message_id_as_thread_id(self):
        assert _resolve_progress_thread_id(
            Platform.TELEGRAM,
            source_thread_id=None,
            event_message_id="12345",
        ) is None


class TestMattermostDisplayHygiene:
    def test_mattermost_requires_platform_opt_in_for_interim_assistant_messages(self):
        """Global interim commentary must not make Mattermost leak scratch notes."""
        user_config = {"display": {"interim_assistant_messages": True}}

        assert _resolve_gateway_display_bool(
            user_config,
            "mattermost",
            "interim_assistant_messages",
            default=True,
            platform=Platform.MATTERMOST,
            require_platform_override_for={Platform.MATTERMOST},
        ) is False

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

    def test_mattermost_requires_platform_opt_in_for_thinking_progress(self):
        """Global thinking_progress must not surface internal analysis in Mattermost."""
        user_config = {"display": {"thinking_progress": True}}

        assert _resolve_gateway_display_bool(
            user_config,
            "mattermost",
            "thinking_progress",
            default=False,
            platform=Platform.MATTERMOST,
            require_platform_override_for={Platform.MATTERMOST},
        ) is False

    def test_mattermost_requires_platform_opt_in_for_show_reasoning(self):
        """Global show_reasoning must not prepend scratch reasoning in Mattermost."""
        user_config = {"display": {"show_reasoning": True}}

        assert _resolve_gateway_display_bool(
            user_config,
            "mattermost",
            "show_reasoning",
            default=False,
            platform=Platform.MATTERMOST,
            require_platform_override_for={Platform.MATTERMOST},
        ) is False

    def test_mattermost_platform_opt_in_can_enable_show_reasoning(self):
        user_config = {
            "display": {
                "show_reasoning": False,
                "platforms": {"mattermost": {"show_reasoning": True}},
            }
        }

        assert _resolve_gateway_display_bool(
            user_config,
            "mattermost",
            "show_reasoning",
            default=False,
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
    def test_apply_env_overrides_mattermost(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "mm-tok-abc123")
        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATTERMOST in config.platforms
        mc = config.platforms[Platform.MATTERMOST]
        assert mc.enabled is True
        assert mc.token == "mm-tok-abc123"
        assert mc.extra.get("url") == "https://mm.example.com"

    def test_mattermost_not_loaded_without_token(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_TOKEN", raising=False)
        monkeypatch.delenv("MATTERMOST_URL", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATTERMOST not in config.platforms

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

    def test_mattermost_url_warning_without_url(self, monkeypatch):
        """MATTERMOST_TOKEN set but MATTERMOST_URL missing should still load."""
        monkeypatch.setenv("MATTERMOST_TOKEN", "mm-tok-abc123")
        monkeypatch.delenv("MATTERMOST_URL", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATTERMOST in config.platforms
        assert config.platforms[Platform.MATTERMOST].extra.get("url") == ""


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

    def test_image_markdown_strips_alt_text(self):
        result = self.adapter.format_message("Here: ![my image](https://x.com/a.jpg) done")
        assert "![" not in result
        assert "https://x.com/a.jpg" in result

    def test_regular_markdown_preserved(self):
        """Regular markdown (bold, italic, code) should be kept as-is."""
        content = "**bold** and *italic* and `code`"
        assert self.adapter.format_message(content) == content

    def test_regular_links_preserved(self):
        """Non-image links should be preserved."""
        content = "[click](https://example.com)"
        assert self.adapter.format_message(content) == content

    def test_plain_text_unchanged(self):
        content = "Hello, world!"
        assert self.adapter.format_message(content) == content

    def test_multiple_images(self):
        content = "![a](http://a.com/1.png) text ![b](http://b.com/2.png)"
        result = self.adapter.format_message(content)
        assert "![" not in result
        assert "http://a.com/1.png" in result
        assert "http://b.com/2.png" in result


class TestMattermostTruncateMessage:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_short_message_single_chunk(self):
        msg = "Hello, world!"
        chunks = self.adapter.truncate_message(msg, 4000)
        assert len(chunks) == 1
        assert chunks[0] == msg

    def test_long_message_splits(self):
        msg = "a " * 2500  # 5000 chars
        chunks = self.adapter.truncate_message(msg, 4000)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4000

    def test_custom_max_length(self):
        msg = "Hello " * 20
        chunks = self.adapter.truncate_message(msg, max_length=50)
        assert all(len(c) <= 50 for c in chunks)

    def test_exactly_at_limit(self):
        msg = "x" * 4000
        chunks = self.adapter.truncate_message(msg, 4000)
        assert len(chunks) == 1


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

        self.adapter._session.post = AsyncMock(return_value=mock_resp)

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
    async def test_send_empty_content_succeeds(self):
        """Empty content should return success without calling the API."""
        result = await self.adapter.send("channel_1", "")
        assert result.success is True

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

        self.adapter._session.post = AsyncMock(return_value=mock_resp)
        self.adapter._session.get = AsyncMock(return_value=mock_get_resp)

        result = await self.adapter.send("channel_1", "Reply!", reply_to="root_post")

        assert result.success is True
        payload = self.adapter._session.post.call_args[1]["json"]
        assert payload["root_id"] == "root_post"

    @pytest.mark.asyncio
    async def test_send_without_thread_no_root_id(self):
        """When reply_mode is 'off', reply_to should NOT set root_id."""
        self.adapter._reply_mode = "off"

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"id": "post789"})
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        self.adapter._session.post = AsyncMock(return_value=mock_resp)

        result = await self.adapter.send("channel_1", "Reply!", reply_to="root_post")

        assert result.success is True
        payload = self.adapter._session.post.call_args[1]["json"]
        assert "root_id" not in payload


    @pytest.mark.asyncio
    async def test_send_uses_metadata_thread_id_for_progress_messages(self):
        """Progress/status messages pass Mattermost thread context via metadata."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "root_post_123", "root_id": ""})
        self.adapter._api_post = AsyncMock(return_value={"id": "progress_post"})

        result = await self.adapter.send(
            "channel_1",
            "⚡ terminal...",
            metadata={"thread_id": "root_post_123"},
        )

        assert result.success is True
        payload = self.adapter._api_post.call_args_list[0][0][1]
        assert payload["root_id"] == "root_post_123"

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

    @pytest.mark.asyncio
    async def test_notify_send_with_invalid_thread_root_falls_back_flat_with_warning(self):
        """Notify-worthy replies may fall back flat so the answer is not lost."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "bad_root", "root_id": ""})
        self.adapter._last_post_status = 400
        self.adapter._last_post_error = "api.context.invalid_param.app_error: invalid root_id"
        self.adapter._api_post = AsyncMock(side_effect=[{}, {"id": "flat_final"}])

        result = await self.adapter.send(
            "channel_1",
            "Final answer body",
            reply_to="bad_root",
            metadata={"notify": True},
        )

        assert result.success is True
        assert result.message_id == "flat_final"
        assert self.adapter._api_post.call_count == 2
        threaded_payload = self.adapter._api_post.call_args_list[0][0][1]
        flat_payload = self.adapter._api_post.call_args_list[1][0][1]
        assert threaded_payload["root_id"] == "bad_root"
        assert "root_id" not in flat_payload
        assert flat_payload["channel_id"] == "channel_1"
        assert "Mattermost thread delivery failed" in flat_payload["message"]
        assert "Final answer body" in flat_payload["message"]

    @pytest.mark.asyncio
    async def test_notify_send_with_server_error_does_not_fall_back_flat(self):
        """Notify fallback is only for broken thread roots, not generic API failures."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_get = AsyncMock(return_value={"id": "root_post", "root_id": ""})
        self.adapter._last_post_status = 500
        self.adapter._last_post_error = "Internal Server Error"
        self.adapter._api_post = AsyncMock(return_value={})

        result = await self.adapter.send(
            "channel_1",
            "Final answer body",
            reply_to="root_post",
            metadata={"notify": True},
        )

        assert result.success is False
        assert self.adapter._api_post.call_count == 1
        payload = self.adapter._api_post.call_args_list[0][0][1]
        assert payload["root_id"] == "root_post"

    @pytest.mark.asyncio
    async def test_progress_send_with_invalid_thread_root_never_falls_back_flat(self):
        """Tool/status/progress bubbles must stay quiet when the thread is broken."""
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

    @pytest.mark.asyncio
    async def test_send_api_failure(self):
        """When API returns error, send should return failure."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.json = AsyncMock(return_value={})
        mock_resp.text = AsyncMock(return_value="Internal Server Error")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        self.adapter._session.post = AsyncMock(return_value=mock_resp)

        result = await self.adapter.send("channel_1", "Hello!")

        assert result.success is False


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
    async def test_ignore_own_messages(self):
        """Messages from the bot's own user_id should be ignored."""
        post_data = {
            "id": "post_self",
            "user_id": "bot_user_id",  # same as bot
            "channel_id": "chan_456",
            "message": "Bot echo",
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
    async def test_ignore_non_posted_events(self):
        """Non-'posted' events should be ignored."""
        event = {
            "event": "typing",
            "data": {"user_id": "user_123"},
        }

        await self.adapter._handle_ws_event(event)
        assert not self.adapter.handle_message.called

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
    async def test_channel_type_mapping(self):
        """channel_type 'D' should map to 'dm'."""
        post_data = {
            "id": "post_dm",
            "user_id": "user_123",
            "channel_id": "chan_dm",
            "message": "DM message",
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
        assert msg_event.source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_thread_id_from_root_id(self):
        """Post with root_id should have thread_id set."""
        post_data = {
            "id": "post_reply",
            "user_id": "user_123",
            "channel_id": "chan_456",
            "message": "@bot_user_id Thread reply",
            "root_id": "root_post_123",
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": "O",
                "sender_name": "@alice",
            },
        }

        await self.adapter._handle_ws_event(event)
        assert self.adapter.handle_message.called
        msg_event = self.adapter.handle_message.call_args[0][0]
        assert msg_event.source.thread_id == "root_post_123"

    @pytest.mark.asyncio
    async def test_invalid_post_json_ignored(self):
        """Invalid JSON in data.post should be silently ignored."""
        event = {
            "event": "posted",
            "data": {
                "post": "not-valid-json{{{",
                "channel_type": "O",
            },
        }

        await self.adapter._handle_ws_event(event)
        assert not self.adapter.handle_message.called


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
    async def test_require_mention_false_responds_to_all(self):
        """MATTERMOST_REQUIRE_MENTION=false: respond to all channel messages."""
        with patch.dict(os.environ, {"MATTERMOST_REQUIRE_MENTION": "false"}):
            await self.adapter._handle_ws_event(self._make_event("hello"))
            assert self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_free_response_channel_responds_without_mention(self):
        """Messages in free-response channels don't need @mention."""
        with patch.dict(os.environ, {"MATTERMOST_FREE_RESPONSE_CHANNELS": "chan_456,chan_789"}):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            await self.adapter._handle_ws_event(self._make_event("hello", channel_id="chan_456"))
            assert self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_non_free_channel_still_requires_mention(self):
        """Channels NOT in free-response list still require @mention."""
        with patch.dict(os.environ, {"MATTERMOST_FREE_RESPONSE_CHANNELS": "chan_789"}):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            await self.adapter._handle_ws_event(self._make_event("hello", channel_id="chan_456"))
            assert not self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_dm_always_responds(self):
        """DMs (channel_type=D) always respond regardless of mention settings."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            await self.adapter._handle_ws_event(self._make_event("hello", channel_type="D"))
            assert self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_mention_stripped_from_text(self):
        """@mention is stripped from message text."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            await self.adapter._handle_ws_event(
                self._make_event("@hermes-bot what is 2+2")
            )
            assert self.adapter.handle_message.called
            msg = self.adapter.handle_message.call_args[0][0]
            assert "@hermes-bot" not in msg.text
            assert "2+2" in msg.text


# ---------------------------------------------------------------------------
# File upload (send_image)
# ---------------------------------------------------------------------------

class TestMattermostFileUpload:
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
        self.adapter._session.get = AsyncMock(return_value=mock_dl_resp)
        post_call_count = 0
        original_post_returns = [mock_upload_resp, mock_post_resp]

        def post_side_effect(*args, **kwargs):
            nonlocal post_call_count
            resp = original_post_returns[min(post_call_count, len(original_post_returns) - 1)]
            post_call_count += 1
            return resp

        self.adapter._session.post = AsyncMock(side_effect=post_side_effect)

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

    @pytest.mark.asyncio
    async def test_duplicate_post_ignored(self):
        """The same post_id within the TTL window should be ignored."""
        post_data = {
            "id": "post_dup",
            "user_id": "user_123",
            "channel_id": "chan_456",
            "message": "@bot_user_id Hello!",
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": "O",
                "sender_name": "@alice",
            },
        }

        # First time: should process
        await self.adapter._handle_ws_event(event)
        assert self.adapter.handle_message.call_count == 1

        # Second time (same post_id): should be deduped
        await self.adapter._handle_ws_event(event)
        assert self.adapter.handle_message.call_count == 1  # still 1

    @pytest.mark.asyncio
    async def test_different_post_ids_both_processed(self):
        """Different post IDs should both be processed."""
        for i, pid in enumerate(["post_a", "post_b"]):
            post_data = {
                "id": pid,
                "user_id": "user_123",
                "channel_id": "chan_456",
                "message": f"@bot_user_id Message {i}",
            }
            event = {
                "event": "posted",
                "data": {
                    "post": json.dumps(post_data),
                    "channel_type": "O",
                    "sender_name": "@alice",
                },
            }
            await self.adapter._handle_ws_event(event)

        assert self.adapter.handle_message.call_count == 2

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

    def test_seen_cache_tracks_post_ids(self):
        """Posts are tracked in the dedup cache."""
        self.adapter._dedup._seen["test_post"] = time.time()
        assert "test_post" in self.adapter._dedup._seen


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestMattermostRequirements:
    def test_check_requirements_with_token_and_url(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "test-token")
        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
        from plugins.platforms.mattermost.adapter import check_mattermost_requirements
        assert check_mattermost_requirements() is True

    def test_check_requirements_without_token(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_TOKEN", raising=False)
        monkeypatch.delenv("MATTERMOST_URL", raising=False)
        from plugins.platforms.mattermost.adapter import check_mattermost_requirements
        assert check_mattermost_requirements() is False

    def test_check_requirements_without_url(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "test-token")
        monkeypatch.delenv("MATTERMOST_URL", raising=False)
        from plugins.platforms.mattermost.adapter import check_mattermost_requirements
        assert check_mattermost_requirements() is False


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
        self.adapter._session.get = AsyncMock(return_value=mock_resp)

        with patch("gateway.platforms.base.cache_image_from_bytes", return_value="/tmp/photo.png"):
            await self.adapter._handle_ws_event(self._make_event(["file1"]))

        msg = self.adapter.handle_message.call_args[0][0]
        assert msg.media_types == ["image/png"]
        assert msg.media_types[0].startswith("image/")

    @pytest.mark.asyncio
    async def test_audio_media_type_is_full_mime(self):
        """An audio attachment should produce 'audio/ogg', not 'audio'."""
        file_info = {"name": "voice.ogg", "mime_type": "audio/ogg"}
        self.adapter._api_get = AsyncMock(return_value=file_info)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"OGG fake")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        self.adapter._session = MagicMock()
        self.adapter._session.get = AsyncMock(return_value=mock_resp)

        with patch("gateway.platforms.base.cache_audio_from_bytes", return_value="/tmp/voice.ogg"), \
             patch("gateway.platforms.base.cache_image_from_bytes"), \
             patch("gateway.platforms.base.cache_document_from_bytes"):
            await self.adapter._handle_ws_event(self._make_event(["file2"]))

        msg = self.adapter.handle_message.call_args[0][0]
        assert msg.media_types == ["audio/ogg"]
        assert msg.media_types[0].startswith("audio/")

    @pytest.mark.asyncio
    async def test_document_media_type_is_full_mime(self):
        """A document attachment should produce 'application/pdf', not 'document'."""
        file_info = {"name": "report.pdf", "mime_type": "application/pdf"}
        self.adapter._api_get = AsyncMock(return_value=file_info)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"PDF fake")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        self.adapter._session = MagicMock()
        self.adapter._session.get = AsyncMock(return_value=mock_resp)

        with patch("gateway.platforms.base.cache_document_from_bytes", return_value="/tmp/report.pdf"), \
             patch("gateway.platforms.base.cache_image_from_bytes"):
            await self.adapter._handle_ws_event(self._make_event(["file3"]))

        msg = self.adapter.handle_message.call_args[0][0]
        assert msg.media_types == ["application/pdf"]
        assert not msg.media_types[0].startswith("image/")
        assert not msg.media_types[0].startswith("audio/")


# ======================================================================
# Sprint 6 — Interactive Approval Buttons Tests
# ======================================================================


class TestMattermostInteractionTokens:
    """Sprint 6.1 — Token signing, creation, and verification."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._hmac_secret = "a" * 32
        self.adapter._callback_base_url = "http://127.0.0.1:8391"

    def test_sign_produces_consistent_output(self):
        """Same payload → same hex digest."""
        sig1 = self.adapter._sign_interaction_token("test:payload")
        sig2 = self.adapter._sign_interaction_token("test:payload")
        assert sig1 == sig2

    def test_sign_different_payloads_different(self):
        """Different payloads → different digests."""
        sig1 = self.adapter._sign_interaction_token("foo:bar")
        sig2 = self.adapter._sign_interaction_token("foo:baz")
        assert sig1 != sig2

    def test_make_token_three_parts(self):
        """Format: <sig>.<rand_hex>.<ts>  (3 segments)."""
        token = self.adapter._make_interaction_token("approval", "ref123", "once")
        parts = token.rsplit(".", 2)
        assert len(parts) == 3
        sig, rand_hex, ts_str = parts
        assert len(sig) == 64  # SHA-256 hex digest is 64 chars
        int(rand_hex, 16)  # does not raise
        int(ts_str)  # does not raise

    def test_verify_accepts_valid_token(self):
        """Freshly created token passes verification."""
        kind, ref_id, choice = "approval", "ref123", "once"
        token = self.adapter._make_interaction_token(kind, ref_id, choice)
        assert self.adapter._verify_interaction_token(kind, ref_id, choice, token) is True

    def test_verify_rejects_expired_token(self):
        """Token past TTL window is rejected."""
        kind, ref_id, choice = "approval", "ref123", "once"
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000
            token = self.adapter._make_interaction_token(kind, ref_id, choice)
            mock_time.return_value = 1000 + self.adapter._INTERACTION_TOKEN_TTL + 1
            assert self.adapter._verify_interaction_token(kind, ref_id, choice, token) is False

    def test_verify_rejects_malformed_token(self):
        """Wrong number of segments → False."""
        assert self.adapter._verify_interaction_token("a", "b", "c", "no-dots") is False
        assert self.adapter._verify_interaction_token("a", "b", "c", "one.two") is False

    def test_verify_rejects_tampered_choice(self):
        """Token signed for one choice fails for a different choice."""
        kind, ref_id, choice = "approval", "ref123", "once"
        token = self.adapter._make_interaction_token(kind, ref_id, choice)
        assert self.adapter._verify_interaction_token(kind, ref_id, "session", token) is False

    def test_verify_rejects_non_hex_rand(self):
        """Non-hex characters in rand_hex → ValueError → False."""
        kind, ref_id, choice = "approval", "ref123", "once"
        token = self.adapter._make_interaction_token(kind, ref_id, choice)
        parts = token.rsplit(".", 2)
        sig, _, ts_str = parts
        bad_token = f"{sig}.ZZZZZZZZZZZZZZZZ.{ts_str}"
        assert self.adapter._verify_interaction_token(kind, ref_id, choice, bad_token) is False

    def test_verify_rejects_invalid_hmac(self):
        """Flipping the signature → HMAC mismatch → False."""
        kind, ref_id, choice = "approval", "ref123", "once"
        token = self.adapter._make_interaction_token(kind, ref_id, choice)
        parts = token.rsplit(".", 2)
        _, rand_hex, ts_str = parts
        bad_sig = "0" * 64
        bad_token = f"{bad_sig}.{rand_hex}.{ts_str}"
        assert self.adapter._verify_interaction_token(kind, ref_id, choice, bad_token) is False


class TestMattermostSendExecApproval:
    """Sprint 6.2 — Approval button payload generation."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._hmac_secret = "b" * 32
        self.adapter._callback_base_url = "http://127.0.0.1:8391"

    @pytest.mark.asyncio
    async def test_configured_sends_four_buttons(self):
        """Payload contains 4 actions with correct types, styles, names."""
        self.adapter._api_post = AsyncMock(return_value={"id": "post_123"})

        result = await self.adapter.send_exec_approval(
            chat_id="chan_test",
            command="dangerous command",
            session_key="sess_abc",
            description="testing",
        )
        assert result.success is True
        assert result.message_id == "post_123"

        payload = self.adapter._api_post.call_args[0][1]
        assert payload["channel_id"] == "chan_test"

        attachments = payload["props"]["attachments"]
        assert len(attachments) == 1
        assert "⚠️" in attachments[0]["pretext"]

        actions = attachments[0]["actions"]
        assert len(actions) == 4

        # Check each button shape
        expected = [
            ("✅ Allow Once", "once", "primary"),
            ("✅ Allow Session", "session", "default"),
            ("✅ Always Allow", "always", "default"),
            ("❌ Deny", "deny", "danger"),
        ]
        for action, (exp_name, exp_choice, exp_style) in zip(actions, expected):
            assert action["name"] == exp_name
            assert action["type"] == "button"
            assert action["style"] == exp_style
            assert action["integration"]["url"] == "http://127.0.0.1:8391/mattermost/interactions"
            assert action["integration"]["context"]["kind"] == "approval"
            assert action["integration"]["context"]["choice"] == exp_choice
            assert "approval_id" in action["integration"]["context"]
            assert "token" in action["integration"]["context"]

    @pytest.mark.asyncio
    async def test_not_configured_returns_failure(self):
        """No callback URL → success=False."""
        self.adapter._callback_base_url = ""
        result = await self.adapter.send_exec_approval(
            chat_id="chan_test",
            command="test",
            session_key="sess_abc",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_thread_mode_sets_root_id(self):
        """Threaded reply includes root_id in payload."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_post = AsyncMock(return_value={"id": "post_123"})
        self.adapter._resolve_root_id = AsyncMock(return_value="root_456")

        result = await self.adapter.send_exec_approval(
            chat_id="chan_test",
            command="test",
            session_key="sess_abc",
            metadata={"reply_to_message_id": "parent_post"},
        )
        assert result.success is True
        self.adapter._resolve_root_id.assert_called_once_with("parent_post")
        payload = self.adapter._api_post.call_args[0][1]
        assert payload["root_id"] == "root_456"

    @pytest.mark.asyncio
    async def test_flat_mode_omits_root_id(self):
        """Flat mode → no root_id in payload."""
        self.adapter._reply_mode = "off"
        self.adapter._api_post = AsyncMock(return_value={"id": "post_123"})

        await self.adapter.send_exec_approval(
            chat_id="chan_test",
            command="test",
            session_key="sess_abc",
            metadata={"reply_to_message_id": "parent_post"},
        )
        payload = self.adapter._api_post.call_args[0][1]
        assert "root_id" not in payload

    @pytest.mark.asyncio
    async def test_state_stored_correctly(self):
        """After send, _approval_state has the right keys."""
        self.adapter._api_post = AsyncMock(return_value={"id": "post_456"})

        await self.adapter.send_exec_approval(
            chat_id="chan_test",
            command="test",
            session_key="sess_abc",
        )

        # There should be exactly one entry in _approval_state
        assert len(self.adapter._approval_state) == 1
        approval_id = next(iter(self.adapter._approval_state))
        state = self.adapter._approval_state[approval_id]
        assert state["session_key"] == "sess_abc"
        assert state["chat_id"] == "chan_test"
        assert state["message_id"] == "post_456"

    @pytest.mark.asyncio
    async def test_api_failure_returns_failure(self):
        """Empty API response → success=False."""
        self.adapter._api_post = AsyncMock(return_value={})
        result = await self.adapter.send_exec_approval(
            chat_id="chan_test",
            command="test",
            session_key="sess_abc",
        )
        assert result.success is False


class TestMattermostSendSlashConfirm:
    """Sprint 6.3 — Slash-confirm button payload generation."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._hmac_secret = "c" * 32
        self.adapter._callback_base_url = "http://127.0.0.1:8391"

    @pytest.mark.asyncio
    async def test_configured_sends_three_buttons(self):
        """Payload contains 3 actions with kind='confirm'."""
        self.adapter._api_post = AsyncMock(return_value={"id": "post_sc_123"})

        result = await self.adapter.send_slash_confirm(
            chat_id="chan_test",
            title="Run command?",
            message="This will delete all data.",
            session_key="sess_abc",
            confirm_id="confirm_001",
        )
        assert result.success is True
        assert result.message_id == "post_sc_123"

        payload = self.adapter._api_post.call_args[0][1]
        assert payload["channel_id"] == "chan_test"
        assert "Run command?" in payload["message"]

        attachments = payload["props"]["attachments"]
        assert len(attachments) == 1
        actions = attachments[0]["actions"]
        assert len(actions) == 3

        expected = [
            ("✅ Approve Once", "once", "primary"),
            ("✅ Always Approve", "always", "default"),
            ("❌ Cancel", "cancel", "danger"),
        ]
        for action, (exp_name, exp_choice, exp_style) in zip(actions, expected):
            assert action["name"] == exp_name
            assert action["type"] == "button"
            assert action["style"] == exp_style
            assert action["integration"]["context"]["kind"] == "confirm"
            assert action["integration"]["context"]["choice"] == exp_choice
            assert action["integration"]["context"]["confirm_id"] == "confirm_001"
            assert "token" in action["integration"]["context"]

    @pytest.mark.asyncio
    async def test_not_configured_returns_failure(self):
        """No callback URL → success=False."""
        self.adapter._callback_base_url = ""
        result = await self.adapter.send_slash_confirm(
            chat_id="chan_test",
            title="Test",
            message="test",
            session_key="sess_abc",
            confirm_id="confirm_001",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_thread_mode_sets_root_id(self):
        """Threaded reply includes root_id."""
        self.adapter._reply_mode = "thread"
        self.adapter._api_post = AsyncMock(return_value={"id": "post_sc_123"})
        self.adapter._resolve_root_id = AsyncMock(return_value="root_789")

        result = await self.adapter.send_slash_confirm(
            chat_id="chan_test",
            title="Test",
            message="test",
            session_key="sess_abc",
            confirm_id="confirm_001",
            metadata={"reply_to_message_id": "parent_post"},
        )
        assert result.success is True
        payload = self.adapter._api_post.call_args[0][1]
        assert payload["root_id"] == "root_789"

    @pytest.mark.asyncio
    async def test_state_stored_correctly(self):
        """_slash_confirm_state has correct mapping."""
        self.adapter._api_post = AsyncMock(return_value={"id": "post_sc_456"})

        await self.adapter.send_slash_confirm(
            chat_id="chan_test",
            title="Test",
            message="test",
            session_key="sess_abc",
            confirm_id="confirm_001",
        )

        assert "confirm_001" in self.adapter._slash_confirm_state
        state = self.adapter._slash_confirm_state["confirm_001"]
        assert state["session_key"] == "sess_abc"
        assert state["chat_id"] == "chan_test"
        assert state["message_id"] == "post_sc_456"


class TestMattermostInteractionHandler:
    """Sprint 6.4 — Inbound interaction handler logic."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._hmac_secret = "d" * 32
        self.adapter._callback_base_url = "http://127.0.0.1:8391"
        self.adapter._verify_interaction_token = MagicMock(return_value=True)
        self.adapter._is_interactive_user_authorized = MagicMock(return_value=True)

    def _make_mock_request(self, payload: dict) -> MagicMock:
        """Build a mock aiohttp request that returns the given JSON payload."""
        req = MagicMock()
        req.read = AsyncMock(return_value=json.dumps(payload).encode())
        return req

    @pytest.mark.asyncio
    async def test_valid_token_resolves_approval(self):
        """Valid approval token → resolve_gateway_approval called."""
        self.adapter._approval_state["appr_001"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
        }
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_001",
                    "choice": "once",
                    "token": "dummy_token",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        # Returns an update message replacing the post
        assert "update" in body
        mock_resolve.assert_called_once_with("sess_abc", "once")

    @pytest.mark.asyncio
    async def test_unauthorized_user_returns_error(self):
        """Unauthorized user → 200 with 'not authorized', state preserved."""
        self.adapter._is_interactive_user_authorized = MagicMock(return_value=False)
        self.adapter._approval_state["appr_002"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
        }
        req = self._make_mock_request({
            "user_id": "unknown_user",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_002",
                    "choice": "once",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "not authorized" in body.get("text", "").lower()
        mock_resolve.assert_not_called()
        # State preserved — not popped
        assert "appr_002" in self.adapter._approval_state

    @pytest.mark.asyncio
    async def test_stale_approval_returns_already_resolved(self):
        """No state for approval_id → 'already resolved'."""
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "stale_appr",
                    "choice": "once",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "already been resolved" in body.get("text", "").lower()
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_channel_mismatch_preserves_state(self):
        """Different chat_id in request vs state → state preserved, not resolved."""
        self.adapter._approval_state["appr_003"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_original",
            "message_id": "msg_123",
        }
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_different",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_003",
                    "choice": "once",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "channel mismatch" in body.get("text", "").lower()
        mock_resolve.assert_not_called()
        # State was put back
        assert "appr_003" in self.adapter._approval_state

    @pytest.mark.asyncio
    async def test_different_user_rejected(self):
        """Button-clicker != requester → rejected, state preserved."""
        self.adapter._approval_state["appr_usr"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
            "requester_user_id": "requester_1",
        }
        req = self._make_mock_request({
            "user_id": "different_user",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_usr",
                    "choice": "once",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "Only the user who requested" in body.get("text", "")
        mock_resolve.assert_not_called()
        # State was put back
        assert "appr_usr" in self.adapter._approval_state

    @pytest.mark.asyncio
    async def test_same_user_allowed(self):
        """Button-clicker == requester → approved, state consumed."""
        self.adapter._approval_state["appr_same"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
            "requester_user_id": "user_1",
        }
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_same",
                    "choice": "once",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "update" in body
        mock_resolve.assert_called_once_with("sess_abc", "once")
        assert "appr_same" not in self.adapter._approval_state

    @pytest.mark.asyncio
    async def test_deny_choice_resolved_correctly(self):
        """choice='deny' → resolve_gateway_approval called with 'deny'."""
        self.adapter._approval_state["appr_004"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
        }
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_004",
                    "choice": "deny",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "❌" in body["update"]["message"]
        mock_resolve.assert_called_once_with("sess_abc", "deny")

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self):
        """Expired/invalid token → 200 with error text, not resolved."""
        self.adapter._verify_interaction_token = MagicMock(return_value=False)
        self.adapter._approval_state["appr_005"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
        }
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "approval",
                    "approval_id": "appr_005",
                    "choice": "once",
                    "token": "bad_token",
                },
            },
        })

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        # The actual text says "expired or invalid" — match the full phrase
        assert "expired" in body.get("text", "").lower() or "invalid" in body.get("text", "").lower()
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_routes_confirm_kind(self):
        """kind='confirm' → slash-confirm resolver path."""
        self.adapter._slash_confirm_state["confirm_001"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_confirm",
        }
        req = self._make_mock_request({
            "user_id": "user_1",
            "channel_id": "chan_test",
            "data": {
                "context": {
                    "kind": "confirm",
                    "confirm_id": "confirm_001",
                    "choice": "once",
                    "token": "dummy",
                },
            },
        })

        with patch("tools.slash_confirm.resolve", AsyncMock(return_value="ok")) as mock_resolve:
            response = await self.adapter._handle_interaction_request(req)

        assert response.status == 200
        body = json.loads(response.body)
        assert "update" in body
        mock_resolve.assert_called_once_with("sess_abc", "confirm_001", "once")


class TestMattermostSlashConfirmResolution:
    """Sprint 6.5 — Slash-confirm resolution from button clicks."""

    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_valid_confirm_id_resolves(self):
        """Existing confirm_id → tools.slash_confirm.resolve called."""
        self.adapter._slash_confirm_state["confirm_001"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
        }
        with patch("tools.slash_confirm.resolve", AsyncMock(return_value="done")) as mock_resolve:
            response = await self.adapter._resolve_slash_confirm_interaction(
                "confirm_001", "once", "user_1", "chan_test",
            )

        assert response.status == 200
        body = json.loads(response.body)
        assert "update" in body
        assert "✅" in body["update"]["message"]
        mock_resolve.assert_called_once_with("sess_abc", "confirm_001", "once")

    @pytest.mark.asyncio
    async def test_stale_confirm_id_returns_already_resolved(self):
        """No state for confirm_id → 'already resolved'."""
        response = await self.adapter._resolve_slash_confirm_interaction(
            "nonexistent", "once", "user_1", "chan_test",
        )
        assert response.status == 200
        body = json.loads(response.body)
        assert "already been resolved" in body.get("text", "").lower()

    @pytest.mark.asyncio
    async def test_replay_rejected(self):
        """Second click after pop → 'already resolved'."""
        self.adapter._slash_confirm_state["confirm_002"] = {
            "session_key": "sess_abc",
            "chat_id": "chan_test",
            "message_id": "msg_123",
        }
        with patch("tools.slash_confirm.resolve", AsyncMock(return_value="ok")):
            await self.adapter._resolve_slash_confirm_interaction(
                "confirm_002", "once", "user_1", "chan_test",
            )

        # State is now empty after pop. Second call should be rejected.
        response = await self.adapter._resolve_slash_confirm_interaction(
            "confirm_002", "once", "user_1", "chan_test",
        )
        body = json.loads(response.body)
        assert "already been resolved" in body.get("text", "").lower()


class TestMattermostInteractionConfigFlow:
    """Sprint 6.6 — Config plumbing tests."""

    def test_apply_yaml_config_returns_interactions_block(self):
        """When interactions block present, _apply_yaml_config returns it."""
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        mattermost_cfg = {
            "interactions": {
                "callback_url": "https://example.com/callback",
                "hmac_secret": "my-secret-key",
                "listen_host": "0.0.0.0",
                "listen_port": 9999,
            },
        }
        result = _apply_yaml_config({}, mattermost_cfg)
        assert result == {"interactions": mattermost_cfg["interactions"]}

    def test_apply_yaml_config_returns_none_without_interactions(self):
        """No interactions block → returns None."""
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        result = _apply_yaml_config({}, {"token": "abc"})
        assert result is None

    def test_apply_yaml_config_hmac_bridge_sets_env(self, monkeypatch):
        """The HMAC secret is bridged into MATTERMOST_INTERACTIONS_HMAC_SECRET."""
        monkeypatch.delenv("MATTERMOST_INTERACTIONS_HMAC_SECRET", raising=False)
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        mattermost_cfg = {
            "interactions": {"hmac_secret": "my-secret"},
        }
        _apply_yaml_config({}, mattermost_cfg)
        assert os.environ["MATTERMOST_INTERACTIONS_HMAC_SECRET"] == "my-secret"

    def test_apply_yaml_config_env_takes_precedence(self, monkeypatch):
        """Env var already set → YAML is NOT bridged (no override)."""
        monkeypatch.setenv("MATTERMOST_INTERACTIONS_HMAC_SECRET", "env-var-value")
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        mattermost_cfg = {
            "interactions": {"hmac_secret": "yaml-value"},
        }
        _apply_yaml_config({}, mattermost_cfg)
        # Env still has the original value
        assert os.environ["MATTERMOST_INTERACTIONS_HMAC_SECRET"] == "env-var-value"

    @pytest.mark.asyncio
    async def test_connect_reads_interaction_config(self):
        """connect() reads interactions from config.extra and env."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "url": "https://mm.example.com",
                "interactions": {
                    "callback_url": "https://cb.example.com",
                    "listen_host": "0.0.0.0",
                    "listen_port": 9999,
                },
            },
        )
        # Must be ≥ 32 bytes or connect() will disable buttons
        long_secret = "a" * 32
        with patch.dict(
            os.environ,
            {
                "MATTERMOST_INTERACTIONS_HMAC_SECRET": long_secret,
            },
            clear=False,
        ):
            adapter = MattermostAdapter(config)
            # Mock _api_get and _start_interaction_server so connect doesn't
            # make real network calls
            adapter._start_interaction_server = AsyncMock()
            adapter._api_get = AsyncMock(return_value={
                "id": "bot_user_id",
                "username": "hermes-bot",
            })

            result = await adapter.connect()

        assert result is True
        assert adapter._callback_base_url == "https://cb.example.com"
        assert adapter._listen_host == "0.0.0.0"
        assert adapter._listen_port == 9999
        assert adapter._hmac_secret == long_secret
        adapter._start_interaction_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_disabled_without_secret(self):
        """No HMAC secret → interaction server not started."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "url": "https://mm.example.com",
                "interactions": {
                    "callback_url": "https://cb.example.com",
                },
            },
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_INTERACTIONS_HMAC_SECRET", None)
            adapter = MattermostAdapter(config)
            adapter._api_get = AsyncMock(return_value={
                "id": "bot_user_id",
                "username": "hermes-bot",
            })
            adapter._start_interaction_server = AsyncMock()

            result = await adapter.connect()

        assert result is True
        assert adapter._callback_base_url == "https://cb.example.com"
        assert adapter._hmac_secret == ""  # empty secret
        adapter._start_interaction_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_disabled_with_short_secret(self):
        """HMAC secret < 32 bytes → interaction server not started, warning logged."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "url": "https://mm.example.com",
                "interactions": {
                    "callback_url": "https://cb.example.com",
                    "hmac_secret": "short",
                },
            },
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_INTERACTIONS_HMAC_SECRET", None)
            adapter = MattermostAdapter(config)
            adapter._api_get = AsyncMock(return_value={
                "id": "bot_user_id",
                "username": "hermes-bot",
            })
            adapter._start_interaction_server = AsyncMock()

            result = await adapter.connect()

        assert result is True
        # callback_url is cleared because HMAC is too short
        assert adapter._callback_base_url == ""
        adapter._start_interaction_server.assert_not_called()


class TestMattermostInteractionIntegration:
    """Sprint 6.7 — End-to-end integration test with a real interaction server."""

    @pytest.mark.asyncio
    async def test_full_callback_flow(self):
        """
        Start interaction server → send approval → POST callback → verify resolution.
        """
        # Use a high port to avoid colliding with the default 8391
        server_port = 18391

        adapter = _make_adapter()
        adapter._callback_base_url = f"http://127.0.0.1:{server_port}"
        adapter._hmac_secret = "e" * 32
        adapter._listen_host = "127.0.0.1"
        adapter._listen_port = server_port
        adapter._session = aiohttp.ClientSession()

        # Start the interaction server
        await adapter._start_interaction_server()
        assert adapter._interaction_site is not None

        try:
            # Mock _api_post so send_exec_approval doesn't hit the real API
            adapter._api_post = AsyncMock(return_value={"id": "post_approval_123"})

            # Send an approval prompt
            result = await adapter.send_exec_approval(
                chat_id="chan_test",
                command="dangerous command",
                session_key="sess_abc",
                description="test",
            )
            assert result.success is True

            # Extract the button context from the API call payload
            call_payload = adapter._api_post.call_args[0][1]
            attachment = call_payload["props"]["attachments"][0]
            first_action = attachment["actions"][0]
            context = first_action["integration"]["context"]

            # Simulate Mattermost sending a callback POST to our server
            callback_payload = {
                "user_id": "test_user",
                "channel_id": "chan_test",
                "data": {
                    "context": context,
                },
            }

            with patch(
                "tools.approval.resolve_gateway_approval", return_value=1,
            ) as mock_resolve:
                async with aiohttp.ClientSession() as client:
                    resp = await client.post(
                        f"http://127.0.0.1:{server_port}/mattermost/interactions",
                        json=callback_payload,
                    )
                    assert resp.status == 200
                    resp_body = await resp.json()
                    assert "update" in resp_body

                # Verify the approval was resolved
                mock_resolve.assert_called_once_with("sess_abc", "once")

        finally:
            # Clean up the interaction server
            if adapter._interaction_site:
                await adapter._interaction_site.stop()
            if adapter._interaction_runner:
                await adapter._interaction_runner.cleanup()
            if adapter._session and not adapter._session.closed:
                await adapter._session.close()


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


@pytest.mark.asyncio
async def test_mattermost_dm_post_does_not_seed_thread_root():
    adapter = _make_adapter()
    adapter._reply_mode = "thread"
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter.handle_message = AsyncMock()
    post_data = {
        "id": "dm_post_123",
        "user_id": "user_123",
        "channel_id": "dm_chan",
        "message": "hello",
        "root_id": "",
    }
    event = {
        "event": "posted",
        "data": {
            "post": json.dumps(post_data),
            "channel_type": "D",
            "sender_name": "@alice",
        },
    }

    await adapter._handle_ws_event(event)

    msg_event = adapter.handle_message.call_args[0][0]
    assert msg_event.source.thread_id is None
    assert msg_event.source.message_id == "dm_post_123"

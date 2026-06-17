"""Tests for Mattermost platform adapter."""
import json
import os
import time
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
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

        self.adapter._session.post = MagicMock(return_value=mock_resp)
        self.adapter._session.get = MagicMock(return_value=mock_get_resp)

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

        self.adapter._session.post = MagicMock(return_value=mock_resp)

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

        self.adapter._session.post = MagicMock(return_value=mock_resp)

        result = await self.adapter.send("channel_1", "Hello!")

        assert result.success is False


    @pytest.mark.asyncio
    async def test_send_image_file_uses_metadata_thread_id(self, tmp_path):
        """Local file uploads should keep Mattermost thread context from metadata."""
        self.adapter._reply_mode = "thread"
        image_path = tmp_path / "example.png"
        image_path.write_bytes(b"png")
        self.adapter._upload_file = AsyncMock(return_value="file_123")
        self.adapter._api_get = AsyncMock(return_value={"id": "root_post_123", "root_id": ""})
        self.adapter._api_post = AsyncMock(return_value={"id": "post_with_file"})

        result = await self.adapter.send_image_file(
            "channel_1",
            str(image_path),
            metadata={"thread_id": "root_post_123"},
        )

        assert result.success is True
        payload = self.adapter._api_post.call_args[0][1]
        assert payload["root_id"] == "root_post_123"
        assert payload["file_ids"] == ["file_123"]
        assert payload["message"] == "📎 example.png"

    @pytest.mark.asyncio
    async def test_send_multiple_images_uses_metadata_thread_id(self, tmp_path):
        """Batched MEDIA image uploads should stay inside the Mattermost thread."""
        self.adapter._reply_mode = "thread"
        image_path = tmp_path / "example.png"
        image_path.write_bytes(b"png")
        self.adapter._upload_file = AsyncMock(return_value="file_123")
        self.adapter._api_get = AsyncMock(return_value={"id": "root_post_123", "root_id": ""})
        self.adapter._api_post = AsyncMock(return_value={"id": "post_with_file"})

        await self.adapter.send_multiple_images(
            "channel_1",
            [(f"file://{image_path}", "")],
            metadata={"thread_id": "root_post_123"},
        )

        payload = self.adapter._api_post.call_args[0][1]
        assert payload["root_id"] == "root_post_123"
        assert payload["file_ids"] == ["file_123"]
        assert payload["message"] == "📎 example.png"


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

    @pytest.mark.asyncio
    async def test_leading_space_normal_text_is_preserved(self):
        """Only command-shaped mobile messages should be normalized."""
        post_data = {
            "id": "post_text",
            "user_id": "user_123",
            "channel_id": "chan_dm",
            "message": " hello",
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
        assert msg_event.text == " hello"
        assert msg_event.message_type is MessageType.TEXT

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

    @pytest.mark.asyncio
    @patch("tools.url_safety.is_safe_url", return_value=True)
    async def test_send_image_without_caption_uses_visible_filename(self, _mock_safe):
        """URL image uploads without alt text still produce a visible post body."""
        mock_dl_resp = AsyncMock()
        mock_dl_resp.status = 200
        mock_dl_resp.read = AsyncMock(return_value=b"\x89PNG\x00fake-image-data")
        mock_dl_resp.content_type = "image/png"
        mock_dl_resp.__aenter__ = AsyncMock(return_value=mock_dl_resp)
        mock_dl_resp.__aexit__ = AsyncMock(return_value=False)
        self.adapter._session.get = MagicMock(return_value=mock_dl_resp)
        self.adapter._upload_file = AsyncMock(return_value="file_abc123")
        self.adapter._post_preserving_thread = AsyncMock(
            return_value={"id": "post_with_file"}
        )

        result = await self.adapter.send_image(
            "channel_1", "https://img.example.com/cat.png"
        )

        assert result.success is True
        payload = self.adapter._post_preserving_thread.call_args.args[1]
        assert payload["message"] == "📎 cat.png"
        assert payload["file_ids"] == ["file_abc123"]


class TestMattermostStandaloneFileUpload:
    def test_file_post_message_preserves_explicit_caption(self):
        from plugins.platforms.mattermost.adapter import _file_post_message

        assert _file_post_message("Here is the file", ["example.png"]) == "Here is the file"
        spaced_caption = "  Keep leading/trailing whitespace  \n"
        assert _file_post_message(spaced_caption, ["example.png"]) == spaced_caption

    def test_file_post_message_uses_sanitized_filename_fallbacks(self):
        from plugins.platforms.mattermost.adapter import _file_post_message

        assert _file_post_message("", [" example.png "]) == "📎 example.png"
        assert _file_post_message(None, ["@channel\nimage.png"]) == "📎 @\u200bchannel image.png"
        assert _file_post_message("", ["one.png", "two.png"]) == "📎 one.png\n📎 two.png"
        assert _file_post_message("", ["", "   "]) == "📎 Attachment"

    @pytest.mark.asyncio
    async def test_file_only_post_uses_visible_sanitized_filename(
        self, tmp_path, monkeypatch
    ):
        import aiohttp
        from plugins.platforms.mattermost.adapter import _standalone_send

        media_path = tmp_path / "@channel voice.ogg"
        media_path.write_bytes(b"OggS test")
        requests = []

        class Response:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self):
                return self._payload

            async def text(self):
                return ""

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def post(self, url, **kwargs):
                requests.append({"url": url, **kwargs})
                if url.endswith("/files"):
                    return Response(201, {"file_infos": [{"id": "file-1"}]})
                return Response(201, {"id": "post-1"})

        monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: Session())
        monkeypatch.setattr(
            "gateway.platforms.base.resolve_proxy_url", lambda **_kwargs: None
        )
        monkeypatch.setattr(
            "gateway.platforms.base.proxy_kwargs_for_aiohttp",
            lambda _proxy: ({}, {}),
        )

        result = await _standalone_send(
            SimpleNamespace(
                token="test-token",
                extra={"url": "https://mm.example.com"},
            ),
            "channel-id",
            "",
            thread_id="root-post-id",
            media_files=[(str(media_path), True)],
        )

        assert result.get("error") is None, result
        assert result["success"] is True
        post_payload = requests[-1]["json"]
        assert post_payload == {
            "channel_id": "channel-id",
            "message": "📎 @\u200bchannel voice.ogg",
            "root_id": "root-post-id",
            "file_ids": ["file-1"],
        }

    @pytest.mark.asyncio
    async def test_standalone_upload_batches_files_and_sends_caption_once(
        self, tmp_path, monkeypatch
    ):
        import aiohttp
        from plugins.platforms.mattermost.adapter import _standalone_send

        media_files = []
        for index in range(6):
            media_path = tmp_path / f"report-{index}.txt"
            media_path.write_text(f"report {index}", encoding="utf-8")
            media_files.append((str(media_path), False))

        upload_count = 0
        post_payloads = []

        class Response:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self):
                return self._payload

            async def text(self):
                return ""

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def post(self, url, **kwargs):
                nonlocal upload_count
                if url.endswith("/files"):
                    upload_count += 1
                    return Response(
                        201, {"file_infos": [{"id": f"file-{upload_count}"}]}
                    )
                post_payloads.append(kwargs["json"])
                return Response(201, {"id": f"post-{len(post_payloads)}"})

        monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: Session())
        monkeypatch.setattr(
            "gateway.platforms.base.resolve_proxy_url", lambda **_kwargs: None
        )
        monkeypatch.setattr(
            "gateway.platforms.base.proxy_kwargs_for_aiohttp",
            lambda _proxy: ({}, {}),
        )

        result = await _standalone_send(
            SimpleNamespace(
                token="test-token",
                extra={"url": "https://mm.example.com"},
            ),
            "channel-id",
            "  exact caption  ",
            thread_id="root-post-id",
            media_files=media_files,
        )

        assert result.get("error") is None, result
        assert result["message_id"] == "post-2"
        assert upload_count == 6
        assert post_payloads == [
            {
                "channel_id": "channel-id",
                "message": "  exact caption  ",
                "root_id": "root-post-id",
                "file_ids": ["file-1", "file-2", "file-3", "file-4", "file-5"],
            },
            {
                "channel_id": "channel-id",
                "message": "📎 report-5.txt",
                "root_id": "root-post-id",
                "file_ids": ["file-6"],
            },
        ]

    @pytest.mark.asyncio
    async def test_send_message_file_only_reaches_standalone_sender(self, tmp_path):
        from tools.send_message_tool import _send_to_platform

        media_path = tmp_path / "voice.ogg"
        media_path.write_bytes(b"OggS test")
        media_files = [(str(media_path), True)]
        sender = AsyncMock(return_value={"success": True, "message_id": "post-1"})
        entry = SimpleNamespace(max_message_length=4000, standalone_sender_fn=sender)
        pconfig = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"url": "https://mm.example.com"},
        )

        with patch("gateway.run._gateway_runner_ref", return_value=None), patch(
            "gateway.platform_registry.platform_registry.get", return_value=entry
        ):
            result = await _send_to_platform(
                Platform.MATTERMOST,
                pconfig,
                "channel-id",
                "",
                thread_id="root-post-id",
                media_files=media_files,
            )

        assert result == {"success": True, "message_id": "post-1"}
        sender.assert_awaited_once_with(
            pconfig,
            "channel-id",
            "",
            thread_id="root-post-id",
            media_files=media_files,
            force_document=False,
        )

    @pytest.mark.asyncio
    async def test_send_message_long_text_attaches_media_only_to_final_chunk(
        self, tmp_path
    ):
        from tools.send_message_tool import _send_to_platform

        media_path = tmp_path / "report.txt"
        media_path.write_text("report", encoding="utf-8")
        media_files = [(str(media_path), False)]
        sender = AsyncMock(return_value={"success": True, "message_id": "post-1"})
        entry = SimpleNamespace(max_message_length=4000, standalone_sender_fn=sender)
        pconfig = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"url": "https://mm.example.com"},
        )

        with patch("gateway.run._gateway_runner_ref", return_value=None), patch(
            "gateway.platform_registry.platform_registry.get", return_value=entry
        ):
            result = await _send_to_platform(
                Platform.MATTERMOST,
                pconfig,
                "channel-id",
                "x" * 5000,
                media_files=media_files,
            )

        assert result == {"success": True, "message_id": "post-1"}
        assert sender.await_count == 2
        assert sender.await_args_list[0].kwargs["media_files"] is None
        assert sender.await_args_list[1].kwargs["media_files"] == media_files

    @pytest.mark.asyncio
    async def test_send_message_file_only_reaches_live_adapter(self, tmp_path):
        from tools.send_message_tool import _send_to_platform

        media_path = tmp_path / "voice.ogg"
        media_path.write_bytes(b"OggS test")
        live_adapter = SimpleNamespace(
            send_voice=AsyncMock(
                return_value=SimpleNamespace(
                    success=True, message_id="post-1", error=None
                )
            ),
            send_document=AsyncMock(),
        )
        runner = SimpleNamespace(adapters={Platform.MATTERMOST: live_adapter})
        entry = SimpleNamespace(max_message_length=4000, standalone_sender_fn=AsyncMock())
        pconfig = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"url": "https://mm.example.com"},
        )

        with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
            "gateway.platform_registry.platform_registry.get", return_value=entry
        ):
            result = await _send_to_platform(
                Platform.MATTERMOST,
                pconfig,
                "channel-id",
                "",
                thread_id="root-post-id",
                media_files=[(str(media_path), True)],
            )

        assert result == {"success": True, "message_id": "post-1"}
        live_adapter.send_voice.assert_awaited_once_with(
            chat_id="channel-id",
            audio_path=str(media_path),
            caption="",
            metadata={"thread_id": "root-post-id"},
        )

    @pytest.mark.asyncio
    async def test_send_message_missing_live_media_preserves_text(self, tmp_path):
        from tools.send_message_tool import _send_to_platform

        missing_path = tmp_path / "missing-report.txt"
        live_adapter = SimpleNamespace(
            send=AsyncMock(
                return_value=SimpleNamespace(
                    success=True, message_id="text-post", error=None
                )
            ),
            send_voice=AsyncMock(),
            send_document=AsyncMock(
                return_value=SimpleNamespace(
                    success=True, message_id=None, error=None
                )
            ),
        )
        runner = SimpleNamespace(adapters={Platform.MATTERMOST: live_adapter})
        entry = SimpleNamespace(max_message_length=4000, standalone_sender_fn=AsyncMock())
        pconfig = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"url": "https://mm.example.com"},
        )

        with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
            "gateway.platform_registry.platform_registry.get", return_value=entry
        ):
            result = await _send_to_platform(
                Platform.MATTERMOST,
                pconfig,
                "channel-id",
                "keep this text",
                thread_id="root-post-id",
                media_files=[(str(missing_path), False)],
            )

        assert result == {"success": True, "message_id": "text-post"}
        live_adapter.send_document.assert_awaited_once_with(
            chat_id="channel-id",
            file_path=str(missing_path),
            caption="keep this text",
            metadata={"thread_id": "root-post-id"},
        )
        live_adapter.send.assert_awaited_once_with(
            chat_id="channel-id",
            content="keep this text",
            metadata={"thread_id": "root-post-id"},
        )


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
        assert check_mattermost_requirements() is True

    def test_check_requirements_without_url(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "test-token")
        monkeypatch.delenv("MATTERMOST_URL", raising=False)
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

    def test_validate_config_rejects_missing_url(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_URL", raising=False)
        from plugins.platforms.mattermost.adapter import validate_mattermost_config

        config = PlatformConfig(enabled=True, token="cfg-token", extra={})
        assert validate_mattermost_config(config) is False


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
        self.adapter._session.get = MagicMock(return_value=mock_resp)

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
        self.adapter._session.get = MagicMock(return_value=mock_resp)

        with patch("gateway.platforms.base.cache_document_from_bytes", return_value="/tmp/report.pdf"), \
             patch("gateway.platforms.base.cache_image_from_bytes"):
            await self.adapter._handle_ws_event(self._make_event(["file3"]))

        msg = self.adapter.handle_message.call_args[0][0]
        assert msg.media_types == ["application/pdf"]
        assert not msg.media_types[0].startswith("image/")
        assert not msg.media_types[0].startswith("audio/")



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

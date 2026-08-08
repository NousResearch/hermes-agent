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



"""Tests for Mattermost platform adapter."""
import json
import os
import time
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


# ---------------------------------------------------------------------------
# Thread auto-follow (Slack parity)
# ---------------------------------------------------------------------------


class TestMattermostThreadAutoResponse:
    """Tests for the Slack-parity thread-follow feature.

    After the bot is @mentioned in a non-DM channel, follow-up messages in
    that same thread must be processed WITHOUT requiring a re-mention, unless
    strict_mention=true is set.
    """

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._bot_user_id = "bot_user_id"
        self.adapter._bot_username = "hermes-bot"
        self.adapter._reply_mode = "thread"
        self.adapter.handle_message = AsyncMock()

    def _make_event(
        self,
        message,
        post_id="post_root",
        root_id=None,
        channel_type="O",
        channel_id="chan_456",
    ):
        post_data = {
            "id": post_id,
            "user_id": "user_123",
            "channel_id": channel_id,
            "message": message,
        }
        if root_id is not None:
            post_data["root_id"] = root_id
        return {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": channel_type,
                "sender_name": "@alice",
            },
        }

    # 1. Mention registers the thread.
    @pytest.mark.asyncio
    async def test_mention_registers_thread(self):
        """After an @mention in an 'O' channel, thread_id lands in _mentioned_threads."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            os.environ.pop("MATTERMOST_STRICT_MENTION", None)
            await self.adapter._handle_ws_event(
                self._make_event("@hermes-bot hello", post_id="root_post_1")
            )
        assert "root_post_1" in self.adapter._mentioned_threads

    # 2. Follow-up without mention in a mentioned thread IS processed.
    @pytest.mark.asyncio
    async def test_followup_in_mentioned_thread_is_processed(self):
        """Second message in the same thread (no mention) must reach handle_message."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            os.environ.pop("MATTERMOST_STRICT_MENTION", None)
            # First: @mention on the root post.
            await self.adapter._handle_ws_event(
                self._make_event("@hermes-bot hello", post_id="root_post_2")
            )
            self.adapter.handle_message.reset_mock()
            # Second: reply in the same thread, no mention.
            await self.adapter._handle_ws_event(
                self._make_event(
                    "follow-up without mention",
                    post_id="reply_post_2",
                    root_id="root_post_2",
                )
            )
        assert self.adapter.handle_message.called

    # 3. Follow-up in an UNRELATED thread without mention is dropped.
    @pytest.mark.asyncio
    async def test_followup_in_unrelated_thread_is_dropped(self):
        """Reply in a different thread without mention must NOT reach handle_message."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            os.environ.pop("MATTERMOST_STRICT_MENTION", None)
            # Mention thread A.
            await self.adapter._handle_ws_event(
                self._make_event("@hermes-bot hello", post_id="root_thread_a")
            )
            self.adapter.handle_message.reset_mock()
            # Reply in thread B (different root_id).
            await self.adapter._handle_ws_event(
                self._make_event(
                    "message in other thread",
                    post_id="reply_in_b",
                    root_id="root_thread_b",
                )
            )
        assert not self.adapter.handle_message.called

    # 4. strict_mention=true disables auto-follow.
    @pytest.mark.asyncio
    async def test_strict_mention_disables_auto_follow(self):
        """With MATTERMOST_STRICT_MENTION=true, follow-ups without mention are dropped,
        and @mention does NOT register the thread."""
        with patch.dict(
            os.environ,
            {"MATTERMOST_STRICT_MENTION": "true"},
            clear=False,
        ):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            # @mention — should respond but NOT register the thread.
            await self.adapter._handle_ws_event(
                self._make_event("@hermes-bot hello", post_id="strict_root")
            )
            assert self.adapter.handle_message.called
            assert "strict_root" not in self.adapter._mentioned_threads

            self.adapter.handle_message.reset_mock()
            # Follow-up in that thread without mention — must be dropped.
            await self.adapter._handle_ws_event(
                self._make_event(
                    "follow-up without mention",
                    post_id="strict_reply",
                    root_id="strict_root",
                )
            )
        assert not self.adapter.handle_message.called

    # 4b. strict_mention via config.extra (not env var).
    @pytest.mark.asyncio
    async def test_strict_mention_via_config_extra(self):
        """strict_mention=True in config.extra also disables auto-follow."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"url": "https://mm.example.com", "strict_mention": True},
        )
        adapter = MattermostAdapter(config)
        adapter._bot_user_id = "bot_user_id"
        adapter._bot_username = "hermes-bot"
        adapter._reply_mode = "thread"
        adapter.handle_message = AsyncMock()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATTERMOST_REQUIRE_MENTION", None)
            os.environ.pop("MATTERMOST_STRICT_MENTION", None)

            # @mention should go through but not register thread.
            await adapter._handle_ws_event(
                self._make_event("@hermes-bot hello", post_id="cfg_root")
            )
            assert adapter.handle_message.called
            assert "cfg_root" not in adapter._mentioned_threads

            adapter.handle_message.reset_mock()
            # Follow-up without mention — must be dropped.
            await adapter._handle_ws_event(
                self._make_event(
                    "follow-up",
                    post_id="cfg_reply",
                    root_id="cfg_root",
                )
            )
        assert not adapter.handle_message.called

    # 5. _has_active_session_for_thread key matches real flow for public "O" channel.
    def test_session_key_matches_real_flow_for_public_channel(self):
        """The key built by _has_active_session_for_thread must be byte-identical
        to the key build_source produces for an 'O' (public) channel, i.e. with
        chat_type='channel'.  A hardcoded 'group' key would NOT match (the
        original bug) — this test proves the map is used.
        """
        from gateway.session import SessionSource, build_session_key
        from gateway.config import Platform

        channel_id = "public_chan_xyz"
        thread_id = "thread_999"
        channel_type_raw = "O"  # public channel

        # Build the key that the real message flow produces.
        real_source = SessionSource(
            platform=Platform.MATTERMOST,
            chat_id=channel_id,
            chat_type="channel",  # _CHANNEL_TYPE_MAP["O"] == "channel"
            user_id=None,
            thread_id=thread_id,
        )
        real_key = build_session_key(
            real_source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
        )

        # Build what a hardcoded "group" key would look like (the OLD bug).
        buggy_source = SessionSource(
            platform=Platform.MATTERMOST,
            chat_id=channel_id,
            chat_type="group",  # ← incorrect for "O" channel
            user_id=None,
            thread_id=thread_id,
        )
        buggy_key = build_session_key(
            buggy_source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
        )

        # Seed a fake session store with the REAL key.
        fake_store = MagicMock()
        fake_store.config = MagicMock()
        fake_store.config.group_sessions_per_user = True
        fake_store.config.thread_sessions_per_user = False
        fake_store._ensure_loaded = MagicMock()
        fake_store._entries = {real_key: object()}

        self.adapter._session_store = fake_store

        # The correct lookup (using _CHANNEL_TYPE_MAP) must succeed.
        assert self.adapter._has_active_session_for_thread(
            channel_id=channel_id,
            thread_id=thread_id,
            channel_type_raw=channel_type_raw,
        ) is True

        # Prove the buggy hardcoded key would NOT have matched — the test that
        # would have caught the original silent no-op.
        assert real_key != buggy_key, (
            "Sanity: real key and buggy key must differ for this test to be meaningful"
        )
        assert buggy_key not in fake_store._entries

    # 6. thread_sessions_per_user=True: restart lookup key must include sender_id.
    def test_session_key_includes_sender_id_when_thread_sessions_per_user(self):
        """When thread_sessions_per_user=True, build_session_key appends the
        participant id to the key. The restart lookup MUST thread the real
        sender_id from the inbound post through to _has_active_session_for_thread,
        or the recomputed key always omits it and misses every per-user thread
        session created under this mode (the original bug: user_id=None)."""
        from gateway.session import SessionSource, build_session_key
        from gateway.config import Platform

        channel_id = "public_chan_tspu"
        thread_id = "thread_tspu_1"
        channel_type_raw = "O"
        sender_id = "user_123"

        # Key produced at session CREATION time (real flow, sender_id present).
        creation_source = SessionSource(
            platform=Platform.MATTERMOST,
            chat_id=channel_id,
            chat_type="channel",
            user_id=sender_id,
            thread_id=thread_id,
        )
        creation_key = build_session_key(
            creation_source,
            group_sessions_per_user=True,
            thread_sessions_per_user=True,
        )

        fake_store = MagicMock()
        fake_store.config = MagicMock()
        fake_store.config.group_sessions_per_user = True
        fake_store.config.thread_sessions_per_user = True
        fake_store._ensure_loaded = MagicMock()
        fake_store._entries = {creation_key: object()}
        fake_store._resolve_profile_for_key = MagicMock(return_value=None)

        self.adapter._session_store = fake_store

        # Lookup WITH sender_id threaded through must find the session.
        assert self.adapter._has_active_session_for_thread(
            channel_id=channel_id,
            thread_id=thread_id,
            channel_type_raw=channel_type_raw,
            sender_id=sender_id,
        ) is True

        # Sanity: the key without sender_id (the old sender_id=None bug) must
        # NOT match, proving sender_id is load-bearing for this mode.
        no_sender_source = SessionSource(
            platform=Platform.MATTERMOST,
            chat_id=channel_id,
            chat_type="channel",
            user_id=None,
            thread_id=thread_id,
        )
        no_sender_key = build_session_key(
            no_sender_source,
            group_sessions_per_user=True,
            thread_sessions_per_user=True,
        )
        assert no_sender_key != creation_key
        assert no_sender_key not in fake_store._entries

    # 7. Secondary-profile routing: restart lookup key must include the
    #    profile namespace used at session creation.
    def test_session_key_includes_profile_namespace_for_secondary_adapter(self):
        """A secondary-profile adapter creates sessions under 'agent:<profile>'
        (see SessionStore._resolve_profile_for_key / build_session_key's
        ``profile`` kwarg). The restart lookup must recompute the SAME
        namespaced key, or it silently searches 'agent:main' and always misses
        secondary-profile sessions (the original bug: profile was never
        threaded into the recomputed key)."""
        from gateway.session import SessionSource, build_session_key
        from gateway.config import Platform

        channel_id = "public_chan_profile"
        thread_id = "thread_profile_1"
        channel_type_raw = "O"
        profile_name = "coder"

        creation_source = SessionSource(
            platform=Platform.MATTERMOST,
            chat_id=channel_id,
            chat_type="channel",
            user_id=None,
            thread_id=thread_id,
        )
        creation_key = build_session_key(
            creation_source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
            profile=profile_name,
        )

        fake_store = MagicMock()
        fake_store.config = MagicMock()
        fake_store.config.group_sessions_per_user = True
        fake_store.config.thread_sessions_per_user = False
        fake_store._ensure_loaded = MagicMock()
        fake_store._entries = {creation_key: object()}
        fake_store._resolve_profile_for_key = MagicMock(return_value=profile_name)

        self.adapter._session_store = fake_store

        assert self.adapter._has_active_session_for_thread(
            channel_id=channel_id,
            thread_id=thread_id,
            channel_type_raw=channel_type_raw,
        ) is True

        # Sanity: the unnamespaced ("agent:main") key must differ and must not
        # be present — proving the profile namespace is load-bearing.
        unnamespaced_key = build_session_key(
            creation_source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
        )
        assert unnamespaced_key != creation_key
        assert unnamespaced_key not in fake_store._entries


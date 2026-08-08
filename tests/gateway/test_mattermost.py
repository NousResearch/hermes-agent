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
# First-turn thread-context seeding (#37695)
# ---------------------------------------------------------------------------


def _thread_post(pid, uid, msg, create_at, root_id="root_1", username=None):
    p = {
        "id": pid,
        "user_id": uid,
        "message": msg,
        "create_at": create_at,
        "root_id": root_id,
    }
    if username is not None:
        p["username"] = username
    return p


@pytest.mark.asyncio
async def test_fetch_thread_context_formats_prior_posts_with_attribution():
    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    adapter._api_get = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", "first message", 100, "", username="alice"),
                "p2": _thread_post("p2", "u2", "second message", 200, "root_1", username="bob"),
                "p3": _thread_post("p3", "u1", "the trigger", 300, "root_1", username="alice"),
            }
        }
    )

    out = await adapter._fetch_thread_context(
        root_id="root_1",
        current_post_id="p3",
        channel_id="chan",
        chat_type="channel",
    )

    # Trigger post (p3) is excluded; prior posts are seeded in chronological order.
    assert "[alice]: first message" in out
    assert "[bob]: second message" in out
    assert "the trigger" not in out
    assert out.startswith("[Thread context")
    assert out.rstrip().endswith("[End of thread context]")


@pytest.mark.asyncio
async def test_fetch_thread_context_excludes_own_bot_and_commands():
    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    adapter._api_get = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", "human question", 100, "", username="alice"),
                "bp": _thread_post("bp", "bot_user_id", "bot reply", 150, "root_1", username="hermes"),
                "cp": _thread_post("cp", "u1", "/new", 175, "root_1", username="alice"),
                "p2": _thread_post("p2", "u1", "trigger", 200, "root_1", username="alice"),
            }
        }
    )

    out = await adapter._fetch_thread_context(
        root_id="root_1",
        current_post_id="p2",
        channel_id="chan",
        chat_type="channel",
    )

    assert "human question" in out
    assert "bot reply" not in out      # own prior replies excluded (circular)
    assert "/new" not in out           # slash-commands excluded


@pytest.mark.asyncio
async def test_fetch_thread_context_tags_unverified_senders():
    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    # Only u1 is authorized; u2 is not.
    adapter.set_authorization_check(lambda uid, ct, cid: uid == "u1")
    adapter._api_get = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", "verified msg", 100, "", username="alice"),
                "p2": _thread_post("p2", "u2", "stranger msg", 200, "root_1", username="mallory"),
                "p3": _thread_post("p3", "u1", "trigger", 300, "root_1", username="alice"),
            }
        }
    )

    out = await adapter._fetch_thread_context(
        root_id="root_1",
        current_post_id="p3",
        channel_id="chan",
        chat_type="channel",
    )

    assert "[alice]: verified msg" in out
    assert "[unverified] [mallory]: stranger msg" in out
    # Header switches to the security-aware variant when any post is unverified.
    assert "[unverified]" in out.split("\n")[0]


@pytest.mark.asyncio
async def test_fetch_thread_context_empty_root_returns_blank():
    adapter = _make_adapter()
    adapter._api_get = AsyncMock(return_value={})
    out = await adapter._fetch_thread_context(
        root_id="", current_post_id="p1", channel_id="c", chat_type="channel"
    )
    assert out == ""


@pytest.mark.asyncio
async def test_fetch_thread_context_caches_by_root():
    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    api = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", "hello", 100, "", username="alice"),
                "p2": _thread_post("p2", "u1", "trigger", 200, "root_1", username="alice"),
            }
        }
    )
    adapter._api_get = api

    a = await adapter._fetch_thread_context("root_1", "p2", "chan", "channel")
    b = await adapter._fetch_thread_context("root_1", "p2", "chan", "channel")
    assert a == b
    # Only one thread fetch despite two calls (TTL cache).
    assert api.await_count == 1


@pytest.mark.asyncio
async def test_handle_ws_event_seeds_channel_context_on_first_thread_turn():
    adapter = _make_adapter()
    adapter._reply_mode = "thread"
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter.handle_message = AsyncMock()
    # No session store => first turn.
    adapter._api_get = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", "earlier context", 100, "", username="alice"),
                "reply_1": _thread_post("reply_1", "u1", "@hermes-bot help", 200, "root_1", username="alice"),
            }
        }
    )
    post_data = {
        "id": "reply_1",
        "user_id": "u1",
        "channel_id": "chan_456",
        "message": "@hermes-bot help",
        "root_id": "root_1",
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
    assert msg_event.channel_context is not None
    assert "earlier context" in msg_event.channel_context


@pytest.mark.asyncio
async def test_handle_ws_event_no_seed_for_root_post():
    adapter = _make_adapter()
    adapter._reply_mode = "thread"
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter.handle_message = AsyncMock()
    adapter._api_get = AsyncMock()
    post_data = {
        "id": "top_1",
        "user_id": "u1",
        "channel_id": "chan_456",
        "message": "@hermes-bot new topic",
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
    # Root posts have no prior thread history to seed.
    assert msg_event.channel_context is None
    adapter._api_get.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_ws_event_thread_context_off_disables_seeding():
    adapter = _make_adapter()
    adapter._reply_mode = "thread"
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter._thread_context_mode = "off"
    adapter.handle_message = AsyncMock()
    adapter._api_get = AsyncMock()
    post_data = {
        "id": "reply_1",
        "user_id": "u1",
        "channel_id": "chan_456",
        "message": "@hermes-bot help",
        "root_id": "root_1",
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
    assert msg_event.channel_context is None
    adapter._api_get.assert_not_awaited()


def test_apply_yaml_config_bridges_thread_context(monkeypatch):
    from plugins.platforms.mattermost.adapter import _apply_yaml_config

    monkeypatch.delenv("MATTERMOST_THREAD_CONTEXT", raising=False)
    _apply_yaml_config({"thread_context": "off"}, {"thread_context": "off"})
    assert os.environ["MATTERMOST_THREAD_CONTEXT"] == "off"


# ---------------------------------------------------------------------------
# thread_context YAML-boolean normalization (#37695 review follow-up)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (False, "off"),   # YAML 1.1 parses bare `off`/`no`/`false` as bool False
        ("off", "off"),
        ("false", "off"),
        ("no", "off"),
        (0, "off"),
        ("0", "off"),
        (True, "on"),
        ("on", "on"),
        ("true", "on"),
        ("", "on"),       # empty falls through to default
        (None, "on"),
    ],
)
def test_normalize_onoff(raw, expected):
    from plugins.platforms.mattermost.adapter import _normalize_onoff
    assert _normalize_onoff(raw, default="on") == expected


@pytest.mark.parametrize("raw", [False, "off", "false", "no", 0, "0"])
def test_apply_yaml_config_yaml_boolean_off_disables(monkeypatch, raw):
    from plugins.platforms.mattermost.adapter import _apply_yaml_config

    monkeypatch.delenv("MATTERMOST_THREAD_CONTEXT", raising=False)
    _apply_yaml_config({"thread_context": raw}, {"thread_context": raw})
    assert os.environ["MATTERMOST_THREAD_CONTEXT"] == "off"


def test_adapter_thread_context_off_via_yaml_boolean():
    """config.extra carrying YAML boolean False must disable seeding."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com", "thread_context": False},
    )
    adapter = MattermostAdapter(config)
    assert adapter._thread_context_mode == "off"


# ---------------------------------------------------------------------------
# Bounded retrieval + prompt-input caps (#37695 review follow-up)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_thread_context_bounds_the_request():
    """The thread fetch must bound retrieval server-side, not just slice in memory."""
    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    api = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", "hello", 100, "", username="alice"),
                "p2": _thread_post("p2", "u1", "trigger", 200, "root_1", username="alice"),
            }
        }
    )
    adapter._api_get = api

    await adapter._fetch_thread_context(
        root_id="root_1",
        current_post_id="p2",
        channel_id="chan",
        chat_type="channel",
        limit=30,
    )

    called_path = api.await_args[0][0]
    assert called_path.startswith("posts/root_1/thread")
    # Retrieval is bounded at the request: perPage caps how many posts the
    # server materializes, direction=up returns the most recent tail.
    assert "perPage=31" in called_path
    assert "direction=up" in called_path


@pytest.mark.asyncio
async def test_fetch_thread_context_caps_per_post_length():
    """A single over-long post is truncated so it cannot dominate the block."""
    from plugins.platforms.mattermost.adapter import MAX_THREAD_CONTEXT_POST_CHARS

    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    long_text = "x" * (MAX_THREAD_CONTEXT_POST_CHARS + 500)
    adapter._api_get = AsyncMock(
        return_value={
            "posts": {
                "root_1": _thread_post("root_1", "u1", long_text, 100, "", username="alice"),
                "p2": _thread_post("p2", "u1", "trigger", 200, "root_1", username="alice"),
            }
        }
    )

    out = await adapter._fetch_thread_context(
        root_id="root_1",
        current_post_id="p2",
        channel_id="chan",
        chat_type="channel",
    )

    # The rendered post body is truncated to the per-post cap plus an ellipsis.
    assert "x" * MAX_THREAD_CONTEXT_POST_CHARS in out
    assert "x" * (MAX_THREAD_CONTEXT_POST_CHARS + 1) not in out
    assert "[…]" in out


@pytest.mark.asyncio
async def test_fetch_thread_context_caps_total_length_keeping_newest():
    """Total rendered context is bounded; oldest posts drop first."""
    from plugins.platforms.mattermost.adapter import (
        MAX_THREAD_CONTEXT_POST_CHARS,
        MAX_THREAD_CONTEXT_TOTAL_CHARS,
    )

    adapter = _make_adapter()
    adapter._bot_user_id = "bot_user_id"
    # Enough near-max posts that the total budget forces trimming.
    body = "a" * MAX_THREAD_CONTEXT_POST_CHARS
    posts = {"root_1": _thread_post("root_1", "u1", "OLDEST", 50, "", username="alice")}
    n = (MAX_THREAD_CONTEXT_TOTAL_CHARS // MAX_THREAD_CONTEXT_POST_CHARS) + 3
    for i in range(n):
        pid = f"p{i}"
        posts[pid] = _thread_post(pid, "u1", body, 100 + i, "root_1", username="alice")
    posts["newest_msg"] = _thread_post(
        "newest_msg", "u1", "NEWEST-KEPT", 100 + n, "root_1", username="alice"
    )
    posts["trig"] = _thread_post("trig", "u1", "trigger", 100 + n + 1, "root_1", username="alice")
    adapter._api_get = AsyncMock(return_value={"posts": posts})

    out = await adapter._fetch_thread_context(
        root_id="root_1",
        current_post_id="trig",
        channel_id="chan",
        chat_type="channel",
        limit=n + 5,
    )

    # Total stays within budget (plus header/footer framing lines).
    assert len(out) <= MAX_THREAD_CONTEXT_TOTAL_CHARS + 500
    # Newest content is retained; oldest is trimmed to fit.
    assert "NEWEST-KEPT" in out
    assert "OLDEST" not in out


def test_apply_yaml_config_does_not_bridge_strict_mention(monkeypatch):
    """The unrelated MATTERMOST_STRICT_MENTION dead bridge must be gone."""
    from plugins.platforms.mattermost.adapter import _apply_yaml_config

    monkeypatch.delenv("MATTERMOST_STRICT_MENTION", raising=False)
    _apply_yaml_config({"strict_mention": True}, {"strict_mention": True})
    assert "MATTERMOST_STRICT_MENTION" not in os.environ


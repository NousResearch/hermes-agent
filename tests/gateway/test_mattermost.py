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

    def test_load_gateway_config_bridges_auto_thread_settings(
        self,
        tmp_path,
        monkeypatch,
    ):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "mattermost:\n"
            "  auto_thread: true\n"
            "  dm_auto_thread: false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("MATTERMOST_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_DM_AUTO_THREAD", raising=False)

        # Importing the bundled plugin registers its YAML bridge.
        import plugins.platforms.mattermost.adapter  # noqa: F401
        from gateway.config import load_gateway_config

        load_gateway_config()

        assert os.getenv("MATTERMOST_AUTO_THREAD") == "true"
        assert os.getenv("MATTERMOST_DM_AUTO_THREAD") == "false"


# ---------------------------------------------------------------------------
# Adapter format / truncate
# ---------------------------------------------------------------------------

def _make_adapter(extra=None):
    """Create a MattermostAdapter with mocked config."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    adapter_extra = {"url": "https://mm.example.com"}
    adapter_extra.update(extra or {})
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra=adapter_extra,
    )
    adapter = MattermostAdapter(config)
    return adapter


class TestMattermostThreadConfig:
    def test_threading_defaults_to_flat(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_DM_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_REPLY_MODE", raising=False)

        adapter = _make_adapter()

        assert adapter._auto_thread is False
        assert adapter._dm_auto_thread is False

    def test_legacy_thread_mode_enables_both_policies(self, monkeypatch, caplog):
        monkeypatch.delenv("MATTERMOST_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_DM_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_REPLY_MODE", raising=False)

        with caplog.at_level("WARNING"):
            adapter = _make_adapter({"reply_mode": "thread"})

        assert adapter._auto_thread is True
        assert adapter._dm_auto_thread is True
        assert "deprecated" in caplog.text

    def test_new_config_overrides_legacy_thread_mode(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_DM_AUTO_THREAD", raising=False)

        adapter = _make_adapter(
            {
                "reply_mode": "thread",
                "auto_thread": False,
                "dm_auto_thread": False,
            }
        )

        assert adapter._auto_thread is False
        assert adapter._dm_auto_thread is False

    def test_new_env_overrides_config_and_legacy(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_AUTO_THREAD", "false")
        monkeypatch.setenv("MATTERMOST_DM_AUTO_THREAD", "true")

        adapter = _make_adapter(
            {
                "reply_mode": "thread",
                "auto_thread": True,
                "dm_auto_thread": False,
            }
        )

        assert adapter._auto_thread is False
        assert adapter._dm_auto_thread is True

    def test_yaml_bridge_sets_both_auto_thread_values(self, monkeypatch):
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        monkeypatch.delenv("MATTERMOST_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_DM_AUTO_THREAD", raising=False)

        _apply_yaml_config(
            {},
            {"auto_thread": True, "dm_auto_thread": False},
        )

        assert os.getenv("MATTERMOST_AUTO_THREAD") == "true"
        assert os.getenv("MATTERMOST_DM_AUTO_THREAD") == "false"

    def test_yaml_bridge_does_not_override_environment(self, monkeypatch):
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        monkeypatch.setenv("MATTERMOST_AUTO_THREAD", "false")
        monkeypatch.setenv("MATTERMOST_DM_AUTO_THREAD", "true")

        _apply_yaml_config(
            {},
            {"auto_thread": True, "dm_auto_thread": False},
        )

        assert os.getenv("MATTERMOST_AUTO_THREAD") == "false"
        assert os.getenv("MATTERMOST_DM_AUTO_THREAD") == "true"


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
        """When auto_thread is enabled, reply_to should become root_id."""
        self.adapter._auto_thread = True
        self.adapter._remember_channel_type("channel_1", "channel")

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
    async def test_send_with_thread_reply_resolves_to_root(self):
        """A reply post must be resolved to the original thread root."""
        self.adapter._api_get = AsyncMock(
            return_value={"id": "user_reply", "root_id": "thread_root"}
        )
        self.adapter._api_post = AsyncMock(return_value={"id": "post789"})

        result = await self.adapter.send(
            "channel_1",
            "Reply!",
            reply_to="user_reply",
        )

        assert result.success is True
        payload = self.adapter._api_post.await_args.args[1]
        assert payload["root_id"] == "thread_root"

    @pytest.mark.asyncio
    async def test_resolve_root_id_caches_successful_lookups(self):
        self.adapter._api_get = AsyncMock(
            return_value={"id": "user_reply", "root_id": "thread_root"}
        )

        first = await self.adapter._resolve_root_id("user_reply")
        second = await self.adapter._resolve_root_id("user_reply")

        assert first == second == "thread_root"
        self.adapter._api_get.assert_awaited_once_with("posts/user_reply")

    @pytest.mark.asyncio
    async def test_resolve_root_id_does_not_cache_failed_lookup(self):
        self.adapter._api_get = AsyncMock(
            side_effect=[
                {},
                {"id": "user_reply", "root_id": "thread_root"},
            ]
        )

        first = await self.adapter._resolve_root_id("user_reply")
        second = await self.adapter._resolve_root_id("user_reply")

        assert first is None
        assert second == "thread_root"
        assert self.adapter._api_get.await_count == 2

    @pytest.mark.asyncio
    async def test_thread_metadata_root_is_resolved_once_then_cached(self):
        """A metadata root still costs one lookup, but only one per thread."""
        self.adapter._api_get = AsyncMock(
            return_value={"id": "thread_root", "root_id": ""}
        )
        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})

        for _ in range(2):
            result = await self.adapter.send(
                "channel_1", "Reply!", "thread_root", {"thread_id": "thread_root"},
            )
            assert result.success
            assert self.adapter._api_post.await_args.args[1]["root_id"] == "thread_root"
        self.adapter._api_get.assert_awaited_once_with("posts/thread_root")

    @pytest.mark.asyncio
    async def test_thread_metadata_reply_id_resolves_to_thread_root(self):
        """A recorded thread_id pointing at a reply must not become root_id."""
        self.adapter._api_get = AsyncMock(
            return_value={"id": "user_reply", "root_id": "thread_root"}
        )
        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})

        result = await self.adapter.send(
            "channel_1", "Reply!", metadata={"thread_id": "user_reply"},
        )

        assert result.success
        assert self.adapter._api_post.await_args.args[1]["root_id"] == "thread_root"

    @pytest.mark.asyncio
    async def test_send_prefers_and_resolves_metadata_root_over_reply_to(self):
        """send() posts to the metadata root even when reply_to is stale."""
        self.adapter._api_get = AsyncMock(
            return_value={"id": "metadata_reply", "root_id": "thread_root"}
        )
        self.adapter._api_post = AsyncMock(return_value={"id": "post789"})

        result = await self.adapter.send(
            "channel_1",
            "Reply!",
            reply_to="stale_reply",
            metadata={"thread_id": "metadata_reply"},
        )

        assert result.success is True
        payload = self.adapter._api_post.await_args.args[1]
        assert payload["root_id"] == "thread_root"
        self.adapter._api_get.assert_awaited_once_with("posts/metadata_reply")

    @pytest.mark.asyncio
    async def test_thread_metadata_survives_transient_lookup_failure(self):
        """Keep the thread on a failed lookup rather than flattening it."""
        self.adapter._api_get = AsyncMock(return_value={})
        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})

        result = await self.adapter.send(
            "channel_1", "Reply!", metadata={"thread_id": "thread_root"},
        )

        assert result.success
        assert self.adapter._api_post.await_args.args[1]["root_id"] == "thread_root"

    @pytest.mark.asyncio
    async def test_thread_metadata_falls_back_when_reply_lookup_fails(self):
        self.adapter._api_get = AsyncMock(return_value={})

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "channel_1", "Reply!",
            "user_reply",
            {"thread_id": "thread_root"},
        )

        assert self.adapter._api_post.await_args.args[1]["root_id"] == "thread_root"

    @pytest.mark.asyncio
    async def test_send_typing_uses_thread_parent_id(self):
        self.adapter._api_post = AsyncMock(return_value={})

        await self.adapter.send_typing(
            "channel_1",
            metadata={"thread_id": "thread_root"},
        )

        self.adapter._api_post.assert_awaited_once_with(
            "users//typing",
            {"channel_id": "channel_1", "parent_id": "thread_root"},
        )

    @pytest.mark.asyncio
    async def test_send_typing_preserves_explicit_thread_when_auto_thread_off(self):
        self.adapter._auto_thread = False
        self.adapter._dm_auto_thread = False
        self.adapter._api_post = AsyncMock(return_value={})

        await self.adapter.send_typing(
            "channel_1",
            metadata={"thread_id": "thread_root"},
        )

        self.adapter._api_post.assert_awaited_once_with(
            "users//typing",
            {"channel_id": "channel_1", "parent_id": "thread_root"},
        )


    @pytest.mark.asyncio
    async def test_progress_send_with_invalid_thread_root_never_falls_back_flat(self):
        """Tool/status/progress bubbles must stay quiet when the thread is broken."""
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
        self.adapter._auto_thread = True
        self.adapter._remember_channel_type("channel_1", "channel")
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


class TestMattermostAutoThreadRouting:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_uncached_channel_uses_channel_auto_thread_policy(self):
        self.adapter._auto_thread = True
        self.adapter._dm_auto_thread = False
        self.adapter._api_get = AsyncMock(
            side_effect=[
                {"id": "top_post", "root_id": ""},
                {"id": "channel_1", "type": "O"},
            ]
        )

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "channel_1", "Reply!",
            "top_post",
            None,
        )

        assert self.adapter._api_post.await_args.args[1].get("root_id") == "top_post"
        assert self.adapter._channel_type_cache["channel_1"] == "channel"
        assert [entry.args for entry in self.adapter._api_get.await_args_list] == [
            ("posts/top_post",),
            ("channels/channel_1",),
        ]

    @pytest.mark.asyncio
    async def test_uncached_channel_can_disable_auto_thread(self):
        self.adapter._auto_thread = False
        self.adapter._dm_auto_thread = True
        self.adapter._api_get = AsyncMock(
            side_effect=[
                {"id": "top_post", "root_id": ""},
                {"id": "channel_1", "type": "O"},
            ]
        )

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "channel_1", "Reply!",
            "top_post",
            None,
        )

        assert "root_id" not in self.adapter._api_post.await_args.args[1]

    @pytest.mark.asyncio
    async def test_uncached_dm_uses_dm_auto_thread_policy(self):
        self.adapter._auto_thread = False
        self.adapter._dm_auto_thread = True
        self.adapter._api_get = AsyncMock(
            side_effect=[
                {"id": "dm_post", "root_id": ""},
                {"id": "dm_channel", "type": "D"},
            ]
        )

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "dm_channel", "Reply!",
            "dm_post",
            None,
        )

        assert self.adapter._api_post.await_args.args[1].get("root_id") == "dm_post"
        assert self.adapter._channel_type_cache["dm_channel"] == "dm"

    @pytest.mark.asyncio
    async def test_uncached_dm_can_disable_auto_thread(self):
        self.adapter._auto_thread = True
        self.adapter._dm_auto_thread = False
        self.adapter._api_get = AsyncMock(
            side_effect=[
                {"id": "dm_post", "root_id": ""},
                {"id": "dm_channel", "type": "D"},
            ]
        )

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "dm_channel", "Reply!",
            "dm_post",
            None,
        )

        assert "root_id" not in self.adapter._api_post.await_args.args[1]

    @pytest.mark.asyncio
    async def test_explicit_dm_thread_metadata_is_always_preserved(self):
        self.adapter._auto_thread = False
        self.adapter._dm_auto_thread = False
        # dm_root is already a thread root, so resolution returns it unchanged.
        self.adapter._api_get = AsyncMock(return_value={"id": "dm_root", "root_id": ""})

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "dm_channel", "Reply!",
            "dm_reply",
            {"thread_id": "dm_root", "chat_type": "dm"},
        )

        assert self.adapter._api_post.await_args.args[1]["root_id"] == "dm_root"
        # Only the metadata root is looked up — reply_to is never consulted.
        self.adapter._api_get.assert_awaited_once_with("posts/dm_root")

    @pytest.mark.asyncio
    async def test_existing_reply_root_is_preserved_without_metadata(self):
        self.adapter._auto_thread = False
        self.adapter._dm_auto_thread = False
        self.adapter._api_get = AsyncMock(
            return_value={"id": "dm_reply", "root_id": "dm_root"}
        )

        self.adapter._api_post = AsyncMock(return_value={"id": "sent"})
        await self.adapter.send(
            "dm_channel", "Reply!",
            "dm_reply",
            None,
        )

        assert self.adapter._api_post.await_args.args[1].get("root_id") == "dm_root"
        self.adapter._api_get.assert_awaited_once_with("posts/dm_reply")


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

    @pytest.mark.parametrize(
        (
            "channel_type",
            "root_id",
            "auto_thread",
            "dm_auto_thread",
            "expected_thread_id",
        ),
        [
            ("O", "", True, False, "post_policy"),
            ("O", "", False, True, None),
            ("D", "", False, True, "post_policy"),
            ("D", "", True, False, None),
            ("D", "existing_dm_root", False, False, "existing_dm_root"),
        ],
    )
    @pytest.mark.asyncio
    async def test_inbound_thread_policy(
        self,
        channel_type,
        root_id,
        auto_thread,
        dm_auto_thread,
        expected_thread_id,
    ):
        self.adapter._auto_thread = auto_thread
        self.adapter._dm_auto_thread = dm_auto_thread
        message = "DM message" if channel_type == "D" else "@hermes-bot channel message"
        post_data = {
            "id": "post_policy",
            "user_id": "user_123",
            "channel_id": "chan_policy",
            "message": message,
            "root_id": root_id,
        }
        event = {
            "event": "posted",
            "data": {
                "post": json.dumps(post_data),
                "channel_type": channel_type,
                "sender_name": "@alice",
            },
        }

        await self.adapter._handle_ws_event(event)

        msg_event = self.adapter.handle_message.call_args[0][0]
        assert msg_event.source.thread_id == expected_thread_id


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
    async def test_missing_local_file_remains_a_silent_noop(self, tmp_path):
        self.adapter._api_post = AsyncMock()

        result = await self.adapter._send_local_file(
            "channel_1",
            str(tmp_path / "missing-secret.txt"),
            None,
            None,
        )

        assert result.success is True
        assert result.message_id is None
        self.adapter._api_post.assert_not_awaited()

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
    adapter._auto_thread = True
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

    def test_scoped_thread_yaml_load_seeds_extra_without_env_leak(
        self, tmp_path, monkeypatch, multiplex_scope
    ):
        from gateway.config import load_gateway_config
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MATTERMOST_AUTO_THREAD", raising=False)
        monkeypatch.delenv("MATTERMOST_DM_AUTO_THREAD", raising=False)
        (tmp_path / "config.yaml").write_text(
            "mattermost:\n  auto_thread: true\n  dm_auto_thread: false\n",
            encoding="utf-8",
        )
        multiplex_scope()

        config = load_gateway_config().platforms[Platform.MATTERMOST]
        assert config.extra["auto_thread"] is True
        assert config.extra["dm_auto_thread"] is False
        adapter = MattermostAdapter(config)
        assert (adapter._auto_thread, adapter._dm_auto_thread) == (True, False)
        assert "MATTERMOST_AUTO_THREAD" not in os.environ
        assert "MATTERMOST_DM_AUTO_THREAD" not in os.environ

    @pytest.mark.parametrize("scoped_env, extra, expected", [
        ({}, {}, (False, False)),
        ({}, {"auto_thread": True, "dm_auto_thread": False}, (True, False)),
        ({"MATTERMOST_AUTO_THREAD": "false", "MATTERMOST_DM_AUTO_THREAD": "true"},
         {"auto_thread": True, "dm_auto_thread": False}, (False, True)),
        ({"MATTERMOST_REPLY_MODE": "thread"}, {}, (True, True)),
    ])
    def test_thread_policy_uses_only_own_profile(
        self, monkeypatch, multiplex_scope, scoped_env, extra, expected
    ):
        monkeypatch.setenv("MATTERMOST_AUTO_THREAD", "true")
        monkeypatch.setenv("MATTERMOST_DM_AUTO_THREAD", "false")
        monkeypatch.setenv("MATTERMOST_REPLY_MODE", "thread")
        multiplex_scope(scoped_env)

        adapter = _make_adapter(extra)

        assert (adapter._auto_thread, adapter._dm_auto_thread) == expected

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


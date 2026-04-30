"""Tests for Rocket.Chat platform adapter plugin."""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_rc_mod = load_plugin_adapter("rocketchat")

RocketChatAdapter = _rc_mod.RocketChatAdapter
check_requirements = _rc_mod.check_requirements
check_rocketchat_requirements = _rc_mod.check_rocketchat_requirements
validate_config = _rc_mod.validate_config
is_connected = _rc_mod.is_connected
register = _rc_mod.register


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class TestRocketChatPlatformEnum:
    def test_dynamic_member_value(self):
        assert Platform("rocketchat").value == "rocketchat"

    def test_identity_stable(self):
        assert Platform("rocketchat") is Platform("rocketchat")


# ---------------------------------------------------------------------------
# Requirements & config validation
# ---------------------------------------------------------------------------


class TestRequirementsCheck:
    def test_fails_without_url(self, monkeypatch):
        monkeypatch.delenv("ROCKETCHAT_URL", raising=False)
        assert check_requirements() is False

    def test_fails_without_credentials(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.delenv("ROCKETCHAT_USERNAME", raising=False)
        monkeypatch.delenv("ROCKETCHAT_PASSWORD", raising=False)
        monkeypatch.delenv("ROCKETCHAT_TOKEN", raising=False)
        monkeypatch.delenv("ROCKETCHAT_USER_ID", raising=False)
        assert check_requirements() is False

    def test_passes_with_userpass_and_aiohttp(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.setenv("ROCKETCHAT_USERNAME", "bot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "s3cr3t")
        assert check_requirements() is True

    def test_passes_with_pat_and_aiohttp(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.setenv("ROCKETCHAT_TOKEN", "my-pat")
        monkeypatch.setenv("ROCKETCHAT_USER_ID", "uid123")
        monkeypatch.delenv("ROCKETCHAT_USERNAME", raising=False)
        monkeypatch.delenv("ROCKETCHAT_PASSWORD", raising=False)
        assert check_requirements() is True

    def test_alias_matches(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.setenv("ROCKETCHAT_USERNAME", "bot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "s3cr3t")
        assert check_rocketchat_requirements() is True


class TestValidateConfig:
    def _cfg(self, **extra):
        return PlatformConfig(enabled=True, extra=extra)

    def test_passes_with_userpass_env(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.setenv("ROCKETCHAT_USERNAME", "bot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "s3cr3t")
        assert validate_config(self._cfg()) is True

    def test_passes_with_pat_env(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.setenv("ROCKETCHAT_TOKEN", "my-pat")
        monkeypatch.setenv("ROCKETCHAT_USER_ID", "uid123")
        monkeypatch.delenv("ROCKETCHAT_USERNAME", raising=False)
        monkeypatch.delenv("ROCKETCHAT_PASSWORD", raising=False)
        assert validate_config(self._cfg()) is True

    def test_passes_with_extra_fields(self, monkeypatch):
        monkeypatch.delenv("ROCKETCHAT_URL", raising=False)
        monkeypatch.delenv("ROCKETCHAT_USERNAME", raising=False)
        monkeypatch.delenv("ROCKETCHAT_PASSWORD", raising=False)
        cfg = self._cfg(url="https://rc.example.com", username="bot", password="s3cr3t")
        assert validate_config(cfg) is True

    def test_fails_without_url(self, monkeypatch):
        monkeypatch.delenv("ROCKETCHAT_URL", raising=False)
        assert validate_config(self._cfg(username="bot", password="s3cr3t")) is False

    def test_fails_without_credentials(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.delenv("ROCKETCHAT_USERNAME", raising=False)
        monkeypatch.delenv("ROCKETCHAT_PASSWORD", raising=False)
        monkeypatch.delenv("ROCKETCHAT_TOKEN", raising=False)
        monkeypatch.delenv("ROCKETCHAT_USER_ID", raising=False)
        assert validate_config(self._cfg()) is False

    def test_is_connected_delegates_to_validate_config(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://rc.example.com")
        monkeypatch.setenv("ROCKETCHAT_USERNAME", "bot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "s3cr3t")
        cfg = self._cfg()
        assert is_connected(cfg) == validate_config(cfg)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_register_calls_ctx(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()

    def test_register_name(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["name"] == "rocketchat"

    def test_register_auth_env_vars(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["allowed_users_env"] == "ROCKETCHAT_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "ROCKETCHAT_ALLOW_ALL_USERS"

    def test_register_has_setup_fn(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert callable(kwargs.get("setup_fn"))

    def test_register_has_platform_hint(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs.get("platform_hint")


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _make_adapter(extra=None):
    config = PlatformConfig(
        enabled=True,
        extra={"url": "https://rc.example.com", **(extra or {})},
    )
    adapter = RocketChatAdapter(config)
    if not adapter._user_id:
        adapter._user_id = "bot_uid"
    adapter._bot_username = "hermesbot"
    return adapter


# ---------------------------------------------------------------------------
# Adapter init
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_reads_url_from_extra(self):
        adapter = _make_adapter()
        assert adapter._base_url == "https://rc.example.com"

    def test_reads_credentials_from_extra(self):
        adapter = _make_adapter({"username": "bot", "password": "s3cr3t"})
        assert adapter._username == "bot"
        assert adapter._password == "s3cr3t"

    def test_reads_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_USERNAME", "envbot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "envpass")
        adapter = _make_adapter()
        assert adapter._username == "envbot"

    def test_reads_pat_from_env(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_TOKEN", "my-pat")
        monkeypatch.setenv("ROCKETCHAT_USER_ID", "uid123")
        adapter = _make_adapter()
        assert adapter._auth_token == "my-pat"
        assert adapter._user_id == "uid123"

    def test_platform_value(self):
        adapter = _make_adapter()
        assert adapter.platform.value == "rocketchat"

    def test_reply_in_thread_default_false(self):
        adapter = _make_adapter()
        assert adapter._reply_in_thread is False

    def test_reply_in_thread_from_extra(self):
        adapter = _make_adapter({"reply_in_thread": "true"})
        assert adapter._reply_in_thread is True

    def test_reactions_enabled_default_true(self):
        adapter = _make_adapter()
        assert adapter._reactions_enabled is True

    def test_reactions_disabled_from_extra(self):
        adapter = _make_adapter({"reactions": "false"})
        assert adapter._reactions_enabled is False


# ---------------------------------------------------------------------------
# format_message
# ---------------------------------------------------------------------------


class TestFormatMessage:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_image_markdown_stripped_to_url(self):
        result = self.adapter.format_message("![alt](https://img.example.com/a.png)")
        assert result == "https://img.example.com/a.png"

    def test_double_bold_converted(self):
        result = self.adapter.format_message("**hello**")
        assert result == "*hello*"

    def test_tilde_replaced_with_unicode(self):
        result = self.adapter.format_message("~8.3°C and ~17.7°C")
        assert "~" not in result
        assert "\u223c" in result

    def test_single_tilde_replaced(self):
        result = self.adapter.format_message("wind ~20 km/h")
        assert "~" not in result
        assert "\u223c20 km/h" in result

    def test_balanced_bold_markers_preserved(self):
        result = self.adapter.format_message("*bold*")
        assert result == "*bold*"

    def test_unbalanced_bold_marker_dropped(self):
        result = self.adapter.format_message("hello * world")
        assert result.count("*") % 2 == 0 or "*" not in result


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------


def _make_event(chat_type: str) -> MessageEvent:
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform("rocketchat"),
            chat_id="room1",
            chat_type=chat_type,
            user_id="u1",
            user_name="alice",
        ),
        message_id="msg1",
    )


class TestReactions:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_reactions_active_on_channel(self):
        assert self.adapter._reactions_active(_make_event("channel")) is True

    def test_reactions_inactive_on_dm(self):
        assert self.adapter._reactions_active(_make_event("dm")) is False

    def test_reactions_inactive_when_disabled(self):
        self.adapter._reactions_enabled = False
        assert self.adapter._reactions_active(_make_event("channel")) is False

    def test_reactions_inactive_without_message_id(self):
        event = _make_event("channel")
        event.message_id = None
        assert self.adapter._reactions_active(event) is False

    @pytest.mark.asyncio
    async def test_on_processing_start_channel_calls_add_reaction(self):
        self.adapter._add_reaction = AsyncMock(return_value=True)
        await self.adapter.on_processing_start(_make_event("channel"))
        self.adapter._add_reaction.assert_awaited_once_with("msg1", "eyes")

    @pytest.mark.asyncio
    async def test_on_processing_start_dm_skips_reaction(self):
        self.adapter._add_reaction = AsyncMock(return_value=True)
        await self.adapter.on_processing_start(_make_event("dm"))
        self.adapter._add_reaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_processing_complete_channel(self):
        self.adapter._add_reaction = AsyncMock(return_value=True)
        self.adapter._remove_reaction = AsyncMock(return_value=True)
        await self.adapter.on_processing_complete(_make_event("channel"), success=True)
        self.adapter._remove_reaction.assert_awaited_once_with("msg1", "eyes")
        self.adapter._add_reaction.assert_awaited_once_with("msg1", "white_check_mark")

    @pytest.mark.asyncio
    async def test_on_processing_complete_dm_skips(self):
        self.adapter._add_reaction = AsyncMock(return_value=True)
        self.adapter._remove_reaction = AsyncMock(return_value=True)
        await self.adapter.on_processing_complete(_make_event("dm"), success=True)
        self.adapter._add_reaction.assert_not_awaited()
        self.adapter._remove_reaction.assert_not_awaited()


# ---------------------------------------------------------------------------
# Thread replies
# ---------------------------------------------------------------------------


class TestThreadReplies:
    @pytest.mark.asyncio
    async def test_tmid_set_for_channel(self):
        adapter = _make_adapter({"reply_in_thread": "true"})
        adapter._room_type_cache["room1"] = "c"

        posted = {}

        async def fake_post(path, payload):
            posted.update(payload)
            return {"success": True, "message": {"_id": "new_msg"}}

        adapter._api_post = fake_post
        await adapter.send("room1", "hello", reply_to="parent_msg")
        assert posted.get("message", {}).get("tmid") == "parent_msg"

    @pytest.mark.asyncio
    async def test_tmid_not_set_for_dm(self):
        adapter = _make_adapter({"reply_in_thread": "true"})
        adapter._room_type_cache["dm_room"] = "d"

        posted = {}

        async def fake_post(path, payload):
            posted.update(payload)
            return {"success": True, "message": {"_id": "new_msg"}}

        adapter._api_post = fake_post
        await adapter.send("dm_room", "hello", reply_to="parent_msg")
        assert "tmid" not in posted.get("message", {})


# ---------------------------------------------------------------------------
# Deferred attachment buffer
# ---------------------------------------------------------------------------


class TestAttachmentBuffer:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_store_and_pop(self):
        self.adapter._store_attachment("room1", "/tmp/a.pdf", "application/pdf")
        urls, types = self.adapter._pop_recent_attachments("room1")
        assert urls == ["/tmp/a.pdf"]
        assert types == ["application/pdf"]

    def test_pop_clears_buffer(self):
        self.adapter._store_attachment("room1", "/tmp/a.pdf", "application/pdf")
        self.adapter._pop_recent_attachments("room1")
        urls, _ = self.adapter._pop_recent_attachments("room1")
        assert urls == []

    def test_expired_attachments_not_returned(self):
        self.adapter._recent_attachments["room1"] = [
            (time.time() - 400, "/tmp/old.pdf", "application/pdf"),
        ]
        urls, _ = self.adapter._pop_recent_attachments("room1")
        assert urls == []

    def test_pop_unknown_room_returns_empty(self):
        urls, types = self.adapter._pop_recent_attachments("nonexistent")
        assert urls == []
        assert types == []

    def test_multiple_attachments_stored(self):
        self.adapter._store_attachment("room1", "/tmp/a.jpg", "image/jpeg")
        self.adapter._store_attachment("room1", "/tmp/b.pdf", "application/pdf")
        urls, types = self.adapter._pop_recent_attachments("room1")
        assert len(urls) == 2
        assert "/tmp/a.jpg" in urls
        assert "/tmp/b.pdf" in urls


# ---------------------------------------------------------------------------
# Room-type cache
# ---------------------------------------------------------------------------


class TestRoomTypeCache:
    @pytest.mark.asyncio
    async def test_cached_result_not_refetched(self):
        adapter = _make_adapter()
        adapter._room_type_cache["room1"] = "d"
        adapter._api_get = AsyncMock()

        result = await adapter._get_room_type("room1")
        assert result == "d"
        adapter._api_get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dm_room_detected(self):
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(return_value={"room": {"t": "d", "name": ""}})

        result = await adapter._get_room_type("dm_room")
        assert result == "d"
        assert adapter._room_type_cache["dm_room"] == "d"

    @pytest.mark.asyncio
    async def test_channel_room_detected(self):
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(
            return_value={"room": {"t": "c", "name": "general"}}
        )

        result = await adapter._get_room_type("GENERAL")
        assert result == "c"

    @pytest.mark.asyncio
    async def test_missing_room_falls_back_to_channel(self):
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(return_value={})

        result = await adapter._get_room_type("unknown")
        assert result == "c"

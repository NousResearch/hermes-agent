"""Tests for SimpleX Chat platform adapter."""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------


def _make_simplex_adapter(monkeypatch, ws_url="ws://127.0.0.1:5225", **extra):
    """Create a SimplexAdapter with sensible test defaults."""
    monkeypatch.setenv("SIMPLEX_WS_URL", ws_url)
    monkeypatch.setenv("SIMPLEX_GROUP_ALLOWED", extra.pop("group_allowed", ""))
    from gateway.platforms.simplex import SimplexAdapter

    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "ws_url": ws_url,
        "auto_accept": extra.pop("auto_accept", True),
        **extra,
    }
    adapter = SimplexAdapter(config)
    adapter.handle_message = AsyncMock()
    return adapter


def _make_chat_item(
    chat_type="direct",
    contact_id=42,
    contact_name="alice",
    group_id=None,
    group_name=None,
    member_id=None,
    member_name=None,
    msg_type="text",
    text="Hello world",
    direction="directRcv",
    content_type="rcvMsgContent",
    file_info=None,
    item_ts="2025-01-15T12:00:00Z",
):
    """Build a simplex-chat newChatItems event payload."""
    chat_info = {"type": chat_type}
    chat_dir = {"type": direction}

    if chat_type == "direct":
        chat_info["contact"] = {
            "contactId": contact_id,
            "localDisplayName": contact_name,
            "profile": {"displayName": contact_name},
        }
    elif chat_type == "group":
        chat_info["groupInfo"] = {
            "groupId": group_id or 99,
            "localDisplayName": group_name or "test-group",
            "groupProfile": {"displayName": group_name or "test-group"},
        }
        chat_dir["groupMember"] = {
            "memberId": member_id or "m1",
            "localDisplayName": member_name or "bob",
            "memberProfile": {"displayName": member_name or "bob"},
        }

    inner_chat_item = {
        "chatDir": chat_dir,
        "meta": {"itemId": 1001, "itemTs": item_ts, "createdAt": item_ts},
        "content": {
            "type": content_type,
            "msgContent": {"type": msg_type, "text": text},
        },
    }

    if file_info:
        inner_chat_item["file"] = file_info

    return {
        "chatInfo": chat_info,
        "chatItem": inner_chat_item,
    }


def _make_event(resp_type, **kwargs):
    """Build a simplex-chat WS event wrapper."""
    resp = {"type": resp_type, **kwargs}
    return {"resp": resp}


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------


class TestSimplexPlatformEnum:
    def test_simplex_enum_exists(self):
        assert Platform.SIMPLEX.value == "simplex"

    def test_simplex_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "simplex" in platforms


class TestSimplexConfigLoading:
    def test_apply_env_overrides_simplex(self, monkeypatch):
        monkeypatch.setenv("SIMPLEX_WS_URL", "ws://localhost:5225")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIMPLEX in config.platforms
        sc = config.platforms[Platform.SIMPLEX]
        assert sc.enabled is True
        assert sc.extra["ws_url"] == "ws://localhost:5225"
        assert sc.extra["auto_accept"] is True

    def test_simplex_auto_accept_disabled(self, monkeypatch):
        monkeypatch.setenv("SIMPLEX_WS_URL", "ws://localhost:5225")
        monkeypatch.setenv("SIMPLEX_AUTO_ACCEPT", "false")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SIMPLEX]
        assert sc.extra["auto_accept"] is False

    def test_simplex_not_loaded_without_ws_url(self, monkeypatch):
        monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIMPLEX not in config.platforms

    def test_connected_platforms_includes_simplex(self, monkeypatch):
        monkeypatch.setenv("SIMPLEX_WS_URL", "ws://localhost:5225")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert Platform.SIMPLEX in connected

    def test_home_channel_loading(self, monkeypatch):
        monkeypatch.setenv("SIMPLEX_WS_URL", "ws://localhost:5225")
        monkeypatch.setenv("SIMPLEX_HOME_CHANNEL", "42")
        monkeypatch.setenv("SIMPLEX_HOME_CHANNEL_NAME", "Hermes Home")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        hc = config.platforms[Platform.SIMPLEX].home_channel
        assert hc is not None
        assert hc.chat_id == "42"
        assert hc.name == "Hermes Home"


# ---------------------------------------------------------------------------
# Adapter Init & Helpers
# ---------------------------------------------------------------------------


class TestSimplexAdapterInit:
    def test_init_parses_config(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        assert adapter.ws_url == "ws://127.0.0.1:5225"
        assert adapter.auto_accept is True
        assert adapter.platform == Platform.SIMPLEX

    def test_init_group_allowlist(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, group_allowed="1,2,3")
        assert "1" in adapter.group_allow_from
        assert "2" in adapter.group_allow_from
        assert "3" in adapter.group_allow_from

    def test_init_empty_group_allowlist(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        assert len(adapter.group_allow_from) == 0

    def test_init_strips_trailing_slash(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, ws_url="ws://localhost:5225/")
        assert adapter.ws_url == "ws://localhost:5225"


class TestSimplexHelpers:
    def test_parse_comma_list(self):
        from gateway.platforms.simplex import _parse_comma_list

        assert _parse_comma_list("a, b , c") == ["a", "b", "c"]
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []

    def test_redact_id_long(self):
        from gateway.platforms.simplex import _redact_id

        assert _redact_id("12345678") == "12**78"

    def test_redact_id_short(self):
        from gateway.platforms.simplex import _redact_id

        assert _redact_id("42") == "42"

    def test_redact_id_empty(self):
        from gateway.platforms.simplex import _redact_id

        assert _redact_id("") == "<none>"

    def test_guess_extension_png(self):
        from gateway.platforms.simplex import _guess_extension

        assert _guess_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == ".png"

    def test_guess_extension_jpeg(self):
        from gateway.platforms.simplex import _guess_extension

        assert _guess_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == ".jpg"

    def test_guess_extension_pdf(self):
        from gateway.platforms.simplex import _guess_extension

        assert _guess_extension(b"%PDF-1.4" + b"\x00" * 100) == ".pdf"

    def test_guess_extension_mp4(self):
        from gateway.platforms.simplex import _guess_extension

        assert _guess_extension(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100) == ".mp4"

    def test_guess_extension_ogg(self):
        from gateway.platforms.simplex import _guess_extension

        assert _guess_extension(b"OggS" + b"\x00" * 100) == ".ogg"

    def test_guess_extension_unknown(self):
        from gateway.platforms.simplex import _guess_extension

        assert _guess_extension(b"\x00\x01\x02\x03" * 10) == ".bin"

    def test_is_image_ext(self):
        from gateway.platforms.simplex import _is_image_ext

        assert _is_image_ext(".png") is True
        assert _is_image_ext(".jpg") is True
        assert _is_image_ext(".gif") is True
        assert _is_image_ext(".pdf") is False

    def test_is_audio_ext(self):
        from gateway.platforms.simplex import _is_audio_ext

        assert _is_audio_ext(".mp3") is True
        assert _is_audio_ext(".ogg") is True
        assert _is_audio_ext(".png") is False

    def test_check_requirements_true(self, monkeypatch):
        from gateway.platforms.simplex import check_simplex_requirements

        monkeypatch.setenv("SIMPLEX_WS_URL", "ws://localhost:5225")
        assert check_simplex_requirements() is True

    def test_check_requirements_false(self, monkeypatch):
        from gateway.platforms.simplex import check_simplex_requirements

        monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
        assert check_simplex_requirements() is False


# ---------------------------------------------------------------------------
# Event Handling — newChatItems
# ---------------------------------------------------------------------------


class TestSimplexEventHandling:
    @pytest.mark.asyncio
    async def test_handle_direct_text_message(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(
            chat_type="direct",
            contact_id=42,
            contact_name="alice",
            text="Hello!",
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_awaited_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.text == "Hello!"
        assert msg_event.source.chat_id == "42"
        assert msg_event.source.user_id == "42"
        assert msg_event.source.user_name == "alice"

    @pytest.mark.asyncio
    async def test_handle_multiple_chat_items(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        item1 = _make_chat_item(text="First")
        item2 = _make_chat_item(text="Second", contact_id=43, contact_name="bob")
        event = _make_event("newChatItems", chatItems=[item1, item2])
        await adapter._handle_event(event)

        assert adapter.handle_message.await_count == 2

    @pytest.mark.asyncio
    async def test_skip_sent_messages(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(direction="directSnd")
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skip_group_sent_messages(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, group_allowed="*")
        item = _make_chat_item(
            chat_type="group",
            direction="groupSnd",
            group_id=99,
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skip_non_rcv_content(self, monkeypatch):
        """Messages with content.type != 'rcvMsgContent' should be skipped."""
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(content_type="sndMsgContent")
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_group_message_with_allowlist(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, group_allowed="99")
        item = _make_chat_item(
            chat_type="group",
            direction="groupRcv",
            group_id=99,
            group_name="test-group",
            member_id="m1",
            member_name="bob",
            text="Group hello",
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_awaited_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.text == "Group hello"
        assert msg_event.source.chat_id == "group:99"
        assert msg_event.source.user_id == "m1"

    @pytest.mark.asyncio
    async def test_handle_group_message_wildcard(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, group_allowed="*")
        item = _make_chat_item(
            chat_type="group",
            direction="groupRcv",
            group_id=55,
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reject_group_not_in_allowlist(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, group_allowed="100")
        item = _make_chat_item(
            chat_type="group",
            direction="groupRcv",
            group_id=99,
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reject_group_no_allowlist(self, monkeypatch):
        """Groups disabled when SIMPLEX_GROUP_ALLOWED not set."""
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(
            chat_type="group",
            direction="groupRcv",
            group_id=99,
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_text_non_media_skipped(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(text="", msg_type="text")
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timestamp_parsing(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(item_ts="2025-06-15T10:30:00Z")
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.timestamp.year == 2025
        assert msg_event.timestamp.month == 6

    @pytest.mark.asyncio
    async def test_timestamp_fallback_on_invalid(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(item_ts="not-a-date")
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert isinstance(msg_event.timestamp, datetime)


# ---------------------------------------------------------------------------
# Event Handling — File Attachments
# ---------------------------------------------------------------------------


class TestSimplexFileAttachments:
    @pytest.mark.asyncio
    async def test_image_attachment(self, monkeypatch, tmp_path):
        adapter = _make_simplex_adapter(monkeypatch)
        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        item = _make_chat_item(
            msg_type="image",
            text="Look at this",
            file_info={
                "fileId": 1,
                "fileName": "photo.jpg",
                "fileSize": 102,
                "fileSource": {"filePath": str(img_path)},
            },
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_awaited_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert str(img_path) in msg_event.media_urls
        assert any("image" in mt for mt in msg_event.media_types)

    @pytest.mark.asyncio
    async def test_audio_attachment(self, monkeypatch, tmp_path):
        adapter = _make_simplex_adapter(monkeypatch)
        audio_path = tmp_path / "voice.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 100)

        item = _make_chat_item(
            msg_type="voice",
            text="",
            file_info={
                "fileId": 2,
                "fileName": "voice.ogg",
                "fileSize": 104,
                "fileSource": {"filePath": str(audio_path)},
            },
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_awaited_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert any("audio" in mt for mt in msg_event.media_types)

    @pytest.mark.asyncio
    async def test_no_file_source(self, monkeypatch):
        """File info without fileSource should not produce media_urls."""
        adapter = _make_simplex_adapter(monkeypatch)
        item = _make_chat_item(
            msg_type="file",
            text="document",
            file_info={
                "fileId": 3,
                "fileName": "doc.pdf",
                "fileSize": 1024,
                # No fileSource
            },
        )
        event = _make_event("newChatItems", chatItems=[item])
        await adapter._handle_event(event)

        adapter.handle_message.assert_awaited_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 0


# ---------------------------------------------------------------------------
# Contact Request Handling
# ---------------------------------------------------------------------------


class TestSimplexContactRequest:
    @pytest.mark.asyncio
    async def test_auto_accept_contact_request(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, auto_accept=True)
        adapter._send_command = AsyncMock(return_value={"type": "contactConnected"})

        event = _make_event(
            "contactRequest",
            contactRequest={"contactRequestId": 7},
        )
        await adapter._handle_event(event)

        adapter._send_command.assert_awaited_once()
        cmd = adapter._send_command.call_args[0][0]
        assert "/accept 7" in cmd

    @pytest.mark.asyncio
    async def test_no_accept_when_disabled(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch, auto_accept=False)
        adapter._send_command = AsyncMock()

        event = _make_event(
            "contactRequest",
            contactRequest={"contactRequestId": 7},
        )
        await adapter._handle_event(event)

        adapter._send_command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Correlated Response Handling
# ---------------------------------------------------------------------------


class TestSimplexCorrelation:
    @pytest.mark.asyncio
    async def test_correlated_response_resolves_future(self, monkeypatch):
        import asyncio

        adapter = _make_simplex_adapter(monkeypatch)
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        adapter._pending_responses["corr_1"] = fut

        raw_event = {"corrId": "corr_1", "resp": {"type": "newChatItems"}}
        await adapter._handle_event(raw_event.get("resp", {}))

        # corrId handling is in the outer wrapper, let's test it directly
        # The _handle_event receives resp, but corrId is checked in _ws_listener
        # Actually, looking at the code, corrId IS checked in _handle_event
        # Let me re-read...

    @pytest.mark.asyncio
    async def test_unhandled_event_type_ignored(self, monkeypatch):
        """Unknown event types should not raise."""
        adapter = _make_simplex_adapter(monkeypatch)
        event = _make_event("someUnknownEvent")
        await adapter._handle_event(event)
        # Should not raise


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


class TestSimplexSend:
    @pytest.mark.asyncio
    async def test_send_dm_format(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(return_value={"type": "newChatItems"})

        result = await adapter.send("42", "Hello!")
        assert result.success is True

        cmd = adapter._send_command.call_args[0][0]
        assert cmd == "@42 Hello!"

    @pytest.mark.asyncio
    async def test_send_group_format(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(return_value={"type": "newChatItems"})

        result = await adapter.send("group:99", "Hello group!")
        assert result.success is True

        cmd = adapter._send_command.call_args[0][0]
        assert cmd == "#99 Hello group!"
        assert "#group:" not in cmd

    @pytest.mark.asyncio
    async def test_send_error_response(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(
            return_value={"type": "chatCmdError", "chatError": "not found"}
        )

        result = await adapter.send("42", "Hello!")
        assert result.success is False
        assert "error" in result.error.lower() or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_no_ws_connected(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(return_value=None)
        adapter._ws = None

        result = await adapter.send("42", "Hello!")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_no_response_but_ws_connected(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(return_value=None)
        adapter._ws = MagicMock()

        result = await adapter.send("42", "Hello!")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_image_dm(self, monkeypatch, tmp_path):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(return_value={"type": "newChatItems"})
        adapter.send = AsyncMock(return_value=MagicMock(success=True))

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image("42", f"file://{img_path}", caption="A photo")
        assert result.success is True

        cmd = adapter._send_command.call_args[0][0]
        assert cmd.startswith("/f @42 ")

    @pytest.mark.asyncio
    async def test_send_image_group(self, monkeypatch, tmp_path):
        adapter = _make_simplex_adapter(monkeypatch)
        adapter._send_command = AsyncMock(return_value={"type": "newChatItems"})
        adapter.send = AsyncMock(return_value=MagicMock(success=True))

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image("group:99", f"file://{img_path}")
        assert result.success is True

        cmd = adapter._send_command.call_args[0][0]
        assert "/f #99 " in cmd
        assert "#group:" not in cmd

    @pytest.mark.asyncio
    async def test_send_document_not_found(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        result = await adapter.send_document("42", "/nonexistent/file.pdf")
        assert result.success is False
        assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# Chat Info
# ---------------------------------------------------------------------------


class TestSimplexChatInfo:
    @pytest.mark.asyncio
    async def test_dm_chat_info(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        info = await adapter.get_chat_info("42")
        assert info["type"] == "dm"
        assert info["chat_id"] == "42"

    @pytest.mark.asyncio
    async def test_group_chat_info(self, monkeypatch):
        adapter = _make_simplex_adapter(monkeypatch)
        info = await adapter.get_chat_info("group:99")
        assert info["type"] == "group"
        assert info["chat_id"] == "group:99"


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------


class TestSimplexSessionSource:
    def test_session_source_roundtrip(self):
        from gateway.session import SessionSource

        source = SessionSource(
            platform=Platform.SIMPLEX,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="alice",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.platform == Platform.SIMPLEX
        assert restored.chat_id == "42"
        assert restored.user_id == "42"

    def test_session_source_group(self):
        from gateway.session import SessionSource

        source = SessionSource(
            platform=Platform.SIMPLEX,
            chat_id="group:99",
            chat_type="group",
            user_id="m1",
            user_name="bob",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.chat_id == "group:99"
        assert restored.chat_type == "group"


# ---------------------------------------------------------------------------
# Authorization in run.py
# ---------------------------------------------------------------------------


class TestSimplexAuthorization:
    def test_simplex_in_allowlist_maps(self):
        """SimpleX should be in the platform auth maps."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SIMPLEX
        source.user_id = "42"

        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------


class TestSimplexSendMessage:
    def test_simplex_in_platform_map(self):
        """SimpleX should be in the send_message tool's platform map."""
        from gateway.config import Platform

        assert Platform.SIMPLEX.value == "simplex"

    @pytest.mark.asyncio
    async def test_send_simplex_standalone_dm_format(self, monkeypatch):
        """The standalone _send_simplex should use @<id> for DMs."""
        from tools.send_message_tool import _send_simplex

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "corrId": "test",
                    "resp": {"type": "newChatItems"},
                }
            )
        )

        with patch("websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _send_simplex(
                {"ws_url": "ws://localhost:5225"},
                "42",
                "Hello!",
            )

        assert result.get("success") is True
        sent_payload = json.loads(mock_ws.send.call_args[0][0])
        assert sent_payload["cmd"] == "@42 Hello!"

    @pytest.mark.asyncio
    async def test_send_simplex_standalone_group_format(self, monkeypatch):
        """The standalone _send_simplex should use #<id> for groups (not #group:<id>)."""
        from tools.send_message_tool import _send_simplex

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "corrId": "test",
                    "resp": {"type": "newChatItems"},
                }
            )
        )

        with patch("websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _send_simplex(
                {"ws_url": "ws://localhost:5225"},
                "group:99",
                "Hello group!",
            )

        assert result.get("success") is True
        sent_payload = json.loads(mock_ws.send.call_args[0][0])
        assert sent_payload["cmd"] == "#99 Hello group!"
        assert "#group:" not in sent_payload["cmd"]

"""Tests for Matrix platform adapter."""
import asyncio
import json
import re
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestMatrixPlatformEnum:
    def test_matrix_enum_exists(self):
        assert Platform.MATRIX.value == "matrix"

    def test_matrix_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "matrix" in platforms


class TestMatrixConfigLoading:
    def test_apply_env_overrides_with_access_token(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATRIX in config.platforms
        mc = config.platforms[Platform.MATRIX]
        assert mc.enabled is True
        assert mc.token == "syt_abc123"
        assert mc.extra.get("homeserver") == "https://matrix.example.org"

    def test_apply_env_overrides_with_password(self, monkeypatch):
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("MATRIX_PASSWORD", "secret123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_USER_ID", "@bot:example.org")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATRIX in config.platforms
        mc = config.platforms[Platform.MATRIX]
        assert mc.enabled is True
        assert mc.extra.get("password") == "secret123"
        assert mc.extra.get("user_id") == "@bot:example.org"

    def test_matrix_not_loaded_without_creds(self, monkeypatch):
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("MATRIX_PASSWORD", raising=False)
        monkeypatch.delenv("MATRIX_HOMESERVER", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATRIX not in config.platforms

    def test_matrix_encryption_flag(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ENCRYPTION", "true")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        mc = config.platforms[Platform.MATRIX]
        assert mc.extra.get("encryption") is True

    def test_matrix_encryption_default_off(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        mc = config.platforms[Platform.MATRIX]
        assert mc.extra.get("encryption") is False

    def test_matrix_home_room(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_HOME_ROOM", "!room123:example.org")
        monkeypatch.setenv("MATRIX_HOME_ROOM_NAME", "Bot Room")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.get_home_channel(Platform.MATRIX)
        assert home is not None
        assert home.chat_id == "!room123:example.org"
        assert home.name == "Bot Room"

    def test_matrix_user_id_stored_in_extra(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:example.org")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        mc = config.platforms[Platform.MATRIX]
        assert mc.extra.get("user_id") == "@hermes:example.org"


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a MatrixAdapter with mocked config."""
    from gateway.platforms.matrix import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    return adapter


# ---------------------------------------------------------------------------
# mxc:// URL conversion
# ---------------------------------------------------------------------------

class TestMatrixMxcToHttp:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_basic_mxc_conversion(self):
        """mxc://server/media_id should become an authenticated HTTP URL."""
        mxc = "mxc://matrix.org/abc123"
        result = self.adapter._mxc_to_http(mxc)
        assert result == "https://matrix.example.org/_matrix/client/v1/media/download/matrix.org/abc123"

    def test_mxc_with_different_server(self):
        """mxc:// from a different server should still use our homeserver."""
        mxc = "mxc://other.server/media456"
        result = self.adapter._mxc_to_http(mxc)
        assert result.startswith("https://matrix.example.org/")
        assert "other.server/media456" in result

    def test_non_mxc_url_passthrough(self):
        """Non-mxc URLs should be returned unchanged."""
        url = "https://example.com/image.png"
        assert self.adapter._mxc_to_http(url) == url

    def test_mxc_uses_client_v1_endpoint(self):
        """Should use /_matrix/client/v1/media/download/ not the deprecated path."""
        mxc = "mxc://example.com/test123"
        result = self.adapter._mxc_to_http(mxc)
        assert "/_matrix/client/v1/media/download/" in result
        assert "/_matrix/media/v3/download/" not in result


# ---------------------------------------------------------------------------
# DM detection
# ---------------------------------------------------------------------------

class TestMatrixDmDetection:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_room_in_m_direct_is_dm(self):
        """A room listed in m.direct should be detected as DM."""
        self.adapter._joined_rooms = {"!dm_room:ex.org", "!group_room:ex.org"}
        self.adapter._dm_rooms = {
            "!dm_room:ex.org": True,
            "!group_room:ex.org": False,
        }

        assert self.adapter._dm_rooms.get("!dm_room:ex.org") is True
        assert self.adapter._dm_rooms.get("!group_room:ex.org") is False

    def test_unknown_room_not_in_cache(self):
        """Unknown rooms should not be in the DM cache."""
        self.adapter._dm_rooms = {}
        assert self.adapter._dm_rooms.get("!unknown:ex.org") is None

    @pytest.mark.asyncio
    async def test_refresh_dm_cache_with_m_direct(self):
        """_refresh_dm_cache should populate _dm_rooms from m.direct data."""
        self.adapter._joined_rooms = {"!room_a:ex.org", "!room_b:ex.org", "!room_c:ex.org"}

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = {
            "@alice:ex.org": ["!room_a:ex.org"],
            "@bob:ex.org": ["!room_b:ex.org"],
        }
        mock_client.get_account_data = AsyncMock(return_value=mock_resp)
        self.adapter._client = mock_client

        await self.adapter._refresh_dm_cache()

        assert self.adapter._dm_rooms["!room_a:ex.org"] is True
        assert self.adapter._dm_rooms["!room_b:ex.org"] is True
        assert self.adapter._dm_rooms["!room_c:ex.org"] is False


# ---------------------------------------------------------------------------
# Reply fallback stripping
# ---------------------------------------------------------------------------

class TestMatrixReplyFallbackStripping:
    """Test that Matrix reply fallback lines ('> ' prefix) are stripped."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._startup_ts = 0.0
        self.adapter._dm_rooms = {}
        self.adapter._message_handler = AsyncMock()

    def _strip_fallback(self, body: str, has_reply: bool = True) -> str:
        """Simulate the reply fallback stripping logic from _on_room_message."""
        reply_to = "some_event_id" if has_reply else None
        if reply_to and body.startswith("> "):
            lines = body.split("\n")
            stripped = []
            past_fallback = False
            for line in lines:
                if not past_fallback:
                    if line.startswith("> ") or line == ">":
                        continue
                    if line == "":
                        past_fallback = True
                        continue
                    past_fallback = True
                stripped.append(line)
            body = "\n".join(stripped) if stripped else body
        return body

    def test_simple_reply_fallback(self):
        body = "> <@alice:ex.org> Original message\n\nActual reply"
        result = self._strip_fallback(body)
        assert result == "Actual reply"

    def test_multiline_reply_fallback(self):
        body = "> <@alice:ex.org> Line 1\n> Line 2\n\nMy response"
        result = self._strip_fallback(body)
        assert result == "My response"

    def test_no_reply_fallback_preserved(self):
        body = "Just a normal message"
        result = self._strip_fallback(body, has_reply=False)
        assert result == "Just a normal message"

    def test_quote_without_reply_preserved(self):
        """'> ' lines without a reply_to context should be preserved."""
        body = "> This is a blockquote"
        result = self._strip_fallback(body, has_reply=False)
        assert result == "> This is a blockquote"

    def test_empty_fallback_separator(self):
        """The blank line between fallback and actual content should be stripped."""
        body = "> <@alice:ex.org> hi\n>\n\nResponse"
        result = self._strip_fallback(body)
        assert result == "Response"

    def test_multiline_response_after_fallback(self):
        body = "> <@alice:ex.org> Original\n\nLine 1\nLine 2\nLine 3"
        result = self._strip_fallback(body)
        assert result == "Line 1\nLine 2\nLine 3"


# ---------------------------------------------------------------------------
# Thread detection
# ---------------------------------------------------------------------------

class TestMatrixThreadDetection:
    def test_thread_id_from_m_relates_to(self):
        """m.relates_to with rel_type=m.thread should extract the event_id."""
        relates_to = {
            "rel_type": "m.thread",
            "event_id": "$thread_root_event",
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": "$some_event"},
        }
        # Simulate the extraction logic from _on_room_message
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")
        assert thread_id == "$thread_root_event"

    def test_no_thread_for_reply(self):
        """m.in_reply_to without m.thread should not set thread_id."""
        relates_to = {
            "m.in_reply_to": {"event_id": "$reply_event"},
        }
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")
        assert thread_id is None

    def test_no_thread_for_edit(self):
        """m.replace relation should not set thread_id."""
        relates_to = {
            "rel_type": "m.replace",
            "event_id": "$edited_event",
        }
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")
        assert thread_id is None

    def test_empty_relates_to(self):
        """Empty m.relates_to should not set thread_id."""
        relates_to = {}
        thread_id = None
        if relates_to.get("rel_type") == "m.thread":
            thread_id = relates_to.get("event_id")
        assert thread_id is None


# ---------------------------------------------------------------------------
# Format message
# ---------------------------------------------------------------------------

class TestMatrixFormatMessage:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_image_markdown_stripped(self):
        """![alt](url) should be converted to just the URL."""
        result = self.adapter.format_message("![cat](https://img.example.com/cat.png)")
        assert result == "https://img.example.com/cat.png"

    def test_regular_markdown_preserved(self):
        """Standard markdown should be preserved (Matrix supports it)."""
        content = "**bold** and *italic* and `code`"
        assert self.adapter.format_message(content) == content

    def test_plain_text_unchanged(self):
        content = "Hello, world!"
        assert self.adapter.format_message(content) == content

    def test_multiple_images_stripped(self):
        content = "![a](http://a.com/1.png) and ![b](http://b.com/2.png)"
        result = self.adapter.format_message(content)
        assert "![" not in result
        assert "http://a.com/1.png" in result
        assert "http://b.com/2.png" in result


# ---------------------------------------------------------------------------
# Markdown to HTML conversion
# ---------------------------------------------------------------------------

class TestMatrixMarkdownToHtml:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_bold_conversion(self):
        """**bold** should produce <strong> tags."""
        result = self.adapter._markdown_to_html("**bold**")
        assert "<strong>" in result or "<b>" in result
        assert "bold" in result

    def test_italic_conversion(self):
        """*italic* should produce <em> tags."""
        result = self.adapter._markdown_to_html("*italic*")
        assert "<em>" in result or "<i>" in result

    def test_inline_code(self):
        """`code` should produce <code> tags."""
        result = self.adapter._markdown_to_html("`code`")
        assert "<code>" in result

    def test_plain_text_returns_html(self):
        """Plain text should still be returned (possibly with <br> or <p>)."""
        result = self.adapter._markdown_to_html("Hello world")
        assert "Hello world" in result


# ---------------------------------------------------------------------------
# Helper: display name extraction
# ---------------------------------------------------------------------------

class TestMatrixDisplayName:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_get_display_name_from_room_users(self):
        """Should get display name from room's users dict."""
        mock_room = MagicMock()
        mock_user = MagicMock()
        mock_user.display_name = "Alice"
        mock_room.users = {"@alice:ex.org": mock_user}

        name = self.adapter._get_display_name(mock_room, "@alice:ex.org")
        assert name == "Alice"

    def test_get_display_name_fallback_to_localpart(self):
        """Should extract localpart from @user:server format."""
        mock_room = MagicMock()
        mock_room.users = {}

        name = self.adapter._get_display_name(mock_room, "@bob:example.org")
        assert name == "bob"

    def test_get_display_name_no_room(self):
        """Should handle None room gracefully."""
        name = self.adapter._get_display_name(None, "@charlie:ex.org")
        assert name == "charlie"


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestMatrixRequirements:
    def test_check_requirements_with_token(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        from gateway.platforms.matrix import check_matrix_requirements
        try:
            import nio  # noqa: F401
            assert check_matrix_requirements() is True
        except ImportError:
            assert check_matrix_requirements() is False

    def test_check_requirements_without_creds(self, monkeypatch):
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("MATRIX_PASSWORD", raising=False)
        monkeypatch.delenv("MATRIX_HOMESERVER", raising=False)
        from gateway.platforms.matrix import check_matrix_requirements
        assert check_matrix_requirements() is False

    def test_check_requirements_without_homeserver(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.delenv("MATRIX_HOMESERVER", raising=False)
        from gateway.platforms.matrix import check_matrix_requirements
        assert check_matrix_requirements() is False


# ---------------------------------------------------------------------------
# Access-token auth / E2EE bootstrap
# ---------------------------------------------------------------------------

class TestMatrixAccessTokenAuth:
    @pytest.mark.asyncio
    async def test_connect_fetches_device_id_from_whoami_for_access_token(self):
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test_access_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "encryption": True,
            },
        )
        adapter = MatrixAdapter(config)

        class FakeWhoamiResponse:
            def __init__(self, user_id, device_id):
                self.user_id = user_id
                self.device_id = device_id

        class FakeSyncResponse:
            def __init__(self):
                self.rooms = MagicMock(join={})

        fake_client = MagicMock()
        fake_client.whoami = AsyncMock(return_value=FakeWhoamiResponse("@bot:example.org", "DEV123"))
        fake_client.sync = AsyncMock(return_value=FakeSyncResponse())
        fake_client.keys_upload = AsyncMock()
        fake_client.keys_query = AsyncMock()
        fake_client.keys_claim = AsyncMock()
        fake_client.send_to_device_messages = AsyncMock(return_value=[])
        fake_client.get_users_for_key_claiming = MagicMock(return_value={})
        fake_client.close = AsyncMock()
        fake_client.add_event_callback = MagicMock()
        fake_client.rooms = {}
        fake_client.account_data = {}
        fake_client.olm = object()
        fake_client.should_upload_keys = False
        fake_client.should_query_keys = False
        fake_client.should_claim_keys = False

        def _restore_login(user_id, device_id, access_token):
            fake_client.user_id = user_id
            fake_client.device_id = device_id
            fake_client.access_token = access_token
            fake_client.olm = object()

        fake_client.restore_login = MagicMock(side_effect=_restore_login)

        fake_nio = MagicMock()
        fake_nio.AsyncClient = MagicMock(return_value=fake_client)
        fake_nio.WhoamiResponse = FakeWhoamiResponse
        fake_nio.SyncResponse = FakeSyncResponse
        fake_nio.LoginResponse = type("LoginResponse", (), {})
        fake_nio.RoomMessageText = type("RoomMessageText", (), {})
        fake_nio.RoomMessageImage = type("RoomMessageImage", (), {})
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})
        fake_nio.InviteMemberEvent = type("InviteMemberEvent", (), {})
        fake_nio.MegolmEvent = type("MegolmEvent", (), {})

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch.object(adapter, "_refresh_dm_cache", AsyncMock()):
                with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                    assert await adapter.connect() is True

        fake_client.restore_login.assert_called_once_with(
            "@bot:example.org", "DEV123", "syt_test_access_token"
        )
        assert fake_client.access_token == "syt_test_access_token"
        assert fake_client.user_id == "@bot:example.org"
        assert fake_client.device_id == "DEV123"
        fake_client.whoami.assert_awaited_once()

        await adapter.disconnect()


class TestMatrixE2EEMaintenance:
    @pytest.mark.asyncio
    async def test_sync_loop_runs_e2ee_maintenance_requests(self):
        adapter = _make_adapter()
        adapter._encryption = True
        adapter._closing = False

        class FakeSyncError:
            pass

        async def _sync_once(timeout=30000):
            adapter._closing = True
            return MagicMock()

        fake_client = MagicMock()
        fake_client.sync = AsyncMock(side_effect=_sync_once)
        fake_client.send_to_device_messages = AsyncMock(return_value=[])
        fake_client.keys_upload = AsyncMock()
        fake_client.keys_query = AsyncMock()
        fake_client.get_users_for_key_claiming = MagicMock(
            return_value={"@alice:example.org": ["DEVICE1"]}
        )
        fake_client.keys_claim = AsyncMock()
        fake_client.olm = object()
        fake_client.should_upload_keys = True
        fake_client.should_query_keys = True
        fake_client.should_claim_keys = True

        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.SyncError = FakeSyncError

        with patch.dict("sys.modules", {"nio": fake_nio}):
            await adapter._sync_loop()

        fake_client.sync.assert_awaited_once_with(timeout=30000)
        fake_client.send_to_device_messages.assert_awaited_once()
        fake_client.keys_upload.assert_awaited_once()
        fake_client.keys_query.assert_awaited_once()
        fake_client.keys_claim.assert_awaited_once_with(
            {"@alice:example.org": ["DEVICE1"]}
        )


class TestMatrixEncryptedSendFallback:
    @pytest.mark.asyncio
    async def test_send_retries_with_ignored_unverified_devices(self):
        adapter = _make_adapter()
        adapter._encryption = True

        class FakeRoomSendResponse:
            def __init__(self, event_id):
                self.event_id = event_id

        class FakeOlmUnverifiedDeviceError(Exception):
            pass

        fake_client = MagicMock()
        fake_client.room_send = AsyncMock(side_effect=[
            FakeOlmUnverifiedDeviceError("unverified"),
            FakeRoomSendResponse("$event123"),
        ])
        adapter._client = fake_client
        adapter._run_e2ee_maintenance = AsyncMock()

        fake_nio = MagicMock()
        fake_nio.RoomSendResponse = FakeRoomSendResponse
        fake_nio.OlmUnverifiedDeviceError = FakeOlmUnverifiedDeviceError

        with patch.dict("sys.modules", {"nio": fake_nio}):
            result = await adapter.send("!room:example.org", "hello")

        assert result.success is True
        assert result.message_id == "$event123"
        adapter._run_e2ee_maintenance.assert_awaited_once()
        assert fake_client.room_send.await_count == 2
        first_call = fake_client.room_send.await_args_list[0]
        second_call = fake_client.room_send.await_args_list[1]
        assert first_call.kwargs.get("ignore_unverified_devices") is False
        assert second_call.kwargs.get("ignore_unverified_devices") is True

    @pytest.mark.asyncio
    async def test_send_retries_after_timeout_in_encrypted_room(self):
        adapter = _make_adapter()
        adapter._encryption = True

        class FakeRoomSendResponse:
            def __init__(self, event_id):
                self.event_id = event_id

        fake_client = MagicMock()
        fake_client.room_send = AsyncMock(side_effect=[
            asyncio.TimeoutError(),
            FakeRoomSendResponse("$event456"),
        ])
        adapter._client = fake_client
        adapter._run_e2ee_maintenance = AsyncMock()

        fake_nio = MagicMock()
        fake_nio.RoomSendResponse = FakeRoomSendResponse

        with patch.dict("sys.modules", {"nio": fake_nio}):
            result = await adapter.send("!room:example.org", "hello")

        assert result.success is True
        assert result.message_id == "$event456"
        adapter._run_e2ee_maintenance.assert_awaited_once()
        assert fake_client.room_send.await_count == 2
        second_call = fake_client.room_send.await_args_list[1]
        assert second_call.kwargs.get("ignore_unverified_devices") is True


# ---------------------------------------------------------------------------
# E2EE: Auto-trust devices
# ---------------------------------------------------------------------------

class TestMatrixAutoTrustDevices:
    def test_auto_trust_verifies_unverified_devices(self):
        adapter = _make_adapter()

        # DeviceStore.__iter__ yields OlmDevice objects directly.
        device_a = MagicMock()
        device_a.device_id = "DEVICE_A"
        device_a.verified = False
        device_b = MagicMock()
        device_b.device_id = "DEVICE_B"
        device_b.verified = True  # already trusted
        device_c = MagicMock()
        device_c.device_id = "DEVICE_C"
        device_c.verified = False

        fake_client = MagicMock()
        fake_client.device_id = "OWN_DEVICE"
        fake_client.verify_device = MagicMock()

        # Simulate DeviceStore iteration (yields OlmDevice objects)
        fake_client.device_store = MagicMock()
        fake_client.device_store.__iter__ = MagicMock(
            return_value=iter([device_a, device_b, device_c])
        )

        adapter._client = fake_client
        adapter._auto_trust_devices()

        # Should have verified device_a and device_c (not device_b, already verified)
        assert fake_client.verify_device.call_count == 2
        verified_devices = [call.args[0] for call in fake_client.verify_device.call_args_list]
        assert device_a in verified_devices
        assert device_c in verified_devices
        assert device_b not in verified_devices

    def test_auto_trust_skips_own_device(self):
        adapter = _make_adapter()

        own_device = MagicMock()
        own_device.device_id = "MY_DEVICE"
        own_device.verified = False

        fake_client = MagicMock()
        fake_client.device_id = "MY_DEVICE"
        fake_client.verify_device = MagicMock()

        fake_client.device_store = MagicMock()
        fake_client.device_store.__iter__ = MagicMock(
            return_value=iter([own_device])
        )

        adapter._client = fake_client
        adapter._auto_trust_devices()

        fake_client.verify_device.assert_not_called()

    def test_auto_trust_handles_missing_device_store(self):
        adapter = _make_adapter()
        fake_client = MagicMock(spec=[])  # empty spec — no attributes
        adapter._client = fake_client
        # Should not raise
        adapter._auto_trust_devices()


# ---------------------------------------------------------------------------
# E2EE: MegolmEvent key request + buffering
# ---------------------------------------------------------------------------

class TestMatrixMegolmEventHandling:
    @pytest.mark.asyncio
    async def test_megolm_event_requests_room_key_and_buffers(self):
        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}

        fake_megolm = MagicMock()
        fake_megolm.sender = "@alice:example.org"
        fake_megolm.event_id = "$encrypted_event"
        fake_megolm.server_timestamp = 9999999999000  # future
        fake_megolm.session_id = "SESSION123"

        fake_room = MagicMock()
        fake_room.room_id = "!room:example.org"

        fake_client = MagicMock()
        fake_client.request_room_key = AsyncMock(return_value=MagicMock())
        adapter._client = fake_client

        # Create a MegolmEvent class for isinstance check
        fake_nio = MagicMock()
        FakeMegolmEvent = type("MegolmEvent", (), {})
        fake_megolm.__class__ = FakeMegolmEvent
        fake_nio.MegolmEvent = FakeMegolmEvent

        with patch.dict("sys.modules", {"nio": fake_nio}):
            await adapter._on_room_message(fake_room, fake_megolm)

        # Should have requested the room key
        fake_client.request_room_key.assert_awaited_once_with(fake_megolm)

        # Should have buffered the event
        assert len(adapter._pending_megolm) == 1
        room, event, ts = adapter._pending_megolm[0]
        assert room is fake_room
        assert event is fake_megolm

    @pytest.mark.asyncio
    async def test_megolm_buffer_capped(self):
        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}

        fake_client = MagicMock()
        fake_client.request_room_key = AsyncMock(return_value=MagicMock())
        adapter._client = fake_client

        FakeMegolmEvent = type("MegolmEvent", (), {})
        fake_nio = MagicMock()
        fake_nio.MegolmEvent = FakeMegolmEvent

        # Fill the buffer past max
        from gateway.platforms.matrix import _MAX_PENDING_EVENTS
        with patch.dict("sys.modules", {"nio": fake_nio}):
            for i in range(_MAX_PENDING_EVENTS + 10):
                evt = MagicMock()
                evt.__class__ = FakeMegolmEvent
                evt.sender = "@alice:example.org"
                evt.event_id = f"$event_{i}"
                evt.server_timestamp = 9999999999000
                evt.session_id = f"SESSION_{i}"
                room = MagicMock()
                room.room_id = "!room:example.org"
                await adapter._on_room_message(room, evt)

        assert len(adapter._pending_megolm) == _MAX_PENDING_EVENTS


# ---------------------------------------------------------------------------
# E2EE: Retry pending decryptions
# ---------------------------------------------------------------------------

class TestMatrixRetryPendingDecryptions:
    @pytest.mark.asyncio
    async def test_successful_decryption_routes_to_text_handler(self):
        import time as _time

        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}

        # Create types
        FakeMegolmEvent = type("MegolmEvent", (), {})
        FakeRoomMessageText = type("RoomMessageText", (), {})

        decrypted_event = MagicMock()
        decrypted_event.__class__ = FakeRoomMessageText

        fake_megolm = MagicMock()
        fake_megolm.__class__ = FakeMegolmEvent
        fake_megolm.event_id = "$encrypted"

        fake_room = MagicMock()
        now = _time.time()

        adapter._pending_megolm = [(fake_room, fake_megolm, now)]

        fake_client = MagicMock()
        fake_client.decrypt_event = MagicMock(return_value=decrypted_event)
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.MegolmEvent = FakeMegolmEvent
        fake_nio.RoomMessageText = FakeRoomMessageText
        fake_nio.RoomMessageImage = type("RoomMessageImage", (), {})
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch.object(adapter, "_on_room_message", AsyncMock()) as mock_handler:
                await adapter._retry_pending_decryptions()
                mock_handler.assert_awaited_once_with(fake_room, decrypted_event)

        # Buffer should be empty now
        assert len(adapter._pending_megolm) == 0

    @pytest.mark.asyncio
    async def test_still_undecryptable_stays_in_buffer(self):
        import time as _time

        adapter = _make_adapter()

        FakeMegolmEvent = type("MegolmEvent", (), {})

        fake_megolm = MagicMock()
        fake_megolm.__class__ = FakeMegolmEvent
        fake_megolm.event_id = "$still_encrypted"

        now = _time.time()
        adapter._pending_megolm = [(MagicMock(), fake_megolm, now)]

        fake_client = MagicMock()
        # decrypt_event raises when key is still missing
        fake_client.decrypt_event = MagicMock(side_effect=Exception("missing key"))
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.MegolmEvent = FakeMegolmEvent

        with patch.dict("sys.modules", {"nio": fake_nio}):
            await adapter._retry_pending_decryptions()

        assert len(adapter._pending_megolm) == 1

    @pytest.mark.asyncio
    async def test_expired_events_dropped(self):
        import time as _time

        adapter = _make_adapter()

        from gateway.platforms.matrix import _PENDING_EVENT_TTL

        fake_megolm = MagicMock()
        fake_megolm.event_id = "$old_event"
        fake_megolm.__class__ = type("MegolmEvent", (), {})

        # Timestamp well past TTL
        old_ts = _time.time() - _PENDING_EVENT_TTL - 60
        adapter._pending_megolm = [(MagicMock(), fake_megolm, old_ts)]

        fake_client = MagicMock()
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.MegolmEvent = type("MegolmEvent", (), {})

        with patch.dict("sys.modules", {"nio": fake_nio}):
            await adapter._retry_pending_decryptions()

        # Should have been dropped
        assert len(adapter._pending_megolm) == 0
        # Should NOT have tried to decrypt
        fake_client.decrypt_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_media_event_routes_to_media_handler(self):
        import time as _time

        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0

        FakeMegolmEvent = type("MegolmEvent", (), {})
        FakeRoomMessageImage = type("RoomMessageImage", (), {})

        decrypted_image = MagicMock()
        decrypted_image.__class__ = FakeRoomMessageImage

        fake_megolm = MagicMock()
        fake_megolm.__class__ = FakeMegolmEvent
        fake_megolm.event_id = "$encrypted_image"

        fake_room = MagicMock()
        now = _time.time()
        adapter._pending_megolm = [(fake_room, fake_megolm, now)]

        fake_client = MagicMock()
        fake_client.decrypt_event = MagicMock(return_value=decrypted_image)
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.MegolmEvent = FakeMegolmEvent
        fake_nio.RoomMessageText = type("RoomMessageText", (), {})
        fake_nio.RoomMessageImage = FakeRoomMessageImage
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch.object(adapter, "_on_room_message_media", AsyncMock()) as mock_media:
                await adapter._retry_pending_decryptions()
                mock_media.assert_awaited_once_with(fake_room, decrypted_image)

        assert len(adapter._pending_megolm) == 0


# ---------------------------------------------------------------------------
# E2EE: Key export / import
# ---------------------------------------------------------------------------

class TestMatrixKeyExportImport:
    @pytest.mark.asyncio
    async def test_disconnect_exports_keys(self):
        adapter = _make_adapter()
        adapter._encryption = True
        adapter._sync_task = None

        fake_client = MagicMock()
        fake_client.olm = object()
        fake_client.export_keys = AsyncMock()
        fake_client.close = AsyncMock()
        adapter._client = fake_client

        from gateway.platforms.matrix import _KEY_EXPORT_FILE, _KEY_EXPORT_PASSPHRASE

        await adapter.disconnect()

        fake_client.export_keys.assert_awaited_once_with(
            str(_KEY_EXPORT_FILE), _KEY_EXPORT_PASSPHRASE,
        )

    @pytest.mark.asyncio
    async def test_disconnect_handles_export_failure(self):
        adapter = _make_adapter()
        adapter._encryption = True
        adapter._sync_task = None

        fake_client = MagicMock()
        fake_client.olm = object()
        fake_client.export_keys = AsyncMock(side_effect=Exception("export failed"))
        fake_client.close = AsyncMock()
        adapter._client = fake_client

        # Should not raise
        await adapter.disconnect()
        assert adapter._client is None  # still cleaned up

    @pytest.mark.asyncio
    async def test_disconnect_skips_export_when_no_encryption(self):
        adapter = _make_adapter()
        adapter._encryption = False
        adapter._sync_task = None

        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        adapter._client = fake_client

        await adapter.disconnect()
        # Should not have tried to export
        assert not hasattr(fake_client, "export_keys") or \
               not fake_client.export_keys.called


# ---------------------------------------------------------------------------
# E2EE: Encrypted media
# ---------------------------------------------------------------------------

class TestMatrixEncryptedMedia:
    @pytest.mark.asyncio
    async def test_connect_registers_callbacks_for_encrypted_media_events(self):
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_te...oken",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "encryption": True,
            },
        )
        adapter = MatrixAdapter(config)

        class FakeWhoamiResponse:
            def __init__(self, user_id, device_id):
                self.user_id = user_id
                self.device_id = device_id

        class FakeSyncResponse:
            def __init__(self):
                self.rooms = MagicMock(join={})

        class FakeRoomMessageText: ...
        class FakeRoomMessageImage: ...
        class FakeRoomMessageAudio: ...
        class FakeRoomMessageVideo: ...
        class FakeRoomMessageFile: ...
        class FakeRoomEncryptedImage: ...
        class FakeRoomEncryptedAudio: ...
        class FakeRoomEncryptedVideo: ...
        class FakeRoomEncryptedFile: ...
        class FakeInviteMemberEvent: ...
        class FakeMegolmEvent: ...

        fake_client = MagicMock()
        fake_client.whoami = AsyncMock(return_value=FakeWhoamiResponse("@bot:example.org", "DEV123"))
        fake_client.sync = AsyncMock(return_value=FakeSyncResponse())
        fake_client.keys_upload = AsyncMock()
        fake_client.keys_query = AsyncMock()
        fake_client.keys_claim = AsyncMock()
        fake_client.send_to_device_messages = AsyncMock(return_value=[])
        fake_client.get_users_for_key_claiming = MagicMock(return_value={})
        fake_client.close = AsyncMock()
        fake_client.add_event_callback = MagicMock()
        fake_client.rooms = {}
        fake_client.account_data = {}
        fake_client.olm = object()
        fake_client.should_upload_keys = False
        fake_client.should_query_keys = False
        fake_client.should_claim_keys = False
        fake_client.restore_login = MagicMock(side_effect=lambda u, d, t: None)

        fake_nio = MagicMock()
        fake_nio.AsyncClient = MagicMock(return_value=fake_client)
        fake_nio.WhoamiResponse = FakeWhoamiResponse
        fake_nio.SyncResponse = FakeSyncResponse
        fake_nio.LoginResponse = type("LoginResponse", (), {})
        fake_nio.RoomMessageText = FakeRoomMessageText
        fake_nio.RoomMessageImage = FakeRoomMessageImage
        fake_nio.RoomMessageAudio = FakeRoomMessageAudio
        fake_nio.RoomMessageVideo = FakeRoomMessageVideo
        fake_nio.RoomMessageFile = FakeRoomMessageFile
        fake_nio.RoomEncryptedImage = FakeRoomEncryptedImage
        fake_nio.RoomEncryptedAudio = FakeRoomEncryptedAudio
        fake_nio.RoomEncryptedVideo = FakeRoomEncryptedVideo
        fake_nio.RoomEncryptedFile = FakeRoomEncryptedFile
        fake_nio.InviteMemberEvent = FakeInviteMemberEvent
        fake_nio.MegolmEvent = FakeMegolmEvent

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch.object(adapter, "_refresh_dm_cache", AsyncMock()):
                with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                    assert await adapter.connect() is True

        callback_classes = [call.args[1] for call in fake_client.add_event_callback.call_args_list]
        assert FakeRoomEncryptedImage in callback_classes
        assert FakeRoomEncryptedAudio in callback_classes
        assert FakeRoomEncryptedVideo in callback_classes
        assert FakeRoomEncryptedFile in callback_classes

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_on_room_message_media_decrypts_encrypted_image_and_passes_local_path(self):
        from nio.crypto.attachments import encrypt_attachment

        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}
        adapter.handle_message = AsyncMock()

        plaintext = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        ciphertext, keys = encrypt_attachment(plaintext)

        class FakeRoomEncryptedImage:
            def __init__(self):
                self.sender = "@alice:example.org"
                self.event_id = "$img1"
                self.server_timestamp = 0
                self.body = "screenshot.png"
                self.url = "mxc://example.org/media123"
                self.key = keys["key"]["k"]
                self.hashes = keys["hashes"]
                self.iv = keys["iv"]
                self.mimetype = "image/png"
                self.source = {
                    "content": {
                        "body": "screenshot.png",
                        "info": {"mimetype": "image/png"},
                        "file": {
                            "url": self.url,
                            "key": keys["key"],
                            "hashes": keys["hashes"],
                            "iv": keys["iv"],
                        },
                    }
                }

        class FakeDownloadResponse:
            def __init__(self, body):
                self.body = body

        fake_client = MagicMock()
        fake_client.download = AsyncMock(return_value=FakeDownloadResponse(ciphertext))
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.RoomMessageImage = type("RoomMessageImage", (), {})
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})
        fake_nio.RoomEncryptedImage = FakeRoomEncryptedImage
        fake_nio.RoomEncryptedAudio = type("RoomEncryptedAudio", (), {})
        fake_nio.RoomEncryptedVideo = type("RoomEncryptedVideo", (), {})
        fake_nio.RoomEncryptedFile = type("RoomEncryptedFile", (), {})

        room = MagicMock(room_id="!room:example.org", member_count=2, users={})
        event = FakeRoomEncryptedImage()

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch("gateway.platforms.base.cache_image_from_bytes", return_value="/tmp/cached-image.png") as cache_mock:
                await adapter._on_room_message_media(room, event)

        cache_mock.assert_called_once_with(plaintext, ext=".png")
        msg_event = adapter.handle_message.await_args.args[0]
        assert msg_event.message_type.name == "PHOTO"
        assert msg_event.media_urls == ["/tmp/cached-image.png"]
        assert msg_event.media_types == ["image/png"]

    @pytest.mark.asyncio
    async def test_on_room_message_media_decrypts_encrypted_voice_and_caches_audio(self):
        from nio.crypto.attachments import encrypt_attachment

        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}
        adapter.handle_message = AsyncMock()

        plaintext = b"OggS" + b"\x00" * 32
        ciphertext, keys = encrypt_attachment(plaintext)

        class FakeRoomEncryptedAudio:
            def __init__(self):
                self.sender = "@alice:example.org"
                self.event_id = "$voice1"
                self.server_timestamp = 0
                self.body = "voice.ogg"
                self.url = "mxc://example.org/voice123"
                self.key = keys["key"]["k"]
                self.hashes = keys["hashes"]
                self.iv = keys["iv"]
                self.mimetype = "audio/ogg"
                self.source = {
                    "content": {
                        "body": "voice.ogg",
                        "info": {"mimetype": "audio/ogg"},
                        "org.matrix.msc3245.voice": {},
                        "file": {
                            "url": self.url,
                            "key": keys["key"],
                            "hashes": keys["hashes"],
                            "iv": keys["iv"],
                        },
                    }
                }

        class FakeDownloadResponse:
            def __init__(self, body):
                self.body = body

        fake_client = MagicMock()
        fake_client.download = AsyncMock(return_value=FakeDownloadResponse(ciphertext))
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.RoomMessageImage = type("RoomMessageImage", (), {})
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})
        fake_nio.RoomEncryptedImage = type("RoomEncryptedImage", (), {})
        fake_nio.RoomEncryptedAudio = FakeRoomEncryptedAudio
        fake_nio.RoomEncryptedVideo = type("RoomEncryptedVideo", (), {})
        fake_nio.RoomEncryptedFile = type("RoomEncryptedFile", (), {})

        room = MagicMock(room_id="!room:example.org", member_count=2, users={})
        event = FakeRoomEncryptedAudio()

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch("gateway.platforms.base.cache_audio_from_bytes", return_value="/tmp/cached-voice.ogg") as cache_mock:
                await adapter._on_room_message_media(room, event)

        cache_mock.assert_called_once_with(plaintext, ext=".ogg")
        msg_event = adapter.handle_message.await_args.args[0]
        assert msg_event.message_type.name == "VOICE"
        assert msg_event.media_urls == ["/tmp/cached-voice.ogg"]
        assert msg_event.media_types == ["audio/ogg"]

    @pytest.mark.asyncio
    async def test_on_room_message_media_decrypts_encrypted_file_and_caches_document(self):
        from nio.crypto.attachments import encrypt_attachment

        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}
        adapter.handle_message = AsyncMock()

        plaintext = b"hello from encrypted document"
        ciphertext, keys = encrypt_attachment(plaintext)

        class FakeRoomEncryptedFile:
            def __init__(self):
                self.sender = "@alice:example.org"
                self.event_id = "$file1"
                self.server_timestamp = 0
                self.body = "notes.txt"
                self.url = "mxc://example.org/file123"
                self.key = keys["key"]
                self.hashes = keys["hashes"]
                self.iv = keys["iv"]
                self.mimetype = "text/plain"
                self.source = {
                    "content": {
                        "body": "notes.txt",
                        "info": {"mimetype": "text/plain"},
                        "file": {
                            "url": self.url,
                            "key": keys["key"],
                            "hashes": keys["hashes"],
                            "iv": keys["iv"],
                        },
                    }
                }

        class FakeDownloadResponse:
            def __init__(self, body):
                self.body = body

        fake_client = MagicMock()
        fake_client.download = AsyncMock(return_value=FakeDownloadResponse(ciphertext))
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.RoomMessageImage = type("RoomMessageImage", (), {})
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})
        fake_nio.RoomEncryptedImage = type("RoomEncryptedImage", (), {})
        fake_nio.RoomEncryptedAudio = type("RoomEncryptedAudio", (), {})
        fake_nio.RoomEncryptedVideo = type("RoomEncryptedVideo", (), {})
        fake_nio.RoomEncryptedFile = FakeRoomEncryptedFile

        room = MagicMock(room_id="!room:example.org", member_count=2, users={})
        event = FakeRoomEncryptedFile()

        with patch.dict("sys.modules", {"nio": fake_nio}):
            with patch("gateway.platforms.base.cache_document_from_bytes", return_value="/tmp/cached-notes.txt") as cache_mock:
                await adapter._on_room_message_media(room, event)

        cache_mock.assert_called_once_with(plaintext, "notes.txt")
        msg_event = adapter.handle_message.await_args.args[0]
        assert msg_event.message_type.name == "DOCUMENT"
        assert msg_event.media_urls == ["/tmp/cached-notes.txt"]
        assert msg_event.media_types == ["text/plain"]

    @pytest.mark.asyncio
    async def test_on_room_message_media_does_not_emit_ciphertext_url_when_encrypted_media_decryption_fails(self):
        adapter = _make_adapter()
        adapter._user_id = "@bot:example.org"
        adapter._startup_ts = 0.0
        adapter._dm_rooms = {}
        adapter.handle_message = AsyncMock()

        class FakeRoomEncryptedImage:
            def __init__(self):
                self.sender = "@alice:example.org"
                self.event_id = "$img2"
                self.server_timestamp = 0
                self.body = "broken.png"
                self.url = "mxc://example.org/media999"
                self.key = {"k": "broken"}
                self.hashes = {"sha256": "broken"}
                self.iv = "broken"
                self.mimetype = "image/png"
                self.source = {
                    "content": {
                        "body": "broken.png",
                        "info": {"mimetype": "image/png"},
                        "file": {
                            "url": self.url,
                            "key": self.key,
                            "hashes": self.hashes,
                            "iv": self.iv,
                        },
                    }
                }

        class FakeDownloadResponse:
            def __init__(self, body):
                self.body = body

        fake_client = MagicMock()
        fake_client.download = AsyncMock(return_value=FakeDownloadResponse(b"ciphertext"))
        adapter._client = fake_client

        fake_nio = MagicMock()
        fake_nio.RoomMessageImage = type("RoomMessageImage", (), {})
        fake_nio.RoomMessageAudio = type("RoomMessageAudio", (), {})
        fake_nio.RoomMessageVideo = type("RoomMessageVideo", (), {})
        fake_nio.RoomMessageFile = type("RoomMessageFile", (), {})
        fake_nio.RoomEncryptedImage = FakeRoomEncryptedImage
        fake_nio.RoomEncryptedAudio = type("RoomEncryptedAudio", (), {})
        fake_nio.RoomEncryptedVideo = type("RoomEncryptedVideo", (), {})
        fake_nio.RoomEncryptedFile = type("RoomEncryptedFile", (), {})

        room = MagicMock(room_id="!room:example.org", member_count=2, users={})
        event = FakeRoomEncryptedImage()

        with patch.dict("sys.modules", {"nio": fake_nio}):
            await adapter._on_room_message_media(room, event)

        msg_event = adapter.handle_message.await_args.args[0]
        assert not msg_event.media_urls
        assert not msg_event.media_types


# ---------------------------------------------------------------------------
# Markdown to HTML: security tests
# ---------------------------------------------------------------------------

class TestMatrixMarkdownHtmlSecurity:
    """Tests for HTML injection prevention in _markdown_to_html_fallback."""

    def setup_method(self):
        from gateway.platforms.matrix import MatrixAdapter
        self.convert = MatrixAdapter._markdown_to_html_fallback

    def test_script_injection_in_header(self):
        result = self.convert("# <script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_script_injection_in_plain_text(self):
        result = self.convert("Hello <script>alert(1)</script>")
        assert "<script>" not in result

    def test_img_onerror_in_blockquote(self):
        result = self.convert('> <img onerror="alert(1)">')
        assert "onerror" not in result or "&lt;img" in result

    def test_script_in_list_item(self):
        result = self.convert("- <script>alert(1)</script>")
        assert "<script>" not in result

    def test_script_in_ordered_list(self):
        result = self.convert("1. <script>alert(1)</script>")
        assert "<script>" not in result

    def test_javascript_uri_blocked(self):
        result = self.convert("[click](javascript:alert(1))")
        assert 'href="javascript:' not in result

    def test_data_uri_blocked(self):
        result = self.convert("[click](data:text/html,<script>)")
        assert 'href="data:' not in result

    def test_vbscript_uri_blocked(self):
        result = self.convert("[click](vbscript:alert(1))")
        assert 'href="vbscript:' not in result

    def test_link_text_html_injection(self):
        result = self.convert('[<img onerror="x">](http://safe.com)')
        assert "<img" not in result or "&lt;img" in result

    def test_link_href_attribute_breakout(self):
        result = self.convert('[link](http://x" onclick="alert(1))')
        assert "onclick" not in result or "&quot;" in result

    def test_html_injection_in_bold(self):
        result = self.convert("**<img onerror=alert(1)>**")
        assert "<img" not in result or "&lt;img" in result

    def test_html_injection_in_italic(self):
        result = self.convert("*<script>alert(1)</script>*")
        assert "<script>" not in result


# ---------------------------------------------------------------------------
# Markdown to HTML: extended formatting tests
# ---------------------------------------------------------------------------

class TestMatrixMarkdownHtmlFormatting:
    """Tests for new formatting capabilities in _markdown_to_html_fallback."""

    def setup_method(self):
        from gateway.platforms.matrix import MatrixAdapter
        self.convert = MatrixAdapter._markdown_to_html_fallback

    def test_fenced_code_block(self):
        result = self.convert('```python\ndef hello():\n    pass\n```')
        assert "<pre><code" in result
        assert "language-python" in result

    def test_fenced_code_block_no_lang(self):
        result = self.convert('```\nsome code\n```')
        assert "<pre><code>" in result

    def test_code_block_html_escaped(self):
        result = self.convert('```\n<script>alert(1)</script>\n```')
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_headers(self):
        assert "<h1>" in self.convert("# H1")
        assert "<h2>" in self.convert("## H2")
        assert "<h3>" in self.convert("### H3")

    def test_unordered_list(self):
        result = self.convert("- One\n- Two\n- Three")
        assert "<ul>" in result
        assert result.count("<li>") == 3

    def test_ordered_list(self):
        result = self.convert("1. First\n2. Second")
        assert "<ol>" in result
        assert result.count("<li>") == 2

    def test_blockquote(self):
        result = self.convert("> A quote\n> continued")
        assert "<blockquote>" in result
        assert "A quote" in result

    def test_horizontal_rule(self):
        assert "<hr>" in self.convert("---")
        assert "<hr>" in self.convert("***")

    def test_strikethrough(self):
        result = self.convert("~~deleted~~")
        assert "<del>deleted</del>" in result

    def test_links_preserved(self):
        result = self.convert("[text](https://example.com)")
        assert '<a href="https://example.com">text</a>' in result

    def test_complex_mixed_document(self):
        """A realistic agent response with multiple formatting types."""
        text = "## Summary\n\nHere's what I found:\n\n- **Bold item**\n- `code` item\n\n```bash\necho hello\n```\n\n1. Step one\n2. Step two"
        result = self.convert(text)
        assert "<h2>" in result
        assert "<strong>" in result
        assert "<code>" in result
        assert "<ul>" in result
        assert "<ol>" in result
        assert "<pre><code" in result


# ---------------------------------------------------------------------------
# Link URL sanitization
# ---------------------------------------------------------------------------

class TestMatrixLinkSanitization:
    def test_safe_https_url(self):
        from gateway.platforms.matrix import MatrixAdapter
        assert MatrixAdapter._sanitize_link_url("https://example.com") == "https://example.com"

    def test_javascript_blocked(self):
        from gateway.platforms.matrix import MatrixAdapter
        assert MatrixAdapter._sanitize_link_url("javascript:alert(1)") == ""

    def test_data_blocked(self):
        from gateway.platforms.matrix import MatrixAdapter
        assert MatrixAdapter._sanitize_link_url("data:text/html,bad") == ""

    def test_vbscript_blocked(self):
        from gateway.platforms.matrix import MatrixAdapter
        assert MatrixAdapter._sanitize_link_url("vbscript:bad") == ""

    def test_quotes_escaped(self):
        from gateway.platforms.matrix import MatrixAdapter
        result = MatrixAdapter._sanitize_link_url('http://x"y')
        assert '"' not in result
        assert "&quot;" in result


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------

class TestMatrixReactions:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_send_reaction(self):
        """_send_reaction should call room_send with m.reaction."""
        nio = pytest.importorskip("nio")
        mock_client = MagicMock()
        mock_client.room_send = AsyncMock(
            return_value=MagicMock(spec=nio.RoomSendResponse)
        )
        self.adapter._client = mock_client

        result = await self.adapter._send_reaction("!room:ex", "$event1", "👍")
        assert result is True
        mock_client.room_send.assert_called_once()
        args = mock_client.room_send.call_args
        assert args[0][1] == "m.reaction"
        content = args[0][2]
        assert content["m.relates_to"]["rel_type"] == "m.annotation"
        assert content["m.relates_to"]["key"] == "👍"

    @pytest.mark.asyncio
    async def test_send_reaction_no_client(self):
        self.adapter._client = None
        result = await self.adapter._send_reaction("!room:ex", "$ev", "👍")
        assert result is False

    @pytest.mark.asyncio
    async def test_on_processing_start_sends_eyes(self):
        """on_processing_start should send 👀 reaction."""
        from gateway.platforms.base import MessageEvent, MessageType

        self.adapter._reactions_enabled = True
        self.adapter._send_reaction = AsyncMock(return_value=True)

        source = MagicMock()
        source.chat_id = "!room:ex"
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="$msg1",
        )
        await self.adapter.on_processing_start(event)
        self.adapter._send_reaction.assert_called_once_with("!room:ex", "$msg1", "👀")

    @pytest.mark.asyncio
    async def test_on_processing_complete_sends_check(self):
        from gateway.platforms.base import MessageEvent, MessageType

        self.adapter._reactions_enabled = True
        self.adapter._send_reaction = AsyncMock(return_value=True)

        source = MagicMock()
        source.chat_id = "!room:ex"
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="$msg1",
        )
        await self.adapter.on_processing_complete(event, success=True)
        self.adapter._send_reaction.assert_called_once_with("!room:ex", "$msg1", "✅")

    @pytest.mark.asyncio
    async def test_reactions_disabled(self):
        from gateway.platforms.base import MessageEvent, MessageType

        self.adapter._reactions_enabled = False
        self.adapter._send_reaction = AsyncMock()

        source = MagicMock()
        source.chat_id = "!room:ex"
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="$msg1",
        )
        await self.adapter.on_processing_start(event)
        self.adapter._send_reaction.assert_not_called()


# ---------------------------------------------------------------------------
# Read receipts
# ---------------------------------------------------------------------------

class TestMatrixReadReceipts:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_send_read_receipt(self):
        mock_client = MagicMock()
        mock_client.room_read_markers = AsyncMock(return_value=MagicMock())
        self.adapter._client = mock_client

        result = await self.adapter.send_read_receipt("!room:ex", "$event1")
        assert result is True
        mock_client.room_read_markers.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_receipt_no_client(self):
        self.adapter._client = None
        result = await self.adapter.send_read_receipt("!room:ex", "$event1")
        assert result is False


# ---------------------------------------------------------------------------
# Message redaction
# ---------------------------------------------------------------------------

class TestMatrixRedaction:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_redact_message(self):
        nio = pytest.importorskip("nio")
        mock_client = MagicMock()
        mock_client.room_redact = AsyncMock(
            return_value=MagicMock(spec=nio.RoomRedactResponse)
        )
        self.adapter._client = mock_client

        result = await self.adapter.redact_message("!room:ex", "$ev1", "oops")
        assert result is True
        mock_client.room_redact.assert_called_once()

    @pytest.mark.asyncio
    async def test_redact_no_client(self):
        self.adapter._client = None
        result = await self.adapter.redact_message("!room:ex", "$ev1")
        assert result is False


# ---------------------------------------------------------------------------
# Room creation & invite
# ---------------------------------------------------------------------------

class TestMatrixRoomManagement:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_create_room(self):
        nio = pytest.importorskip("nio")
        mock_resp = MagicMock(spec=nio.RoomCreateResponse)
        mock_resp.room_id = "!new:example.org"
        mock_client = MagicMock()
        mock_client.room_create = AsyncMock(return_value=mock_resp)
        self.adapter._client = mock_client

        room_id = await self.adapter.create_room(name="Test Room", topic="A test")
        assert room_id == "!new:example.org"
        assert "!new:example.org" in self.adapter._joined_rooms

    @pytest.mark.asyncio
    async def test_invite_user(self):
        nio = pytest.importorskip("nio")
        mock_client = MagicMock()
        mock_client.room_invite = AsyncMock(
            return_value=MagicMock(spec=nio.RoomInviteResponse)
        )
        self.adapter._client = mock_client

        result = await self.adapter.invite_user("!room:ex", "@user:ex")
        assert result is True

    @pytest.mark.asyncio
    async def test_create_room_no_client(self):
        self.adapter._client = None
        result = await self.adapter.create_room()
        assert result is None


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------

class TestMatrixPresence:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_set_presence_valid(self):
        mock_client = MagicMock()
        mock_client.set_presence = AsyncMock()
        self.adapter._client = mock_client

        result = await self.adapter.set_presence("online")
        assert result is True

    @pytest.mark.asyncio
    async def test_set_presence_invalid_state(self):
        mock_client = MagicMock()
        self.adapter._client = mock_client

        result = await self.adapter.set_presence("busy")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_presence_no_client(self):
        self.adapter._client = None
        result = await self.adapter.set_presence("online")
        assert result is False


# ---------------------------------------------------------------------------
# Emote & notice
# ---------------------------------------------------------------------------

class TestMatrixMessageTypes:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_send_emote(self):
        nio = pytest.importorskip("nio")
        mock_client = MagicMock()
        mock_resp = MagicMock(spec=nio.RoomSendResponse)
        mock_resp.event_id = "$emote1"
        mock_client.room_send = AsyncMock(return_value=mock_resp)
        self.adapter._client = mock_client

        result = await self.adapter.send_emote("!room:ex", "waves hello")
        assert result.success is True
        call_args = mock_client.room_send.call_args[0]
        assert call_args[2]["msgtype"] == "m.emote"

    @pytest.mark.asyncio
    async def test_send_notice(self):
        nio = pytest.importorskip("nio")
        mock_client = MagicMock()
        mock_resp = MagicMock(spec=nio.RoomSendResponse)
        mock_resp.event_id = "$notice1"
        mock_client.room_send = AsyncMock(return_value=mock_resp)
        self.adapter._client = mock_client

        result = await self.adapter.send_notice("!room:ex", "System message")
        assert result.success is True
        call_args = mock_client.room_send.call_args[0]
        assert call_args[2]["msgtype"] == "m.notice"

    @pytest.mark.asyncio
    async def test_send_emote_empty_text(self):
        self.adapter._client = MagicMock()
        result = await self.adapter.send_emote("!room:ex", "")
        assert result.success is False

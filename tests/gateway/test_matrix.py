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

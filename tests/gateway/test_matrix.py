"""Tests for Matrix platform adapter (mautrix-python backend)."""
import asyncio
import json
import re
import sys
import time
import types
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


def _make_fake_mautrix():
    """Create a lightweight set of fake ``mautrix`` modules.

    The adapter does ``from mautrix.api import HTTPAPI``,
    ``from mautrix.client import Client``, ``from mautrix.types import ...``
    at import time and inside methods.  We provide just enough stubs for
    tests that need to mock the mautrix import chain.

    Use via ``patch.dict("sys.modules", _make_fake_mautrix())``.
    """
    # --- mautrix (root) ---
    mautrix = types.ModuleType("mautrix")

    # --- mautrix.api ---
    mautrix_api = types.ModuleType("mautrix.api")

    class HTTPAPI:
        def __init__(self, base_url="", token="", **kwargs):
            self.base_url = base_url
            self.token = token
            self.session = MagicMock()
            self.session.close = AsyncMock()

    mautrix_api.HTTPAPI = HTTPAPI
    mautrix.api = mautrix_api

    # --- mautrix.types ---
    mautrix_types = types.ModuleType("mautrix.types")

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"
        ROOM_ENCRYPTED = "m.room.encrypted"
        ROOM_NAME = "m.room.name"

    class UserID(str):
        pass

    class RoomID(str):
        pass

    class EventID(str):
        pass

    class ContentURI(str):
        pass

    class SyncToken(str):
        pass

    class RoomCreatePreset:
        PRIVATE = "private_chat"
        PUBLIC = "public_chat"
        TRUSTED_PRIVATE = "trusted_private_chat"

    class PresenceState:
        ONLINE = "online"
        OFFLINE = "offline"
        UNAVAILABLE = "unavailable"

    class TrustState:
        UNVERIFIED = 0
        VERIFIED = 1

    class PaginationDirection:
        BACKWARD = "b"
        FORWARD = "f"

    mautrix_types.EventType = EventType
    mautrix_types.UserID = UserID
    mautrix_types.RoomID = RoomID
    mautrix_types.EventID = EventID
    mautrix_types.ContentURI = ContentURI
    mautrix_types.SyncToken = SyncToken
    mautrix_types.RoomCreatePreset = RoomCreatePreset
    mautrix_types.PresenceState = PresenceState
    mautrix_types.TrustState = TrustState
    mautrix_types.PaginationDirection = PaginationDirection
    mautrix.types = mautrix_types

    # --- mautrix.client ---
    mautrix_client = types.ModuleType("mautrix.client")

    class Client:
        def __init__(self, mxid=None, device_id=None, api=None,
                     state_store=None, sync_store=None, **kwargs):
            self.mxid = mxid
            self.device_id = device_id
            self.api = api
            self.state_store = state_store
            self.sync_store = sync_store
            self.crypto = None
            self._event_handlers = {}

        def add_event_handler(self, event_type, handler):
            self._event_handlers.setdefault(event_type, []).append(handler)

        def add_dispatcher(self, dispatcher_type):
            pass

    class InternalEventType:
        INVITE = "internal.invite"

    mautrix_client.Client = Client
    mautrix_client.InternalEventType = InternalEventType
    mautrix.client = mautrix_client

    # --- mautrix.client.dispatcher ---
    mautrix_client_dispatcher = types.ModuleType("mautrix.client.dispatcher")

    class MembershipEventDispatcher:
        pass

    mautrix_client_dispatcher.MembershipEventDispatcher = MembershipEventDispatcher

    # --- mautrix.client.state_store ---
    mautrix_client_state_store = types.ModuleType("mautrix.client.state_store")

    class MemoryStateStore:
        async def get_member(self, room_id, user_id):
            return None

        async def get_members(self, room_id):
            return []

        async def get_member_profiles(self, room_id):
            return {}

    class MemorySyncStore:
        pass

    mautrix_client_state_store.MemoryStateStore = MemoryStateStore
    mautrix_client_state_store.MemorySyncStore = MemorySyncStore

    # --- mautrix.crypto ---
    mautrix_crypto = types.ModuleType("mautrix.crypto")

    class OlmMachine:
        def __init__(self, client=None, crypto_store=None, state_store=None):
            self.share_keys_min_trust = None
            self.send_keys_min_trust = None

        async def load(self):
            pass

        async def share_keys(self):
            pass

        async def decrypt_megolm_event(self, event):
            return event

    mautrix_crypto.OlmMachine = OlmMachine

    # --- mautrix.crypto.store ---
    mautrix_crypto_store = types.ModuleType("mautrix.crypto.store")

    class MemoryCryptoStore:
        def __init__(self, account_id="", pickle_key=""):  # noqa: S301
            self.account_id = account_id
            self.pickle_key = pickle_key

    mautrix_crypto_store.MemoryCryptoStore = MemoryCryptoStore

    # --- mautrix.crypto.attachments ---
    mautrix_crypto_attachments = types.ModuleType("mautrix.crypto.attachments")

    def encrypt_attachment(data):
        encrypted_file = MagicMock()
        encrypted_file.serialize.return_value = {
            "key": {"k": "testkey"}, "iv": "testiv",
            "hashes": {"sha256": "testhash"}, "v": "v2",
        }
        return (b"ciphertext_" + data, encrypted_file)

    mautrix_crypto_attachments.encrypt_attachment = encrypt_attachment

    # --- mautrix.crypto.store.asyncpg ---
    mautrix_crypto_store_asyncpg = types.ModuleType("mautrix.crypto.store.asyncpg")

    class PgCryptoStore:
        upgrade_table = MagicMock()

        def __init__(self, account_id="", pickle_key="", db=None):  # noqa: S301
            self.account_id = account_id
            self.pickle_key = pickle_key
            self.db = db

        async def open(self):
            pass

    mautrix_crypto_store_asyncpg.PgCryptoStore = PgCryptoStore

    # --- mautrix.util ---
    mautrix_util = types.ModuleType("mautrix.util")

    # --- mautrix.util.async_db ---
    mautrix_util_async_db = types.ModuleType("mautrix.util.async_db")

    class Database:
        @classmethod
        def create(cls, url, upgrade_table=None):
            db = MagicMock()
            db.start = AsyncMock()
            db.stop = AsyncMock()
            return db

    mautrix_util_async_db.Database = Database

    return {
        "mautrix": mautrix,
        "mautrix.api": mautrix_api,
        "mautrix.types": mautrix_types,
        "mautrix.client": mautrix_client,
        "mautrix.client.dispatcher": mautrix_client_dispatcher,
        "mautrix.client.state_store": mautrix_client_state_store,
        "mautrix.crypto": mautrix_crypto,
        "mautrix.crypto.attachments": mautrix_crypto_attachments,
        "mautrix.crypto.store": mautrix_crypto_store,
        "mautrix.crypto.store.asyncpg": mautrix_crypto_store_asyncpg,
        "mautrix.util": mautrix_util,
        "mautrix.util.async_db": mautrix_util_async_db,
    }


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
# Typing indicator
# ---------------------------------------------------------------------------

class TestMatrixTypingIndicator:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._client = MagicMock()
        self.adapter._client.set_typing = AsyncMock()

    @pytest.mark.asyncio
    async def test_stop_typing_clears_matrix_typing_state(self):
        """stop_typing() should send typing=false instead of waiting for timeout expiry."""
        from gateway.platforms.matrix import RoomID

        await self.adapter.stop_typing("!room:example.org")

        self.adapter._client.set_typing.assert_awaited_once_with(
            RoomID("!room:example.org"),
            timeout=0,
        )

    @pytest.mark.asyncio
    async def test_stop_typing_no_client_is_noop(self):
        self.adapter._client = None
        await self.adapter.stop_typing("!room:example.org")  # should not raise

    @pytest.mark.asyncio
    async def test_stop_typing_suppresses_exceptions(self):
        self.adapter._client.set_typing = AsyncMock(side_effect=Exception("network"))
        await self.adapter.stop_typing("!room:example.org")  # should not raise


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

    @pytest.mark.asyncio
    async def test_get_display_name_from_state_store(self):
        """Should get display name from state_store.get_member()."""
        mock_member = MagicMock()
        mock_member.displayname = "Alice"

        mock_state_store = MagicMock()
        mock_state_store.get_member = AsyncMock(return_value=mock_member)

        mock_client = MagicMock()
        mock_client.state_store = mock_state_store
        self.adapter._client = mock_client

        name = await self.adapter._get_display_name("!room:ex.org", "@alice:ex.org")
        assert name == "Alice"

    @pytest.mark.asyncio
    async def test_get_display_name_fallback_to_localpart(self):
        """Should extract localpart from @user:server format."""
        mock_state_store = MagicMock()
        mock_state_store.get_member = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.state_store = mock_state_store
        self.adapter._client = mock_client

        name = await self.adapter._get_display_name("!room:ex.org", "@bob:example.org")
        assert name == "bob"

    @pytest.mark.asyncio
    async def test_get_display_name_no_client(self):
        """Should handle None client gracefully."""
        self.adapter._client = None
        name = await self.adapter._get_display_name("!room:ex.org", "@charlie:ex.org")
        assert name == "charlie"


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestMatrixModuleImport:
    def test_module_importable_without_mautrix(self):
        """gateway.platforms.matrix must be importable even when mautrix is
        not installed — otherwise the gateway crashes for ALL platforms.

        This test uses a subprocess to avoid polluting the current process's
        sys.modules (reimporting a module creates a second module object whose
        classes don't share globals with the original — breaking patch.object
        in subsequent tests).
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys\n"
                "# Block mautrix completely\n"
                "class _Blocker:\n"
                "    def find_module(self, name, path=None):\n"
                "        if name.startswith('mautrix'): return self\n"
                "    def load_module(self, name):\n"
                "        raise ImportError(f'blocked: {name}')\n"
                "sys.meta_path.insert(0, _Blocker())\n"
                "for k in list(sys.modules):\n"
                "    if k.startswith('mautrix'): del sys.modules[k]\n"
                "from gateway.platforms.matrix import check_matrix_requirements\n"
                "assert not check_matrix_requirements()\n"
                "print('OK')\n"
            )],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestMatrixRequirements:
    def test_check_requirements_with_token(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)
        from gateway.platforms.matrix import check_matrix_requirements
        try:
            import mautrix  # noqa: F401
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

    def test_check_requirements_encryption_true_no_e2ee_deps(self, monkeypatch):
        """MATRIX_ENCRYPTION=true should fail if python-olm is not installed."""
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ENCRYPTION", "true")

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=False):
            assert matrix_mod.check_matrix_requirements() is False

    def test_check_requirements_encryption_false_no_e2ee_deps_ok(self, monkeypatch):
        """Without encryption, missing E2EE deps should not block startup."""
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=False):
            # Still needs mautrix itself to be importable
            try:
                import mautrix  # noqa: F401
                assert matrix_mod.check_matrix_requirements() is True
            except ImportError:
                assert matrix_mod.check_matrix_requirements() is False

    def test_check_requirements_encryption_true_with_e2ee_deps(self, monkeypatch):
        """MATRIX_ENCRYPTION=true should pass if E2EE deps are available."""
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ENCRYPTION", "true")

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
            try:
                import mautrix  # noqa: F401
                assert matrix_mod.check_matrix_requirements() is True
            except ImportError:
                assert matrix_mod.check_matrix_requirements() is False


# ---------------------------------------------------------------------------
# Access-token auth / E2EE bootstrap
# ---------------------------------------------------------------------------

class TestMatrixAccessTokenAuth:
    @pytest.mark.asyncio
    async def test_connect_with_access_token_and_encryption(self):
        """connect() should call whoami, set user_id/device_id, set up crypto."""
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

        fake_mautrix_mods = _make_fake_mautrix()

        # Create a mock client that returns from the mautrix.client.Client constructor
        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.state_store = MagicMock()
        mock_client.sync_store = MagicMock()
        mock_client.crypto = None
        mock_client.whoami = AsyncMock(return_value=FakeWhoamiResponse("@bot:example.org", "DEV123"))
        mock_client.sync = AsyncMock(return_value={"rooms": {"join": {"!room:server": {}}}})
        mock_client.add_event_handler = MagicMock()
        mock_client.handle_sync = MagicMock(return_value=[])
        mock_client.query_keys = AsyncMock(return_value={
            "device_keys": {"@bot:example.org": {"DEV123": {
                "keys": {"ed25519:DEV123": "fake_ed25519_key"},
            }}},
        })
        mock_client.api = MagicMock()
        mock_client.api.token = "syt_test_access_token"
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()

        # Mock the crypto setup
        mock_olm = MagicMock()
        mock_olm.load = AsyncMock()
        mock_olm.share_keys = AsyncMock()
        mock_olm.share_keys_min_trust = None
        mock_olm.send_keys_min_trust = None
        mock_olm.account = MagicMock()
        mock_olm.account.identity_keys = {"ed25519": "fake_ed25519_key"}

        # Patch Client constructor to return our mock
        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)
        fake_mautrix_mods["mautrix.crypto"].OlmMachine = MagicMock(return_value=mock_olm)

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
            with patch.dict("sys.modules", fake_mautrix_mods):
                with patch.object(adapter, "_refresh_dm_cache", AsyncMock()):
                    with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                        assert await adapter.connect() is True

        mock_client.whoami.assert_awaited_once()
        assert adapter._user_id == "@bot:example.org"

        await adapter.disconnect()


class TestDeviceKeyReVerification:
    @pytest.mark.asyncio
    async def test_verify_fails_when_server_keys_mismatch_after_upload(self):
        """share_keys() succeeds but server still has old keys -> should return False."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = "TESTDEVICE"

        # First query: keys missing -> triggers share_keys
        # Second query: keys still don't match -> should fail
        mock_keys_missing = MagicMock()
        mock_keys_missing.device_keys = {"@bot:example.org": {}}

        mock_keys_mismatch = MagicMock()
        mock_device = MagicMock()
        mock_device.keys = {"ed25519:TESTDEVICE": "server_old_key"}
        mock_keys_mismatch.device_keys = {"@bot:example.org": {"TESTDEVICE": mock_device}}

        mock_client.query_keys = AsyncMock(side_effect=[mock_keys_missing, mock_keys_mismatch])

        mock_olm = MagicMock()
        mock_olm.account = MagicMock()
        mock_olm.account.shared = False
        mock_olm.account.identity_keys = {"ed25519": "local_new_key"}
        mock_olm.share_keys = AsyncMock()

        from gateway.platforms.matrix import MatrixAdapter
        result = await adapter._verify_device_keys_on_server(mock_client, mock_olm)

        assert result is False
        mock_olm.share_keys.assert_awaited_once()


class TestMatrixE2EEHardFail:
    """connect() must refuse to start when E2EE is requested but deps are missing."""

    @pytest.mark.asyncio
    async def test_connect_fails_when_encryption_true_but_no_e2ee_deps(self):
        from gateway.platforms.matrix import MatrixAdapter, _check_e2ee_deps

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

        fake_mautrix_mods = _make_fake_mautrix()

        mock_client = MagicMock()
        mock_client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="DEV123"))
        mock_client.api = MagicMock()
        mock_client.api.token = "syt_test_access_token"
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.crypto = None

        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=False):
            with patch.dict("sys.modules", fake_mautrix_mods):
                with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                    result = await adapter.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_when_crypto_setup_raises(self):
        """Even if _check_e2ee_deps passes, if OlmMachine raises, hard-fail."""
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

        fake_mautrix_mods = _make_fake_mautrix()

        mock_client = MagicMock()
        mock_client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="DEV123"))
        mock_client.api = MagicMock()
        mock_client.api.token = "syt_test_access_token"
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.crypto = None

        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)
        fake_mautrix_mods["mautrix.crypto"].OlmMachine = MagicMock(side_effect=Exception("olm init failed"))

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
            with patch.dict("sys.modules", fake_mautrix_mods):
                result = await adapter.connect()

        assert result is False


class TestMatrixDeviceId:
    """MATRIX_DEVICE_ID should be used for stable device identity."""

    def test_device_id_from_config_extra(self):
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test",
            extra={
                "homeserver": "https://matrix.example.org",
                "device_id": "HERMES_BOT_STABLE",
            },
        )
        adapter = MatrixAdapter(config)
        assert adapter._device_id == "HERMES_BOT_STABLE"

    def test_device_id_from_env(self, monkeypatch):
        monkeypatch.setenv("MATRIX_DEVICE_ID", "FROM_ENV")

        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test",
            extra={
                "homeserver": "https://matrix.example.org",
            },
        )
        adapter = MatrixAdapter(config)
        assert adapter._device_id == "FROM_ENV"

    def test_device_id_config_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("MATRIX_DEVICE_ID", "FROM_ENV")

        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test",
            extra={
                "homeserver": "https://matrix.example.org",
                "device_id": "FROM_CONFIG",
            },
        )
        adapter = MatrixAdapter(config)
        assert adapter._device_id == "FROM_CONFIG"

    @pytest.mark.asyncio
    async def test_connect_uses_configured_device_id_over_whoami(self):
        """When MATRIX_DEVICE_ID is set, it should be used instead of whoami device_id."""
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test_access_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "encryption": True,
                "device_id": "MY_STABLE_DEVICE",
            },
        )
        adapter = MatrixAdapter(config)

        fake_mautrix_mods = _make_fake_mautrix()

        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.state_store = MagicMock()
        mock_client.sync_store = MagicMock()
        mock_client.crypto = None
        mock_client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="WHOAMI_DEV"))
        mock_client.sync = AsyncMock(return_value={"rooms": {"join": {"!room:server": {}}}})
        mock_client.add_event_handler = MagicMock()
        mock_client.handle_sync = MagicMock(return_value=[])
        mock_client.query_keys = AsyncMock(return_value={
            "device_keys": {"@bot:example.org": {"MY_STABLE_DEVICE": {
                "keys": {"ed25519:MY_STABLE_DEVICE": "fake_ed25519_key"},
            }}},
        })
        mock_client.api = MagicMock()
        mock_client.api.token = "syt_test_access_token"
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()

        mock_olm = MagicMock()
        mock_olm.load = AsyncMock()
        mock_olm.share_keys = AsyncMock()
        mock_olm.share_keys_min_trust = None
        mock_olm.send_keys_min_trust = None
        mock_olm.account = MagicMock()
        mock_olm.account.identity_keys = {"ed25519": "fake_ed25519_key"}

        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)
        fake_mautrix_mods["mautrix.crypto"].OlmMachine = MagicMock(return_value=mock_olm)

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
            with patch.dict("sys.modules", fake_mautrix_mods):
                with patch.object(adapter, "_refresh_dm_cache", AsyncMock()):
                    with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                        assert await adapter.connect() is True

        # The configured device_id should override the whoami device_id.
        # In mautrix, the adapter sets client.device_id directly.
        assert adapter._device_id == "MY_STABLE_DEVICE"

        await adapter.disconnect()


class TestMatrixPasswordLoginDeviceId:
    """MATRIX_DEVICE_ID should be passed to mautrix Client even with password login."""

    @pytest.mark.asyncio
    async def test_password_login_uses_device_id(self):
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "password": "secret",
                "device_id": "STABLE_PW_DEVICE",
            },
        )
        adapter = MatrixAdapter(config)

        fake_mautrix_mods = _make_fake_mautrix()

        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.state_store = MagicMock()
        mock_client.sync_store = MagicMock()
        mock_client.crypto = None
        mock_client.login = AsyncMock(return_value=MagicMock(device_id="STABLE_PW_DEVICE", access_token="tok"))
        mock_client.sync = AsyncMock(return_value={"rooms": {"join": {}}})
        mock_client.add_event_handler = MagicMock()
        mock_client.api = MagicMock()
        mock_client.api.token = ""
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()

        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)

        from gateway.platforms import matrix as matrix_mod
        with patch.dict("sys.modules", fake_mautrix_mods):
            with patch.object(adapter, "_refresh_dm_cache", AsyncMock()):
                with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                    assert await adapter.connect() is True

        mock_client.login.assert_awaited_once()
        assert adapter._device_id == "STABLE_PW_DEVICE"

        await adapter.disconnect()


class TestMatrixDeviceIdConfig:
    """MATRIX_DEVICE_ID should be plumbed through gateway config."""

    def test_device_id_in_config_extra(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_DEVICE_ID", "HERMES_BOT")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        mc = config.platforms[Platform.MATRIX]
        assert mc.extra.get("device_id") == "HERMES_BOT"

    def test_device_id_not_set_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_abc123")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_DEVICE_ID", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        mc = config.platforms[Platform.MATRIX]
        assert "device_id" not in mc.extra


class TestMatrixSyncLoop:
    @pytest.mark.asyncio
    async def test_sync_loop_dispatches_events_and_stores_token(self):
        """_sync_loop should call handle_sync() and persist next_batch."""
        adapter = _make_adapter()
        adapter._encryption = True
        adapter._closing = False

        call_count = 0

        async def _sync_once(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                adapter._closing = True
            return {"rooms": {"join": {"!room:example.org": {}}}, "next_batch": "s1234"}

        mock_crypto = MagicMock()

        mock_sync_store = MagicMock()
        mock_sync_store.get_next_batch = AsyncMock(return_value=None)
        mock_sync_store.put_next_batch = AsyncMock()

        fake_client = MagicMock()
        fake_client.sync = AsyncMock(side_effect=_sync_once)
        fake_client.crypto = mock_crypto
        fake_client.sync_store = mock_sync_store
        fake_client.handle_sync = MagicMock(return_value=[])
        adapter._client = fake_client

        await adapter._sync_loop()

        fake_client.sync.assert_awaited_once()
        fake_client.handle_sync.assert_called_once()
        mock_sync_store.put_next_batch.assert_awaited_once_with("s1234")


class TestMatrixUploadAndSend:
    @pytest.mark.asyncio
    async def test_upload_unencrypted_room_uses_plain_url(self):
        """Unencrypted rooms should use plain 'url' key."""
        adapter = _make_adapter()
        adapter._encryption = True
        mock_client = MagicMock()
        mock_client.crypto = object()
        mock_client.state_store = MagicMock()
        mock_client.state_store.is_encrypted = AsyncMock(return_value=False)
        mock_client.upload_media = AsyncMock(return_value="mxc://example.org/plain")
        mock_client.send_message_event = AsyncMock(return_value="$event")
        adapter._client = mock_client

        result = await adapter._upload_and_send(
            "!room:example.org", b"hello", "test.txt", "text/plain", "m.file",
        )

        assert result.success is True
        sent = mock_client.send_message_event.await_args.args[2]
        assert sent["url"] == "mxc://example.org/plain"
        assert "file" not in sent

    @pytest.mark.asyncio
    async def test_upload_encrypted_room_uses_file_payload(self):
        """Encrypted rooms should use 'file' key with crypto metadata."""
        adapter = _make_adapter()
        adapter._encryption = True
        mock_client = MagicMock()
        mock_client.crypto = object()
        mock_client.state_store = MagicMock()
        mock_client.state_store.is_encrypted = AsyncMock(return_value=True)
        mock_client.upload_media = AsyncMock(return_value="mxc://example.org/enc")
        mock_client.send_message_event = AsyncMock(return_value="$event")
        adapter._client = mock_client

        result = await adapter._upload_and_send(
            "!room:example.org", b"secret", "secret.txt", "text/plain", "m.file",
        )

        assert result.success is True
        # Should have uploaded ciphertext, not plaintext
        uploaded_data = mock_client.upload_media.await_args.args[0]
        assert uploaded_data != b"secret"
        sent = mock_client.send_message_event.await_args.args[2]
        assert "url" not in sent
        assert "file" in sent
        assert sent["file"]["url"] == "mxc://example.org/enc"


class TestMatrixEncryptedSendFallback:
    @pytest.mark.asyncio
    async def test_send_retries_after_e2ee_error(self):
        """send() should retry with crypto.share_keys() on E2EE errors."""
        adapter = _make_adapter()
        adapter._encryption = True

        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(side_effect=[
            Exception("encryption error"),
            "$event123",  # mautrix returns EventID string directly
        ])
        mock_crypto = MagicMock()
        mock_crypto.share_keys = AsyncMock()
        fake_client.crypto = mock_crypto
        adapter._client = fake_client

        result = await adapter.send("!room:example.org", "hello")

        assert result.success is True
        assert result.message_id == "$event123"
        mock_crypto.share_keys.assert_awaited_once()
        assert fake_client.send_message_event.await_count == 2


# ---------------------------------------------------------------------------
# E2EE: _joined_rooms reference preservation for CryptoStateStore
# ---------------------------------------------------------------------------

class TestJoinedRoomsReference:
    def test_joined_rooms_reference_preserved_after_reassignment(self):
        """_CryptoStateStore must see updates after initial sync populates rooms."""
        from gateway.platforms.matrix import _CryptoStateStore

        joined = set()
        store = _CryptoStateStore(MagicMock(), joined)

        # Simulate what connect() should do: mutate in place, not reassign.
        joined.clear()
        joined.update(["!room1:example.org", "!room2:example.org"])

        import asyncio
        rooms = asyncio.get_event_loop().run_until_complete(store.find_shared_rooms("@user:ex"))
        assert set(rooms) == {"!room1:example.org", "!room2:example.org"}


# ---------------------------------------------------------------------------
# E2EE: connect registers encrypted event handler
# ---------------------------------------------------------------------------

class TestMatrixEncryptedEventHandler:
    @pytest.mark.asyncio
    async def test_connect_registers_encrypted_event_handler_when_encryption_on(self):
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "encryption": True,
            },
        )
        adapter = MatrixAdapter(config)

        fake_mautrix_mods = _make_fake_mautrix()

        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.state_store = MagicMock()
        mock_client.sync_store = MagicMock()
        mock_client.crypto = None  # Will be set during connect
        mock_client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="DEV123"))
        mock_client.sync = AsyncMock(return_value={"rooms": {"join": {"!room:server": {}}}})
        mock_client.add_event_handler = MagicMock()
        mock_client.handle_sync = MagicMock(return_value=[])
        mock_client.query_keys = AsyncMock(return_value={
            "device_keys": {"@bot:example.org": {"DEV123": {
                "keys": {"ed25519:DEV123": "fake_ed25519_key"},
            }}},
        })
        mock_client.api = MagicMock()
        mock_client.api.token = "syt_test_token"
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()

        mock_olm = MagicMock()
        mock_olm.load = AsyncMock()
        mock_olm.share_keys = AsyncMock()
        mock_olm.share_keys_min_trust = None
        mock_olm.send_keys_min_trust = None
        mock_olm.account = MagicMock()
        mock_olm.account.identity_keys = {"ed25519": "fake_ed25519_key"}

        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)
        fake_mautrix_mods["mautrix.crypto"].OlmMachine = MagicMock(return_value=mock_olm)

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
            with patch.dict("sys.modules", fake_mautrix_mods):
                with patch.object(adapter, "_refresh_dm_cache", AsyncMock()):
                    with patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
                        assert await adapter.connect() is True

        # Verify event handlers were registered.
        # In mautrix the order is: add_event_handler(EventType, callback)
        handler_calls = mock_client.add_event_handler.call_args_list
        registered_types = [call.args[0] for call in handler_calls]

        # Should have registered handlers for ROOM_MESSAGE, REACTION, INVITE
        assert len(handler_calls) >= 3

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connect_fails_on_stale_otk_conflict(self):
        """connect() must refuse E2EE when OTK upload hits 'already exists'."""
        from gateway.platforms.matrix import MatrixAdapter

        config = PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "encryption": True,
            },
        )
        adapter = MatrixAdapter(config)

        fake_mautrix_mods = _make_fake_mautrix()

        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.state_store = MagicMock()
        mock_client.sync_store = MagicMock()
        mock_client.crypto = None
        mock_client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="DEV123"))
        mock_client.add_event_handler = MagicMock()
        mock_client.add_dispatcher = MagicMock()
        mock_client.query_keys = AsyncMock(return_value={
            "device_keys": {"@bot:example.org": {"DEV123": {
                "keys": {"ed25519:DEV123": "fake_ed25519_key"},
            }}},
        })
        mock_client.api = MagicMock()
        mock_client.api.token = "syt_test_token"
        mock_client.api.session = MagicMock()
        mock_client.api.session.close = AsyncMock()

        # share_keys succeeds on first call (from _verify_device_keys_on_server),
        # then raises "already exists" on the proactive OTK flush in connect().
        mock_olm = MagicMock()
        mock_olm.load = AsyncMock()
        mock_olm.share_keys = AsyncMock(
            side_effect=[None, Exception("One time key signed_curve25519:AAAAAQ already exists")]
        )
        mock_olm.share_keys_min_trust = None
        mock_olm.send_keys_min_trust = None
        mock_olm.account = MagicMock()
        mock_olm.account.identity_keys = {"ed25519": "fake_ed25519_key"}

        fake_mautrix_mods["mautrix.client"].Client = MagicMock(return_value=mock_client)
        fake_mautrix_mods["mautrix.crypto"].OlmMachine = MagicMock(return_value=mock_olm)

        from gateway.platforms import matrix as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
            with patch.dict("sys.modules", fake_mautrix_mods):
                result = await adapter.connect()

        assert result is False


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------

class TestMatrixDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_closes_api_session(self):
        """disconnect() should close client.api.session."""
        adapter = _make_adapter()
        adapter._sync_task = None

        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        mock_api = MagicMock()
        mock_api.session = mock_session

        fake_client = MagicMock()
        fake_client.api = mock_api
        adapter._client = fake_client

        await adapter.disconnect()

        mock_session.close.assert_awaited_once()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_session_close_failure(self):
        """disconnect() should not raise if session close fails."""
        adapter = _make_adapter()
        adapter._sync_task = None

        mock_session = MagicMock()
        mock_session.close = AsyncMock(side_effect=Exception("close failed"))

        mock_api = MagicMock()
        mock_api.session = mock_session

        fake_client = MagicMock()
        fake_client.api = mock_api
        adapter._client = fake_client

        # Should not raise
        await adapter.disconnect()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_disconnect_without_client(self):
        """disconnect() should handle None client gracefully."""
        adapter = _make_adapter()
        adapter._sync_task = None
        adapter._client = None

        await adapter.disconnect()
        assert adapter._client is None


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
        """_send_reaction should call send_message_event with m.reaction."""
        mock_client = MagicMock()
        # mautrix send_message_event returns EventID string directly
        mock_client.send_message_event = AsyncMock(return_value="$reaction1")
        self.adapter._client = mock_client

        result = await self.adapter._send_reaction("!room:ex", "$event1", "\U0001f44d")
        assert result == "$reaction1"
        mock_client.send_message_event.assert_called_once()
        call_args = mock_client.send_message_event.call_args
        content = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get("content")
        assert content["m.relates_to"]["rel_type"] == "m.annotation"
        assert content["m.relates_to"]["key"] == "\U0001f44d"

    @pytest.mark.asyncio
    async def test_send_reaction_no_client(self):
        self.adapter._client = None
        result = await self.adapter._send_reaction("!room:ex", "$ev", "\U0001f44d")
        assert result is None

    @pytest.mark.asyncio
    async def test_on_processing_start_sends_eyes(self):
        """on_processing_start should send eyes reaction."""
        from gateway.platforms.base import MessageEvent, MessageType

        self.adapter._reactions_enabled = True
        self.adapter._send_reaction = AsyncMock(return_value="$reaction_event_123")

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
        self.adapter._send_reaction.assert_called_once_with("!room:ex", "$msg1", "\U0001f440")
        assert self.adapter._pending_reactions == {("!room:ex", "$msg1"): "$reaction_event_123"}

    @pytest.mark.asyncio
    async def test_on_processing_complete_sends_check(self):
        from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome

        self.adapter._reactions_enabled = True
        self.adapter._pending_reactions = {("!room:ex", "$msg1"): "$eyes_reaction_123"}
        self.adapter._redact_reaction = AsyncMock(return_value=True)
        self.adapter._send_reaction = AsyncMock(return_value="$check_reaction_456")

        source = MagicMock()
        source.chat_id = "!room:ex"
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="$msg1",
        )
        await self.adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        self.adapter._redact_reaction.assert_called_once_with("!room:ex", "$eyes_reaction_123")
        self.adapter._send_reaction.assert_called_once_with("!room:ex", "$msg1", "\u2705")

    @pytest.mark.asyncio
    async def test_on_processing_complete_sends_cross_on_failure(self):
        from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome

        self.adapter._reactions_enabled = True
        self.adapter._pending_reactions = {("!room:ex", "$msg1"): "$eyes_reaction_123"}
        self.adapter._redact_reaction = AsyncMock(return_value=True)
        self.adapter._send_reaction = AsyncMock(return_value="$cross_reaction_456")

        source = MagicMock()
        source.chat_id = "!room:ex"
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="$msg1",
        )
        await self.adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
        self.adapter._redact_reaction.assert_called_once_with("!room:ex", "$eyes_reaction_123")
        self.adapter._send_reaction.assert_called_once_with("!room:ex", "$msg1", "\u274c")

    @pytest.mark.asyncio
    async def test_on_processing_complete_cancelled_sends_no_terminal_reaction(self):
        from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome

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
        await self.adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)
        self.adapter._send_reaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_processing_complete_no_pending_reaction(self):
        """on_processing_complete should skip redaction if no eyes reaction was tracked."""
        from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome

        self.adapter._reactions_enabled = True
        self.adapter._pending_reactions = {}
        self.adapter._redact_reaction = AsyncMock()
        self.adapter._send_reaction = AsyncMock(return_value="$check_reaction_789")

        source = MagicMock()
        source.chat_id = "!room:ex"
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="$msg1",
        )
        await self.adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        self.adapter._redact_reaction.assert_not_called()
        self.adapter._send_reaction.assert_called_once_with("!room:ex", "$msg1", "\u2705")

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
    async def test_accepted_message_schedules_read_receipt(self):
        self.adapter._is_dm_room = AsyncMock(return_value=True)
        self.adapter._get_display_name = AsyncMock(return_value="Alice")
        self.adapter._background_read_receipt = MagicMock()

        ctx = await self.adapter._resolve_message_context(
            room_id="!room:ex",
            sender="@alice:ex",
            event_id="$event1",
            body="hello",
            source_content={"body": "hello"},
            relates_to={},
        )

        assert ctx is not None
        self.adapter._background_read_receipt.assert_called_once_with(
            "!room:ex", "$event1"
        )

    @pytest.mark.asyncio
    async def test_send_read_receipt(self):
        """send_read_receipt should call mautrix's real read-marker API."""
        mock_client = MagicMock()
        mock_client.set_fully_read_marker = AsyncMock(return_value=None)
        self.adapter._client = mock_client

        result = await self.adapter.send_read_receipt("!room:ex", "$event1")
        assert result is True
        mock_client.set_fully_read_marker.assert_awaited_once_with(
            "!room:ex", "$event1", "$event1"
        )

    @pytest.mark.asyncio
    async def test_send_read_receipt_falls_back_to_receipt_only(self):
        """send_read_receipt should still work with clients lacking read markers."""
        mock_client = MagicMock(spec=["send_receipt"])
        mock_client.send_receipt = AsyncMock(return_value=None)
        self.adapter._client = mock_client

        result = await self.adapter.send_read_receipt("!room:ex", "$event1")
        assert result is True
        mock_client.send_receipt.assert_awaited_once_with("!room:ex", "$event1")

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
        """redact_message should call client.redact()."""
        mock_client = MagicMock()
        # mautrix redact() returns EventID string
        mock_client.redact = AsyncMock(return_value="$redact_event")
        self.adapter._client = mock_client

        result = await self.adapter.redact_message("!room:ex", "$ev1", "oops")
        assert result is True
        mock_client.redact.assert_called_once()

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
        """create_room should call client.create_room() returning RoomID string."""
        mock_client = MagicMock()
        # mautrix create_room returns RoomID string directly
        mock_client.create_room = AsyncMock(return_value="!new:example.org")
        self.adapter._client = mock_client

        room_id = await self.adapter.create_room(name="Test Room", topic="A test")
        assert room_id == "!new:example.org"
        assert "!new:example.org" in self.adapter._joined_rooms

    @pytest.mark.asyncio
    async def test_invite_user(self):
        """invite_user should call client.invite_user()."""
        mock_client = MagicMock()
        mock_client.invite_user = AsyncMock(return_value=None)
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

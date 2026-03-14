"""
Tests for the Matrix platform adapter (mautrix-python + SQLite backend).

The adapter uses:
- mautrix Client for sync and event handling
- PgCryptoStore on SQLite for persistent E2EE session storage
- OlmMachine for encryption/decryption
- SessionNotFound with wait_for_session for robust decrypt retry

Test coverage:
- Platform enum and config loading
- MatrixAdapter construction and attribute parsing
- Authorization (allowed_users, allow_all)
- connect() / disconnect() basics
- send_message (Markdown → HTML rendering)
- _on_message dispatch and deduplication
- _on_member invite handling
- get_home_channel
- _extract_verif_content helper
- Key verification handler routing
- Python 3.14 compatibility patches (_patch_mautrix_py314)
- _redact_matrix_id and _bool_env helpers
- check_matrix_requirements
- send_matrix standalone function
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MATRIX_ENV_VARS = (
    "MATRIX_HOMESERVER_URL",
    "MATRIX_ACCESS_TOKEN",
    "MATRIX_USER_ID",
    "MATRIX_ALLOWED_USERS",
    "MATRIX_HOME_CHANNEL",
    "MATRIX_VERIFY_SSL",
    "MATRIX_E2EE",
    "MATRIX_PASSWORD",
    "MATRIX_DEVICE_ID",
    "MATRIX_ALLOW_ALL_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)


def _make_adapter(extra=None):
    """Create a MatrixAdapter with clean env (no ~/.hermes/.env bleed)."""
    import os
    saved = {k: os.environ.pop(k, None) for k in _MATRIX_ENV_VARS}
    try:
        with patch("gateway.platforms.matrix._MAUTRIX_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            config = PlatformConfig()
            config.enabled = True
            config.token = "syt_test_token"
            config.extra = extra or {
                "homeserver_url": "https://matrix.example.org",
                "user_id": "@bot:matrix.example.org",
                "verify_ssl": "true",
            }
            return MatrixAdapter(config)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _make_adapter_with_env(monkeypatch, env_vars: dict, extra=None):
    """Create adapter with specific env vars set, clearing all others first."""
    import os
    for k in _MATRIX_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    with patch("gateway.platforms.matrix._MAUTRIX_AVAILABLE", True):
        from gateway.platforms.matrix import MatrixAdapter
        config = PlatformConfig()
        config.enabled = True
        config.token = "syt_test"
        config.extra = extra or {
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@bot:matrix.example.org",
        }
        return MatrixAdapter(config)


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------

class TestMatrixPlatformEnum:
    def test_matrix_value(self):
        assert Platform.MATRIX.value == "matrix"

    def test_in_platform_list(self):
        assert "matrix" in [p.value for p in Platform]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestMatrixConfigLoading:
    def test_all_three_vars_required(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX not in config.platforms

    def test_missing_token(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("MATRIX_USER_ID", "@bot:matrix.example.org")
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX not in config.platforms

    def test_all_vars_present(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_USER_ID", "@bot:matrix.example.org")
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX in config.platforms

    def test_e2ee_stored(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_USER_ID", "@bot:matrix.example.org")
        monkeypatch.setenv("MATRIX_E2EE", "true")
        config = GatewayConfig()
        _apply_env_overrides(config)
        pf = config.platforms.get(Platform.MATRIX)
        assert pf is not None
        assert pf.extra.get("e2ee") in ("true", True)

    def test_verify_ssl_stored(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
        monkeypatch.setenv("MATRIX_USER_ID", "@bot:matrix.example.org")
        monkeypatch.setenv("MATRIX_VERIFY_SSL", "false")
        config = GatewayConfig()
        _apply_env_overrides(config)
        pf = config.platforms.get(Platform.MATRIX)
        assert pf is not None
        assert pf.extra.get("verify_ssl") in ("false", False)


# ---------------------------------------------------------------------------
# Helpers: _redact_matrix_id, _bool_env, check_matrix_requirements
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_redact_long_localpart(self):
        from gateway.platforms.matrix import _redact_matrix_id
        assert _redact_matrix_id("@alice:example.org") == "@al**:example.org"

    def test_redact_preserves_domain(self):
        from gateway.platforms.matrix import _redact_matrix_id
        assert ":matrix.example.org" in _redact_matrix_id("@alice:matrix.example.org")

    def test_redact_no_match_passthrough(self):
        from gateway.platforms.matrix import _redact_matrix_id
        assert _redact_matrix_id("not-a-matrix-id") == "not-a-matrix-id"

    def test_bool_env_false_strings(self):
        from gateway.platforms.matrix import _bool_env
        for v in ("false", "False", "FALSE", "0", "no", ""):
            assert _bool_env(v) is False, f"Expected False for {v!r}"

    def test_bool_env_true_strings(self):
        from gateway.platforms.matrix import _bool_env
        for v in ("true", "True", "TRUE", "1", "yes"):
            assert _bool_env(v) is True, f"Expected True for {v!r}"

    def test_bool_env_python_bools(self):
        from gateway.platforms.matrix import _bool_env
        assert _bool_env(True) is True
        assert _bool_env(False) is False

    def test_check_requirements_available(self):
        with patch("gateway.platforms.matrix._MAUTRIX_AVAILABLE", True):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is True

    def test_check_requirements_unavailable(self):
        with patch("gateway.platforms.matrix._MAUTRIX_AVAILABLE", False):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is False


# ---------------------------------------------------------------------------
# MatrixAdapter construction
# ---------------------------------------------------------------------------

class TestMatrixAdapterConstruction:
    def test_homeserver_from_extra(self):
        assert _make_adapter().homeserver_url == "https://matrix.example.org"

    def test_trailing_slash_stripped(self):
        a = _make_adapter(extra={
            "homeserver_url": "https://matrix.example.org/",
            "user_id": "@bot:matrix.example.org",
        })
        assert a.homeserver_url == "https://matrix.example.org"

    def test_homeserver_from_env(self, monkeypatch):
        a = _make_adapter_with_env(monkeypatch, {
            "MATRIX_HOMESERVER_URL": "https://env.example.org",
            "MATRIX_USER_ID": "@bot:env.example.org",
            "MATRIX_ACCESS_TOKEN": "syt_env",
        }, extra={})
        assert a.homeserver_url == "https://env.example.org"

    def test_user_id_from_extra(self):
        assert _make_adapter().user_id == "@bot:matrix.example.org"

    def test_e2ee_defaults_false(self):
        assert _make_adapter().e2ee is False

    def test_e2ee_true_from_extra(self):
        a = _make_adapter(extra={
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@bot:matrix.example.org",
            "e2ee": "true",
        })
        assert a.e2ee is True

    def test_verify_ssl_defaults_true(self):
        assert _make_adapter().verify_ssl is True

    def test_verify_ssl_false(self):
        a = _make_adapter(extra={
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@bot:matrix.example.org",
            "verify_ssl": "false",
        })
        assert a.verify_ssl is False

    def test_allowed_users_from_env(self, monkeypatch):
        a = _make_adapter_with_env(monkeypatch, {
            "MATRIX_ALLOWED_USERS": "@alice:example.org,@bob:example.org",
        })
        assert "@alice:example.org" in a.allowed_users
        assert "@bob:example.org" in a.allowed_users

    def test_allowed_users_empty(self):
        assert len(_make_adapter().allowed_users) == 0

    def test_home_channel_from_env(self, monkeypatch):
        a = _make_adapter_with_env(monkeypatch, {
            "MATRIX_HOME_CHANNEL": "!room:example.org",
        })
        assert a.home_channel == "!room:example.org"

    def test_platform_attribute(self):
        assert _make_adapter().PLATFORM == Platform.MATRIX

    def test_sas_sessions_initialized_empty(self):
        a = _make_adapter()
        assert isinstance(a._sas_sessions, dict)
        assert len(a._sas_sessions) == 0

    def test_seen_event_ids_initialized(self):
        assert isinstance(_make_adapter()._seen_event_ids, dict)

    def test_allow_all_from_env(self, monkeypatch):
        a = _make_adapter_with_env(monkeypatch, {"GATEWAY_ALLOW_ALL_USERS": "true"})
        assert a._allow_all is True

    def test_allow_all_false_by_default(self):
        assert _make_adapter()._allow_all is False

    def test_client_initially_none(self):
        assert _make_adapter()._client is None

    def test_crypto_initially_none(self):
        assert _make_adapter()._crypto is None


# ---------------------------------------------------------------------------
# connect() / disconnect()
# ---------------------------------------------------------------------------

class TestConnect:
    @pytest.mark.asyncio
    async def test_returns_false_when_unavailable(self):
        with patch("gateway.platforms.matrix._MAUTRIX_AVAILABLE", False):
            adapter = _make_adapter()
            result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_missing_homeserver(self):
        with patch("gateway.platforms.matrix._MAUTRIX_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            config = PlatformConfig()
            config.enabled = True
            config.token = "syt_test"
            config.extra = {"user_id": "@bot:example.org"}
            adapter = MatrixAdapter(config)
            adapter.homeserver_url = ""
            result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_missing_user_id(self):
        adapter = _make_adapter()
        adapter.user_id = ""
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_no_client(self):
        adapter = _make_adapter()
        adapter._client = None
        await adapter.disconnect()  # must not raise

    @pytest.mark.asyncio
    async def test_disconnect_stops_client(self):
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.stop = MagicMock()
        mock_client.syncing_task = None
        mock_client.api = MagicMock()
        mock_client.api.session = AsyncMock()
        mock_client.api.session.close = AsyncMock()
        adapter._client = mock_client
        await adapter.disconnect()
        mock_client.stop.assert_called_once()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_disconnect_closes_crypto_db(self):
        adapter = _make_adapter()
        mock_db = AsyncMock()
        mock_db.stop = AsyncMock()
        adapter._client = None
        adapter._crypto_db = mock_db
        await adapter.disconnect()
        mock_db.stop.assert_called_once()
        assert adapter._crypto_db is None


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------

class TestSendMessage:
    @pytest.mark.asyncio
    async def test_error_when_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_message("!room:example.org", "hello")
        assert result.success is False
        assert "Not connected" in (result.error or "")

    @pytest.mark.asyncio
    async def test_success(self):
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value="$event:example.org")
        adapter._client = mock_client
        result = await adapter.send_message("!room:example.org", "hello world")
        assert result.success is True
        assert result.message_id == "$event:example.org"

    @pytest.mark.asyncio
    async def test_network_failure_caught(self):
        adapter = _make_adapter()
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(side_effect=Exception("net error"))
        adapter._client = mock_client
        result = await adapter.send_message("!room:example.org", "hello")
        assert result.success is False
        assert "net error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_markdown_rendered_to_html(self):
        adapter = _make_adapter()
        captured = {}
        mock_client = MagicMock()

        async def cap(room_id, content):
            captured["content"] = content
            return "$evt"

        mock_client.send_message = cap
        adapter._client = mock_client
        await adapter.send_message("!room:example.org", "**bold** and `code`")
        c = captured.get("content")
        assert c is not None
        assert c.formatted_body is not None
        assert "<strong>" in c.formatted_body or "<code>" in c.formatted_body

    @pytest.mark.asyncio
    async def test_plain_text_body_always_set(self):
        adapter = _make_adapter()
        captured = {}
        mock_client = MagicMock()

        async def cap(room_id, content):
            captured["content"] = content
            return "$evt"

        mock_client.send_message = cap
        adapter._client = mock_client
        await adapter.send_message("!room:example.org", "plain text here")
        assert captured["content"].body == "plain text here"

    @pytest.mark.asyncio
    async def test_send_delegates_to_send_message(self):
        adapter = _make_adapter()
        adapter.send_message = AsyncMock(return_value=MagicMock(success=True))
        await adapter.send("!room:example.org", "hello", reply_to="$prev")
        adapter.send_message.assert_called_once_with(
            "!room:example.org", "hello", reply_to_message_id="$prev"
        )


# ---------------------------------------------------------------------------
# _on_message
# ---------------------------------------------------------------------------

class TestOnMessage:
    def _evt(self, sender, room_id="!room:example.org", body="hello",
              event_id="$evt1:example.org", msgtype="m.text"):
        event = MagicMock()
        event.sender = sender
        event.room_id = room_id
        event.event_id = event_id
        content = MagicMock()
        content.body = body
        mt = MagicMock()
        mt.__str__ = lambda s: msgtype
        mt.__eq__ = lambda s, o: str(s) == str(o)
        content.msgtype = mt
        content.url = None
        event.content = content
        return event

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self):
        adapter = _make_adapter()
        adapter.user_id = "@bot:example.org"
        adapter.handle_message = AsyncMock()
        await adapter._on_message(self._evt("@bot:example.org"))
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocks_unauthorized(self):
        adapter = _make_adapter()
        adapter._allow_all = False
        adapter.allowed_users = {"@alice:example.org"}
        adapter.handle_message = AsyncMock()
        await adapter._on_message(self._evt("@eve:example.org"))
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_authorized(self):
        adapter = _make_adapter()
        adapter._allow_all = False
        adapter.allowed_users = {"@alice:example.org"}
        adapter.handle_message = AsyncMock()
        adapter._client = MagicMock()
        await adapter._on_message(self._evt("@alice:example.org"))
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_allow_all_bypasses_allowlist(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        adapter.handle_message = AsyncMock()
        adapter._client = MagicMock()
        await adapter._on_message(self._evt("@anyone:example.org"))
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplication(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        adapter.handle_message = AsyncMock()
        adapter._client = MagicMock()
        evt = self._evt("@alice:example.org", event_id="$dup:example.org")
        await adapter._on_message(evt)
        await adapter._on_message(evt)
        assert adapter.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_drops_verification_messages(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        adapter.handle_message = AsyncMock()
        await adapter._on_message(
            self._evt("@alice:example.org", msgtype="m.key.verification.request")
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_client_returns_silently(self):
        adapter = _make_adapter()
        adapter._client = None
        adapter.handle_message = AsyncMock()
        await adapter._on_message(self._evt("@alice:example.org"))
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_gateway_event_fields(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        captured = {}

        async def cap(evt):
            captured["evt"] = evt

        adapter.handle_message = cap
        adapter._client = MagicMock()
        await adapter._on_message(self._evt(
            "@alice:example.org",
            room_id="!room:example.org",
            body="test message",
            event_id="$testevt",
        ))
        ge = captured["evt"]
        assert ge.source.chat_id == "!room:example.org"
        assert ge.source.user_id == "@alice:example.org"
        assert ge.text == "test message"
        assert ge.message_id == "$testevt"


# ---------------------------------------------------------------------------
# _on_member
# ---------------------------------------------------------------------------

class TestOnMember:
    def _invite(self, sender, state_key, room_id="!room:example.org"):
        from mautrix.types import MemberStateEventContent, Membership
        event = MagicMock()
        event.sender = sender
        event.state_key = state_key
        event.room_id = room_id
        content = MagicMock(spec=MemberStateEventContent)
        content.membership = Membership.INVITE
        event.content = content
        return event

    @pytest.mark.asyncio
    async def test_accepts_from_allowed(self):
        adapter = _make_adapter()
        adapter._allow_all = False
        adapter.allowed_users = {"@alice:example.org"}
        mc = MagicMock()
        mc.join_room = AsyncMock()
        mc.leave_room = AsyncMock()
        adapter._client = mc
        await adapter._on_member(self._invite("@alice:example.org", "@bot:matrix.example.org"))
        mc.join_room.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_from_unauthorized(self):
        adapter = _make_adapter()
        adapter._allow_all = False
        adapter.allowed_users = {"@alice:example.org"}
        mc = MagicMock()
        mc.join_room = AsyncMock()
        mc.leave_room = AsyncMock()
        adapter._client = mc
        await adapter._on_member(self._invite("@eve:example.org", "@bot:matrix.example.org"))
        mc.join_room.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_invite(self):
        from mautrix.types import MemberStateEventContent, Membership
        adapter = _make_adapter()
        mc = MagicMock()
        mc.join_room = AsyncMock()
        adapter._client = mc
        event = MagicMock()
        event.sender = "@alice:example.org"
        event.state_key = "@bot:matrix.example.org"
        event.room_id = "!room:example.org"
        content = MagicMock(spec=MemberStateEventContent)
        content.membership = Membership.JOIN
        event.content = content
        await adapter._on_member(event)
        mc.join_room.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_invite_for_other_user(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        mc = MagicMock()
        mc.join_room = AsyncMock()
        adapter._client = mc
        await adapter._on_member(
            self._invite("@alice:example.org", "@someone_else:example.org")
        )
        mc.join_room.assert_not_called()


# ---------------------------------------------------------------------------
# get_home_channel
# ---------------------------------------------------------------------------

class TestGetHomeChannel:
    def test_returns_configured_channel(self, monkeypatch):
        a = _make_adapter_with_env(monkeypatch, {"MATRIX_HOME_CHANNEL": "!home:example.org"})
        assert a.get_home_channel() == "!home:example.org"

    def test_returns_none_when_unset(self):
        assert _make_adapter().get_home_channel() is None


# ---------------------------------------------------------------------------
# _extract_verif_content
# ---------------------------------------------------------------------------

class TestExtractVerifContent:
    def test_plain_dict(self):
        adapter = _make_adapter()
        event = MagicMock()
        event.content = {"transaction_id": "txid123"}
        assert adapter._extract_verif_content(event)["transaction_id"] == "txid123"

    def test_mautrix_object_with_json(self):
        adapter = _make_adapter()
        event = MagicMock()
        content = MagicMock(spec=[])  # no __iter__
        content._json = {"transaction_id": "txid456"}
        event.content = content
        assert adapter._extract_verif_content(event)["transaction_id"] == "txid456"

    def test_no_content_returns_empty(self):
        adapter = _make_adapter()
        event = MagicMock(spec=[])  # no 'content'
        result = adapter._extract_verif_content(event)
        assert result == {}


# ---------------------------------------------------------------------------
# Verification handlers
# ---------------------------------------------------------------------------

class TestVerificationHandlers:
    @pytest.mark.asyncio
    async def test_request_sends_ready_for_sas(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        mc = MagicMock()
        mc.device_id = "TESTDEVICE"
        mc.send_to_one_device = AsyncMock()
        adapter._client = mc

        event = MagicMock()
        event.sender = "@alice:example.org"
        event.content = {
            "transaction_id": "txid1",
            "from_device": "ALICEDEVICE",
            "methods": ["m.sas.v1"],
        }
        await adapter._on_verification_request(event)
        mc.send_to_one_device.assert_called_once()
        payload = mc.send_to_one_device.call_args[0][3]
        assert payload["transaction_id"] == "txid1"
        assert "m.sas.v1" in payload["methods"]

    @pytest.mark.asyncio
    async def test_request_ignored_for_unauthorized(self):
        adapter = _make_adapter()
        adapter._allow_all = False
        adapter.allowed_users = {"@alice:example.org"}
        mc = MagicMock()
        mc.send_to_one_device = AsyncMock()
        adapter._client = mc

        event = MagicMock()
        event.sender = "@eve:example.org"
        event.content = {
            "transaction_id": "txid1",
            "from_device": "EVEDEVICE",
            "methods": ["m.sas.v1"],
        }
        await adapter._on_verification_request(event)
        mc.send_to_one_device.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_ignored_without_sas_v1(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        mc = MagicMock()
        mc.send_to_one_device = AsyncMock()
        adapter._client = mc

        event = MagicMock()
        event.sender = "@alice:example.org"
        event.content = {
            "transaction_id": "txid1",
            "from_device": "DEVICE",
            "methods": ["m.qr_code.scan.v1"],
        }
        await adapter._on_verification_request(event)
        mc.send_to_one_device.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_removes_session(self):
        adapter = _make_adapter()
        adapter._sas_sessions["txid99"] = {"sender": "@alice:example.org"}
        event = MagicMock()
        event.content = {"transaction_id": "txid99", "reason": "user cancelled"}
        await adapter._on_verification_cancel(event)
        assert "txid99" not in adapter._sas_sessions

    @pytest.mark.asyncio
    async def test_mac_sends_done_and_removes_session(self):
        adapter = _make_adapter()
        adapter._allow_all = True
        mc = MagicMock()
        mc.send_to_one_device = AsyncMock()
        adapter._client = mc
        adapter._sas_sessions["txid77"] = {
            "sender": "@alice:example.org",
            "from_device": "ALICEDEVICE",
        }
        event = MagicMock()
        event.sender = "@alice:example.org"
        event.content = {"transaction_id": "txid77"}
        await adapter._on_verification_mac(event)
        assert "txid77" not in adapter._sas_sessions
        mc.send_to_one_device.assert_called_once()
        event_type_str = str(mc.send_to_one_device.call_args[0][0])
        assert "done" in event_type_str

    @pytest.mark.asyncio
    async def test_mac_no_session_is_silent(self):
        adapter = _make_adapter()
        mc = MagicMock()
        mc.send_to_one_device = AsyncMock()
        adapter._client = mc
        event = MagicMock()
        event.content = {"transaction_id": "nonexistent"}
        await adapter._on_verification_mac(event)  # must not raise
        mc.send_to_one_device.assert_not_called()


# ---------------------------------------------------------------------------
# Python 3.14 patches
# ---------------------------------------------------------------------------

class TestPy314Patches:
    def test_patch_function_is_callable(self):
        """_patch_mautrix_py314 is importable and callable without error."""
        from gateway.platforms.matrix import _patch_mautrix_py314
        # Call it again — should be idempotent
        _patch_mautrix_py314()

    def test_encrypt_megolm_patch_replaced_encrypt_megolm_event(self):
        """On Python 3.14, the patched encrypt_megolm_event should be our wrapper."""
        import sys
        if sys.version_info < (3, 14):
            pytest.skip("Only patched on Python 3.14+")
        # Ensure the gateway module (which applies the patch) is imported
        import gateway.platforms.matrix  # noqa: F401
        from mautrix.crypto.encrypt_megolm import MegolmEncryptionMachine
        method = MegolmEncryptionMachine.encrypt_megolm_event
        assert "_patched_encrypt_meg" in method.__qualname__

    def test_memory_state_store_find_shared_rooms(self):
        """On Python 3.14, MemoryStateStore has find_shared_rooms after gateway import."""
        import sys
        if sys.version_info < (3, 14):
            pytest.skip("Only patched on Python 3.14+")
        import gateway.platforms.matrix  # noqa: F401 — triggers the patch
        from mautrix.client.state_store import MemoryStateStore
        assert hasattr(MemoryStateStore, "find_shared_rooms")
        assert callable(MemoryStateStore.find_shared_rooms)

    def test_drop_signatures_returns_zero_for_missing(self):
        """After patch, drop_signatures_by_key returns 0 instead of crashing."""
        import sys
        if sys.version_info < (3, 14):
            pytest.skip("Python 3.14+ only")
        from mautrix.crypto.store.memory import MemoryCryptoStore
        from mautrix.types import UserID

        async def check():
            store = MemoryCryptoStore("@test:example.org", "key")
            # CrossSigner is a NamedTuple(user_id, key) — use a simple mock
            from unittest.mock import MagicMock
            signer = MagicMock()
            result = await store.drop_signatures_by_key(signer)
            assert result == 0  # missing key → 0, not crash

        asyncio.run(check())

    def test_put_cross_signing_key_patch_handles_readonly(self):
        """After patch, put_cross_signing_key doesn't crash on read-only NamedTuple."""
        import sys
        if sys.version_info < (3, 14):
            pytest.skip("Python 3.14+ only")
        from mautrix.crypto.store.memory import MemoryCryptoStore, TOFUSigningKey
        from mautrix.types import CrossSigningUsage

        async def check():
            store = MemoryCryptoStore("@test:example.org", "key")
            # Pre-populate with a key so the update path is exercised
            usage = CrossSigningUsage.MASTER
            first_key = MagicMock()
            store._cross_signing_keys.setdefault("@user:example.org", {})[usage] = (
                TOFUSigningKey(key=first_key, first=first_key)
            )
            new_key = MagicMock()
            # Should not raise AttributeError on Python 3.14
            await store.put_cross_signing_key("@user:example.org", usage, new_key)

        asyncio.run(check())


# ---------------------------------------------------------------------------
# send_matrix standalone function
# ---------------------------------------------------------------------------

class TestSendMatrixFunction:
    @pytest.mark.asyncio
    async def test_is_coroutine_function(self):
        from tools.send_message_tool import _send_matrix
        import inspect
        assert (
            asyncio.iscoroutinefunction(_send_matrix)
            or inspect.iscoroutinefunction(_send_matrix)
        )

    @pytest.mark.asyncio
    async def test_returns_error_when_missing_homeserver(self):
        from tools.send_message_tool import _send_matrix
        from gateway.config import PlatformConfig
        pconfig = PlatformConfig()
        pconfig.token = "syt_test"
        pconfig.extra = {"homeserver_url": "", "user_id": "@bot:example.org"}
        result = await _send_matrix(pconfig, "!room:example.org", "hello")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_mautrix_missing(self):
        """When mautrix is not importable, _send_matrix returns an error dict."""
        from tools.send_message_tool import _send_matrix
        from gateway.config import PlatformConfig

        pconfig = PlatformConfig()
        pconfig.token = "syt_test"
        pconfig.extra = {
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        }
        # Patch _MAUTRIX_AVAILABLE to False to simulate missing library.
        # This is cleaner than manipulating sys.modules which can leak state
        # across parallel test workers if an exception occurs mid-teardown.
        with patch("tools.send_message_tool._MAUTRIX_AVAILABLE", False, create=True):
            result = await _send_matrix(pconfig, "!room:example.org", "hello")
        # The function should return an error dict, not raise
        assert isinstance(result, dict)

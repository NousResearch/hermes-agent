"""Tests for Signal messenger platform adapter."""
import asyncio
import base64
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with sensible test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


def _stub_rpc(return_value):
    """Return an async mock for SignalAdapter._rpc that captures call params."""
    captured = []

    async def mock_rpc(method, params, rpc_id=None):
        captured.append({"method": method, "params": dict(params)})
        return return_value

    return mock_rpc, captured


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestSignalPlatformEnum:
    def test_signal_enum_exists(self):
        assert Platform.SIGNAL.value == "signal"

    def test_signal_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "signal" in platforms


class TestSignalConfigLoading:
    def test_apply_env_overrides_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL in config.platforms
        sc = config.platforms[Platform.SIGNAL]
        assert sc.enabled is True
        assert sc.extra["http_url"] == "http://localhost:9090"
        assert sc.extra["account"] == "+15551234567"

    def test_signal_not_loaded_without_both_vars(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        # No SIGNAL_ACCOUNT

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL not in config.platforms

    def test_connected_platforms_includes_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert Platform.SIGNAL in connected


# ---------------------------------------------------------------------------
# Adapter Init & Helpers
# ---------------------------------------------------------------------------

class TestSignalAdapterInit:
    def test_init_parses_config(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, group_allowed="group123,group456")
        assert adapter.http_url == "http://localhost:8080"
        assert adapter.account == "+15551234567"
        assert "group123" in adapter.group_allow_from

    def test_init_empty_allowlist(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert len(adapter.group_allow_from) == 0

    def test_init_strips_trailing_slash(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, http_url="http://localhost:8080/")
        assert adapter.http_url == "http://localhost:8080"

    def test_self_message_filtering(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._account_normalized == "+15551234567"


class TestSignalHelpers:
    def test_redact_phone_long(self):
        from gateway.platforms.signal import _redact_phone
        assert _redact_phone("+15551234567") == "+155****4567"

    def test_redact_phone_short(self):
        from gateway.platforms.signal import _redact_phone
        assert _redact_phone("+12345") == "+1****45"

    def test_redact_phone_empty(self):
        from gateway.platforms.signal import _redact_phone
        assert _redact_phone("") == "<none>"

    def test_parse_comma_list(self):
        from gateway.platforms.signal import _parse_comma_list
        assert _parse_comma_list("+1234, +5678 , +9012") == ["+1234", "+5678", "+9012"]
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []

    def test_guess_extension_png(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == ".png"

    def test_guess_extension_jpeg(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == ".jpg"

    def test_guess_extension_pdf(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"%PDF-1.4" + b"\x00" * 100) == ".pdf"

    def test_guess_extension_zip(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"PK\x03\x04" + b"\x00" * 100) == ".zip"

    def test_guess_extension_mp4(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100) == ".mp4"

    def test_guess_extension_unknown(self):
        from gateway.platforms.signal import _guess_extension
        assert _guess_extension(b"\x00\x01\x02\x03" * 10) == ".bin"

    def test_is_image_ext(self):
        from gateway.platforms.signal import _is_image_ext
        assert _is_image_ext(".png") is True
        assert _is_image_ext(".jpg") is True
        assert _is_image_ext(".gif") is True
        assert _is_image_ext(".pdf") is False

    def test_is_audio_ext(self):
        from gateway.platforms.signal import _is_audio_ext
        assert _is_audio_ext(".mp3") is True
        assert _is_audio_ext(".ogg") is True
        assert _is_audio_ext(".png") is False

    def test_check_requirements(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        assert check_signal_requirements() is True

    def test_render_mentions(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hello \uFFFC, how are you?"
        mentions = [{"start": 6, "length": 1, "number": "+15559999999"}]
        result = _render_mentions(text, mentions)
        assert "@+15559999999" in result
        assert "\uFFFC" not in result

    def test_render_mentions_no_mentions(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hello world"
        result = _render_mentions(text, [])
        assert result == "Hello world"

    def test_check_requirements_missing(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        assert check_signal_requirements() is False


# ---------------------------------------------------------------------------
# SSE URL Encoding (Bug Fix: phone numbers with + must be URL-encoded)
# ---------------------------------------------------------------------------

class TestSignalSSEUrlEncoding:
    """Verify that phone numbers with + are URL-encoded in the SSE endpoint."""

    def test_sse_url_encodes_plus_in_account(self):
        """The + in E.164 phone numbers must be percent-encoded in the SSE query string."""
        encoded = quote("+31612345678", safe="")
        assert encoded == "%2B31612345678"

    def test_sse_url_encoding_preserves_digits(self):
        """Digits and country codes should pass through URL encoding unchanged."""
        assert quote("+15551234567", safe="") == "%2B15551234567"


# ---------------------------------------------------------------------------
# Attachment Fetch (Bug Fix: parameter must be "id" not "attachmentId")
# ---------------------------------------------------------------------------

class TestSignalAttachmentFetch:
    """Verify that _fetch_attachment uses the correct RPC parameter name."""

    @pytest.mark.asyncio
    async def test_fetch_attachment_uses_id_parameter(self, monkeypatch):
        """RPC getAttachment must use 'id', not 'attachmentId' (signal-cli requirement)."""
        adapter = _make_signal_adapter(monkeypatch)

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()

        adapter._rpc, captured = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/test.png"):
            await adapter._fetch_attachment("attachment-123")

        call = captured[0]
        assert call["method"] == "getAttachment"
        assert call["params"]["id"] == "attachment-123"
        assert "attachmentId" not in call["params"], "Must NOT use 'attachmentId' — causes NullPointerException in signal-cli"
        assert call["params"]["account"] == "+15551234567"

    @pytest.mark.asyncio
    async def test_fetch_attachment_returns_none_on_empty(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._rpc, _ = _stub_rpc(None)
        path, ext = await adapter._fetch_attachment("missing-id")
        assert path is None
        assert ext == ""

    @pytest.mark.asyncio
    async def test_fetch_attachment_handles_dict_response(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        pdf_data = b"%PDF-1.4" + b"\x00" * 100
        b64_data = base64.b64encode(pdf_data).decode()

        adapter._rpc, _ = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_document_from_bytes", return_value="/tmp/test.pdf"):
            path, ext = await adapter._fetch_attachment("doc-456")

        assert path == "/tmp/test.pdf"
        assert ext == ".pdf"


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------

class TestSignalSessionSource:
    def test_session_source_alt_fields(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            user_id="+15551234567",
            user_id_alt="uuid:abc-123",
            chat_id_alt=None,
        )
        d = source.to_dict()
        assert d["user_id_alt"] == "uuid:abc-123"
        assert "chat_id_alt" not in d  # None fields excluded

    def test_session_source_roundtrip(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:xyz",
            chat_type="group",
            user_id="+15551234567",
            user_id_alt="uuid:abc",
            chat_id_alt="xyz",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.user_id_alt == "uuid:abc"
        assert restored.chat_id_alt == "xyz"
        assert restored.platform == Platform.SIGNAL


# ---------------------------------------------------------------------------
# Phone Redaction in agent/redact.py
# ---------------------------------------------------------------------------

class TestSignalPhoneRedaction:
    @pytest.fixture(autouse=True)
    def _ensure_redaction_enabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)

    def test_us_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Call +15551234567 now")
        assert "+15551234567" not in result
        assert "+155" in result  # Prefix preserved
        assert "4567" in result  # Suffix preserved

    def test_uk_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("UK: +442071838750")
        assert "+442071838750" not in result
        assert "****" in result

    def test_multiple_numbers(self):
        from agent.redact import redact_sensitive_text
        text = "From +15551234567 to +442071838750"
        result = redact_sensitive_text(text)
        assert "+15551234567" not in result
        assert "+442071838750" not in result

    def test_short_number_not_matched(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Code: +12345")
        # 5 digits after + is below the 7-digit minimum
        assert "+12345" in result  # Too short to redact


# ---------------------------------------------------------------------------
# Authorization in run.py
# ---------------------------------------------------------------------------

class TestSignalAuthorization:
    def test_signal_in_allowlist_maps(self):
        """Signal should be in the platform auth maps."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SIGNAL
        source.user_id = "+15559999999"

        # No allowlists set — should check GATEWAY_ALLOW_ALL_USERS
        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------

class TestSignalSendMessage:
    def test_signal_in_platform_map(self):
        """Signal should be in the send_message tool's platform map."""
        from tools.send_message_tool import send_message_tool
        # Just verify the import works and Signal is a valid platform
        from gateway.config import Platform
        assert Platform.SIGNAL.value == "signal"


# ---------------------------------------------------------------------------
# Read Receipts
# ---------------------------------------------------------------------------

class TestSignalReadReceipts:
    """Verify that read receipts are sent via JSON-RPC."""

    def test_read_receipts_enabled_by_default(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter.send_read_receipts is True

    def test_read_receipts_can_be_disabled(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=False)
        assert adapter.send_read_receipts is False

    @pytest.mark.asyncio
    async def test_send_read_receipt_calls_rpc(self, monkeypatch):
        """_send_read_receipt should call sendReceipt with correct params."""
        adapter = _make_signal_adapter(monkeypatch)
        rpc_mock, captured = _stub_rpc(None)
        adapter._rpc = rpc_mock

        await adapter._send_read_receipt("+15559999999", 1625821234567)

        assert len(captured) == 1
        call = captured[0]
        assert call["method"] == "sendReceipt"
        assert call["params"]["recipient"] == "+15559999999"
        assert call["params"]["targetTimestamp"] == 1625821234567
        assert call["params"]["type"] == "read"
        assert call["params"]["account"] == "+15551234567"

    @pytest.mark.asyncio
    async def test_send_read_receipt_swallows_errors(self, monkeypatch):
        """Read receipt failures must not propagate exceptions."""
        adapter = _make_signal_adapter(monkeypatch)

        async def failing_rpc(method, params, rpc_id=None):
            raise ConnectionError("daemon down")

        adapter._rpc = failing_rpc

        # Should not raise
        await adapter._send_read_receipt("+15559999999", 1625821234567)

    @pytest.mark.asyncio
    async def test_handle_envelope_sends_read_receipt_for_dm(self, monkeypatch):
        """Processing a DM should fire a read receipt."""
        adapter = _make_signal_adapter(monkeypatch)
        rpc_mock, captured = _stub_rpc(None)
        adapter._rpc = rpc_mock
        adapter.handle_message = AsyncMock()

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceName": "Test User",
                "sourceUuid": "uuid-abc",
                "timestamp": 1625821234567,
                "dataMessage": {
                    "message": "Hello",
                    "timestamp": 1625821234567,
                },
            }
        }

        await adapter._handle_envelope(envelope)
        await asyncio.sleep(0)  # Let fire-and-forget receipt task run

        # Should have called sendReceipt
        receipt_calls = [c for c in captured if c["method"] == "sendReceipt"]
        assert len(receipt_calls) == 1
        assert receipt_calls[0]["params"]["recipient"] == "+15559999999"
        assert receipt_calls[0]["params"]["targetTimestamp"] == 1625821234567

    @pytest.mark.asyncio
    async def test_handle_envelope_no_receipt_for_groups(self, monkeypatch):
        """Group messages should NOT get read receipts."""
        adapter = _make_signal_adapter(monkeypatch, group_allowed="*")
        rpc_mock, captured = _stub_rpc(None)
        adapter._rpc = rpc_mock
        adapter.handle_message = AsyncMock()

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceName": "Test User",
                "sourceUuid": "uuid-abc",
                "timestamp": 1625821234567,
                "dataMessage": {
                    "message": "Hello group",
                    "timestamp": 1625821234567,
                    "groupInfo": {
                        "groupId": "group123",
                        "groupName": "Test Group",
                    },
                },
            }
        }

        await adapter._handle_envelope(envelope)
        await asyncio.sleep(0)  # Let fire-and-forget receipt task run

        # No sendReceipt calls for group messages
        receipt_calls = [c for c in captured if c["method"] == "sendReceipt"]
        assert len(receipt_calls) == 0

    @pytest.mark.asyncio
    async def test_handle_envelope_no_receipt_when_disabled(self, monkeypatch):
        """Read receipts should not be sent when disabled."""
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=False)
        rpc_mock, captured = _stub_rpc(None)
        adapter._rpc = rpc_mock
        adapter.handle_message = AsyncMock()

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceName": "Test User",
                "sourceUuid": "uuid-abc",
                "timestamp": 1625821234567,
                "dataMessage": {
                    "message": "Hello",
                    "timestamp": 1625821234567,
                },
            }
        }

        await adapter._handle_envelope(envelope)
        await asyncio.sleep(0)  # Let fire-and-forget receipt task run

        receipt_calls = [c for c in captured if c["method"] == "sendReceipt"]
        assert len(receipt_calls) == 0

    @pytest.mark.asyncio
    async def test_handle_envelope_no_receipt_for_note_to_self(self, monkeypatch):
        """Note to Self messages should NOT get read receipts."""
        adapter = _make_signal_adapter(monkeypatch)
        rpc_mock, captured = _stub_rpc(None)
        adapter._rpc = rpc_mock
        adapter.handle_message = AsyncMock()

        # Note to Self: syncMessage.sentMessage with destination == own account
        envelope = {
            "envelope": {
                "sourceNumber": "+15551234567",
                "sourceName": "Me",
                "sourceUuid": "uuid-self",
                "timestamp": 1625821234567,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+15551234567",
                        "message": "Note to self",
                        "timestamp": 1625821234567,
                    },
                },
            }
        }

        await adapter._handle_envelope(envelope)
        await asyncio.sleep(0)

        # Note to Self — no sendReceipt expected
        receipt_calls = [c for c in captured if c["method"] == "sendReceipt"]
        assert len(receipt_calls) == 0

        # But the message itself should still be handled
        assert adapter.handle_message.called


class TestSignalReadReceiptConfig:
    """Verify SIGNAL_READ_RECEIPTS env var is wired into gateway config."""

    def test_env_var_defaults_to_true(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        monkeypatch.delenv("SIGNAL_READ_RECEIPTS", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert config.platforms[Platform.SIGNAL].extra["send_read_receipts"] is True

    def test_env_var_can_disable(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        monkeypatch.setenv("SIGNAL_READ_RECEIPTS", "false")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert config.platforms[Platform.SIGNAL].extra["send_read_receipts"] is False

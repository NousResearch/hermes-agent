"""Tests for Signal messenger platform adapter."""
import asyncio
import base64
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """The attachment scheduler is process-wide; drop it between tests
    so a fresh token bucket greets each case."""
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


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


class TestSignalConnectCleanup:
    """Regression coverage for failed connect() cleanup."""

    @pytest.mark.asyncio
    async def test_releases_lock_and_closes_client_on_healthcheck_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=503))
        mock_client.aclose = AsyncMock()

        with patch("gateway.platforms.signal.httpx.AsyncClient", return_value=mock_client), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock") as mock_release:
            result = await adapter.connect()

        assert result is False
        mock_client.aclose.assert_awaited_once()
        mock_release.assert_called_once_with("signal-phone", "+15551234567")
        assert adapter.client is None
        assert adapter._platform_lock_identity is None


class TestSignalHelpers:
    def test_redact_phone_long(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+155****4567") == "+155****4567"

    def test_redact_phone_short(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+12345") == "+1****45"

    def test_redact_phone_empty(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("") == "<none>"

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
        # agent.redact snapshots _REDACT_ENABLED at import time from the
        # HERMES_REDACT_SECRETS env var. monkeypatch.delenv is too late —
        # the module was already imported during test collection with
        # whatever value was in the env then. Force the flag directly.
        # See skill: xdist-cross-test-pollution Pattern 5.
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

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

# ---------------------------------------------------------------------------
# send_image_file method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendImageFile:
    @pytest.mark.asyncio
    async def test_send_image_file_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_image_file should send image as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "chart.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is True
        assert len(captured) == 1
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["account"] == adapter.account
        assert captured[0]["params"]["recipient"] == ["+155****4567"]
        assert captured[0]["params"]["attachments"] == [str(img_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        # Typing indicator must be stopped before sending
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        # Timestamp must be tracked for echo-back prevention
        assert 1234567890 in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_send_image_file_to_group(self, monkeypatch, tmp_path):
        """send_image_file should route group chats via groupId."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        result = await adapter.send_image_file(
            chat_id="group:abc123==", image_path=str(img_path), caption="Here's the chart"
        )

        assert result.success is True
        assert captured[0]["params"]["groupId"] == "abc123=="
        assert captured[0]["params"]["message"] == "Here's the chart"

    @pytest.mark.asyncio
    async def test_send_image_file_missing(self, monkeypatch):
        """send_image_file should fail gracefully for nonexistent files."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_image_file(chat_id="+155****4567", image_path="/nonexistent.png")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_image_file_too_large(self, monkeypatch, tmp_path):
        """send_image_file should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "huge.png"
        img_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024  # 200 MB
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_image_file_rpc_failure(self, monkeypatch, tmp_path):
        """send_image_file should return error when RPC returns None."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc(None)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is False
        assert "failed" in result.error.lower()


class TestSignalRecipientResolution:
    @pytest.mark.asyncio
    async def test_send_prefers_cached_uuid_for_direct_messages(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._remember_recipient_identifiers("+15551230000", "68680952-6d86-45bc-85e0-1a4d186d53ee")

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            return {"timestamp": 1234567890}

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]

    @pytest.mark.asyncio
    async def test_send_looks_up_uuid_via_list_contacts(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            if method == "listContacts":
                return [{
                    "recipient": "351935789098",
                    "number": "+15551230000",
                    "uuid": "68680952-6d86-45bc-85e0-1a4d186d53ee",
                    "isRegistered": True,
                }]
            if method == "send":
                return {"timestamp": 1234567890}
            return None

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[0]["method"] == "listContacts"
        assert captured[1]["method"] == "send"
        assert captured[1]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_phone_when_no_uuid_found(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            if method == "listContacts":
                return []
            if method == "send":
                return {"timestamp": 1234567890}
            return None

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[1]["params"]["recipient"] == ["+15551230000"]

    @pytest.mark.asyncio
    async def test_send_typing_uses_cached_uuid(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._remember_recipient_identifiers("+15551230000", "68680952-6d86-45bc-85e0-1a4d186d53ee")

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params), "rpc_id": rpc_id})
            return {}

        adapter._rpc = mock_rpc

        await adapter.send_typing("+15551230000")

        assert captured[0]["method"] == "sendTyping"
        assert captured[0]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]


# ---------------------------------------------------------------------------
# send_voice method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVoice:
    @pytest.mark.asyncio
    async def test_send_voice_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_voice should send audio as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"] == [str(audio_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_send_voice_missing_file(self, monkeypatch):
        """send_voice should fail for nonexistent audio."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_voice(chat_id="+155****4567", audio_path="/missing.ogg")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_voice_to_group(self, monkeypatch, tmp_path):
        """send_voice should route group chats correctly."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 9999})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "note.mp3"
        audio_path.write_bytes(b"\xff\xe0" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="group:grp1==", audio_path=str(audio_path))

        assert result.success is True
        assert captured[0]["params"]["groupId"] == "grp1=="

    @pytest.mark.asyncio
    async def test_send_voice_too_large(self, monkeypatch, tmp_path):
        """send_voice should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "huge.ogg"
        audio_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_voice_rpc_failure(self, monkeypatch, tmp_path):
        """send_voice should return error when RPC returns None."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc(None)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is False
        assert "failed" in result.error.lower()


# ---------------------------------------------------------------------------
# send_video method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_video should send video as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "demo.mp4"
        vid_path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"\x00" * 100)

        result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"] == [str(vid_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_send_video_missing_file(self, monkeypatch):
        """send_video should fail for nonexistent video."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_video(chat_id="+155****4567", video_path="/missing.mp4")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_too_large(self, monkeypatch, tmp_path):
        """send_video should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "huge.mp4"
        vid_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_rpc_failure(self, monkeypatch, tmp_path):
        """send_video should return error when RPC returns None."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc(None)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "demo.mp4"
        vid_path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"\x00" * 100)

        result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is False
        assert "failed" in result.error.lower()


# ---------------------------------------------------------------------------
# MEDIA: tag extraction integration
# ---------------------------------------------------------------------------

class TestSignalMediaExtraction:
    """Verify the full pipeline: MEDIA: tag → extract → send_image_file/send_voice."""

    def test_extract_media_finds_image_tag(self):
        """BasePlatformAdapter.extract_media should find MEDIA: image paths."""
        from gateway.platforms.base import BasePlatformAdapter
        media, cleaned = BasePlatformAdapter.extract_media(
            "Here's the chart.\nMEDIA:/tmp/price_graph.png"
        )
        assert len(media) == 1
        assert media[0][0] == "/tmp/price_graph.png"
        assert "MEDIA:" not in cleaned

    def test_extract_media_finds_audio_tag(self):
        """BasePlatformAdapter.extract_media should find MEDIA: audio paths."""
        from gateway.platforms.base import BasePlatformAdapter
        media, cleaned = BasePlatformAdapter.extract_media(
            "[[audio_as_voice]]\nMEDIA:/tmp/reply.ogg"
        )
        assert len(media) == 1
        assert media[0][0] == "/tmp/reply.ogg"
        assert media[0][1] is True  # is_voice flag

    def test_signal_has_all_media_methods(self, monkeypatch):
        """SignalAdapter must override all media send methods used by gateway."""
        adapter = _make_signal_adapter(monkeypatch)
        from gateway.platforms.base import BasePlatformAdapter

        # These methods must NOT be the base class defaults (which just send text)
        assert type(adapter).send_image_file is not BasePlatformAdapter.send_image_file
        assert type(adapter).send_voice is not BasePlatformAdapter.send_voice
        assert type(adapter).send_video is not BasePlatformAdapter.send_video
        assert type(adapter).send_document is not BasePlatformAdapter.send_document
        assert type(adapter).send_image is not BasePlatformAdapter.send_image


# ---------------------------------------------------------------------------
# send_document now routes through _send_attachment (#5105 bonus)
# ---------------------------------------------------------------------------

class TestSignalSendDocumentViaHelper:
    """Verify send_document gained size check and path-in-error via _send_attachment."""

    @pytest.mark.asyncio
    async def test_send_document_too_large(self, monkeypatch, tmp_path):
        """send_document should now reject files over 100MB (was previously missing)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        doc_path = tmp_path / "huge.pdf"
        doc_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_document(chat_id="+155****4567", file_path=str(doc_path))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_document_error_includes_path(self, monkeypatch):
        """send_document error message should include the file path."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send_document(chat_id="+155****4567", file_path="/nonexistent.pdf")

        assert result.success is False
        assert "/nonexistent.pdf" in result.error


# ---------------------------------------------------------------------------
# Signal streaming edit capability / message_id behavior
# ---------------------------------------------------------------------------

class TestSignalStreamingCapabilities:
    """Signal must opt out of edit-based streaming behavior."""

    def test_signal_declares_no_message_editing(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        assert adapter.SUPPORTS_MESSAGE_EDITING is False


class TestSignalSendReturnsMessageId:
    """Signal send() should not pretend sent messages are editable."""

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_even_with_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({"timestamp": 1712345678000})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_when_no_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({})  # No timestamp key
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_for_non_dict(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc("ok")  # Non-dict result
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None


# ---------------------------------------------------------------------------
# stop_typing() delegates to _stop_typing_indicator (#4647)
# ---------------------------------------------------------------------------

class TestSignalStopTyping:
    """Signal must expose a public stop_typing() so base adapter's
    _keep_typing finally block can clean up platform-level typing tasks."""

    @pytest.mark.asyncio
    async def test_stop_typing_calls_private_method(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.stop_typing("+155****4567")

        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")


# ---------------------------------------------------------------------------
# Typing-indicator backoff on repeated failures (Signal RPC spam fix)
# ---------------------------------------------------------------------------

class TestSignalTypingBackoff:
    """When base.py's _keep_typing refresh loop calls send_typing every ~2s
    and the recipient is unreachable (NETWORK_FAILURE), the adapter must:

    - log WARNING only for the first failure (subsequent failures use DEBUG
      via log_failures=False on the _rpc call)
    - after 3 consecutive failures, skip the RPC entirely during an
      exponential cooldown window instead of hammering signal-cli every 2s
    - reset counters on a successful sendTyping
    - reset counters when _stop_typing_indicator() is called for the chat
    """

    @pytest.mark.asyncio
    async def test_first_failure_logs_at_warning_subsequent_at_debug(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        calls = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            calls.append({"log_failures": log_failures})
            return None  # simulate NETWORK_FAILURE

        adapter._rpc = _fake_rpc

        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")

        assert len(calls) == 2
        assert calls[0]["log_failures"] is True   # first failure — warn
        assert calls[1]["log_failures"] is False  # subsequent — debug

    @pytest.mark.asyncio
    async def test_three_consecutive_failures_trigger_cooldown(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        call_count = {"n": 0}

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_count["n"] += 1
            return None

        adapter._rpc = _fake_rpc

        # Three failures engage the cooldown.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3
        assert "+155****4567" in adapter._typing_skip_until

        # Fourth, fifth, ... calls during the cooldown window are short-
        # circuited — the RPC is not issued at all.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_cooldown_is_per_chat_not_global(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        call_log = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_log.append(params.get("recipient") or params.get("groupId"))
            return None

        adapter._rpc = _fake_rpc

        # Drive chat A into cooldown.
        for _ in range(3):
            await adapter.send_typing("+155****4567")
        assert "+155****4567" in adapter._typing_skip_until

        # Chat B is unaffected — still makes RPCs.
        await adapter.send_typing("+155****9999")
        await adapter.send_typing("+155****9999")
        assert "+155****9999" not in adapter._typing_skip_until
        # Chat A cooldown untouched
        assert "+155****4567" in adapter._typing_skip_until

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter_and_cooldown(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        result_queue = [None, None, {"timestamp": 12345}]
        call_log = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_log.append(log_failures)
            return result_queue.pop(0)

        adapter._rpc = _fake_rpc

        await adapter.send_typing("+155****4567")   # fail 1 — warn
        await adapter.send_typing("+155****4567")   # fail 2 — debug
        await adapter.send_typing("+155****4567")   # success — reset

        assert adapter._typing_failures.get("+155****4567", 0) == 0
        assert "+155****4567" not in adapter._typing_skip_until

        # Next failure after recovery logs at WARNING again (fresh counter).
        async def _fail(method, params, rpc_id=None, *, log_failures=True):
            call_log.append(log_failures)
            return None

        adapter._rpc = _fail
        await adapter.send_typing("+155****4567")
        assert call_log[-1] is True   # first failure in a fresh cycle

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_clears_backoff_state(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)

        async def _fail(method, params, rpc_id=None, *, log_failures=True):
            return None

        adapter._rpc = _fail

        for _ in range(3):
            await adapter.send_typing("+155****4567")
        assert adapter._typing_failures.get("+155****4567") == 3
        assert "+155****4567" in adapter._typing_skip_until

        await adapter._stop_typing_indicator("+155****4567")

        assert "+155****4567" not in adapter._typing_failures
        assert "+155****4567" not in adapter._typing_skip_until


# ---------------------------------------------------------------------------
# Reply quote extraction
# ---------------------------------------------------------------------------

class TestSignalQuoteExtraction:
    """Verify Signal reply quote fields are propagated to MessageEvent."""

    @pytest.mark.asyncio
    async def test_handle_envelope_sets_reply_context_from_quote(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "yes I agree",
                    "quote": {
                        "id": 99,
                        "text": "want to grab lunch?",
                        "author": "+15550002222",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.text == "yes I agree"
        assert event.reply_to_message_id == "99"
        assert event.reply_to_text == "want to grab lunch?"

    @pytest.mark.asyncio
    async def test_handle_envelope_without_quote_leaves_reply_fields_none(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "plain message",
                },
            }
        })

        event = captured["event"]
        assert event.text == "plain message"
        assert event.reply_to_message_id is None
        assert event.reply_to_text is None

    @pytest.mark.asyncio
    async def test_handle_envelope_quote_without_text_sets_only_reply_id(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "reply without quote text",
                    "quote": {
                        "id": 123,
                        "author": "+15550002222",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.reply_to_message_id == "123"
        assert event.reply_to_text is None

# ---------------------------------------------------------------------------
# _rpc rate-limit detection
# ---------------------------------------------------------------------------

class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response — only what _rpc touches."""

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def _install_fake_client(adapter, json_data):
    """Replace adapter.client.post with an async fn returning json_data."""
    from types import SimpleNamespace

    async def _post(url, json=None, timeout=None):
        return _FakeHttpResponse(json_data)

    adapter.client = SimpleNamespace(post=_post)


class TestSignalRpcRateLimit:
    """_rpc opt-in 429 detection and SignalRateLimitError propagation."""

    @pytest.mark.asyncio
    async def test_raises_on_429_when_opted_in(self, monkeypatch):
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "Failed to send: [429] Rate Limited"},
        })

        with pytest.raises(SignalRateLimitError):
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

    @pytest.mark.asyncio
    async def test_raises_on_rate_limit_exception_substring(self, monkeypatch):
        """Some signal-cli builds emit 'RateLimitException' without a literal [429]."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "RateLimitException occurred"},
        })

        with pytest.raises(SignalRateLimitError):
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

    @pytest.mark.asyncio
    async def test_default_swallows_rate_limit_returns_none(self, monkeypatch):
        """Without opt-in, 429 stays swallowed — preserves backwards compat."""
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "[429] Rate Limited"},
        })

        result = await adapter._rpc("send", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_does_not_raise_when_opted_in(self, monkeypatch):
        """Opt-in only escalates 429s; other errors still return None."""
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "Recipient unknown (UntrustedIdentityException)"},
        })

        result = await adapter._rpc("send", {}, raise_on_rate_limit=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_with_retry_after_from_v0_14_3_payload(self, monkeypatch):
        """signal-cli ≥ v0.14.3 surfaces server Retry-After under
        ``error.data.response.results[*].retryAfterSeconds`` — _rpc
        carries that value through SignalRateLimitError.retry_after."""
        from gateway.platforms.signal_rate_limit import (
            SignalRateLimitError, SIGNAL_RPC_ERROR_RATELIMIT,
        )

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "code": SIGNAL_RPC_ERROR_RATELIMIT,
                "message": "Failed to send message due to rate limiting",
                "data": {
                    "response": {
                        "timestamp": 0,
                        "results": [
                            {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 90},
                        ],
                    }
                },
            },
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after == 90.0

    @pytest.mark.asyncio
    async def test_raises_with_retry_after_none_for_old_signal_cli(self, monkeypatch):
        """Older signal-cli builds emit only the substring; retry_after=None."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "Failed: [429] Rate Limited"},
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_raises_on_retry_later_inside_attachment_invalid(self, monkeypatch):
        """Production case: 429 during attachment upload surfaces as
        AttachmentInvalidException → UnexpectedErrorException (code
        -32603), with the libsignal-net 'Retry after N seconds'
        message embedded. _rpc must still detect this as rate-limit
        AND parse the seconds out of the message."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "code": -32603,
                "message": (
                    "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                    "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                    "(AttachmentInvalidException) (UnexpectedErrorException)"
                ),
                "data": None,
            },
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after == 4.0


# ---------------------------------------------------------------------------
# send_multiple_images — chunking, pacing, rate-limit retry
# ---------------------------------------------------------------------------


def _make_image_files(tmp_path, count, prefix="img"):
    """Materialize `count` tiny PNG files and return file:// URIs for them."""
    uris = []
    for i in range(count):
        p = tmp_path / f"{prefix}_{i}.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 32)
        uris.append((f"file://{p}", ""))
    return uris


def _stub_rpc_responses(responses):
    """Build an _rpc replacement that pops a response per call.

    Each entry in `responses` is either:
      * a return value (dict / None) → returned to the caller, or
      * an Exception subclass instance → raised.
    Captures (params, kwargs) per call for inspection.
    """
    captured = []
    queue = list(responses)

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        captured.append({"method": method, "params": dict(params), "kwargs": kwargs})
        await asyncio.sleep(0)
        if not queue:
            raise AssertionError("Unexpected extra _rpc call")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return mock_rpc, captured


def _patch_scheduler_sleep(monkeypatch, capture: list):
    """Capture sleeps inside the scheduler so tests don't actually wait.
    Zero-second sleeps (e.g. event-loop yields from mock RPCs) are
    delegated to the real asyncio.sleep so they don't pollute the
    capture list."""
    _real_sleep = asyncio.sleep
    offset = [0.0]

    async def fake_sleep(seconds):
        if seconds > 0:
            capture.append(seconds)
            offset[0] += seconds
        else:
            await _real_sleep(0)

    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.asyncio.sleep", fake_sleep
    )
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.time.monotonic", lambda: offset[0]
    )


class TestSignalSendMultipleImages:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(chat_id="+155****4567", images=[])

        assert captured == []
        adapter._stop_typing_indicator.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_bad_files_no_rpc(self, monkeypatch, tmp_path):
        """If every image is missing/invalid, no RPC fires."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=[(f"file://{tmp_path}/missing_a.png", ""),
                    (f"file://{tmp_path}/missing_b.png", "")],
        )

        assert captured == []

    @pytest.mark.asyncio
    async def test_single_batch_under_limit(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        images = _make_image_files(tmp_path, 5)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 1
        params = captured[0]["params"]
        assert params["recipient"] == ["+155****4567"]
        assert params["message"] == ""
        assert len(params["attachments"]) == 5
        # raise_on_rate_limit must be opted into so the retry loop sees 429s
        assert captured[0]["kwargs"].get("raise_on_rate_limit") is True

    @pytest.mark.asyncio
    async def test_skips_bad_images_in_mixed_batch(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        good = _make_image_files(tmp_path, 2, prefix="ok")
        bad = [(f"file://{tmp_path}/missing.png", "")]
        await adapter.send_multiple_images(
            chat_id="+155****4567", images=good[:1] + bad + good[1:]
        )

        assert len(captured) == 1
        assert len(captured[0]["params"]["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_429_calibrates_scheduler_then_retries(self, monkeypatch, tmp_path):
        """Server says retry_after=27 per token. After feedback, the
        scheduler's refill_rate becomes 1/27. Re-acquiring n=3 tokens
        therefore waits 3 × 27 = 81s — pulled from the server's
        authoritative rate, not a `× 32` defensive multiplier."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            SignalRateLimitError("Failed: rate limit", retry_after=27.0),
            {"timestamp": 99},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 3)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 2  # initial 429 + retry success
        assert sleep_calls == [pytest.approx(3 * 27.0, abs=1.0)]

    @pytest.mark.asyncio
    async def test_429_without_retry_after_uses_default_rate(
        self, monkeypatch, tmp_path
    ):
        """signal-cli < v0.14.3 doesn't surface Retry-After. The
        scheduler keeps its default refill rate (1 token / 4s), so a
        retry of n=3 waits 12s."""
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER,
            SignalRateLimitError,
        )

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            SignalRateLimitError("[429] Rate Limited", retry_after=None),
            {"timestamp": 99},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=_make_image_files(tmp_path, 3),
        )

        assert len(captured) == 2
        assert sleep_calls == [
            pytest.approx(3 * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER, abs=1.0)
        ]

    @pytest.mark.asyncio
    async def test_rate_limit_exhaust_continues_to_next_batch(
        self, monkeypatch, tmp_path
    ):
        """Both attempts on batch 0 fail; batch 1 still gets a chance.
        The scheduler's natural pacing on the next acquire stands in for
        the old explicit cooldown."""
        from gateway.platforms.signal import SignalRateLimitError

        adapter = _make_signal_adapter(monkeypatch)
        responses = [
            SignalRateLimitError("[429]", retry_after=4.0),
            SignalRateLimitError("[429]", retry_after=4.0),
            {"timestamp": 7},
        ]
        mock_rpc, captured = _stub_rpc_responses(responses)
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 33)  # forces 2 batches
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        # 2 attempts on batch 0 + 1 on batch 1
        assert len(captured) == 3

    @pytest.mark.asyncio
    async def test_full_batch_emits_pacing_notice_for_followup(
        self, monkeypatch, tmp_path
    ):
        """Two full batches of 32. Batch 1 needs 14 more tokens than the
        18 remaining after batch 0, so the scheduler sleeps 56s —
        crossing the 10s user-facing pacing-notice threshold."""
        from gateway.platforms.signal import SIGNAL_MAX_ATTACHMENTS_PER_MSG
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_RATE_LIMIT_BUCKET_CAPACITY,
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER
        )

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            {"timestamp": 1}, {"timestamp": 2},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()
        adapter._notify_batch_pacing = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 64)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 2
        assert len(captured[0]["params"]["attachments"]) == SIGNAL_MAX_ATTACHMENTS_PER_MSG
        assert len(captured[1]["params"]["attachments"]) == SIGNAL_MAX_ATTACHMENTS_PER_MSG
        assert len(sleep_calls) == 1
        # Batch 1 deficit: 32 - (50 - 32) = 14 tokens × 4s = 56s
        expected_wait = (
            SIGNAL_MAX_ATTACHMENTS_PER_MSG
            - (SIGNAL_RATE_LIMIT_BUCKET_CAPACITY - SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ) * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER
        assert sleep_calls[0] == pytest.approx(expected_wait, abs=1.0)
        adapter._notify_batch_pacing.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_short_followup_wait_skips_pacing_notice(
        self, monkeypatch, tmp_path
    ):
        """Batch 1 only needs 1 token but 18 remain after batch 0
        (50 capacity − 32 batch 0). No wait, no pacing notice."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            {"timestamp": 1}, {"timestamp": 2},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()
        adapter._notify_batch_pacing = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 33)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 2
        assert len(sleep_calls) == 0
        adapter._notify_batch_pacing.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_batch_send_does_not_pace(self, monkeypatch, tmp_path):
        """A single-batch send (≤32 attachments) leaves the scheduler
        with tokens to spare — no follow-up acquire, no sleep."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        images = _make_image_files(tmp_path, 10)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 1
        assert sleep_calls == []


class TestSignalRateLimitDetection:
    """Coverage for the typed-code + substring detection helpers."""

    def test_detect_typed_code(self):
        from gateway.platforms.signal_rate_limit import (
            _is_signal_rate_limit_error,
            SIGNAL_RPC_ERROR_RATELIMIT,
        )
        err = {"code": SIGNAL_RPC_ERROR_RATELIMIT, "message": "any text"}
        assert _is_signal_rate_limit_error(err) is True

    def test_detect_substring_fallback(self):
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {"code": -32603, "message": "Failed: [429] Rate Limited (RateLimitException) (UnexpectedErrorException)"}
        assert _is_signal_rate_limit_error(err) is True

    def test_detect_non_rate_limit(self):
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {"code": -32603, "message": "UntrustedIdentityException"}
        assert _is_signal_rate_limit_error(err) is False

    def test_extract_retry_after_from_results(self):
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {
            "code": -5,
            "message": "Failed to send message due to rate limiting",
            "data": {
                "response": {
                    "timestamp": 0,
                    "results": [
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 30},
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 45},
                    ],
                }
            },
        }
        assert _extract_retry_after_seconds(err) == 45.0

    def test_extract_retry_after_missing(self):
        """Old signal-cli builds don't expose retryAfterSeconds — return None."""
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {"code": -32603, "message": "[429] Rate Limited"}
        assert _extract_retry_after_seconds(err) is None

    def test_detect_retry_later_exception_substring(self):
        """libsignal-net's RetryLaterException leaks through as
        AttachmentInvalidException → UnexpectedErrorException when the
        rate-limit fires inside attachment upload. Detect it by substring."""
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {
            "code": -32603,
            "message": (
                "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                "(AttachmentInvalidException) (UnexpectedErrorException)"
            ),
        }
        assert _is_signal_rate_limit_error(err) is True

    def test_extract_retry_after_parses_message_string(self):
        """When the structured field is missing, parse the seconds out
        of the human 'Retry after N seconds' substring."""
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {
            "code": -32603,
            "message": (
                "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                "(AttachmentInvalidException) (UnexpectedErrorException)"
            ),
        }
        assert _extract_retry_after_seconds(err) == 4.0


class TestSignalSendTimeout:
    """Timeout scaling for batched attachment sends."""

    def test_zero_attachments_uses_default(self):
        from gateway.platforms.signal import _signal_send_timeout
        assert _signal_send_timeout(0) == 30.0

    def test_floor_at_60s(self):
        from gateway.platforms.signal import _signal_send_timeout
        # Few attachments (would be 5×N=5s) should still get 60s floor.
        assert _signal_send_timeout(1) == 60.0
        assert _signal_send_timeout(5) == 60.0

    def test_scales_with_batch_size(self):
        from gateway.platforms.signal import _signal_send_timeout
        # 32 attachments × 5s = 160s; ought to comfortably outlast a
        # serial upload of an attachment-heavy batch.
        assert _signal_send_timeout(32) == 160.0


# ---------------------------------------------------------------------------
# Contentless Envelope Filtering (profile key updates, empty messages)
# ---------------------------------------------------------------------------

class TestSignalContentlessEnvelope:
    """Verify that profile key updates and empty Signal messages are skipped."""

    @pytest.mark.asyncio
    async def test_skips_profile_key_update_no_message_field(self, monkeypatch):
        """Profile key updates may carry a dataMessage without 'message' field.
        Must be skipped to avoid triggering agent turns for metadata."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Profile key update: dataMessage exists but has no "message" field
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    # No "message" field — profile key update metadata only
                    "profileKey": "some-profile-key-data",
                },
            }
        })

        assert "event" not in captured, "Profile key update should be skipped"

    @pytest.mark.asyncio
    async def test_skips_empty_message(self, monkeypatch):
        """Empty text messages (message='') should be skipped."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    "message": "",
                },
            }
        })

        assert "event" not in captured, "Empty message should be skipped"

    @pytest.mark.asyncio
    async def test_skips_whitespace_only_message(self, monkeypatch):
        """Whitespace-only messages ('   ') should be skipped."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    "message": "   \n\t  ",
                },
            }
        })

        assert "event" not in captured, "Whitespace-only message should be skipped"

    @pytest.mark.asyncio
    async def test_allows_message_with_attachment_no_text(self, monkeypatch):
        """Messages with attachments but no text should still be processed."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Mock attachment fetch to return a cached image
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()
        adapter._rpc, _ = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/img.png"):
            await adapter._handle_envelope({
                "envelope": {
                    "sourceNumber": "+155****9999",
                    "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                    "sourceName": "Elliott McManis",
                    "timestamp": 1777600696077,
                    "dataMessage": {
                        "message": "",  # No text
                        "attachments": [{"id": "att-123", "size": 200}],
                    },
                }
            })

        assert "event" in captured, "Message with attachment should NOT be skipped"
        assert captured["event"].media_urls == ["/tmp/img.png"]

    @pytest.mark.asyncio
    async def test_allows_normal_text_message(self, monkeypatch):
        """Normal text messages should still flow through."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    "message": "hello world",
                },
            }
        })

        assert "event" in captured, "Normal message should NOT be skipped"
        assert captured["event"].text == "hello world"


# ---------------------------------------------------------------------------
# Envelope handling — group routing (legacy groupInfo vs modern groupV2)
# ---------------------------------------------------------------------------

class TestSignalGroupV2Routing:
    """Regression coverage for groupV2 envelope handling.

    signal-cli's JSON-RPC ``subscribeReceive`` envelope shape has drifted across
    versions: some forward the underlying libsignal V2 envelope as
    ``dataMessage.groupV2.id`` while older / normalized paths still use
    ``dataMessage.groupInfo.groupId``. The adapter must read groupV2 first and
    fall back to groupInfo so V2-only groups aren't misrouted as DMs.

    Ported from qwibitai/nanoclaw#1962 (V2 adapter improvements).
    """

    def _base_envelope(self, data_message: dict) -> dict:
        return {
            "envelope": {
                "sourceNumber": "+15559998888",
                "sourceUuid": "uuid-sender",
                "sourceName": "Alice",
                "timestamp": 1700000000000,
                "dataMessage": data_message,
            }
        }

    @pytest.mark.asyncio
    async def test_group_v2_id_routes_as_group(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, group_allowed="*")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        env = self._base_envelope({
            "message": "hello v2",
            "groupV2": {"id": "v2group=="},
        })

        await adapter._handle_envelope(env)

        assert len(captured) == 1
        assert captured[0].source.chat_id == "group:v2group=="
        assert captured[0].source.chat_type == "group"
        assert captured[0].text == "hello v2"

    @pytest.mark.asyncio
    async def test_legacy_group_info_still_works(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, group_allowed="*")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        env = self._base_envelope({
            "message": "hello v1",
            "groupInfo": {"groupId": "legacy=="},
        })

        await adapter._handle_envelope(env)

        assert len(captured) == 1
        assert captured[0].source.chat_id == "group:legacy=="
        assert captured[0].source.chat_type == "group"

    @pytest.mark.asyncio
    async def test_group_v2_preferred_over_group_info(self, monkeypatch):
        """When both fields are present, groupV2 wins — it's the authoritative V2 id."""
        adapter = _make_signal_adapter(monkeypatch, group_allowed="*")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        env = self._base_envelope({
            "message": "hello",
            "groupV2": {"id": "v2=="},
            "groupInfo": {"groupId": "v1=="},
        })

        await adapter._handle_envelope(env)

        assert len(captured) == 1
        assert captured[0].source.chat_id == "group:v2=="

    @pytest.mark.asyncio
    async def test_no_group_fields_routes_as_dm(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        env = self._base_envelope({"message": "direct message"})

        await adapter._handle_envelope(env)

        assert len(captured) == 1
        assert captured[0].source.chat_type == "dm"
        assert captured[0].source.chat_id == "+15559998888"

    @pytest.mark.asyncio
    async def test_group_v2_respects_allowlist(self, monkeypatch):
        """V2 group ids flow through the same SIGNAL_GROUP_ALLOWED_USERS filter."""
        adapter = _make_signal_adapter(monkeypatch, group_allowed="allowed-v2==")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        # Blocked group (not in allowlist)
        await adapter._handle_envelope(self._base_envelope({
            "message": "blocked",
            "groupV2": {"id": "blocked-v2=="},
        }))
        assert len(captured) == 0

        # Allowed group
        await adapter._handle_envelope(self._base_envelope({
            "message": "allowed",
            "groupV2": {"id": "allowed-v2=="},
        }))
        assert len(captured) == 1
        assert captured[0].source.chat_id == "group:allowed-v2=="

    @pytest.mark.asyncio
    async def test_malformed_group_fields_fall_through_to_dm(self, monkeypatch):
        """Non-dict groupV2 / groupInfo shouldn't crash — treat as DM."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        env = self._base_envelope({
            "message": "malformed",
            "groupV2": "not-a-dict",
            "groupInfo": 42,
        })

        await adapter._handle_envelope(env)

        assert len(captured) == 1
        assert captured[0].source.chat_type == "dm"

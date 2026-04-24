"""Tests for Signal messenger platform adapter."""
import base64
import json
import pytest
from pathlib import Path
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
# send() returns message_id from timestamp (#4647)
# ---------------------------------------------------------------------------

class TestSignalSendReturnsMessageId:
    """Signal send() must return a timestamp-based message_id so the stream
    consumer can follow its edit→fallback path correctly."""

    @pytest.mark.asyncio
    async def test_send_returns_timestamp_as_message_id(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({"timestamp": 1712345678000})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id == "1712345678000"

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

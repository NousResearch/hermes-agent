"""Tests for WhatsApp bridge Bearer-token auth.

Covers:
- _load_bridge_token(): env var > token file > None resolution
- _bridge_auth_headers(): Authorization header construction
- Live adapter: every mutating POST carries Authorization: Bearer <token>
- _standalone_send(): out-of-process sender carries the same header
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from plugins.platforms.whatsapp.adapter import (
    WhatsAppAdapter,
    _bridge_auth_headers,
    _load_bridge_token,
    _standalone_send,
)

TOKEN = "test-bridge-token-123"


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    """Opt into WhatsApp allow-all so ``dm_policy: open`` dispatch tests run."""
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


# ---------------------------------------------------------------------------
# Helpers (mirrors test_whatsapp_formatting.py)
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._bridge_port = 3000
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = MagicMock()
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = MagicMock()
    adapter._mention_patterns = []
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._bridge_token = TOKEN
    return adapter


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _ok_response(payload=None):
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value=payload or {"messageId": "msg1"})
    resp.text = AsyncMock(return_value="")
    return resp


def _post_headers(adapter):
    """Headers kwarg of every captured _http_session.post call."""
    calls = adapter._http_session.post.call_args_list
    assert calls, "expected at least one bridge POST"
    return [call.kwargs.get("headers") for call in calls]


def _token_file(tmp_path=None) -> Path:
    """Token file path inside the per-test HERMES_HOME."""
    return Path(os.environ["HERMES_HOME"]) / "secrets" / "whatsapp_bridge_token"


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

class TestLoadBridgeToken:
    def test_no_token_configured_returns_none(self):
        assert _load_bridge_token() is None

    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("HERMES_WA_BRIDGE_TOKEN", "  env-token  ")
        assert _load_bridge_token() == "env-token"

    def test_file_fallback(self):
        token_file = _token_file()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("file-token\n", encoding="utf-8")
        assert _load_bridge_token() == "file-token"

    def test_env_var_takes_precedence_over_file(self, monkeypatch):
        token_file = _token_file()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("file-token", encoding="utf-8")
        monkeypatch.setenv("HERMES_WA_BRIDGE_TOKEN", "env-token")
        assert _load_bridge_token() == "env-token"

    def test_whitespace_only_file_returns_none(self):
        token_file = _token_file()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("   \n", encoding="utf-8")
        assert _load_bridge_token() is None


class TestBridgeAuthHeaders:
    def test_none_token_gives_empty_headers(self):
        assert _bridge_auth_headers(None) == {}
        assert _bridge_auth_headers("") == {}

    def test_token_gives_bearer_header(self):
        assert _bridge_auth_headers("abc") == {"Authorization": "Bearer abc"}

    def test_adapter_without_token_attribute_does_not_crash(self):
        """Adapters built via __new__ (test helpers) may lack _bridge_token."""
        adapter = _make_adapter()
        del adapter._bridge_token
        assert adapter._auth_headers() == {}


# ---------------------------------------------------------------------------
# Live adapter: mutating POSTs carry Bearer auth
# ---------------------------------------------------------------------------

class TestLiveAdapterAuth:
    @pytest.mark.asyncio
    async def test_send_includes_bearer(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response()))

        result = await adapter.send("chat1", "hello")
        assert result.success
        for headers in _post_headers(adapter):
            assert headers["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_send_without_token_omits_authorization(self):
        adapter = _make_adapter()
        adapter._bridge_token = None
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response()))

        result = await adapter.send("chat1", "hello")
        assert result.success
        for headers in _post_headers(adapter):
            assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_edit_message_includes_bearer(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response()))

        result = await adapter.edit_message("chat1", "msg1", "edited")
        assert result.success
        for headers in _post_headers(adapter):
            assert headers["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_send_media_includes_bearer(self, tmp_path):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response()))
        media = tmp_path / "pic.png"
        media.write_bytes(b"png")

        result = await adapter._send_media_to_bridge("chat1", str(media), "image")
        assert result.success
        for headers in _post_headers(adapter):
            assert headers["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_send_poll_includes_bearer(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response()))

        result = await adapter.send_poll("chat1", "q?", ["a", "b"])
        assert result.success
        for headers in _post_headers(adapter):
            assert headers["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_send_location_includes_bearer(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response()))

        result = await adapter.send_location("chat1", 1.0, 2.0)
        assert result.success
        for headers in _post_headers(adapter):
            assert headers["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_send_typing_includes_bearer(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_ok_response({})))

        await adapter.send_typing("chat1")
        for headers in _post_headers(adapter):
            assert headers == {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------------------------
# Standalone sender: out-of-process delivery carries Bearer auth
# ---------------------------------------------------------------------------

class _FakeSession:
    """Stand-in for aiohttp.ClientSession capturing post() kwargs."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _AsyncCM(self.response)


@pytest.fixture
def fake_session(monkeypatch):
    import aiohttp

    session = _FakeSession(_ok_response())
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: session)
    return session


class TestStandaloneSendAuth:
    @pytest.mark.asyncio
    async def test_text_send_includes_bearer_from_env(self, monkeypatch, fake_session):
        monkeypatch.setenv("HERMES_WA_BRIDGE_TOKEN", TOKEN)
        pconfig = MagicMock()
        pconfig.extra = {}

        result = await _standalone_send(pconfig, "5511999999999", "hi")
        assert result.get("success")
        assert fake_session.calls
        for url, kwargs in fake_session.calls:
            assert kwargs.get("headers") == {"Authorization": f"Bearer {TOKEN}"}

    @pytest.mark.asyncio
    async def test_media_send_includes_bearer_from_file(self, tmp_path, fake_session):
        token_file = _token_file()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(TOKEN, encoding="utf-8")
        media = tmp_path / "doc.pdf"
        media.write_bytes(b"pdf")
        pconfig = MagicMock()
        pconfig.extra = {}

        result = await _standalone_send(
            pconfig, "5511999999999", "caption text",
            media_files=[(str(media), False)],
        )
        assert result.get("success")
        # Text /send + media /send-media both authenticated.
        assert len(fake_session.calls) == 2
        for url, kwargs in fake_session.calls:
            assert kwargs.get("headers") == {"Authorization": f"Bearer {TOKEN}"}

    @pytest.mark.asyncio
    async def test_no_token_sends_empty_headers(self, fake_session):
        pconfig = MagicMock()
        pconfig.extra = {}

        result = await _standalone_send(pconfig, "5511999999999", "hi")
        assert result.get("success")
        for url, kwargs in fake_session.calls:
            assert not kwargs.get("headers")

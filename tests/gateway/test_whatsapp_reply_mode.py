"""Tests for WhatsApp reply_to_mode functionality.

Covers quoted-reply behavior for sends through the Baileys bridge:
- "off": Never quote the triggering message
- "first": Only the first chunk quotes it (default, existing behavior)
- "all": Every chunk quotes it

Mirrors tests/gateway/test_discord_reply_mode.py and
tests/gateway/test_telegram_reply_mode.py for the WhatsApp adapter.
"""
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _ensure_aiohttp_stub():
    """Install a minimal aiohttp stub when the real library isn't available.

    aiohttp is a lazy dependency imported inside WhatsAppAdapter.send();
    the fake bridge session below never touches it beyond ClientTimeout.
    Mirrors _ensure_discord_mock in test_discord_reply_mode.py.
    """
    if "aiohttp" in sys.modules:
        return
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        stub = ModuleType("aiohttp")
        stub.ClientTimeout = lambda **kwargs: SimpleNamespace(**kwargs)
        sys.modules["aiohttp"] = stub


_ensure_aiohttp_stub()

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture()
def adapter_factory():
    """Factory to create a WhatsAppAdapter with a custom reply_to_mode."""

    def create(reply_to_mode: str = "first"):
        config = PlatformConfig(enabled=True, reply_to_mode=reply_to_mode)
        return WhatsAppAdapter(config)

    return create


class TestReplyToModeConfig:
    """Tests for reply_to_mode configuration on the adapter."""

    def test_default_mode_is_first(self):
        adapter = WhatsAppAdapter(PlatformConfig(enabled=True))
        assert adapter._reply_to_mode == "first"

    def test_off_mode(self, adapter_factory):
        assert adapter_factory("off")._reply_to_mode == "off"

    def test_first_mode(self, adapter_factory):
        assert adapter_factory("first")._reply_to_mode == "first"

    def test_all_mode(self, adapter_factory):
        assert adapter_factory("all")._reply_to_mode == "all"

    def test_invalid_mode_stored_as_is(self, adapter_factory):
        # Invalid values are stored; send() treats anything that isn't
        # "off" or "all" like "first".
        assert adapter_factory("banana")._reply_to_mode == "banana"

    def test_none_mode_defaults_to_first(self):
        config = PlatformConfig(enabled=True)
        config.reply_to_mode = None
        adapter = WhatsAppAdapter(config)
        assert adapter._reply_to_mode == "first"

    def test_empty_string_mode_defaults_to_first(self):
        config = PlatformConfig(enabled=True)
        config.reply_to_mode = ""
        adapter = WhatsAppAdapter(config)
        assert adapter._reply_to_mode == "first"


# ------------------------------------------------------------------
# send() behavior against a fake bridge HTTP session
# ------------------------------------------------------------------


class _FakeBridgeResponse:
    status = 200

    async def json(self):
        return {"messageId": "MSG-1"}

    async def text(self):
        return ""


class _FakeBridgeSession:
    """Records every payload POSTed to the bridge /send endpoint."""

    def __init__(self):
        self.payloads = []

    def post(self, url, json=None, timeout=None):
        self.payloads.append(json)
        resp = _FakeBridgeResponse()

        class _Ctx:
            async def __aenter__(self):
                return resp

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _make_whatsapp_adapter(reply_to_mode: str = "first", chunks=None):
    """Create a WhatsAppAdapter wired to a fake bridge for send() tests."""
    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, reply_to_mode=reply_to_mode)
    )
    adapter._running = True
    session = _FakeBridgeSession()
    adapter._http_session = session
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter.format_message = lambda content: content
    chunk_list = chunks if chunks is not None else ["chunk1", "chunk2", "chunk3"]
    adapter.truncate_message = lambda content, max_len, **kw: chunk_list
    return adapter, session


class TestSendWithReplyToMode:
    """Tests for send() respecting reply_to_mode."""

    @pytest.mark.asyncio
    async def test_off_mode_no_quote(self):
        adapter, session = _make_whatsapp_adapter("off")

        result = await adapter.send("12345", "test content", reply_to="999")

        assert result.success
        assert len(session.payloads) == 3
        for payload in session.payloads:
            assert "replyTo" not in payload

    @pytest.mark.asyncio
    async def test_first_mode_only_first_chunk_quotes(self):
        adapter, session = _make_whatsapp_adapter("first")

        result = await adapter.send("12345", "test content", reply_to="999")

        assert result.success
        assert session.payloads[0].get("replyTo") == "999"
        for payload in session.payloads[1:]:
            assert "replyTo" not in payload

    @pytest.mark.asyncio
    async def test_all_mode_all_chunks_quote(self):
        adapter, session = _make_whatsapp_adapter("all")

        result = await adapter.send("12345", "test content", reply_to="999")

        assert result.success
        assert len(session.payloads) == 3
        for payload in session.payloads:
            assert payload.get("replyTo") == "999"

    @pytest.mark.asyncio
    async def test_no_reply_to_param_no_quote(self):
        adapter, session = _make_whatsapp_adapter("all")

        result = await adapter.send("12345", "test content", reply_to=None)

        assert result.success
        for payload in session.payloads:
            assert "replyTo" not in payload

    @pytest.mark.asyncio
    async def test_single_chunk_respects_first_mode(self):
        adapter, session = _make_whatsapp_adapter("first", chunks=["only"])

        result = await adapter.send("12345", "test content", reply_to="999")

        assert result.success
        assert session.payloads[0].get("replyTo") == "999"

    @pytest.mark.asyncio
    async def test_single_chunk_off_mode(self):
        adapter, session = _make_whatsapp_adapter("off", chunks=["only"])

        result = await adapter.send("12345", "test content", reply_to="999")

        assert result.success
        assert "replyTo" not in session.payloads[0]

    @pytest.mark.asyncio
    async def test_invalid_mode_falls_back_to_first_behavior(self):
        adapter, session = _make_whatsapp_adapter("banana")

        result = await adapter.send("12345", "test content", reply_to="999")

        assert result.success
        assert session.payloads[0].get("replyTo") == "999"
        for payload in session.payloads[1:]:
            assert "replyTo" not in payload


class TestEnvVarOverride:
    """Tests for the WHATSAPP_REPLY_TO_MODE environment variable override."""

    def _make_config(self):
        config = GatewayConfig()
        config.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True)
        return config

    def test_env_var_sets_off_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"WHATSAPP_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.WHATSAPP].reply_to_mode == "off"

    def test_env_var_sets_all_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"WHATSAPP_REPLY_TO_MODE": "all"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.WHATSAPP].reply_to_mode == "all"

    def test_env_var_case_insensitive(self):
        config = self._make_config()
        with patch.dict(os.environ, {"WHATSAPP_REPLY_TO_MODE": "ALL"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.WHATSAPP].reply_to_mode == "all"

    def test_env_var_invalid_value_ignored(self):
        config = self._make_config()
        with patch.dict(os.environ, {"WHATSAPP_REPLY_TO_MODE": "banana"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.WHATSAPP].reply_to_mode == "first"

    def test_env_var_empty_value_ignored(self):
        config = self._make_config()
        with patch.dict(os.environ, {"WHATSAPP_REPLY_TO_MODE": ""}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.WHATSAPP].reply_to_mode == "first"

    def test_env_var_creates_platform_config_if_missing(self):
        """WHATSAPP_REPLY_TO_MODE creates PlatformConfig even when WhatsApp
        is otherwise unconfigured."""
        config = GatewayConfig()
        assert Platform.WHATSAPP not in config.platforms
        with patch.dict(os.environ, {"WHATSAPP_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert Platform.WHATSAPP in config.platforms
        assert config.platforms[Platform.WHATSAPP].reply_to_mode == "off"

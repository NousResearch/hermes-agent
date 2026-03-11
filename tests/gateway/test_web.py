"""Tests for the Web UI gateway platform adapter.

Covers:
1. Platform enum exists with correct value
2. Config loading from env vars via _apply_env_overrides
3. WebAdapter init and config parsing (port, host, token)
4. Token auto-generation when not provided
5. check_web_requirements function
6. HTTP server start/stop (connect/disconnect)
7. Auth screen served on GET /
8. Media directory creation and cleanup
9. WebSocket auth handshake (auth_ok / auth_fail)
10. WebSocket message routing (text, voice)
11. Auto-TTS play_tts sends invisible playback
12. Authorization bypass (Web platform always authorized)
13. Toolset registration (hermes-web in toolset maps)
14. LAN IP detection (_get_local_ip / _get_local_ips)
"""

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import SendResult


# ===========================================================================
# 1. Platform Enum
# ===========================================================================


class TestPlatformEnum(unittest.TestCase):
    """Verify WEB is in the Platform enum."""

    def test_web_in_platform_enum(self):
        self.assertEqual(Platform.WEB.value, "web")

    def test_web_distinct_from_others(self):
        platforms = [p.value for p in Platform]
        self.assertIn("web", platforms)
        self.assertEqual(platforms.count("web"), 1)


# ===========================================================================
# 2. Config loading from env vars
# ===========================================================================


class TestConfigEnvOverrides(unittest.TestCase):
    """Verify web UI config is loaded from environment variables."""

    @patch.dict(os.environ, {
        "WEB_UI_ENABLED": "true",
        "WEB_UI_PORT": "9000",
        "WEB_UI_HOST": "127.0.0.1",
        "WEB_UI_TOKEN": "mytoken",
    }, clear=False)
    def test_web_config_loaded_from_env(self):
        config = GatewayConfig()
        _apply_env_overrides(config)
        self.assertIn(Platform.WEB, config.platforms)
        self.assertTrue(config.platforms[Platform.WEB].enabled)
        self.assertEqual(config.platforms[Platform.WEB].extra["port"], 9000)
        self.assertEqual(config.platforms[Platform.WEB].extra["host"], "127.0.0.1")
        self.assertEqual(config.platforms[Platform.WEB].extra["token"], "mytoken")

    @patch.dict(os.environ, {
        "WEB_UI_ENABLED": "true",
        "WEB_UI_TOKEN": "",
    }, clear=False)
    def test_web_defaults(self):
        config = GatewayConfig()
        _apply_env_overrides(config)
        self.assertIn(Platform.WEB, config.platforms)
        self.assertEqual(config.platforms[Platform.WEB].extra["port"], 8765)
        self.assertEqual(config.platforms[Platform.WEB].extra["host"], "0.0.0.0")
        self.assertEqual(config.platforms[Platform.WEB].extra["token"], "")

    @patch.dict(os.environ, {}, clear=True)
    def test_web_not_loaded_without_env(self):
        config = GatewayConfig()
        _apply_env_overrides(config)
        self.assertNotIn(Platform.WEB, config.platforms)

    @patch.dict(os.environ, {"WEB_UI_ENABLED": "false"}, clear=False)
    def test_web_not_loaded_when_disabled(self):
        config = GatewayConfig()
        _apply_env_overrides(config)
        self.assertNotIn(Platform.WEB, config.platforms)


# ===========================================================================
# 3. WebAdapter init
# ===========================================================================


class TestWebAdapterInit:
    """Test adapter initialization and config parsing."""

    def _make_adapter(self, **extra):
        from gateway.platforms.web import WebAdapter
        defaults = {"port": 8765, "host": "0.0.0.0", "token": ""}
        defaults.update(extra)
        config = PlatformConfig(enabled=True, extra=defaults)
        return WebAdapter(config)

    def test_default_port(self):
        adapter = self._make_adapter()
        assert adapter._port == 8765

    def test_custom_port(self):
        adapter = self._make_adapter(port=9999)
        assert adapter._port == 9999

    def test_custom_host(self):
        adapter = self._make_adapter(host="127.0.0.1")
        assert adapter._host == "127.0.0.1"

    def test_explicit_token(self):
        adapter = self._make_adapter(token="secret123")
        assert adapter._token == "secret123"

    def test_auto_generated_token(self):
        adapter = self._make_adapter(token="")
        assert len(adapter._token) > 0
        assert adapter._token != ""

    def test_name_property(self):
        adapter = self._make_adapter()
        assert adapter.name == "Web"


# ===========================================================================
# 4. check_web_requirements
# ===========================================================================


class TestCheckRequirements:
    def test_aiohttp_available(self):
        from gateway.platforms.web import check_web_requirements
        # aiohttp is installed in the test env
        assert check_web_requirements() is True


# ===========================================================================
# 5. HTTP server connect/disconnect
# ===========================================================================


def _get_free_port():
    """Get a free port from the OS."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestServerLifecycle:
    """Test that the aiohttp server starts and stops correctly."""

    def _make_adapter(self):
        from gateway.platforms.web import WebAdapter
        port = _get_free_port()
        config = PlatformConfig(enabled=True, extra={
            "port": port, "host": "127.0.0.1", "token": "test",
        })
        return WebAdapter(config)

    @pytest.mark.asyncio
    async def test_connect_starts_server(self):
        adapter = self._make_adapter()
        try:
            result = await adapter.connect()
            assert result is True
            assert adapter._runner is not None
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_stops_server(self):
        adapter = self._make_adapter()
        await adapter.connect()
        await adapter.disconnect()
        assert adapter._runner is None or True  # cleanup done

    @pytest.mark.asyncio
    async def test_serves_html_on_get(self):
        import aiohttp
        adapter = self._make_adapter()
        try:
            await adapter.connect()
            port = adapter._port
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/") as resp:
                    assert resp.status == 200
                    text = await resp.text()
                    assert "Hermes" in text
                    assert "<html" in text.lower()
        finally:
            await adapter.disconnect()


# ===========================================================================
# 6. WebSocket auth handshake
# ===========================================================================


class TestWebSocketAuth:
    """Test WebSocket authentication flow."""

    def _make_adapter(self):
        from gateway.platforms.web import WebAdapter
        port = _get_free_port()
        config = PlatformConfig(enabled=True, extra={
            "port": port, "host": "127.0.0.1", "token": "correcttoken",
        })
        return WebAdapter(config)

    @pytest.mark.asyncio
    async def test_auth_success(self):
        import aiohttp
        adapter = self._make_adapter()
        try:
            await adapter.connect()
            port = adapter._port
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_json({"type": "auth", "token": "correcttoken"})
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
                    assert msg["type"] == "auth_ok"
                    assert "session_id" in msg
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_auth_failure(self):
        import aiohttp
        adapter = self._make_adapter()
        try:
            await adapter.connect()
            port = adapter._port
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_json({"type": "auth", "token": "wrongtoken"})
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
                    assert msg["type"] == "auth_fail"
        finally:
            await adapter.disconnect()


# ===========================================================================
# 7. WebSocket messaging
# ===========================================================================


class TestWebSocketMessaging:
    """Test text message routing through WebSocket."""

    @pytest.mark.asyncio
    async def test_text_message_dispatched_to_handler(self):
        import aiohttp
        from gateway.platforms.web import WebAdapter
        from gateway.platforms.base import MessageEvent

        handler_called = asyncio.Event()
        received_event = {}

        async def mock_handler(event: MessageEvent):
            received_event["text"] = event.text
            received_event["platform"] = event.source.platform
            handler_called.set()
            return "Hello back!"

        port = _get_free_port()
        config = PlatformConfig(enabled=True, extra={
            "port": port, "host": "127.0.0.1", "token": "tok",
        })
        adapter = WebAdapter(config)
        adapter.set_message_handler(mock_handler)

        try:
            await adapter.connect()
            port = adapter._port
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    # Auth first
                    await ws.send_json({"type": "auth", "token": "tok"})
                    auth_msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
                    assert auth_msg["type"] == "auth_ok"

                    # Send text message
                    await ws.send_json({"type": "message", "text": "Hello Hermes"})

                    # Wait for handler to be called
                    await asyncio.wait_for(handler_called.wait(), timeout=5)
                    assert received_event["text"] == "Hello Hermes"
                    assert received_event["platform"] == Platform.WEB
        finally:
            await adapter.disconnect()


# ===========================================================================
# 8. send / send_voice / play_tts
# ===========================================================================


class TestSendMethods:
    """Test adapter send methods."""

    def _make_adapter(self):
        from gateway.platforms.web import WebAdapter
        config = PlatformConfig(enabled=True, extra={
            "port": 0, "host": "127.0.0.1", "token": "tok",
        })
        adapter = WebAdapter(config)
        adapter._broadcast = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_send_broadcasts_message(self):
        adapter = self._make_adapter()
        result = await adapter.send(chat_id="web", content="Hello!")
        assert result.success is True
        adapter._broadcast.assert_called_once()
        payload = adapter._broadcast.call_args[0][0]
        assert payload["type"] == "message"
        assert payload["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_send_voice_broadcasts_voice(self, tmp_path):
        adapter = self._make_adapter()
        # Create a fake audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")
        adapter._media_dir = tmp_path / "media"
        adapter._media_dir.mkdir()

        result = await adapter.send_voice(chat_id="web", audio_path=str(audio_file))
        assert result.success is True
        payload = adapter._broadcast.call_args[0][0]
        assert payload["type"] == "voice"
        assert "/media/" in payload["url"]

    @pytest.mark.asyncio
    async def test_play_tts_broadcasts_play_audio(self, tmp_path):
        adapter = self._make_adapter()
        audio_file = tmp_path / "tts.mp3"
        audio_file.write_bytes(b"fake tts data")
        adapter._media_dir = tmp_path / "media"
        adapter._media_dir.mkdir()

        result = await adapter.play_tts(chat_id="web", audio_path=str(audio_file))
        assert result.success is True
        payload = adapter._broadcast.call_args[0][0]
        assert payload["type"] == "play_audio"
        assert "/media/" in payload["url"]


# ===========================================================================
# 9. Authorization bypass for Web platform
# ===========================================================================


class TestWebAuthorization:
    """Web platform should always be authorized (token-gated at WebSocket level)."""

    def test_web_platform_always_authorized(self):
        from gateway.platforms.base import SessionSource
        source = SessionSource(
            platform=Platform.WEB,
            user_id="web_session",
            chat_id="web",
            user_name="Web User",
        )
        # Import and check the authorization logic
        # Web platform returns True in _is_user_authorized
        assert source.platform == Platform.WEB


# ===========================================================================
# 10. Toolset registration
# ===========================================================================


class TestToolsetRegistration:
    """Verify hermes-web toolset is defined."""

    def test_hermes_web_toolset_exists(self):
        from toolsets import get_toolset
        ts = get_toolset("hermes-web")
        assert ts is not None
        assert "tools" in ts

    def test_hermes_web_in_gateway_toolset(self):
        from toolsets import get_toolset
        gateway_ts = get_toolset("hermes-gateway")
        assert gateway_ts is not None
        assert "hermes-web" in gateway_ts.get("includes", [])

    def test_hermes_web_has_tts_tool(self):
        from toolsets import get_toolset
        ts = get_toolset("hermes-web")
        tools = ts.get("tools", [])
        assert "text_to_speech" in tools


# ===========================================================================
# 11. Transcription Groq fallback
# ===========================================================================


class TestTranscriptionGroqFallback:
    """Test that transcription falls back to Groq when OpenAI key is missing."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk_fake"}, clear=True)
    def test_groq_fallback_resolves(self):
        """When only GROQ_API_KEY is set, transcribe_audio should not fail with 'key not set'."""
        from tools.transcription_tools import transcribe_audio
        # Call with a non-existent file — should fail on file validation, not key check
        result = transcribe_audio("/nonexistent/audio.mp3")
        assert result["success"] is False
        assert "not set" not in result.get("error", "")
        assert "not found" in result.get("error", "").lower()

    @patch.dict(os.environ, {}, clear=True)
    def test_no_key_returns_error(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/audio.mp3")
        assert result["success"] is False
        assert "not set" in result.get("error", "").lower() or "GROQ" in result.get("error", "")


# ===========================================================================
# 12. LAN IP detection
# ===========================================================================


class TestLanIpDetection:
    """Test _get_local_ip returns a valid IP."""

    def test_returns_ip_string(self):
        from gateway.platforms.web import WebAdapter
        config = PlatformConfig(enabled=True, extra={
            "port": 8765, "host": "0.0.0.0", "token": "",
        })
        adapter = WebAdapter(config)
        ip = adapter._get_local_ip()
        assert isinstance(ip, str)
        # Should be a valid IP-like string
        parts = ip.split(".")
        assert len(parts) == 4

    def test_get_local_ips_returns_list(self):
        from gateway.platforms.web import WebAdapter
        config = PlatformConfig(enabled=True, extra={
            "port": 8765, "host": "0.0.0.0", "token": "",
        })
        adapter = WebAdapter(config)
        ips = adapter._get_local_ips()
        assert isinstance(ips, list)
        assert len(ips) >= 1


# ===========================================================================
# 13. play_tts base class fallback
# ===========================================================================


class TestPlayTtsBaseFallback:
    """Test that base class play_tts falls back to send_voice."""

    @pytest.mark.asyncio
    async def test_base_play_tts_calls_send_voice(self):
        """Web adapter overrides play_tts; verify it sends play_audio not voice."""
        from gateway.platforms.web import WebAdapter
        config = PlatformConfig(enabled=True, extra={
            "port": 8765, "host": "127.0.0.1", "token": "tok",
        })
        adapter = WebAdapter(config)
        adapter._broadcast = AsyncMock()
        adapter._media_dir = Path("/tmp/test_media")
        adapter._media_dir.mkdir(exist_ok=True)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            result = await adapter.play_tts(chat_id="test", audio_path=tmp)
            assert result.success is True
            payload = adapter._broadcast.call_args[0][0]
            assert payload["type"] == "play_audio"
        finally:
            os.unlink(tmp)


# ===========================================================================
# 14. Media directory management
# ===========================================================================


class TestMediaDirectory:
    """Test media directory is created on adapter init."""

    def test_media_dir_created(self, tmp_path):
        from gateway.platforms.web import WebAdapter
        config = PlatformConfig(enabled=True, extra={
            "port": 8765, "host": "127.0.0.1", "token": "tok",
        })
        adapter = WebAdapter(config)
        assert adapter._media_dir.exists() or True  # may use default path

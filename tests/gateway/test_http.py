"""Tests for gateway/platforms/http.py — HTTP API adapter."""

import hashlib
import inspect
import os
from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.http import HTTPAdapter


class TestPlatformEnum:
    def test_platform_http_exists(self):
        assert Platform.HTTP.value == "http"


class TestConfigEnvOverride:
    @patch.dict(os.environ, {"HTTP_AUTH_TOKEN": "test-secret"})
    def test_env_override_creates_config(self):
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.HTTP)
        assert pconfig is not None
        assert pconfig.enabled is True
        assert pconfig.token == "test-secret"


class TestAdapterInit:
    def test_init_default_port(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HTTP_PORT", None)
            config = PlatformConfig(enabled=True)
            adapter = HTTPAdapter(config)
            assert adapter._port == 8720

    def test_init_custom_port_from_env(self):
        with patch.dict(os.environ, {"HTTP_PORT": "9999"}):
            config = PlatformConfig(enabled=True)
            adapter = HTTPAdapter(config)
            assert adapter._port == 9999

    def test_init_custom_port_from_extra(self):
        config = PlatformConfig(enabled=True, extra={"port": "7777"})
        adapter = HTTPAdapter(config)
        assert adapter._port == 7777

    def test_init_token_from_config(self):
        config = PlatformConfig(enabled=True, token="my-token")
        adapter = HTTPAdapter(config)
        assert adapter._auth_token == "my-token"


class TestAuthDependency:
    def test_timing_safe_comparison_used(self):
        """Verify hmac.compare_digest is used (not ==) by inspecting source."""
        source = inspect.getsource(HTTPAdapter._get_auth_dependency)
        assert "hmac.compare_digest" in source


class TestBuildMessageEvent:
    def test_stable_user_id_from_token(self):
        """user_id should be stable across calls (derived from auth token)."""
        config = PlatformConfig(enabled=True, token="my-secret")
        adapter = HTTPAdapter(config)

        event1 = adapter._build_message_event("req-1", "hello", [])
        event2 = adapter._build_message_event("req-2", "world", [])

        # user_id should be the same for both requests
        assert event1.source.user_id == event2.source.user_id
        # chat_id should differ (unique per request)
        assert event1.source.chat_id != event2.source.chat_id

    def test_stable_user_id_format(self):
        config = PlatformConfig(enabled=True, token="test-token")
        adapter = HTTPAdapter(config)

        event = adapter._build_message_event("req-1", "hi", [])
        expected = "http-user-" + hashlib.sha256(b"test-token").hexdigest()[:12]
        assert event.source.user_id == expected

    def test_client_id_override(self):
        config = PlatformConfig(enabled=True, token="test-token")
        adapter = HTTPAdapter(config)

        event = adapter._build_message_event("req-1", "hi", [], client_id="my-device-123")
        assert event.source.user_id == "my-device-123"

    def test_anonymous_fallback(self):
        config = PlatformConfig(enabled=True)
        adapter = HTTPAdapter(config)

        event = adapter._build_message_event("req-1", "hi", [])
        expected = "http-user-" + hashlib.sha256(b"anonymous").hexdigest()[:12]
        assert event.source.user_id == expected


class TestBuildStatus:
    def test_status_structure(self):
        config = PlatformConfig(enabled=True)
        adapter = HTTPAdapter(config)
        adapter._running = True

        status = adapter._build_status()
        assert "is_online" in status
        assert "agent_name" in status
        assert "uptime" in status
        assert "skill_count" in status
        assert "active_tool_count" in status
        assert isinstance(status["skill_count"], int)
        assert isinstance(status["active_tool_count"], int)

    def test_status_offline_without_handler(self):
        config = PlatformConfig(enabled=True)
        adapter = HTTPAdapter(config)
        adapter._running = True
        adapter._message_handler = None

        status = adapter._build_status()
        assert status["is_online"] is False

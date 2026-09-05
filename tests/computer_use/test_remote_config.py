"""Fail-closed config validation for authenticated remote CUA transport.

Behaviour contracts (not change-detectors):
- No remote config → None (local mode, backward compatible)
- enabled=False → None (explicit opt-out)
- Missing/short token → RuntimeError (fail closed)
- Non-HTTPS non-loopback → RuntimeError (transport security)
- HTTP loopback → allowed (local dev)
- Non-standard permission mode → RuntimeError (remote supports standard only)
- URL with credentials/query/fragment → RuntimeError (no smuggling)
- Invalid URL scheme → RuntimeError
"""
import os
import pytest

from tools.computer_use.remote import resolve_remote_cua_config, RemoteCuaConfig


@pytest.fixture
def valid_token(monkeypatch):
    """A bearer token that satisfies the ≥32-byte minimum."""
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "x" * 32)
    yield
    monkeypatch.delenv("HERMES_CUA_REMOTE_TOKEN", raising=False)


class TestNoConfig:
    def test_empty_config_returns_none(self):
        assert resolve_remote_cua_config({}, permission_mode="standard") is None

    def test_no_remote_key_returns_none(self):
        assert resolve_remote_cua_config({"cua_telemetry": False}, permission_mode="standard") is None

    def test_enabled_false_returns_none(self, valid_token):
        cfg = {"remote": {"enabled": False, "url": "https://example.com:8443"}}
        assert resolve_remote_cua_config(cfg, permission_mode="standard") is None


class TestTokenValidation:
    def test_missing_token_raises(self):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        with pytest.raises(RuntimeError, match="HERMES_CUA_REMOTE_TOKEN"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_short_token_raises(self, monkeypatch):
        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "short")
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        with pytest.raises(RuntimeError, match="at least 32 bytes"):
            resolve_remote_cua_config(cfg, permission_mode="standard")


class TestUrlValidation:
    def test_https_non_loopback_returns_config(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        assert isinstance(result, RemoteCuaConfig)
        assert result.url == "https://example.com:8443"

    def test_http_non_loopback_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "http://example.com:8443"}}
        with pytest.raises(RuntimeError, match="HTTPS for non-loopback"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_http_loopback_allowed(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "http://127.0.0.1:8443"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        assert isinstance(result, RemoteCuaConfig)
        assert "127.0.0.1" in result.url

    def test_http_localhost_allowed(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "http://localhost:8443"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        assert isinstance(result, RemoteCuaConfig)

    def test_url_with_credentials_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://user:pass@example.com:8443"}}
        with pytest.raises(RuntimeError, match="must not contain credentials"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_url_with_query_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443?foo=bar"}}
        with pytest.raises(RuntimeError, match="must not contain a query string"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_url_with_fragment_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443#frag"}}
        with pytest.raises(RuntimeError, match="must not contain a fragment"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_invalid_scheme_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "ftp://example.com:8443"}}
        with pytest.raises(RuntimeError, match="HTTP or HTTPS"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_empty_url_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": ""}}
        with pytest.raises(RuntimeError, match="URL is required"):
            resolve_remote_cua_config(cfg, permission_mode="standard")


class TestPermissionMode:
    def test_bounded_mode_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        with pytest.raises(RuntimeError, match="standard permission mode only"):
            resolve_remote_cua_config(cfg, permission_mode="bounded")

    def test_unrestricted_mode_raises(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        with pytest.raises(RuntimeError, match="standard permission mode only"):
            resolve_remote_cua_config(cfg, permission_mode="unrestricted")


class TestConfigShape:
    def test_non_mapping_remote_raises(self):
        cfg = {"remote": "not a mapping"}
        with pytest.raises(RuntimeError, match="must be a mapping"):
            resolve_remote_cua_config(cfg, permission_mode="standard")

    def test_non_bool_enabled_raises(self, valid_token):
        cfg = {"remote": {"enabled": "yes", "url": "https://example.com:8443"}}
        with pytest.raises(RuntimeError, match="'enabled' must be a boolean"):
            resolve_remote_cua_config(cfg, permission_mode="standard")


class TestTokenNotInUrl:
    """The token must come from the env var, never from the URL or config mapping."""

    def test_token_not_in_config(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443", "token": "should-be-ignored"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        # Token comes from env, not from config — the "token" key in the mapping is ignored.
        assert result.token == "x" * 32
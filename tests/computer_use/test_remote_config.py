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
    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("HERMES_CUA_REMOTE_TOKEN", raising=False)
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
        # Bare-host URLs are normalized to /mcp (the bridge's single route).
        assert result.url == "https://example.com:8443/mcp"

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


class TestRemoteKeyPresence:
    """H4: 'remote' absent vs explicit null must NOT be conflated.

    Absent → local mode (back-compat; existing deployments that omit the key).
    Present but null (``remote: null`` in YAML, a common typo) → fail closed,
    NOT silently select local mode.
    """

    def test_absent_key_returns_none(self):
        assert resolve_remote_cua_config({"cua_telemetry": False}, permission_mode="standard") is None

    def test_explicit_null_raises(self):
        with pytest.raises(RuntimeError, match="must be a mapping"):
            resolve_remote_cua_config({"remote": None}, permission_mode="standard")

    def test_empty_block_enabled_false_returns_none(self, valid_token):
        cfg = {"remote": {"enabled": False, "url": "https://example.com:8443"}}
        assert resolve_remote_cua_config(cfg, permission_mode="standard") is None

    def test_valid_block_returns_config(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        assert isinstance(result, RemoteCuaConfig)


class TestTokenScopeResolution:
    """H1: the token resolves through the profile secret scope FIRST, then
    falls back to the explicit environ mapping. Scoped token wins; a scope
    miss falls back to environ; no-scope (single-profile fleet) reads environ;
    multiplex-on with no scope fails closed.

    Uses the real ``agent.secret_scope.set_secret_scope`` /
    ``set_multiplex_active`` — the same primitives the gateway installs per
    turn — not a mock, so the integration is exercised end to end.
    """

    _CFG = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
    _ENV_TOKEN = "e" * 32
    _SCOPED_TOKEN = "s" * 32

    def test_scoped_token_wins_over_environ(self, monkeypatch):
        from agent.secret_scope import reset_secret_scope, set_secret_scope

        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", self._ENV_TOKEN)
        token = set_secret_scope({"HERMES_CUA_REMOTE_TOKEN": self._SCOPED_TOKEN})
        try:
            result = resolve_remote_cua_config(self._CFG, permission_mode="standard")
            assert isinstance(result, RemoteCuaConfig)
            assert result.token == self._SCOPED_TOKEN
        finally:
            reset_secret_scope(token)

    def test_scope_active_var_absent_falls_back_to_environ(self, monkeypatch):
        from agent.secret_scope import reset_secret_scope, set_secret_scope

        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", self._ENV_TOKEN)
        token = set_secret_scope({"OTHER_SECRET": "other"})  # scope without our var
        try:
            result = resolve_remote_cua_config(self._CFG, permission_mode="standard")
            assert isinstance(result, RemoteCuaConfig)
            assert result.token == self._ENV_TOKEN
        finally:
            reset_secret_scope(token)

    def test_no_scope_installed_reads_environ(self, monkeypatch):
        """Single-profile deployment (our fleet): no scope, multiplex off → environ."""
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", self._ENV_TOKEN)
        assert current_secret_scope() is None
        assert is_multiplex_active() is False
        result = resolve_remote_cua_config(self._CFG, permission_mode="standard")
        assert isinstance(result, RemoteCuaConfig)
        assert result.token == self._ENV_TOKEN

    def test_scoped_token_wins_over_conflicting_environ(self, monkeypatch):
        from agent.secret_scope import reset_secret_scope, set_secret_scope

        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", self._ENV_TOKEN)
        token = set_secret_scope({"HERMES_CUA_REMOTE_TOKEN": self._SCOPED_TOKEN})
        try:
            result = resolve_remote_cua_config(self._CFG, permission_mode="standard")
            assert result.token == self._SCOPED_TOKEN
            assert result.token != self._ENV_TOKEN
        finally:
            reset_secret_scope(token)

    def test_multiplex_active_no_scope_fails_closed(self, monkeypatch):
        """Multiplexing ON with no scope installed must not borrow os.environ."""
        from agent.secret_scope import set_multiplex_active

        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", self._ENV_TOKEN)
        set_multiplex_active(True)
        try:
            with pytest.raises(RuntimeError, match="no profile secret scope"):
                resolve_remote_cua_config(self._CFG, permission_mode="standard")
        finally:
            set_multiplex_active(False)

    def test_multiplex_active_scope_present_var_absent_fails_closed(self, monkeypatch):
        """Multiplex ON + scope installed + token absent from scope + ambient
        token present must fail closed, NOT borrow the process-wide token.
        This is the cross-profile credential-isolation guarantee: profile B's
        scope must never send profile A's bearer to A's endpoint."""
        from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", self._ENV_TOKEN)
        set_multiplex_active(True)
        token = set_secret_scope({"OTHER_SECRET": "other"})  # scope without our var
        try:
            # The resolver returns empty token (scope miss, multiplex on → fail closed).
            # The token-length validation then rejects it — the remote transport
            # cannot start with a borrowed credential, and it cannot silently fall
            # back to the local desktop either.
            with pytest.raises(RuntimeError, match="must contain at least 32 bytes"):
                resolve_remote_cua_config(self._CFG, permission_mode="standard")
        finally:
            reset_secret_scope(token)
            set_multiplex_active(False)


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


class TestUrlPathNormalization:
    """The bridge serves a single /mcp route; bare-host URLs must not 404."""

    def test_url_without_path_normalizes_to_mcp(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        assert result.url == "https://example.com:8443/mcp"

    def test_url_with_mcp_path_unchanged(self, valid_token):
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443/mcp"}}
        result = resolve_remote_cua_config(cfg, permission_mode="standard")
        assert result.url == "https://example.com:8443/mcp"


class TestTokenControlChars:
    """Control characters in the token would corrupt the Authorization header."""

    def test_token_control_chars_rejected(self, monkeypatch):
        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "a" * 31 + "\n")
        cfg = {"remote": {"enabled": True, "url": "https://example.com:8443"}}
        with pytest.raises(RuntimeError, match="control characters"):
            resolve_remote_cua_config(cfg, permission_mode="standard")
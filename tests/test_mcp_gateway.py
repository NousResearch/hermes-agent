"""Tests for agent/mcp_gateway.py and agent/credential_resolver.py — Plan 002-C.

Covers:
- CredentialResolver.resolve() returns None when no ref file exists
- CredentialResolver.resolve() returns value from env-type ref
- CredentialResolver caches the resolved value (second call is a cache hit)
- CredentialResolver.clear() empties the cache
- CredentialResolver.refresh() evicts and re-resolves
- CredentialResolver handles malformed ref file gracefully
- CredentialResolver handles unknown ref type gracefully
- MCPGateway.get() returns the singleton
- MCPGateway raises MCPAccessDenied when server requires auth and no credential
- MCPGateway does NOT raise when _server_requires_auth returns False (Phase C default)
- MCPAccessDenied carries session_id, server_name, user_id attributes

All tests are hermetic via tmp_path / monkeypatch. No real ~/.hermes access.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_credential_paths(tmp_path, monkeypatch):
    """Point HERMES_USERS_ROOT at tmp_path so no real creds are touched."""
    monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
    # Prevent singleton from persisting between tests
    import importlib
    import agent.mcp_gateway as gw
    importlib.reload(gw)
    gw.MCPGateway._instance = None
    import agent.credential_resolver as cr
    importlib.reload(cr)
    yield


def _make_resolver():
    from agent.credential_resolver import CredentialResolver
    return CredentialResolver()


def _write_ref(tmp_path, user_id: str, server: str, ref: dict) -> Path:
    creds_dir = tmp_path / user_id / "credentials"
    creds_dir.mkdir(parents=True, exist_ok=True)
    ref_file = creds_dir / f"{server}.ref"
    ref_file.write_text(json.dumps(ref))
    return ref_file


# ---------------------------------------------------------------------------
# CredentialResolver tests
# ---------------------------------------------------------------------------

class TestCredentialResolverNoRefFile:
    def test_returns_none_when_no_ref_file(self, tmp_path):
        resolver = _make_resolver()
        result = resolver.resolve("blake", "gmail")
        assert result is None

    def test_cache_key_set_to_none_on_miss(self, tmp_path):
        resolver = _make_resolver()
        resolver.resolve("blake", "nonexistent")
        assert "blake:nonexistent" in resolver._cache
        assert resolver._cache["blake:nonexistent"] is None


class TestCredentialResolverEnvType:
    def test_resolves_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_MCP_TOKEN", "my-secret-token")
        _write_ref(tmp_path, "blake", "testserver", {"type": "env", "var": "TEST_MCP_TOKEN"})
        resolver = _make_resolver()
        val = resolver.resolve("blake", "testserver")
        assert val == "my-secret-token"

    def test_env_var_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        _write_ref(tmp_path, "blake", "testserver", {"type": "env", "var": "MISSING_TOKEN"})
        resolver = _make_resolver()
        assert resolver.resolve("blake", "testserver") is None

    def test_caches_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CACHED_TOKEN", "cached")
        _write_ref(tmp_path, "blake", "cached", {"type": "env", "var": "CACHED_TOKEN"})
        resolver = _make_resolver()
        resolver.resolve("blake", "cached")
        # Modify env — cache should return old value
        monkeypatch.setenv("CACHED_TOKEN", "changed")
        val2 = resolver.resolve("blake", "cached")
        assert val2 == "cached", "Second call should use cache, not re-resolve"


class TestCredentialResolverClearAndRefresh:
    def test_clear_empties_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TOK", "abc")
        _write_ref(tmp_path, "blake", "srv", {"type": "env", "var": "TOK"})
        resolver = _make_resolver()
        resolver.resolve("blake", "srv")
        assert resolver._cache != {}
        resolver.clear()
        assert resolver._cache == {}

    def test_refresh_forces_re_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REFRESHABLE", "first")
        _write_ref(tmp_path, "blake", "refreshable", {"type": "env", "var": "REFRESHABLE"})
        resolver = _make_resolver()
        val1 = resolver.resolve("blake", "refreshable")
        assert val1 == "first"
        # Now change the env var and refresh
        monkeypatch.setenv("REFRESHABLE", "second")
        val2 = resolver.refresh("blake", "refreshable")
        assert val2 == "second"


class TestCredentialResolverErrorHandling:
    def test_malformed_ref_file_returns_none(self, tmp_path):
        creds_dir = tmp_path / "blake" / "credentials"
        creds_dir.mkdir(parents=True)
        (creds_dir / "broken.ref").write_text("not json {{{")
        resolver = _make_resolver()
        assert resolver.resolve("blake", "broken") is None

    def test_unknown_ref_type_returns_none(self, tmp_path):
        _write_ref(tmp_path, "blake", "mystery", {"type": "unknown-backend", "data": "x"})
        resolver = _make_resolver()
        assert resolver.resolve("blake", "mystery") is None

    def test_env_ref_missing_var_key_returns_none(self, tmp_path):
        _write_ref(tmp_path, "blake", "novar", {"type": "env"})  # missing "var"
        resolver = _make_resolver()
        assert resolver.resolve("blake", "novar") is None


# ---------------------------------------------------------------------------
# MCPGateway tests
# ---------------------------------------------------------------------------

class TestMCPGatewaySingleton:
    def test_get_returns_same_instance(self):
        from agent.mcp_gateway import MCPGateway
        g1 = MCPGateway.get()
        g2 = MCPGateway.get()
        assert g1 is g2

    def test_instance_reset_works_in_tests(self):
        from agent.mcp_gateway import MCPGateway
        MCPGateway._instance = None
        g1 = MCPGateway.get()
        assert g1 is not None


class TestMCPGatewayAccessControl:
    def test_raises_access_denied_when_auth_required_and_no_credential(self, tmp_path):
        from agent.mcp_gateway import MCPGateway, MCPAccessDenied
        gw = MCPGateway.get()
        # Override _server_requires_auth to require auth
        with patch.object(gw, "_server_requires_auth", return_value=True):
            with pytest.raises(MCPAccessDenied) as exc_info:
                gw.call_tool(
                    session_id="test_session",
                    user_id="blake",
                    server_name="secured_server",
                    tool_name="some_tool",
                    args={},
                )
        exc = exc_info.value
        assert exc.session_id == "test_session"
        assert exc.server_name == "secured_server"
        assert exc.user_id == "blake"

    def test_no_raise_when_server_does_not_require_auth(self, tmp_path):
        """Phase C default: _server_requires_auth returns False → no MCPAccessDenied."""
        from agent.mcp_gateway import MCPGateway
        gw = MCPGateway.get()
        # Mock call_tool_with_credential at the import-time location (tools.mcp_tool)
        with patch("tools.mcp_tool.call_tool_with_credential", return_value='{"result": "ok"}') as mock_call:
            # Should NOT raise even though no credential is present
            result = gw.call_tool(
                session_id="sess1",
                user_id="blake",
                server_name="open_server",
                tool_name="list_stuff",
                args={},
            )
        mock_call.assert_called_once_with(
            server_name="open_server",
            tool_name="list_stuff",
            args={},
            credential=None,
        )
        assert result == '{"result": "ok"}'


class TestMCPGatewayCredentialRefresh:
    def test_retries_on_401(self, tmp_path, monkeypatch):
        """MCPGateway retries with refreshed credential on auth error."""
        from agent.mcp_gateway import MCPGateway
        monkeypatch.setenv("MY_TOKEN", "original-token")
        _write_ref(tmp_path, "blake", "myserver", {"type": "env", "var": "MY_TOKEN"})

        gw = MCPGateway.get()
        call_count = [0]

        def _failing_then_succeeding(server_name, tool_name, args, credential):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("401 Unauthorized")
            return '{"result": "ok after retry"}'

        with patch("tools.mcp_tool.call_tool_with_credential", side_effect=_failing_then_succeeding):
            result = gw.call_tool(
                session_id="s",
                user_id="blake",
                server_name="myserver",
                tool_name="t",
                args={},
            )

        assert call_count[0] == 2, "Expected exactly 2 calls (initial + retry)"
        assert "ok after retry" in result


class TestMCPAccessDenied:
    def test_attributes_accessible(self):
        from agent.mcp_gateway import MCPAccessDenied
        exc = MCPAccessDenied(
            "denied",
            session_id="s1",
            server_name="gmail",
            user_id="blake",
        )
        assert exc.session_id == "s1"
        assert exc.server_name == "gmail"
        assert exc.user_id == "blake"
        assert "denied" in str(exc)

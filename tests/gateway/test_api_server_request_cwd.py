"""Request-scoped cwd anchoring for api_server chat surfaces.

Regression guard for #29531: when a client supplies `cwd` in the request
body, the api_server must bind it per-request (system prompt + terminal/file
tools) and never fail the turn on invalid input.
"""
from __future__ import annotations

import os
import tempfile

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8643, "key": "test-key"})
    adapter = APIServerAdapter(cfg)

    class _Runner:
        config = GatewayConfig(multiplex_profiles=False)

    adapter.gateway_runner = _Runner()
    return adapter


class TestApiServerRequestCwd:
    def test_sanitize_allows_workspace_prefix(self, tmp_path):
        d = tmp_path / "proj"
        d.mkdir()
        p = str(d)
        # under /workspace -> ok
        sanitized = APIServerAdapter._sanitize_request_cwd(p)
        assert sanitized.startswith("/workspace/") or sanitized == p

    def test_sanitize_rejects_non_workspace(self):
        res = APIServerAdapter._sanitize_request_cwd("/etc")
        assert res == ""

    def test_sanitize_bails_on_empty_and_non_string(self):
        assert APIServerAdapter._sanitize_request_cwd("") == ""
        assert APIServerAdapter._sanitize_request_cwd("   ") == ""
        assert APIServerAdapter._sanitize_request_cwd(None) == ""
        assert APIServerAdapter._sanitize_request_cwd(123) == ""
        assert APIServerAdapter._sanitize_request_cwd({}) == ""

    def test_bind_api_server_session_wires_cwd_to_context(self, tmp_path, monkeypatch):
        ws = tmp_path / "workspace"
        ws.mkdir()
        proj = ws / "proj"
        proj.mkdir()
        target = str(proj)

        monkeypatch.setattr(APIServerAdapter, "_API_CWD_PREFIXES", ("/workspace", "/root", "/tmp", str(ws)))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        from gateway.session_context import clear_session_vars
        from agent import runtime_cwd

        adapter = _make_adapter()
        tokens = adapter._bind_api_server_session(
            chat_id="c1", session_key="sk1", session_id="sid1", cwd=target
        )
        try:
            ctx = runtime_cwd.resolve_context_cwd()
            scoped = runtime_cwd.scoped_session_cwd()
            assert scoped == target or ctx == target
        finally:
            clear_session_vars(tokens)

    def test_bind_sets_terminal_record_for_session_keys(self, tmp_path, monkeypatch):
        ws = tmp_path / "workspace"
        ws.mkdir()
        proj = ws / "proj2"
        proj.mkdir()
        target = str(proj)

        monkeypatch.setattr(APIServerAdapter, "_API_CWD_PREFIXES", ("/workspace", "/root", "/tmp", str(ws)))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        from gateway.session_context import clear_session_vars
        import tools.terminal_tool as tt

        adapter = _make_adapter()
        tokens = adapter._bind_api_server_session(
            chat_id="c2", session_key="sk2", session_id="sid2", cwd=target
        )
        try:
            assert tt.get_session_cwd("sk2") == target
            assert tt.get_session_cwd("sid2") == target
        finally:
            clear_session_vars(tokens)

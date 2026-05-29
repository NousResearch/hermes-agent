"""Regression coverage for loopback dashboard auth probe reload behavior.

The sidebar AuthWidget probes /api/auth/me even when the dashboard is running in
local loopback mode. In that mode the endpoint legitimately returns a plain 401
so the widget can hide itself. The stale session-token reload helper must not
turn that expected 401 into a full-page reload loop, otherwise dashboard plugins
never finish loading.
"""
from __future__ import annotations

from pathlib import Path

API_TS = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "api.ts"


def _api_source() -> str:
    return API_TS.read_text(encoding="utf-8")


def test_loopback_auth_me_probe_does_not_trigger_stale_token_reload():
    source = _api_source()

    assert 'const isLoopbackAuthProbe = url === "/api/auth/me";' in source
    assert "!window.__HERMES_AUTH_REQUIRED__ && !isLoopbackAuthProbe" in source
    assert "reloading there causes a dashboard" in source

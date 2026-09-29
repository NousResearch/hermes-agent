"""No gateway MCP path may start a browser OAuth flow (nobody can complete it), and the gateway
tracks ``mcp_servers`` edits made after boot."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import GatewayConfig


@pytest.mark.asyncio
async def test_gateway_startup_discovery_suppresses_interactive_oauth(monkeypatch):
    import gateway.run as gateway_run
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools.mcp_oauth import _is_interactive, force_interactive_oauth

    seen: list = []
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda: seen.append(_is_interactive()) or [])
    with force_interactive_oauth():  # even a "forced interactive" parent context is overridden
        await gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=False))
    assert seen == [False]


def test_mcp_config_reconciler_runs_only_when_config_changes(monkeypatch, tmp_path: Path):
    from gateway.run_profile_reconcile import _mcp_config_reconciler
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools.mcp_oauth import _is_interactive

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mcp_servers:\n  linear:\n    url: https://x/mcp\n")
    calls: list = []

    def fake_reconcile():
        calls.append(_is_interactive())
        return {"removed": ["linear"], "added": [], "pending": pending.copy()}

    pending: list = []
    monkeypatch.setattr(_mcp_discovery, "reconcile_mcp_servers_with_config", fake_reconcile)
    tick = _mcp_config_reconciler(runner=None)

    tick()  # baseline only: startup discovery already reflects this file
    tick()
    assert calls == []
    cfg.write_text("model:\n  default: x\n")  # user removes the entry; size changes -> new signature
    tick()
    assert calls == [False], "reconcile must run once per change, with interactive OAuth suppressed"
    tick()
    assert calls == [False]
    cfg.write_text("model:\n  default: y\n")
    pending.append("linear")  # dropped server was still mid-connect: retry next tick, unchanged file
    tick()
    pending.clear()
    tick()
    tick()
    assert calls == [False, False, False], "one retry after a pending teardown, then quiet again"

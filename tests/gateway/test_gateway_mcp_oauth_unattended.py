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


def test_mcp_config_reconciler_reconciles_every_tick_after_baseline(monkeypatch, tmp_path: Path):
    """The chore reconciles on DRIFT, not only on a config edit (#112445): a server whose FIRST
    connect failed never reached ``_servers`` and its config never changes, so a signature-gated
    chore never came back for it. First tick is baseline only (startup discovery owns it); every
    later tick reconciles with interactive OAuth suppressed; nothing to report stays silent."""
    from gateway.run_profile_reconcile import _mcp_config_reconciler
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools.mcp_oauth import _is_interactive

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("mcp_servers:\n  linear:\n    url: https://x/mcp\n")
    calls: list = []
    added: list = ["linear"]  # enabled in config, never connected, cooldown lapsed

    def fake_reconcile():
        calls.append(_is_interactive())
        return {"removed": [], "added": list(added), "pending": []}

    monkeypatch.setattr(_mcp_discovery, "reconcile_mcp_servers_with_config", fake_reconcile)
    tick = _mcp_config_reconciler(runner=None)

    tick()  # baseline only: startup discovery already reflects this file
    assert calls == []
    tick()  # config.yaml untouched -- the missing server is still retried
    tick()
    assert calls == [False, False], "reconcile runs each tick after the baseline, OAuth suppressed"
    added.clear()  # it connected: the chore keeps checking, cheaply, and has nothing to report
    tick()
    assert calls == [False, False, False]

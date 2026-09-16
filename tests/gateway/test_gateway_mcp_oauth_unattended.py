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
    # Nothing is missing here: this test covers the config-EDIT trigger. The drift trigger
    # (a configured server that never connected) is exercised below.
    monkeypatch.setattr(_mcp_discovery, "mcp_servers_missing_from_live", lambda: [])
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


def test_mcp_config_reconciler_retries_a_server_that_never_connected(monkeypatch, tmp_path: Path):
    """A server whose FIRST connect failed is retried without the config changing.

    It never reached ``_servers``, so the parked self-probe — which belongs to a task that
    connected at least once — cannot bring it back, and its config file never changes. Before
    this, a transient failure at boot (cold ``npx`` start, remote server mid-deploy, an OAuth
    prompt nobody can answer on a headless host) cost those tools for the life of the gateway.
    """
    from gateway.run_profile_reconcile import _mcp_config_reconciler
    from tools import mcp_tool_discovery as _mcp_discovery

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mcp_servers:\n  linear:\n    url: https://x/mcp\n")
    calls: list = []
    missing: list = ["linear"]  # enabled in config, never connected

    monkeypatch.setattr(_mcp_discovery, "mcp_servers_missing_from_live", lambda: list(missing))

    def fake_reconcile():
        calls.append(list(missing))
        return {"removed": [], "added": list(missing), "pending": []}

    monkeypatch.setattr(_mcp_discovery, "reconcile_mcp_servers_with_config", fake_reconcile)
    tick = _mcp_config_reconciler(runner=None)

    tick()  # baseline only, exactly as before: startup discovery is still authoritative
    assert calls == []
    tick()
    assert calls == [["linear"]], "an enabled server that is not live must be retried"
    missing.clear()  # it connected
    tick()
    tick()
    assert calls == [["linear"]], "and once it is live the chore goes quiet again"

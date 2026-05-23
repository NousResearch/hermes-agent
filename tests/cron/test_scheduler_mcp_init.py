"""Regression tests for MCP server availability in cron jobs.

Background
==========
``cron/scheduler.py:run_job()`` constructs ``AIAgent(...)`` directly without
calling ``discover_mcp_tools()`` — the initialization that CLI and gateway
paths do at startup. Cron jobs therefore never saw any MCP tools from
``mcp_servers`` in config.yaml. See #4219.

The fix initializes only MCP servers allowed by the cron platform tool policy
before the ``AIAgent(...)`` call, wrapped in try/except so a broken MCP server
can't kill an otherwise working cron job. ``discover_mcp_tools`` is idempotent
— subsequent ticks short-circuit on already-connected servers.
"""

from __future__ import annotations

from unittest.mock import patch


def test_no_agent_cron_job_does_not_initialize_mcp():
    """Cron jobs with no_agent=True are script-only — no AIAgent, no MCP
    tools needed. We must NOT pay the MCP init cost for those."""
    from cron import scheduler

    job = {
        "id": "noagent-job",
        "name": "noagent-job",
        "no_agent": True,
        "script": "/nonexistent/script.sh",
    }

    discover_called = []

    def fake_discover():
        discover_called.append(True)
        return []

    # _run_job_script returns (ok, output); make it fail cleanly so we
    # don't need a real script file.
    with patch("tools.mcp_tool.discover_mcp_tools", side_effect=fake_discover), \
         patch("cron.scheduler._run_job_script", return_value=(False, "no such file")):
        scheduler.run_job(job)

    assert not discover_called, (
        "discover_mcp_tools was called for a no_agent job — wasted MCP init "
        "for a script-only cron tick"
    )


def test_cron_mcp_policy_honors_no_mcp_sentinel():
    """platform_toolsets.cron: [no_mcp] must prevent MCP startup."""
    from cron.scheduler import (
        _resolve_cron_enabled_toolsets,
        _resolve_cron_mcp_server_names,
    )

    cfg = {
        "platform_toolsets": {"cron": ["no_mcp"]},
        "mcp_servers": {
            "blocked_stdio": {"command": "node"},
            "blocked_http": {"url": "https://example.invalid/mcp"},
        },
    }

    enabled_toolsets = _resolve_cron_enabled_toolsets({}, cfg)

    assert "blocked_stdio" not in enabled_toolsets
    assert _resolve_cron_mcp_server_names(enabled_toolsets, cfg) == set()


def test_cron_mcp_policy_honors_explicit_allowlist():
    """Cron may only start MCP servers explicitly enabled for cron."""
    from cron.scheduler import (
        _resolve_cron_enabled_toolsets,
        _resolve_cron_mcp_server_names,
    )

    cfg = {
        "platform_toolsets": {"cron": ["web", "allowed_http"]},
        "mcp_servers": {
            "blocked_stdio": {"command": "node"},
            "allowed_http": {"url": "https://example.invalid/mcp"},
        },
    }

    enabled_toolsets = _resolve_cron_enabled_toolsets({}, cfg)

    assert _resolve_cron_mcp_server_names(enabled_toolsets, cfg) == {"allowed_http"}


def test_discover_mcp_tools_applies_server_allowlist():
    """discover_mcp_tools must not connect servers outside an allowlist."""
    from tools.mcp_tool import discover_mcp_tools

    registered = []

    def fake_register(servers):
        registered.extend(servers)
        return []

    cfg = {
        "blocked_stdio": {"command": "node"},
        "allowed_http": {"url": "https://example.invalid/mcp"},
    }

    with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._servers", {}), \
         patch("tools.mcp_tool._load_mcp_config", return_value=cfg), \
         patch("tools.mcp_tool.register_mcp_servers", side_effect=fake_register):
        discover_mcp_tools(allowed_server_names={"allowed_http"})

    assert registered == ["allowed_http"]

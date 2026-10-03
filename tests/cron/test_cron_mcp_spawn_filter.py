"""Cron workers must spawn only the MCP servers the job's toolsets actually expose (#121536).

``_init_cron_mcp_tools`` used to call ``discover_mcp_tools()`` with no filter, so EVERY cron
agent run started, connected and kept EVERY configured MCP server for the whole run — even for
jobs whose ``enabled_toolsets`` contain the ``no_mcp`` sentinel, and jobs whose allowlist names
only some servers. The toolset filter still hid the tools from the agent; only the processes
were wasted.

``_resolve_cron_enabled_toolsets(job, cfg)`` already computes exactly what the agent will see
(``no_mcp`` strips every MCP name, an explicit MCP allowlist stays as-is, otherwise the merge
layers every enabled server in), so the spawn filter is the intersection of that list with the
configured enabled servers.
"""

from unittest.mock import patch

from cron.scheduler import _init_cron_mcp_tools

_CFG = {
    "mcp_servers": {
        "notion": {"url": "https://mcp.invalid/notion"},
        "slack": {"url": "https://mcp.invalid/slack"},
    }
}


def _record_discover():
    """Patch discover_mcp_tools with a spy that records the call shape (kwarg or zero-arg)."""
    calls = []

    def _fake(allowed_mcp_names=None):
        calls.append(allowed_mcp_names)
        return []

    return calls, patch("tools.mcp_tool_discovery.discover_mcp_tools", side_effect=_fake)


def test_no_mcp_job_spawns_no_mcp_servers():
    calls, spy = _record_discover()
    with spy:
        _init_cron_mcp_tools(
            {"id": "j1", "enabled_toolsets": ["no_mcp", "terminal"]}, _CFG)

    # RED pre-fix: recorded [None] — every configured server was spawned anyway.
    assert calls == [[]]


def test_named_mcp_allowlist_spawns_only_the_named_servers():
    calls, spy = _record_discover()
    with spy:
        _init_cron_mcp_tools(
            {"id": "j2", "enabled_toolsets": ["terminal", "notion"]}, _CFG)

    # "notion" is listed so the merge keeps the allowlist as-is (slack stays out of the
    # agent's view): spawn must match, not "all configured servers".
    # RED pre-fix: recorded [None].
    assert calls == [["notion"]]


def test_default_job_keeps_the_unfiltered_call_shape():
    calls, spy = _record_discover()
    with spy:
        # No per-job override: the platform resolution layers every enabled MCP server in,
        # so the filter equals "everything configured" and the zero-arg call shape stands
        # (out-of-tree callers/tests stub discover_mcp_tools as a zero-arg callable).
        _init_cron_mcp_tools({"id": "j3"}, _CFG)

    assert calls == [None]

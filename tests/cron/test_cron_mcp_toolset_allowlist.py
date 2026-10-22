"""A per-job MCP allowlist must stay narrow: naming one server is not naming all of them.

Regression for #141468. A job names an MCP server by its TOOLSET name (``mcp-<server>``), but the
merge helper compared the job's list against the BARE server names, so the intersection never
matched and the fallback unioned every enabled MCP server into the job's allowlist: a job that
asked for one server ran with all of them.
"""

from __future__ import annotations

import pytest

from cron.scheduler import _merge_mcp_into_per_job_toolsets


@pytest.fixture
def mcp_cfg():
    """Three enabled MCP servers, so a widened allowlist is visible in the result."""
    return {
        "mcp_servers": {
            "filesystem": {"command": "npx", "args": ["-y", "server-filesystem"]},
            "github": {"command": "npx", "args": ["-y", "server-github"]},
            "slack": {"command": "npx", "args": ["-y", "server-slack"]},
        }
    }


def test_a_job_naming_one_mcp_server_keeps_only_that_server(mcp_cfg):
    """The job's allowlist is an allowlist: one named server, not every enabled one."""
    assert _merge_mcp_into_per_job_toolsets(["mcp-filesystem"], mcp_cfg) == ["mcp-filesystem"]


def test_a_named_mcp_server_alongside_a_native_toolset_keeps_both(mcp_cfg):
    assert _merge_mcp_into_per_job_toolsets(["mcp-filesystem", "terminal"], mcp_cfg) == [
        "mcp-filesystem",
        "terminal",
    ]


def test_the_bare_server_name_means_the_same_allowlist(mcp_cfg):
    """The bare alias names the same server, so it also narrows the allowlist."""
    assert _merge_mcp_into_per_job_toolsets(["filesystem"], mcp_cfg) == ["filesystem"]


def test_a_native_toolset_allowlist_still_merges_every_enabled_server(mcp_cfg):
    """Unchanged: a list that names no MCP server keeps every enabled one, so an
    operator's native-toolset allowlist never silently strips MCP tools."""
    assert _merge_mcp_into_per_job_toolsets(["terminal"], mcp_cfg) == [
        "terminal",
        "filesystem",
        "github",
        "slack",
    ]


def test_the_no_mcp_sentinel_still_means_no_mcp_servers(mcp_cfg):
    assert _merge_mcp_into_per_job_toolsets(["no_mcp"], mcp_cfg) == []

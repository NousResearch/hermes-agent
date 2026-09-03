"""Regression tests for canonical MCP toolset names in cron allowlists."""

from unittest.mock import patch

from cron.scheduler import _merge_mcp_into_per_job_toolsets


CFG = {
    "mcp_servers": {
        "finnhub": {"enabled": True},
        "playwright": {"enabled": True},
        "disabled_finnhub": {"enabled": False},
    }
}


def test_canonical_mcp_toolset_name_is_an_explicit_allowlist() -> None:
    """A canonical mcp-* entry must not expand to every enabled MCP server."""
    with patch("hermes_cli.tools_config.enabled_mcp_server_names", return_value={"finnhub", "playwright"}):
        result = _merge_mcp_into_per_job_toolsets(["web", "mcp-finnhub"], CFG)

    assert result == ["web", "mcp-finnhub"]


def test_canonical_mcp_toolset_name_preserves_native_and_mcp_entries() -> None:
    """Canonical names remain intact when the allowlist contains other tools."""
    with patch("hermes_cli.tools_config.enabled_mcp_server_names", return_value={"finnhub", "playwright"}):
        result = _merge_mcp_into_per_job_toolsets(["terminal", "mcp-finnhub", "file"], CFG)

    assert result == ["terminal", "mcp-finnhub", "file"]


def test_disabled_canonical_mcp_toolset_name_is_still_an_explicit_allowlist() -> None:
    """A disabled canonical mcp-* entry must not expand to enabled MCP servers."""
    with patch("hermes_cli.tools_config.enabled_mcp_server_names", return_value={"playwright"}):
        result = _merge_mcp_into_per_job_toolsets(["web", "mcp-disabled_finnhub"], CFG)

    assert result == ["web", "mcp-disabled_finnhub"]


def test_only_canonical_mcp_toolset_name_is_still_an_explicit_allowlist() -> None:
    """A canonical-only allowlist must not be widened to every enabled server."""
    with patch("hermes_cli.tools_config.enabled_mcp_server_names", return_value={"playwright"}):
        result = _merge_mcp_into_per_job_toolsets(["mcp-disabled_finnhub"], CFG)

    assert result == ["mcp-disabled_finnhub"]

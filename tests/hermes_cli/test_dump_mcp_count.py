"""Regression for #38650: hermes dump reports mcp_servers: 0 even when
servers are configured.

The original bug was a wrong config key: ``_count_mcp_servers`` read
``config["mcp"]["servers"]`` (a path that never existed) instead of the
real top-level ``config["mcp_servers"]`` key. With the live registry
now preferred over the config, the function also has to handle the
"discovery hasn't run yet" case (fall back to config) and the
"discovery populated the registry" case (use the live count).
"""

import pytest
from unittest.mock import patch


def _count(config: dict) -> int:
    from hermes_cli.dump import _count_mcp_servers
    return _count_mcp_servers(config)


class TestCountMcpServersFromConfig:
    """When the live registry is empty, fall back to the config dict."""

    def test_zero_when_no_mcp_servers_key(self):
        assert _count({}) == 0

    def test_uses_top_level_mcp_servers_key(self):
        """The real Hermes config uses a top-level ``mcp_servers`` key, not
        ``mcp.servers``. The original implementation read the nested path
        and always returned 0 — that's the #38650 dump bug."""
        config = {
            "mcp_servers": {
                "fetch": {"command": "fetch-mcp"},
                "github": {"url": "https://api.github.com/mcp"},
            }
        }
        assert _count(config) == 2

    def test_does_not_read_legacy_nested_path(self):
        """Sanity check: the legacy ``mcp.servers`` key (which never
        existed in real configs) is NOT consulted. This pins the regression."""
        config = {
            "mcp": {"servers": {"legacy": {"command": "x"}}},
        }
        # The legacy path used to be read; with the fix, this is 0.
        assert _count(config) == 0

    def test_ignores_non_dict_mcp_servers(self):
        """If mcp_servers is somehow a list or string, don't crash."""
        assert _count({"mcp_servers": []}) == 0
        assert _count({"mcp_servers": "oops"}) == 0


class TestCountMcpServersPrefersLiveRegistry:
    """When the live registry is populated, the count comes from there."""

    def test_uses_live_registry_over_config(self):
        """Live registry is the source of truth at dump time — see #38650."""
        config = {"mcp_servers": {"a": {}, "b": {}, "c": {}}}  # 3 configured
        fake_live = {"a": object(), "b": object()}  # only 2 actually connected

        with patch("tools.mcp_tool._servers", fake_live, create=True):
            assert _count(config) == 2

    def test_falls_back_to_config_when_live_registry_empty(self):
        """If discovery hasn't run yet (_servers empty), fall back to the
        config count so the user still sees how many servers are planned."""
        config = {"mcp_servers": {"a": {}, "b": {}}}
        with patch("tools.mcp_tool._servers", {}, create=True):
            assert _count(config) == 2

    def test_live_registry_can_exceed_config_count(self):
        """Edge case: a user manually called register_mcp_servers with
        extra servers. The live count is correct even if config disagrees."""
        config = {"mcp_servers": {"a": {}}}  # 1 in config
        fake_live = {"a": object(), "extra": object()}  # 2 in registry
        with patch("tools.mcp_tool._servers", fake_live, create=True):
            assert _count(config) == 2

    def test_handles_missing_tools_module(self):
        """If tools.mcp_tool can't be imported (broken env, partial install),
        don't crash — fall back to config count."""
        config = {"mcp_servers": {"a": {}}}
        with patch.dict("sys.modules", {"tools.mcp_tool": None}):
            assert _count(config) == 1

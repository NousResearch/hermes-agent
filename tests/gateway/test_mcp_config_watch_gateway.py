"""The gateway auto-reloads MCP connections when config.yaml's ``mcp_servers``
section changes on disk — the always-on analogue of the TUI config watcher
(cli.HermesCLI._check_config_mcp_changes), covered by
tests/cli/test_cli_mcp_config_watch.py.

Exercises GatewayRunner._check_config_mcp_changes (one watcher tick) directly
with a mocked reload, so no real MCP servers are contacted.
"""

from unittest.mock import AsyncMock, patch

import pytest
import yaml


def _make_runner(tmp_path, mcp_servers=None):
    """Bare GatewayRunner with just the watcher state + a mocked reload."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._mcp_watch_mtime = 0.0  # forced stale so the parse path runs
    runner._mcp_watch_servers = dict(mcp_servers or {})
    # _reload_mcp_connections returns (added, removed, reconnected, new_tools).
    runner._reload_mcp_connections = AsyncMock(return_value=(set(), set(), set(), []))

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"mcp_servers": mcp_servers or {}}), encoding="utf-8")
    return runner, cfg


@pytest.mark.asyncio
async def test_new_server_triggers_gateway_reload(tmp_path):
    runner, cfg = _make_runner(tmp_path, mcp_servers={})
    cfg.write_text(
        yaml.dump({"mcp_servers": {"gh": {"url": "https://mcp.example.com"}}}),
        encoding="utf-8",
    )
    with patch("hermes_cli.config.get_config_path", return_value=cfg):
        await runner._check_config_mcp_changes()
    runner._reload_mcp_connections.assert_awaited_once()
    # Watcher state advanced to the new server set.
    assert runner._mcp_watch_servers == {"gh": {"url": "https://mcp.example.com"}}


@pytest.mark.asyncio
async def test_removed_server_triggers_gateway_reload(tmp_path):
    runner, cfg = _make_runner(tmp_path, mcp_servers={"gh": {"url": "x"}})
    cfg.write_text(yaml.dump({"mcp_servers": {}}), encoding="utf-8")
    with patch("hermes_cli.config.get_config_path", return_value=cfg):
        await runner._check_config_mcp_changes()
    runner._reload_mcp_connections.assert_awaited_once()


@pytest.mark.asyncio
async def test_unrelated_section_edit_does_not_reload(tmp_path):
    runner, cfg = _make_runner(tmp_path, mcp_servers={"fs": {"command": "npx"}})
    # Same mcp_servers, a different section changed.
    cfg.write_text(
        yaml.dump({"mcp_servers": {"fs": {"command": "npx"}}, "model": {"default": "x"}}),
        encoding="utf-8",
    )
    with patch("hermes_cli.config.get_config_path", return_value=cfg):
        await runner._check_config_mcp_changes()
    runner._reload_mcp_connections.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_config_is_noop(tmp_path):
    runner, _cfg = _make_runner(tmp_path, mcp_servers={})
    missing = tmp_path / "gone.yaml"
    with patch("hermes_cli.config.get_config_path", return_value=missing):
        await runner._check_config_mcp_changes()
    runner._reload_mcp_connections.assert_not_awaited()

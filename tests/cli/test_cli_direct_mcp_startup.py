"""Tests for direct ``cli.py`` startup MCP discovery parity."""

from __future__ import annotations

import runpy
import sys
from unittest.mock import MagicMock


def test_direct_cli_startup_discovers_mcp_tools(monkeypatch):
    calls: list[str] = []

    fake_mcp_tool = MagicMock()
    fake_mcp_tool.discover_mcp_tools.side_effect = lambda: calls.append("discover")

    fake_fire = MagicMock()
    fake_fire.Fire.side_effect = lambda fn: calls.append(f"fire:{fn.__name__}")

    monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mcp_tool)
    monkeypatch.setitem(sys.modules, "fire", fake_fire)

    runpy.run_module("cli", run_name="__main__")

    assert calls == ["discover", "fire:main"]

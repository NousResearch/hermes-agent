"""Tests for direct ``cli.py`` startup MCP discovery parity."""

from __future__ import annotations

import runpy
import sys
import threading
import time
import types
from contextlib import nullcontext
from unittest.mock import MagicMock


def test_direct_cli_startup_uses_shared_background_mcp_helper(monkeypatch):
    calls: list[str] = []

    fake_mcp_startup = MagicMock()
    fake_mcp_startup.start_background_mcp_discovery.side_effect = (
        lambda **kwargs: calls.append(f"start:{kwargs['thread_name']}")
    )

    fake_fire = MagicMock()
    fake_fire.Fire.side_effect = lambda fn: calls.append(f"fire:{fn.__name__}")

    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_startup", fake_mcp_startup)
    monkeypatch.setitem(sys.modules, "fire", fake_fire)

    runpy.run_module("cli", run_name="__main__")

    assert calls == ["start:hermes-direct-mcp-discovery", "fire:main"]


def test_direct_cli_startup_backgrounds_blocking_mcp_discovery(monkeypatch):
    import cli as cli_mod
    from hermes_cli import mcp_startup

    stop = threading.Event()
    calls: list[str] = []
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread

    def _blocking_discover():
        calls.append("discover")
        stop.wait()

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_blocking_discover),
    )
    mcp_startup._mcp_discovery_started = False
    mcp_startup._mcp_discovery_thread = None

    try:
        start = time.monotonic()
        cli_mod._run_direct_cli_startup()
        calls.append("fire")
        elapsed = time.monotonic() - start

        assert elapsed < 0.2
        assert "fire" in calls
        deadline = time.monotonic() + 3.0
        while "discover" not in calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "discover" in calls
        assert mcp_startup._mcp_discovery_thread is not None
        assert mcp_startup._mcp_discovery_thread.is_alive()
    finally:
        stop.set()
        thread = mcp_startup._mcp_discovery_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread

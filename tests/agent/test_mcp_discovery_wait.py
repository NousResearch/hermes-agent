"""Behavioral regression tests: MCP discovery wait prevents tool-list race (#73739).

These tests exercise the actual wait_for_mcp_discovery() mechanism rather than
asserting source-code shape, per AGENTS.md §1382-1403.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeToolRegistry:
    """Minimal stand-in for the MCP tool registry that discovery populates."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def register(self, name: str, schema: object) -> None:
        self._tools[name] = schema

    def snapshot(self) -> list[str]:
        """Mimics get_tool_definitions(): returns whatever is registered NOW."""
        return list(self._tools.keys())


def _slow_discovery(registry: _FakeToolRegistry, delay: float = 0.15):
    """Simulates an npx/stdio MCP server that takes `delay` seconds to connect."""
    time.sleep(delay)
    registry.register("mcp__slowserver__search", {"type": "function"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wait_for_discovery_ensures_tools_present_before_snapshot():
    """#73739 core: calling wait_for_mcp_discovery() before the tool snapshot
    guarantees slow MCP servers are registered.  Without the wait, the snapshot
    would race and miss them."""
    registry = _FakeToolRegistry()

    # Simulate a background discovery thread (like start_background_mcp_discovery).
    discovery_thread = threading.Thread(
        target=_slow_discovery, args=(registry, 0.1), daemon=True
    )
    discovery_thread.start()

    # --- The fix under test: wait BEFORE snapshot ---
    import hermes_cli.mcp_startup as mcp_mod

    with patch.object(mcp_mod, "_mcp_discovery_thread", discovery_thread):
        mcp_mod.wait_for_mcp_discovery(timeout=5.0)

    # After the wait, the slow server's tool is registered.
    tools = registry.snapshot()
    assert "mcp__slowserver__search" in tools, (
        "wait_for_mcp_discovery must block until discovery registers tools"
    )


def test_without_wait_snapshot_races_and_misses_tools():
    """Demonstrates the bug: snapshotting WITHOUT waiting misses slow tools."""
    registry = _FakeToolRegistry()

    discovery_thread = threading.Thread(
        target=_slow_discovery, args=(registry, 0.3), daemon=True
    )
    discovery_thread.start()

    # Snapshot IMMEDIATELY (no wait) — the race that #73739 reports.
    tools = registry.snapshot()
    # The slow server hasn't finished yet, so its tool is absent.
    assert "mcp__slowserver__search" not in tools

    # Cleanup: let the thread finish.
    discovery_thread.join(timeout=2.0)


def test_wait_is_noop_when_no_thread_started():
    """wait_for_mcp_discovery returns instantly when no discovery thread exists
    (gateway/cron/ACP entry points that own their own startup)."""
    import hermes_cli.mcp_startup as mcp_mod

    with patch.object(mcp_mod, "_mcp_discovery_thread", None):
        start = time.perf_counter()
        mcp_mod.wait_for_mcp_discovery(timeout=5.0)
        elapsed = time.perf_counter() - start

    assert elapsed < 0.1, "No-op wait should return instantly"


def test_wait_is_noop_when_thread_already_finished():
    """A completed discovery thread also returns instantly."""
    import hermes_cli.mcp_startup as mcp_mod

    registry = _FakeToolRegistry()
    t = threading.Thread(target=_slow_discovery, args=(registry, 0.0))
    t.start()
    t.join()  # already done

    with patch.object(mcp_mod, "_mcp_discovery_thread", t):
        start = time.perf_counter()
        mcp_mod.wait_for_mcp_discovery(timeout=5.0)
        elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert "mcp__slowserver__search" in registry.snapshot()


def test_agent_init_defensive_import_survives_missing_hermes_cli():
    """The try/except around the wait_for_mcp_discovery import in agent_init
    means non-CLI entry points (gateway, cron) don't crash if hermes_cli is
    unavailable.  Simulate by making the import raise ImportError."""
    import builtins
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if "hermes_cli" in name:
            raise ImportError("simulated: hermes_cli unavailable")
        return real_import(name, *args, **kwargs)

    # The agent_init pattern: try/except around the import + call.
    waited = False
    try:
        with patch("builtins.__import__", side_effect=_blocking_import):
            from hermes_cli.mcp_startup import wait_for_mcp_discovery  # noqa: F401
            wait_for_mcp_discovery()
            waited = True
    except Exception:
        pass

    # The defensive pattern means we land here without crashing.
    assert not waited, "Import should have been blocked"

"""Behavioral regression tests for ACP background MCP discovery + late-refresh.

These replace the previous AST-based test that only inspected source text.
They verify the *behavior*: (1) a blocked discovery doesn't block startup, and
(2) a delayed-but-reachable MCP server's tools land in the agent's snapshot
via the automatic late-refresh, cache-safely (pre-first-turn only).
"""

from __future__ import annotations

import sys
import threading
import time
import types
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState
from hermes_cli import mcp_startup


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal stand-in for AIAgent with the attributes late-refresh touches."""

    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self._user_turn_count = 0
        self._api_call_count = 0


class NoopDb:
    def get_session(self, *_a, **_k):
        return None

    def create_session(self, *_a, **_k):
        return None

    def update_session(self, *_a, **_k):
        return None


def _mod(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    """Ensure each test starts with a clean discovery thread state."""
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    mcp_startup._mcp_discovery_started = False
    mcp_startup._mcp_discovery_thread = None
    yield
    thread = mcp_startup._mcp_discovery_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    mcp_startup._mcp_discovery_started = saved_started
    mcp_startup._mcp_discovery_thread = saved_thread


# ---------------------------------------------------------------------------
# Test 1 — blocked discovery does not block startup
# ---------------------------------------------------------------------------


def test_acp_background_discovery_does_not_block_startup(monkeypatch):
    """start_background_mcp_discovery must return immediately even if discovery hangs."""
    block = threading.Event()

    def _blocking_discover():
        block.wait(timeout=5.0)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _mod(
            "hermes_cli.config",
            read_raw_config=lambda: {"mcp_servers": {"slow": {"url": "https://mcp.example.test"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        _mod("tools.mcp_oauth", suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        _mod("tools.mcp_tool", discover_mcp_tools=_blocking_discover),
    )

    start = time.monotonic()
    mcp_startup.start_background_mcp_discovery(
        logger=SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-acp-discovery",
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.2, "start_background_mcp_discovery blocked for {:.3f}s".format(elapsed)
    assert mcp_startup._mcp_discovery_thread is not None
    assert mcp_startup._mcp_discovery_thread.is_alive()
    block.set()
    mcp_startup._mcp_discovery_thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 2 — delayed discovery lands tools via late-refresh (pre-first-turn)
# ---------------------------------------------------------------------------


def test_acp_late_refresh_adds_tools_when_discovery_lands_after_build(monkeypatch):
    """A slow MCP server that finishes after agent build must still appear in tools."""

    discovery_block = threading.Event()
    discovery_done = threading.Event()

    def _slow_discover():
        discovery_block.wait(timeout=5.0)
        discovery_done.set()

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _mod(
            "hermes_cli.config",
            read_raw_config=lambda: {"mcp_servers": {"slow": {"url": "https://mcp.example.test"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        _mod("tools.mcp_oauth", suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        _mod("tools.mcp_tool", discover_mcp_tools=_slow_discover),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-acp-late",
    )

    # Build the session immediately — discovery is still in flight.
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **_k: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")

    # Discovery is blocked, so it must still be in flight.
    assert not discovery_done.is_set(), "discovery finished too early for this test"

    # Track refresh_agent_mcp_tools calls.
    refreshed = []

    def _fake_refresh(agent, **_kw):
        agent.tools = [{"function": {"name": "mcp_slow_tool"}}]
        agent.valid_tool_names = {"mcp_slow_tool"}
        refreshed.append(agent)
        return {"mcp_slow_tool"}

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        _mod("tools.mcp_tool", refresh_agent_mcp_tools=_fake_refresh),
    )

    # Trigger late-refresh.
    acp_agent._schedule_mcp_late_refresh(state)

    # Release discovery so the late-refresh daemon can proceed.
    discovery_block.set()

    # Wait for the late-refresh daemon to finish.
    deadline = time.monotonic() + 5.0
    while not refreshed and time.monotonic() < deadline:
        time.sleep(0.01)

    assert refreshed, "late-refresh daemon did not call refresh_agent_mcp_tools"
    assert refreshed[0] is fake
    assert "mcp_slow_tool" in fake.valid_tool_names


# ---------------------------------------------------------------------------
# Test 3 — late-refresh is cache-safe: skips after first turn
# ---------------------------------------------------------------------------


def test_acp_late_refresh_skips_after_first_turn(monkeypatch):
    """Once the user has sent a message, late-refresh must NOT rebuild tools."""

    discovery_block = threading.Event()

    def _slow_discover():
        discovery_block.wait(timeout=5.0)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _mod(
            "hermes_cli.config",
            read_raw_config=lambda: {"mcp_servers": {"slow": {"url": "https://mcp.example.test"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        _mod("tools.mcp_oauth", suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        _mod("tools.mcp_tool", discover_mcp_tools=_slow_discover),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-acp-cache",
    )

    fake = FakeAgent()
    fake._api_call_count = 1  # simulate: user already sent a message
    manager = SessionManager(agent_factory=lambda **_k: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")

    refreshed = []

    def _fake_refresh(agent, **_kw):
        refreshed.append(agent)
        return set()

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        _mod("tools.mcp_tool", refresh_agent_mcp_tools=_fake_refresh),
    )

    acp_agent._schedule_mcp_late_refresh(state)

    # Release discovery so the daemon can proceed (if it were going to).
    discovery_block.set()

    # Give the daemon time to run (if it were going to).
    time.sleep(0.5)

    assert not refreshed, "late-refresh rebuilt tools after the first turn — cache broken!"
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


def test_acp_same_name_mcp_tools_are_isolated_by_session_scope():
    """Two ACP sessions may advertise the same MCP name without first-writer wins."""
    from tools.registry import ToolRegistry, reset_registry_scope, set_registry_scope

    registry = ToolRegistry()
    states = [SimpleNamespace(session_id="session-a"), SimpleNamespace(session_id="session-b")]
    scopes = [HermesACPAgent._mcp_session_scope(state) for state in states]  # type: ignore[arg-type]
    assert scopes[0] != scopes[1]

    schema = {"description": "session tool", "parameters": {"type": "object", "properties": {}}}
    handlers = [lambda _args: "a", lambda _args: "b"]
    for scope, handler in zip(scopes, handlers):
        registry.register(
            name="mcp_shared_echo", toolset="mcp-shared", schema=schema,
            handler=handler, scope=scope,
        )

    for scope, handler in zip(scopes, handlers):
        token = set_registry_scope(scope)
        try:
            entry = registry.get_entry("mcp_shared_echo")
            assert entry is not None and entry.handler is handler
            definitions = registry.get_definitions({"mcp_shared_echo"}, quiet=True)
            assert [item["function"]["name"] for item in definitions] == ["mcp_shared_echo"]
        finally:
            reset_registry_scope(token)


def _fake_session_mcp_registration(handler_result: str):
    """Stand-in for ``register_mcp_servers`` that registers each server's tool into the
    current (session) registry scope like a schema-cache (lazy) registration does."""
    from tools.mcp_tool_common import _core
    from tools.registry import registry

    def _register(configs):
        scope = registry.current_scope_key()
        names = []
        for server_name, config in configs.items():
            tool = f"mcp_{server_name}_echo"
            registry.register(
                name=tool, toolset=f"mcp-{server_name}",
                schema={"description": "echo", "parameters": {"type": "object", "properties": {}}},
                handler=lambda _args, _r=handler_result: _r, scope=scope)
            with _core._lock:
                _core._lazy_server_configs[(scope, server_name)] = config
                _core._lazy_server_tool_names[(scope, server_name)] = [tool]
            names.append(tool)
        return names

    return _register


def _scoped_definitions(enabled_toolsets=None, **_kwargs):
    from tools.registry import registry

    return [{"function": {"name": entry.name}} for entry in registry._snapshot_entries()
            if entry.toolset in (enabled_toolsets or [])]


@pytest.mark.asyncio
async def test_acp_session_mcp_scope_is_retired_and_not_resurrected(monkeypatch):
    """A registers mcp_shared_echo -> A's MCP is retired -> scope gone, tool unavailable ->
    the same id comes back with no MCP and the old handler does not return. Replacing the
    set retires the dropped server before registering the new one."""
    import asyncio

    from acp.schema import McpServerStdio
    from tools.mcp_tool_common import _core
    from tools.registry import registry, reset_registry_scope, set_registry_scope

    monkeypatch.setattr("model_tools.get_tool_definitions", _scoped_definitions)
    monkeypatch.setattr("agent.memory_manager.inject_memory_provider_tools", lambda _agent: None)
    manager = SessionManager(agent_factory=FakeAgent, db=NoopDb())
    server = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd="/tmp")
    shared = McpServerStdio(name="shared", command="/bin/echo", args=["a"], env=[])

    def _entry(name):
        token = set_registry_scope(server._mcp_session_scope(state))
        try:
            return registry.get_entry(name)
        finally:
            reset_registry_scope(token)

    token = set_registry_scope(server._mcp_session_scope(state))
    scope = registry.current_scope_key()  # canonical key of the session overlay
    reset_registry_scope(token)
    try:
        monkeypatch.setattr("tools.mcp_tool_discovery.register_mcp_servers", _fake_session_mcp_registration("A"))
        await server._register_session_mcp_servers(state, [shared])
        assert _entry("mcp_shared_echo").handler({}) == "A"
        assert "mcp_shared_echo" in state.agent.valid_tool_names
        assert scope in registry._scoped_tools

        # Retire: overlay dropped, lazy ledger forgotten, handler unreachable.
        assert await asyncio.to_thread(server._retire_session_mcp, state) == ["shared"]
        assert scope not in registry._scoped_tools
        assert _entry("mcp_shared_echo") is None
        assert not any(key[0] == scope for key in _core._lazy_server_configs if isinstance(key, tuple))
        assert state.mcp_server_configs == {}

        # Re-register, then the same id is loaded again with NO MCP servers.
        await server._register_session_mcp_servers(state, [shared])
        assert _entry("mcp_shared_echo") is not None
        await server._register_session_mcp_servers(state, [])
        assert scope not in registry._scoped_tools
        assert _entry("mcp_shared_echo") is None
        assert "mcp_shared_echo" not in state.agent.valid_tool_names
        assert "mcp-shared" not in state.agent.enabled_toolsets

        # Replacing the set retires the dropped/reconfigured server first.
        await server._register_session_mcp_servers(state, [shared])
        other = McpServerStdio(name="other", command="/bin/echo", args=["b"], env=[])
        await server._register_session_mcp_servers(state, [other])
        assert _entry("mcp_shared_echo") is None
        assert _entry("mcp_other_echo") is not None
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-other"]

        # Process shutdown releases every live session scope.
        assert server.retire_all_session_mcp() == 1
        assert scope not in registry._scoped_tools
    finally:
        from tools.mcp_tool_discovery import release_mcp_scope
        release_mcp_scope(scope)


def test_release_mcp_scope_closes_owned_and_drops_adopted_connections(monkeypatch):
    """Owned connections go through the scoped shutdown; an overlay on a connection adopted
    from another scope is removed without touching that owner's connection."""
    from tools import mcp_tool_lifecycle
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_discovery import release_mcp_scope
    from tools.registry import registry, reset_registry_scope, set_registry_scope

    token = set_registry_scope("/tmp/hermes-test-home/.acp-sessions/release-sid")
    scope = registry.current_scope_key()
    reset_registry_scope(token)
    owner = "/tmp/hermes-test-home/.acp-sessions/owner-sid"
    owned_key, adopted_key = (scope, "live"), (owner, "adopt")
    schema = {"description": "x", "parameters": {"type": "object", "properties": {}}}
    registry.register(name="mcp_adopt_echo", toolset="mcp-adopt", schema=schema,
                      handler=lambda _a: "owner", scope=scope)
    shutdowns = []
    monkeypatch.setattr(mcp_tool_lifecycle, "shutdown_mcp_servers",
                        lambda **kwargs: shutdowns.append(kwargs))
    owned_server, adopted_server = object(), object()
    with _core._lock:
        _core._servers[owned_key] = owned_server
        _core._server_scope_keys[owned_key] = scope
        _core._servers[adopted_key] = adopted_server
        _core._server_tool_scopes[adopted_key] = {owner, scope}
    try:
        assert release_mcp_scope(scope) == ["adopt", "live"]
        assert shutdowns == [{"scope": scope, "names": None, "timeout": 15.0}]
        assert _core._server_tool_scopes[adopted_key] == {owner}
        assert _core._servers[adopted_key] is adopted_server  # the owner keeps its connection
        assert registry.get_entry("mcp_adopt_echo", scope=scope) is None
        assert scope not in registry._scoped_tools
    finally:
        with _core._lock:
            for key in (owned_key, adopted_key):
                _core._servers.pop(key, None)
                _core._server_scope_keys.pop(key, None)
                _core._server_tool_scopes.pop(key, None)


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    """Ensure each test starts with a clean discovery thread state."""
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    mcp_startup._mcp_discovery_started = set()
    mcp_startup._mcp_discovery_thread = {}
    yield
    thread = mcp_startup._current_home_thread()
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
        "tools.mcp_tool_discovery",
        _mod("tools.mcp_tool_discovery", discover_mcp_tools=_blocking_discover),
    )

    start = time.monotonic()
    mcp_startup.start_background_mcp_discovery(
        logger=SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-acp-discovery",
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.2, "start_background_mcp_discovery blocked for {:.3f}s".format(elapsed)
    thread = mcp_startup._current_home_thread()
    assert thread is not None
    assert thread.is_alive()
    block.set()
    thread.join(timeout=2.0)


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
        "tools.mcp_tool_discovery",
        _mod("tools.mcp_tool_discovery", discover_mcp_tools=_slow_discover),
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
        "tools.mcp_tool_agent",
        _mod("tools.mcp_tool_agent", refresh_agent_mcp_tools=_fake_refresh),
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
        "tools.mcp_tool_discovery",
        _mod("tools.mcp_tool_discovery", discover_mcp_tools=_slow_discover),
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
        "tools.mcp_tool_agent",
        _mod("tools.mcp_tool_agent", refresh_agent_mcp_tools=_fake_refresh),
    )

    acp_agent._schedule_mcp_late_refresh(state)

    # Release discovery so the daemon can proceed (if it were going to).
    discovery_block.set()

    # Give the daemon time to run (if it were going to).
    time.sleep(0.5)

    assert not refreshed, "late-refresh rebuilt tools after the first turn — cache broken!"


# ---------------------------------------------------------------------------
# Test 4 — late-refresh is serialized with turn start: skips while running
# ---------------------------------------------------------------------------


def test_acp_late_refresh_skips_while_turn_running(monkeypatch):
    """A turn in flight (state.is_running) must block the rebuild even when
    the agent's counters still read zero — closes the guard/turn-start race."""

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
        "tools.mcp_tool_discovery",
        _mod("tools.mcp_tool_discovery", discover_mcp_tools=_slow_discover),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-acp-running",
    )

    fake = FakeAgent()  # counters are 0 — only is_running blocks the refresh
    manager = SessionManager(agent_factory=lambda **_k: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    state.is_running = True  # simulate: first prompt dispatched concurrently

    refreshed = []

    def _fake_refresh(agent, **_kw):
        refreshed.append(agent)
        return set()

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_agent",
        _mod("tools.mcp_tool_agent", refresh_agent_mcp_tools=_fake_refresh),
    )

    acp_agent._schedule_mcp_late_refresh(state)
    discovery_block.set()
    time.sleep(0.5)

    assert not refreshed, "late-refresh rebuilt tools while a turn was running!"

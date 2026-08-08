"""Tests for mid-turn MCP tool refresh in the conversation loop.

When an MCP server changes its tool set while handling a call (e.g. a server
that swaps between a minimal and a full toolset when a mode-change tool is
called), the registry updates via ``ToolListChanged`` but ``agent.tools``
still holds the stale snapshot.  Worse, the notification races the tool-call
response it follows and can land *after* the post-execution checkpoint, so a
snapshot rebuild alone still misses the change (issue #65428).

The mid-turn sync after ``_execute_tool_calls()`` closes both gaps:
``refresh_agent_tools_after_mcp_calls`` re-polls ``tools/list`` on the
just-called servers whose lists are known dynamic, then rebuilds
``agent.tools`` from the registry.

These test the *contract*: after tool execution, the manifest presented to
the next API call reflects the server's post-call tool list — without
depending on notification arrival order.

The scenario used throughout is a fictitious dynamic server exposing
``swap_in_full_toolset`` in its minimal mode; calling it replaces the tool
set with the full toolset (``probe_state``, ``poke_state``,
``swap_in_minimal_toolset``), and ``swap_in_minimal_toolset`` swaps back.
"""

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _make_agent(tool_names):
    """Minimal agent with the attributes the mid-turn refresh touches."""
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a.enabled_toolsets = None
    a.disabled_toolsets = None
    a._skip_mcp_refresh = False
    a._tool_snapshot_generation = 0
    return a


def _simulate_mid_turn_refresh(agent, called_tool_names=()):
    """Run the mid-turn MCP refresh code path extracted from conversation_loop.py.

    This executes the same logic as the inline block after
    ``_execute_tool_calls()`` in the conversation loop — the try/except
    with the three guards (_skip_mcp_refresh, sys.modules, has_registered).
    """
    try:
        if not getattr(agent, "_skip_mcp_refresh", False):
            if "tools.mcp_tool" in sys.modules:
                from tools.mcp_tool import (
                    has_registered_mcp_tools,
                    refresh_agent_tools_after_mcp_calls,
                )
                if has_registered_mcp_tools():
                    _mcp_added = refresh_agent_tools_after_mcp_calls(
                        agent, set(called_tool_names), quiet_mode=True
                    )
                    return _mcp_added
    except Exception:
        pass
    return set()


class TestMidTurnMcpRefresh:
    """Mid-turn MCP tool refresh contracts (mirrors between-turns tests)."""

    def test_adds_dynamically_registered_tool(self, monkeypatch):
        """After tool execution, a newly registered MCP tool appears in agent.tools."""
        agent = _make_agent(["swap_in_full_toolset"])

        # Simulate: the server swapped its minimal toolset for the full one.
        new_defs = [
            _tool(n)
            for n in ("probe_state", "poke_state", "swap_in_minimal_toolset")
        ]
        import model_tools
        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: list(new_defs))

        with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True):
            added = _simulate_mid_turn_refresh(agent)

        assert "probe_state" in added
        assert "poke_state" in added
        assert "swap_in_minimal_toolset" in added
        assert "swap_in_full_toolset" not in agent.valid_tool_names
        assert "probe_state" in agent.valid_tool_names

    def test_no_change_returns_empty(self, monkeypatch):
        """Unchanged tool set → empty set, snapshot not swapped."""
        agent = _make_agent(["read_file", "terminal"])
        original_tools = agent.tools

        import model_tools
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("read_file"), _tool("terminal")],
        )

        with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True):
            added = _simulate_mid_turn_refresh(agent)

        assert added == set()
        assert agent.tools is original_tools  # no churn

    def test_skipped_when_no_mcp_servers(self, monkeypatch):
        """No MCP servers registered → refresh never walks the registry."""
        agent = _make_agent(["read_file"])

        import model_tools
        gtd = MagicMock()
        monkeypatch.setattr(model_tools, "get_tool_definitions", gtd)

        with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False):
            added = _simulate_mid_turn_refresh(agent)

        assert added == set()
        gtd.assert_not_called()

    def test_skipped_when_skip_flag_set(self, monkeypatch):
        """_skip_mcp_refresh=True → refresh bypassed even with MCP servers."""
        agent = _make_agent(["read_file"])
        agent._skip_mcp_refresh = True

        import model_tools
        gtd = MagicMock()
        monkeypatch.setattr(model_tools, "get_tool_definitions", gtd)

        with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True):
            added = _simulate_mid_turn_refresh(agent)

        assert added == set()
        gtd.assert_not_called()

    def test_detects_tool_swap_same_count(self, monkeypatch):
        """Name-based diff catches add+remove of equal count (a one-for-one swap)."""
        agent = _make_agent(["swap_in_full_toolset"])

        import model_tools
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("swap_in_minimal_toolset")],
        )

        with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True):
            added = _simulate_mid_turn_refresh(agent)

        assert added == {"swap_in_minimal_toolset"}
        assert "swap_in_full_toolset" not in agent.valid_tool_names
        assert "swap_in_minimal_toolset" in agent.valid_tool_names

    def test_exception_is_swallowed(self, monkeypatch):
        """Refresh failure must not crash the tool loop."""
        agent = _make_agent(["read_file"])

        with patch("tools.mcp_tool.has_registered_mcp_tools", side_effect=RuntimeError("boom")):
            # Should not raise
            added = _simulate_mid_turn_refresh(agent)

        assert added == set()
        # Agent tools unchanged
        assert agent.valid_tool_names == {"read_file"}


class _FakeServer:
    """Stand-in for MCPServerTask exposing what the post-call sync touches."""

    def __init__(self, name, *, dynamic=True, has_session=True):
        self.name = name
        self.session = object() if has_session else None
        self._dynamic = dynamic
        self.poll_count = 0

    def _has_dynamic_tool_list(self):
        return self._dynamic

    async def _refresh_tools(self):
        self.poll_count += 1


def _install_servers(monkeypatch, tool_to_server, servers):
    """Point the module-level server maps at fakes for the duration of a test."""
    import tools.mcp_tool as mcp_tool
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", dict(tool_to_server))
    monkeypatch.setattr(mcp_tool, "_servers", {s.name: s for s in servers})


def _run_coro_inline(coro_or_factory, timeout=30):
    """Test double for _run_on_mcp_loop: run the coroutine synchronously."""
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    return asyncio.run(coro)


class TestNotificationOrderingPoll:
    """The tools/list re-poll that defeats the response/notification race.

    The race being simulated: the server already swapped its tool set while
    handling the call, but the ToolListChanged notification hasn't arrived,
    so the registry (here: get_tool_definitions) still serves the OLD list
    until the server is explicitly re-polled.
    """

    def _race_gtd(self, server, old_defs, new_defs):
        """get_tool_definitions that stays stale until the server was polled."""
        return lambda **kw: list(new_defs) if server.poll_count else list(old_defs)

    def test_poll_freshens_manifest_before_next_api_call(self, monkeypatch):
        """Swap-back direction: registry is stale at the checkpoint; the re-poll fixes it."""
        agent = _make_agent([
            "mcp__dyn_server__probe_state",
            "mcp__dyn_server__swap_in_minimal_toolset",
        ])
        server = _FakeServer("dyn_server", dynamic=True)
        _install_servers(
            monkeypatch,
            {
                "mcp__dyn_server__probe_state": "dyn_server",
                "mcp__dyn_server__swap_in_minimal_toolset": "dyn_server",
            },
            [server],
        )

        import model_tools
        old_defs = [
            _tool("mcp__dyn_server__probe_state"),
            _tool("mcp__dyn_server__swap_in_minimal_toolset"),
        ]
        new_defs = [_tool("mcp__dyn_server__swap_in_full_toolset")]
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            self._race_gtd(server, old_defs, new_defs),
        )

        import tools.mcp_tool as mcp_tool
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _run_coro_inline)

        added = mcp_tool.refresh_agent_tools_after_mcp_calls(
            agent, {"mcp__dyn_server__swap_in_minimal_toolset"}
        )

        assert server.poll_count == 1
        assert added == {"mcp__dyn_server__swap_in_full_toolset"}
        assert "mcp__dyn_server__swap_in_full_toolset" in agent.valid_tool_names
        assert "mcp__dyn_server__probe_state" not in agent.valid_tool_names

    def test_sanitized_server_name_still_polls(self, monkeypatch):
        """Hyphenated config name vs sanitized mapping value must still match.

        ``_mcp_tool_server_names`` stores sanitized names (``dyn_server``)
        while ``_servers`` is keyed by the config name (``dyn-server``).
        A naive key lookup silently skips the poll for every such server —
        the exact gap found in live testing of PR #65436.
        """
        agent = _make_agent([
            "mcp__dyn_server__probe_state",
            "mcp__dyn_server__swap_in_minimal_toolset",
        ])
        server = _FakeServer("dyn-server", dynamic=True)
        import tools.mcp_tool as mcp_tool
        monkeypatch.setattr(
            mcp_tool, "_mcp_tool_server_names",
            {"mcp__dyn_server__swap_in_minimal_toolset": "dyn_server"},
        )
        monkeypatch.setattr(mcp_tool, "_servers", {"dyn-server": server})

        import model_tools
        old_defs = [
            _tool("mcp__dyn_server__probe_state"),
            _tool("mcp__dyn_server__swap_in_minimal_toolset"),
        ]
        new_defs = [_tool("mcp__dyn_server__swap_in_full_toolset")]
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            self._race_gtd(server, old_defs, new_defs),
        )
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _run_coro_inline)

        added = mcp_tool.refresh_agent_tools_after_mcp_calls(
            agent, {"mcp__dyn_server__swap_in_minimal_toolset"}
        )

        assert server.poll_count == 1
        assert added == {"mcp__dyn_server__swap_in_full_toolset"}

    def test_static_server_is_not_polled(self, monkeypatch):
        """No listChanged capability and never notified → no extra RPC."""
        agent = _make_agent(["mcp__files__read"])
        server = _FakeServer("files", dynamic=False)
        _install_servers(monkeypatch, {"mcp__files__read": "files"}, [server])

        import model_tools
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("mcp__files__read")],
        )

        import tools.mcp_tool as mcp_tool
        run_loop = MagicMock(side_effect=_run_coro_inline)
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_loop)

        added = mcp_tool.refresh_agent_tools_after_mcp_calls(agent, {"mcp__files__read"})

        run_loop.assert_not_called()
        assert server.poll_count == 0
        assert added == set()

    def test_non_mcp_tools_do_not_poll(self, monkeypatch):
        """Built-in tool names map to no server → no poll, plain refresh only."""
        agent = _make_agent(["terminal"])
        server = _FakeServer("dyn_server", dynamic=True)
        _install_servers(
            monkeypatch,
            {"mcp__dyn_server__swap_in_minimal_toolset": "dyn_server"},
            [server],
        )

        import model_tools
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("terminal")],
        )

        import tools.mcp_tool as mcp_tool
        run_loop = MagicMock(side_effect=_run_coro_inline)
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_loop)

        added = mcp_tool.refresh_agent_tools_after_mcp_calls(agent, {"terminal"})

        run_loop.assert_not_called()
        assert added == set()

    def test_server_without_session_is_skipped(self, monkeypatch):
        """A recycled/disconnected server (session=None) is not polled."""
        agent = _make_agent(["mcp__dyn_server__swap_in_minimal_toolset"])
        server = _FakeServer("dyn_server", dynamic=True, has_session=False)
        _install_servers(
            monkeypatch,
            {"mcp__dyn_server__swap_in_minimal_toolset": "dyn_server"},
            [server],
        )

        import model_tools
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("mcp__dyn_server__swap_in_minimal_toolset")],
        )

        import tools.mcp_tool as mcp_tool
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _run_coro_inline)

        mcp_tool.refresh_agent_tools_after_mcp_calls(
            agent, {"mcp__dyn_server__swap_in_minimal_toolset"}
        )

        assert server.poll_count == 0

    def test_poll_failure_fails_soft(self, monkeypatch):
        """A timed-out poll is swallowed; the snapshot rebuild still runs."""
        agent = _make_agent(["mcp__dyn_server__swap_in_minimal_toolset"])
        server = _FakeServer("dyn_server", dynamic=True)
        _install_servers(
            monkeypatch,
            {"mcp__dyn_server__swap_in_minimal_toolset": "dyn_server"},
            [server],
        )

        import model_tools
        # Registry happens to be fresh despite the failed poll (notification
        # won the race after all) — the rebuild must still pick it up.
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("mcp__dyn_server__swap_in_full_toolset")],
        )

        import tools.mcp_tool as mcp_tool

        def _boom(coro_or_factory, timeout=30):
            if callable(coro_or_factory):
                coro_or_factory().close()
            raise TimeoutError("MCP call timed out")

        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _boom)

        added = mcp_tool.refresh_agent_tools_after_mcp_calls(
            agent, {"mcp__dyn_server__swap_in_minimal_toolset"}
        )

        assert added == {"mcp__dyn_server__swap_in_full_toolset"}

    def test_interrupt_aborts_remaining_polls(self, monkeypatch):
        """InterruptedError stops polling other servers but still rebuilds."""
        agent = _make_agent(["mcp__srv_a__x", "mcp__srv_b__y"])
        server_a = _FakeServer("srv_a", dynamic=True)
        server_b = _FakeServer("srv_b", dynamic=True)
        _install_servers(
            monkeypatch,
            {"mcp__srv_a__x": "srv_a", "mcp__srv_b__y": "srv_b"},
            [server_a, server_b],
        )

        import model_tools
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [_tool("mcp__srv_a__x"), _tool("mcp__srv_b__y")],
        )

        import tools.mcp_tool as mcp_tool
        calls = []

        def _interrupt(coro_or_factory, timeout=30):
            if callable(coro_or_factory):
                coro_or_factory().close()
            calls.append(1)
            raise InterruptedError("User sent a new message")

        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _interrupt)

        added = mcp_tool.refresh_agent_tools_after_mcp_calls(
            agent, {"mcp__srv_a__x", "mcp__srv_b__y"}
        )

        # Servers are polled in sorted-name order; the first raises and the
        # second must not be attempted.
        assert len(calls) == 1
        assert added == set()


class TestDynamicToolListGates:
    """Capability/latch gates deciding which servers get the re-poll."""

    @staticmethod
    def _fake_task(init_result, seen=False):
        """Host object carrying the real gate methods over fake attributes.

        MCPServerTask.__init__ spins up asyncio primitives, so instead of
        instantiating one we borrow the (attribute-only) gate methods onto a
        plain class.
        """
        from tools.mcp_tool import MCPServerTask

        class _Host:
            _advertises_tool_list_changed = MCPServerTask._advertises_tool_list_changed
            _has_dynamic_tool_list = MCPServerTask._has_dynamic_tool_list

        host = _Host()
        host.initialize_result = init_result
        host._tool_list_changed_seen = seen
        return host

    @staticmethod
    def _init_result(list_changed):
        return types.SimpleNamespace(
            capabilities=types.SimpleNamespace(
                tools=types.SimpleNamespace(listChanged=list_changed)
            )
        )

    def test_declared_listchanged_true(self):
        fake = self._fake_task(self._init_result(True))
        assert fake._advertises_tool_list_changed() is True
        assert fake._has_dynamic_tool_list() is True

    def test_declared_listchanged_false(self):
        fake = self._fake_task(self._init_result(False))
        assert fake._advertises_tool_list_changed() is False
        assert fake._has_dynamic_tool_list() is False

    def test_no_capability_info_defaults_to_static(self):
        """Unlike _advertises_tools, missing init info must NOT add RPCs."""
        fake = self._fake_task(None)
        assert fake._advertises_tool_list_changed() is False

    def test_seen_notification_latch_wins(self):
        """A server that notified without declaring listChanged is dynamic."""
        fake = self._fake_task(self._init_result(False), seen=True)
        assert fake._has_dynamic_tool_list() is True

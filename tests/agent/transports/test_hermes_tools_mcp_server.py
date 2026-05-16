"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )

    def test_expected_hermes_specific_tools_listed(self):
        """The Hermes-specific tools should be present so users on the
        codex runtime keep access to them."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for required in (
            "web_search",
            "web_extract",
            "browser_navigate",
            "vision_analyze",
            "image_generate",
            "skill_view",
            "memory",
            "session_search",
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_stateful_agent_loop_tools_not_exposed(self):
        """delegate_task / todo require running AIAgent/TodoStore state,
        so a stateless MCP callback can't drive them. They must NOT be in
        EXPOSED_TOOLS."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for agent_loop_tool in ("delegate_task", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS, (
                f"{agent_loop_tool!r} requires the agent loop context "
                "and can't be reached through a stateless MCP callback"
            )

    def test_stateless_agent_loop_dispatchers_cover_memory_and_session_search(self):
        """memory/session_search are blocked by model_tools.handle_function_call,
        so the MCP server must route them through its local wrappers."""
        from agent.transports.hermes_tools_mcp_server import (
            _STATELESS_AGENT_LOOP_DISPATCHERS,
        )

        assert set(_STATELESS_AGENT_LOOP_DISPATCHERS) == {
            "memory",
            "session_search",
        }

    def test_memory_stateless_dispatch_uses_profile_memory_store(self, tmp_path, monkeypatch):
        """The codex MCP subprocess has no parent AIAgent, but it does inherit
        HERMES_HOME. Its memory wrapper should load and mutate that store."""
        from agent.transports.hermes_tools_mcp_server import _dispatch_memory_stateless

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        result = _dispatch_memory_stateless(
            action="add",
            target="memory",
            content="Codex runtime can write profile memory.",
        )

        assert '"success": true' in result
        memory_file = home / "memories" / "MEMORY.md"
        assert "Codex runtime can write profile memory." in memory_file.read_text()

    def test_session_search_stateless_dispatch_forwards_args(self, monkeypatch):
        from agent.transports import hermes_tools_mcp_server as m

        calls = []

        def fake_session_search(**kwargs):
            calls.append(kwargs)
            return '{"success": true}'

        import tools.session_search_tool as session_search_tool

        monkeypatch.setattr(session_search_tool, "session_search", fake_session_search)

        assert m._dispatch_session_search_stateless(
            query="oauth",
            role_filter="assistant",
            limit="2",
            current_session_id="sid-123",
        ) == '{"success": true}'
        assert calls == [{
            "query": "oauth",
            "role_filter": "assistant",
            "limit": 2,
            "current_session_id": "sid-123",
        }]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, 3),
            ("2", 2),
            (10, 5),
            (0, 1),
            ("not-an-int", 3),
            (True, 3),
        ],
    )
    def test_session_search_limit_is_normalized_before_stateless_dispatch(
        self,
        raw,
        expected,
        monkeypatch,
    ):
        from agent.transports import hermes_tools_mcp_server as m
        import tools.session_search_tool as session_search_tool

        calls = []

        def fake_session_search(**kwargs):
            calls.append(kwargs)
            return '{"success": true}'

        monkeypatch.setattr(session_search_tool, "session_search", fake_session_search)

        assert (
            m._dispatch_session_search_stateless(query="oauth", limit=raw)
            == '{"success": true}'
        )
        assert calls[0]["limit"] == expected

    def test_build_server_routes_stateless_agent_loop_tools_without_handle_function_call(self, monkeypatch):
        """_build_server must route memory/session_search through local wrappers.

        The normal Hermes dispatcher intentionally blocks agent-loop tools outside
        AIAgent. Codex MCP handlers for memory/session_search should therefore
        bypass handle_function_call while ordinary exposed tools still use it.
        """
        from agent.transports import hermes_tools_mcp_server as m
        import model_tools

        class FakeFastMCP:
            def __init__(self, *args, **kwargs):
                self.tools = {}

            def add_tool(self, fn, name, description):
                self.tools[name] = fn

        fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
        setattr(fastmcp_mod, "FastMCP", FakeFastMCP)
        server_mod = types.ModuleType("mcp.server")
        setattr(server_mod, "fastmcp", fastmcp_mod)
        mcp_mod = types.ModuleType("mcp")
        setattr(mcp_mod, "server", server_mod)
        monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
        monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
        monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)

        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda quiet_mode=True: [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"{name} description",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("memory", "session_search", "web_search")
            ],
        )

        handle_calls = []
        monkeypatch.setattr(
            model_tools,
            "handle_function_call",
            lambda name, args: handle_calls.append((name, args)) or "handled",
        )
        monkeypatch.setitem(
            m._STATELESS_AGENT_LOOP_DISPATCHERS,
            "memory",
            lambda **kwargs: f"memory:{kwargs['action']}",
        )
        monkeypatch.setitem(
            m._STATELESS_AGENT_LOOP_DISPATCHERS,
            "session_search",
            lambda **kwargs: f"search:{kwargs['query']}",
        )

        server = m._build_server()

        assert server.tools["memory"](action="list") == "memory:list"
        assert server.tools["session_search"](query="oauth") == "search:oauth"
        assert server.tools["web_search"](query="hermes") == "handled"
        assert handle_calls == [("web_search", {"query": "hermes"})]

    def test_build_server_wraps_stateless_dispatcher_errors_as_json(self, monkeypatch):
        """MCP handlers should return structured errors instead of crashing.

        The stateless wrappers touch profile files/SQLite; if those are
        unavailable in the spawned subprocess, the MCP call should fail as a
        tool result that Codex can read rather than tearing down the server.
        """
        from agent.transports import hermes_tools_mcp_server as m
        import model_tools

        class FakeFastMCP:
            def __init__(self, *args, **kwargs):
                self.tools = {}

            def add_tool(self, fn, name, description):
                self.tools[name] = fn

        fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
        setattr(fastmcp_mod, "FastMCP", FakeFastMCP)
        server_mod = types.ModuleType("mcp.server")
        setattr(server_mod, "fastmcp", fastmcp_mod)
        mcp_mod = types.ModuleType("mcp")
        setattr(mcp_mod, "server", server_mod)
        monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
        monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
        monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
        monkeypatch.setattr(m, "EXPOSED_TOOLS", ("memory",))
        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda quiet_mode=True: [
                {
                    "type": "function",
                    "function": {
                        "name": "memory",
                        "description": "Manage memory",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        def broken_dispatcher(**kwargs):
            raise RuntimeError("memory store unavailable")

        monkeypatch.setitem(
            m._STATELESS_AGENT_LOOP_DISPATCHERS,
            "memory",
            broken_dispatcher,
        )

        server = m._build_server()
        payload = json.loads(server.tools["memory"](action="list"))

        assert payload == {"error": "memory store unavailable", "tool": "memory"}

    def test_build_server_preserves_hermes_parameter_schema(self, monkeypatch):
        """Codex should see Hermes' tool JSON schema, not FastMCP's **kwargs schema."""
        from agent.transports import hermes_tools_mcp_server as m
        import model_tools

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }

        monkeypatch.setattr(m, "EXPOSED_TOOLS", ("web_search",))
        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda quiet_mode=True: [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": schema,
                    },
                }
            ],
        )
        monkeypatch.setattr(
            model_tools,
            "handle_function_call",
            lambda name, args: "handled",
        )

        server = m._build_server()

        assert server._tool_manager._tools["web_search"].parameters == schema

    def test_kanban_worker_tools_exposed(self):
        """Kanban workers run as `hermes chat -q` subprocesses; if they
        come up on the codex_app_server runtime, the worker can do the
        actual work via codex's shell but needs the kanban tools through
        the MCP callback to report back to the kernel. Without these
        tools available, the worker would hang at completion time."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        # Worker handoff tools — every dispatched worker uses at least
        # one of {complete, block, comment} to close out its task.
        for worker_tool in (
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
            "kanban_heartbeat",
        ):
            assert worker_tool in EXPOSED_TOOLS, (
                f"{worker_tool!r} missing from codex callback — kanban "
                "workers on codex_app_server runtime would hang"
            )

    def test_kanban_orchestrator_tools_exposed(self):
        """Orchestrator agents need to dispatch new tasks, query the
        board, and unblock/link tasks. Exposed so an orchestrator on
        codex_app_server can do its job."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for orch_tool in (
            "kanban_create",
            "kanban_show",
            "kanban_list",
            "kanban_unblock",
            "kanban_link",
        ):
            assert orch_tool in EXPOSED_TOOLS, (
                f"{orch_tool!r} missing from codex callback"
            )


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1

"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch



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
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_agent_loop_tools_not_exposed(self):
        """delegate_task / memory / session_search / todo require the
        running AIAgent context to dispatch, so a stateless MCP callback
        can't drive them. They must NOT be in EXPOSED_TOOLS."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for agent_loop_tool in ("delegate_task", "memory", "session_search", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS, (
                f"{agent_loop_tool!r} requires the agent loop context "
                "and can't be reached through a stateless MCP callback"
            )

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

    def test_build_server_preserves_tool_argument_schema(self, monkeypatch):
        """FastMCP must see Hermes' real argument names.

        Regression coverage for Codex app-server sessions where skill_view
        was exposed but calls like skill_view(name="hermes-agent") arrived as
        an empty argument dict, producing "Skill '' not found".
        """
        import agent.transports.hermes_tools_mcp_server as m

        fake_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name"},
                "file_path": {"type": "string"},
            },
            "required": ["name"],
        }

        monkeypatch.setattr(
            "model_tools.get_tool_definitions",
            lambda quiet_mode=True: [
                {
                    "type": "function",
                    "function": {
                        "name": "skill_view",
                        "description": "Load a skill",
                        "parameters": fake_schema,
                    },
                }
            ],
        )

        class FakeToolManager:
            def __init__(self):
                self._tools = {}

            def add_tool(self, fn, name=None, description=None, **kwargs):
                signature = inspect.signature(fn)
                tool = SimpleNamespace(
                    name=name,
                    description=description,
                    parameters={
                        "type": "object",
                        "properties": {
                            key: {} for key in signature.parameters
                        },
                    },
                )
                self._tools[name] = tool
                return tool

        class FakeFastMCP:
            def __init__(self, *args, **kwargs):
                self._tool_manager = FakeToolManager()

            def add_tool(self, fn, **kwargs):
                self._tool_manager.add_tool(fn, **kwargs)

        mcp_mod = types.ModuleType("mcp")
        server_mod = types.ModuleType("mcp.server")
        fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
        fastmcp_mod.FastMCP = FakeFastMCP
        monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
        monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
        monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)

        server = m._build_server()
        tool = server._tool_manager._tools["skill_view"]

        assert "name" in tool.parameters["properties"]
        assert "file_path" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["name"]
        assert "kwargs" not in tool.parameters["properties"]


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


class TestSignatureFromJsonSchema:
    """The synthetic signature is the mechanism that fixes Codex sending empty
    argument dicts; test it directly, not just the resulting tool schema."""

    def test_builds_keyword_only_signature_required_first(self):
        import agent.transports.hermes_tools_mcp_server as m

        schema = {
            "type": "object",
            "properties": {
                "opt": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        sig = m._signature_from_json_schema(schema)

        assert sig is not None
        params = list(sig.parameters.values())
        # Required arg ordered first; every arg keyword-only; no **kwargs.
        assert [p.name for p in params] == ["name", "opt"]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)
        assert sig.parameters["name"].default is inspect.Parameter.empty
        assert sig.parameters["opt"].default is None

    def test_returns_none_for_non_identifier_property(self):
        import agent.transports.hermes_tools_mcp_server as m

        # All-or-nothing: a property name that is not a valid Python identifier
        # disables the synthetic signature (handler falls back to **kwargs).
        schema = {"type": "object", "properties": {"bad-name": {"type": "string"}}}
        assert m._signature_from_json_schema(schema) is None

    def test_returns_none_without_properties(self):
        import agent.transports.hermes_tools_mcp_server as m

        assert m._signature_from_json_schema({"type": "object"}) is None

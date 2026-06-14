"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in most unit tests — that requires the codex
subprocess + client + an event loop. These tests pin both the static contract
and the plugin-override export contract for the bridge.
"""

from __future__ import annotations

import asyncio
import json

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


class TestPluginOverrideExport:
    def _restore_entry(self, entry):
        from tools.registry import registry

        # Remove the protected override first so the original registration is
        # restored as a normal built-in-style entry, not as another protected
        # override that can leak into later tests.
        registry.deregister(entry.name)
        registry.register(
            name=entry.name,
            toolset=entry.toolset,
            schema=entry.schema,
            handler=entry.handler,
            check_fn=entry.check_fn,
            requires_env=entry.requires_env,
            is_async=entry.is_async,
            description=entry.description,
            emoji=entry.emoji,
            max_result_size_chars=entry.max_result_size_chars,
            dynamic_schema_overrides=entry.dynamic_schema_overrides,
        )

    def test_resolve_exposed_tool_names_includes_plugin_tools_in_exposed_toolsets(self):
        """Plugin tools in an exposed toolset are included by the bridge resolver."""
        from agent.transports import hermes_tools_mcp_server as server_mod
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        name = "web_mcp_plugin_probe_resolver"
        registry.register(
            name=name,
            toolset="web",
            schema={
                "name": name,
                "description": "Plugin probe tool in web toolset",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kw: json.dumps({"ok": True}),
            override=True,
        )
        try:
            assert name in server_mod._resolve_exposed_tool_names()
        finally:
            registry.deregister(name)

    def test_codex_mcp_bridge_exports_plugin_tools_in_exposed_toolsets(self):
        """Plugin tools added to an already-exposed toolset should reach Codex MCP."""
        pytest.importorskip("mcp.server.fastmcp")

        from agent.transports import hermes_tools_mcp_server as server_mod
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        name = "web_mcp_plugin_probe"
        registry.register(
            name=name,
            toolset="web",
            schema={
                "name": name,
                "description": "Plugin probe tool in web toolset",
                "parameters": {
                    "type": "object",
                    "properties": {"probe": {"type": "string"}},
                    "required": ["probe"],
                },
            },
            handler=lambda args, **kw: json.dumps({"ok": True}),
            override=True,
        )
        try:
            assert name in server_mod._resolve_exposed_tool_names()

            async def _list():
                mcp = server_mod._build_server()
                return await mcp.list_tools()

            tools = asyncio.run(_list())
            tool = next(t for t in tools if t.name == name)
            assert tool.inputSchema["properties"]["probe"]["type"] == "string"
            assert tool.inputSchema["required"] == ["probe"]
        finally:
            registry.deregister(name)

    def test_codex_mcp_bridge_exports_plugin_override_schema_for_builtin_tool(self):
        """An override=True plugin registration should replace built-in schema in MCP."""
        pytest.importorskip("mcp.server.fastmcp")

        from agent.transports import hermes_tools_mcp_server as server_mod
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        original = registry.get_entry("web_search")
        assert original is not None

        registry.register(
            name="web_search",
            toolset="web",
            schema={
                "name": "web_search",
                "description": "OVERRIDE PROBE DESCRIPTION",
                "parameters": {
                    "type": "object",
                    "properties": {"override_probe": {"type": "boolean"}},
                    "required": ["override_probe"],
                },
            },
            handler=lambda args, **kw: json.dumps({"source": "override"}),
            override=True,
        )
        try:
            async def _list():
                mcp = server_mod._build_server()
                return await mcp.list_tools()

            tools = asyncio.run(_list())
            tool = next(t for t in tools if t.name == "web_search")
            assert tool.description == "OVERRIDE PROBE DESCRIPTION"
            assert "override_probe" in tool.inputSchema["properties"]
            assert tool.inputSchema["required"] == ["override_probe"]
        finally:
            self._restore_entry(original)


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

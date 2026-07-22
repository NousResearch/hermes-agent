"""Tests for the hermes-tools-as-MCP server module surface.

The focused contract tests build the real FastMCP server and drive it through
an in-memory MCP client session. Static tests also pin the curated tool subset
and entry-point behavior.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest


AUTHORITATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {
            "type": "array",
            "items": {"type": "string", "format": "uri"},
            "minItems": 1,
            "maxItems": 5,
        },
        "mode": {"type": "string", "enum": ["fast", "thorough"]},
        "filters": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["domains"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["urls", "mode"],
    "additionalProperties": False,
}


@pytest.fixture
def authoritative_server(monkeypatch):
    """Build a real FastMCP server around deterministic Hermes definitions."""
    import model_tools
    from agent.transports import hermes_tools_mcp_server as m

    definitions = []
    for name, schema in (
        ("schema_probe", AUTHORITATIVE_SCHEMA),
        ("error_probe", {"type": "object", "properties": {}}),
        ("non_string_error_probe", {"type": "object", "properties": {}}),
        ("raised_probe", {"type": "object", "properties": {}}),
        ("text_probe", {}),
        ("object_probe", {"type": "object", "properties": {}}),
        ("scalar_probe", {"type": "object", "properties": {}}),
        ("list_probe", {"type": "object", "properties": {}}),
        ("prebuilt_error_probe", {"type": "object", "properties": {}}),
        ("prebuilt_success_probe", {"type": "object", "properties": {}}),
    ):
        definitions.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Authoritative description for {name}",
                "parameters": schema,
            },
        })

    calls = []

    def dispatch(name, arguments):
        from mcp.types import CallToolResult, TextContent

        calls.append((name, arguments))
        if name == "error_probe":
            return json.dumps(
                {
                    "error": "échec upstream",
                    "code": "E_UPSTREAM",
                    "retry": {"allowed": True, "after_seconds": 30},
                    "tool": name,
                },
                ensure_ascii=False,
            )
        if name == "non_string_error_probe":
            return {
                "error": {"message": "window elapsed"},
                "code": "E_WINDOW",
                "observed_at": datetime(2026, 7, 11, tzinfo=timezone.utc),
            }
        if name == "raised_probe":
            raise RuntimeError("dispatcher exploded")
        if name == "text_probe":
            return "plain text result"
        if name == "object_probe":
            return {"ok": True, "source": "native object"}
        if name == "scalar_probe":
            return "42"
        if name == "list_probe":
            return '[1, {"nested": true}]'
        if name == "prebuilt_error_probe":
            return CallToolResult(
                content=[TextContent(type="text", text="prebuilt failure")],
                isError=True,
            )
        if name == "prebuilt_success_probe":
            return CallToolResult(
                content=[TextContent(type="text", text="prebuilt success")],
                structuredContent={"prebuilt": True},
            )
        return json.dumps({"ok": True, "received": arguments})

    monkeypatch.setattr(
        m, "EXPOSED_TOOLS", tuple(item["function"]["name"] for item in definitions)
    )
    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda **_kwargs: definitions
    )
    monkeypatch.setattr(model_tools, "handle_function_call", dispatch)
    return m._build_server(), calls


async def _list_tools(server):
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server) as client:
        return await client.list_tools()


async def _call_tool(server, name, arguments):
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server) as client:
        return await client.call_tool(name, arguments)


class TestAuthoritativeFastMCPSurface:
    def test_list_tools_preserves_authoritative_schema_exactly(
        self, authoritative_server
    ):
        server, _calls = authoritative_server

        listed = asyncio.run(_list_tools(server))
        tools = {tool.name: tool for tool in listed.tools}

        assert tools["schema_probe"].inputSchema == AUTHORITATIVE_SCHEMA
        assert "kwargs" not in tools["schema_probe"].inputSchema.get("properties", {})
        assert tools["text_probe"].inputSchema == {}
        for name, tool in tools.items():
            assert tool.description == f"Authoritative description for {name}"

    def test_protocol_call_forwards_nested_arguments_and_returns_structured_success(
        self, authoritative_server
    ):
        server, calls = authoritative_server
        arguments = {
            "urls": ["https://example.com/a", "https://example.com/b"],
            "mode": "thorough",
            "filters": {"domains": ["example.com", "example.org"]},
        }

        result = asyncio.run(_call_tool(server, "schema_probe", arguments))

        assert result.isError is False
        assert result.structuredContent == {"ok": True, "received": arguments}
        content = result.content[0]
        assert content.type == "text"
        assert json.loads(content.text) == result.structuredContent
        assert calls == [("schema_probe", arguments)]

    @pytest.mark.parametrize(
        "invalid_arguments",
        [
            pytest.param({}, id="required"),
            pytest.param(
                {"urls": ["https://example.com"], "mode": "invalid"}, id="enum"
            ),
            pytest.param(
                {
                    "urls": ["https://example.com"],
                    "mode": "fast",
                    "unexpected": True,
                },
                id="additional-properties",
            ),
            pytest.param(
                {"urls": "https://example.com", "mode": "fast"}, id="array-type"
            ),
            pytest.param({"urls": [], "mode": "fast"}, id="array-bounds"),
            pytest.param(
                {
                    "urls": ["https://example.com"],
                    "mode": "fast",
                    "filters": {},
                },
                id="nested-required-and-one-of",
            ),
            pytest.param(
                {
                    "urls": ["https://example.com"],
                    "mode": "fast",
                    "filters": {"domains": [1]},
                },
                id="nested-array-items",
            ),
            pytest.param({"urls": ["not a uri"], "mode": "fast"}, id="uri-format"),
        ],
    )
    def test_protocol_rejects_schema_invalid_calls_without_dispatch(
        self, authoritative_server, invalid_arguments
    ):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, "schema_probe", invalid_arguments))

        assert result.isError is True
        assert calls == []

    def test_protocol_maps_hermes_error_object_to_mcp_error(self, authoritative_server):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, "error_probe", {}))

        assert result.isError is True
        content = result.content[0]
        assert content.type == "text"
        diagnostic = {
            "error": "échec upstream",
            "code": "E_UPSTREAM",
            "retry": {"allowed": True, "after_seconds": 30},
            "tool": "error_probe",
        }
        assert content.text == json.dumps(diagnostic, ensure_ascii=False)
        assert result.structuredContent == diagnostic
        assert calls == [("error_probe", {})]

    def test_protocol_preserves_non_string_non_json_native_error_diagnostics(
        self, authoritative_server
    ):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, "non_string_error_probe", {}))

        assert result.isError is True
        content = result.content[0]
        assert content.type == "text"
        diagnostic = json.loads(content.text)
        assert diagnostic == {
            "error": {"message": "window elapsed"},
            "code": "E_WINDOW",
            "observed_at": "2026-07-11 00:00:00+00:00",
        }
        assert result.structuredContent == diagnostic
        assert calls == [("non_string_error_probe", {})]

    def test_protocol_maps_dispatch_exceptions_to_structured_mcp_error(
        self, authoritative_server
    ):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, "raised_probe", {}))

        diagnostic = {"error": "dispatcher exploded", "tool": "raised_probe"}
        assert result.isError is True
        content = result.content[0]
        assert content.type == "text"
        assert json.loads(content.text) == diagnostic
        assert result.structuredContent == diagnostic
        assert calls == [("raised_probe", {})]

    def test_protocol_keeps_plain_string_results_compatible(self, authoritative_server):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, "text_probe", {}))

        assert result.isError is False
        assert result.structuredContent is None
        content = result.content[0]
        assert content.type == "text"
        assert content.text == "plain text result"
        assert calls == [("text_probe", {})]

    def test_protocol_keeps_native_object_results_structured(
        self, authoritative_server
    ):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, "object_probe", {}))

        assert result.isError is False
        assert result.structuredContent == {"ok": True, "source": "native object"}
        assert calls == [("object_probe", {})]

    @pytest.mark.parametrize(
        ("tool_name", "expected_text"),
        [("scalar_probe", "42"), ("list_probe", '[1, {"nested": true}]')],
    )
    def test_protocol_keeps_json_scalar_and_list_strings_as_text(
        self, authoritative_server, tool_name, expected_text
    ):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, tool_name, {}))

        assert result.isError is False
        assert result.structuredContent is None
        content = result.content[0]
        assert content.type == "text"
        assert content.text == expected_text
        assert calls == [(tool_name, {})]

    @pytest.mark.parametrize(
        ("tool_name", "is_error", "text", "structured"),
        [
            ("prebuilt_error_probe", True, "prebuilt failure", None),
            ("prebuilt_success_probe", False, "prebuilt success", {"prebuilt": True}),
        ],
    )
    def test_protocol_preserves_existing_call_tool_results(
        self, authoritative_server, tool_name, is_error, text, structured
    ):
        server, calls = authoritative_server

        result = asyncio.run(_call_tool(server, tool_name, {}))

        assert result.isError is is_error
        content = result.content[0]
        assert content.type == "text"
        assert content.text == text
        assert result.structuredContent == structured
        assert calls == [(tool_name, {})]


class TestCodexFacingBoundary:
    def test_codex_app_server_ingests_authoritative_schema(self, tmp_path, monkeypatch):
        """Have Codex ingest the real stdio server's ``tools/list``.

        ``mcpServerStatus/list`` is the smallest supported app-server boundary
        that returns Codex's parsed tool inventory; no provider/model turn is
        needed. Skip only when the optional Codex CLI is unavailable.
        """
        codex_bin = shutil.which("codex")
        if codex_bin is None:
            pytest.skip("Codex CLI is not installed")

        import model_tools
        from agent.transports.codex_app_server import (
            CodexAppServerClient,
            check_codex_binary,
        )
        from hermes_cli.codex_runtime_plugin_migration import migrate

        ok, detail = check_codex_binary(codex_bin)
        if not ok:
            pytest.skip(detail)

        definitions = {
            item["function"]["name"]: item["function"]
            for item in model_tools.get_tool_definitions(quiet_mode=True)
            if item.get("type") == "function"
        }
        # Skills are dependency-free and therefore present even when optional
        # web/browser providers are not configured in CI.
        tool_name = "skills_list"
        authoritative = definitions[tool_name]["parameters"]

        # The generated stdio entry must import this worktree rather than any
        # separately installed Hermes checkout.
        monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parents[3]))
        codex_home = tmp_path / "codex-home"
        report = migrate(
            {},
            codex_home=codex_home,
            discover_plugins=False,
            default_permission_profile=None,
            expose_hermes_tools=True,
        )
        assert report.errors == []

        client = CodexAppServerClient(codex_bin=codex_bin, codex_home=str(codex_home))
        try:
            client.initialize(timeout=15)
            status = client.request(
                "mcpServerStatus/list",
                {"detail": "toolsAndAuthOnly"},
                timeout=60,
            )
        finally:
            client.close()

        servers = {item["name"]: item for item in status["data"]}
        codex_schema = servers["hermes-tools"]["tools"][tool_name]["inputSchema"]
        assert codex_schema == authoritative
        assert "kwargs" not in codex_schema.get("properties", {})


class TestModuleSurface:
    def test_fastmcp_adapter_rejects_unreviewed_sdk_version(self, monkeypatch):
        from agent.transports import hermes_tools_mcp_server as m

        monkeypatch.setattr(m, "distribution_version", lambda _name: "1.26.1")

        with pytest.raises(
            RuntimeError, match=r"requires mcp==1\.26\.0; found 1\.26\.1"
        ):
            m._FastMCP126SchemaAdapter(object())

    def test_fastmcp_adapter_rejects_missing_private_shape(self, monkeypatch):
        from agent.transports import hermes_tools_mcp_server as m

        monkeypatch.setattr(m, "distribution_version", lambda _name: "1.26.0")

        with pytest.raises(RuntimeError, match="_tool_manager.get_tool"):
            m._FastMCP126SchemaAdapter(object())

    def test_fastmcp_adapter_rejects_changed_registered_tool_shape(self, monkeypatch):
        from agent.transports import hermes_tools_mcp_server as m

        class Manager:
            @staticmethod
            def get_tool(_name):
                return object()

        class Server:
            _tool_manager = Manager()

        monkeypatch.setattr(m, "distribution_version", lambda _name: "1.26.0")
        adapter = m._FastMCP126SchemaAdapter(Server())

        with pytest.raises(RuntimeError, match="parameters and fn_metadata.arg_model"):
            adapter.install("probe", {})

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
            "terminal",
            "shell",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "process",
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


class TestMain:
    @pytest.mark.parametrize(
        "error",
        [ImportError("mcp not installed"), RuntimeError("unsupported mcp shape")],
    )
    def test_main_returns_2_when_server_cannot_build(self, monkeypatch, error):
        """Dependency and compatibility failures should not start the server."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise error

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

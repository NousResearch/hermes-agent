"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from typing import get_args

from agent.transports.hermes_tools_mcp_server import (
    _signature_from_schema,
)


class TestSignatureFromSchema:
    """Test the JSON Schema -> Python signature conversion."""

    def test_simple_required_string_param(self):
        """A required string param becomes str with no default."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        sig, annots = _signature_from_schema(schema)

        assert len(sig.parameters) == 1
        param = sig.parameters["query"]
        assert param.name == "query"
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert annots["query"] == str
        assert param.default is inspect.Parameter.empty



    def test_skip_private_params(self):
        """Params starting with '_' are excluded from the signature."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "_internal": {"type": "string"},
            },
            "required": ["query", "_internal"],
        }
        sig, annots = _signature_from_schema(schema)

        assert "_internal" not in sig.parameters
        assert "_internal" not in annots
        assert "query" in sig.parameters

    def test_all_json_types(self):
        """All JSON schema types map to correct Python types."""
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            },
            "required": ["s", "i", "n", "b", "a", "o"],
        }
        sig, annots = _signature_from_schema(schema)

        assert annots["s"] == str
        assert annots["i"] == int
        assert annots["n"] == float
        assert annots["b"] == bool
        assert annots["a"] == list
        assert annots["o"] == dict








class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """Codex's default MCP surface excludes its native file/shell tools."""
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

    def test_claude_sdk_profile_adds_bounded_read_only_file_tools(self):
        """Claude gets inspection only; no filesystem mutation tool is exposed."""
        from agent.transports.hermes_tools_mcp_server import (
            EXPOSED_TOOLS,
            exposed_tools_for_profile,
        )

        tools = set(exposed_tools_for_profile("claude-agent-sdk"))
        assert {"read_file", "search_files"} <= tools
        assert "skill_manage" not in tools
        assert "skill_manage" not in EXPOSED_TOOLS
        assert not tools & {
            "terminal", "shell", "write_file", "patch", "process",
            "git_add", "git_commit", "git_push",
        }

    def test_unknown_profile_fails_closed_to_codex_default(self):
        """Invalid launcher input must not expand the curated surface."""
        from agent.transports.hermes_tools_mcp_server import (
            EXPOSED_TOOLS,
            exposed_tools_for_profile,
        )

        assert exposed_tools_for_profile("not-a-profile") == EXPOSED_TOOLS




class TestClaudeSdkMcpIntegration:
    def test_claude_profile_lists_and_executes_bounded_file_tools(self, tmp_path):
        """The real stdio server exposes only bounded inspection additions."""
        fixture = tmp_path / "probe.txt"
        fixture.write_text("alpha bounded inspection\nbeta\n", encoding="utf-8")

        async def exercise():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            env = dict(os.environ)
            env["PYTHONPATH"] = str(
                __import__("pathlib").Path(__file__).resolve().parents[3]
            )
            env["HERMES_HOME"] = str(tmp_path / "hermes-home")
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "agent.transports.hermes_tools_mcp_server",
                    "--profile",
                    "claude-agent-sdk",
                ],
                env=env,
                cwd=tmp_path,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    listed = await client.list_tools()
                    names = {tool.name for tool in listed.tools}
                    assert {"read_file", "search_files"} <= names
                    assert not names & {
                        "terminal", "shell", "write_file", "patch", "process",
                        "git_add", "git_commit", "git_push",
                    }
                    read_result = await client.call_tool("read_file", {"path": "probe.txt"})
                    search_result = await client.call_tool(
                        "search_files", {"pattern": "bounded inspection", "path": "."}
                    )
                    return read_result, search_result

        read_result, search_result = asyncio.run(exercise())
        assert "alpha bounded inspection" in str(read_result.content)
        assert "probe.txt" in str(search_result.content)


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

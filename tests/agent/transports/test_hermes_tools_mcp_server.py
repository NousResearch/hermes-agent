"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import inspect

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
        _sig, annots = _signature_from_schema(schema)

        assert annots["s"] == str
        assert annots["i"] == int
        assert annots["n"] == float
        assert annots["b"] == bool
        assert annots["a"] == list
        assert annots["o"] == dict


class TestModuleSurface:

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


    def test_every_exposed_tool_is_a_core_registered_tool(self):
        """Each exposed name must dispatch through handle_function_call in a plain Hermes process:
        a name only a plugin registers (the camofox plugin's browser_navigate & co.) is not core."""
        import model_tools
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        missing = set(EXPOSED_TOOLS) - set(model_tools.get_all_tool_names())
        assert not missing, f"EXPOSED_TOOLS names no core tool registers: {sorted(missing)}"


class _FakeMCPServer:
    """Records add_tool calls in place of mcp.server.MCPServer."""

    def __init__(self, *args, **kwargs):
        self.tools = []
        self.handlers = {}

    def add_tool(self, handler, *, name, description):
        self.tools.append((name, description))
        self.handlers[name] = handler


def _install_fake_mcp_server(monkeypatch):
    """Point ``from mcp.server import MCPServer`` at _FakeMCPServer."""
    import sys
    import types

    try:
        import mcp.server as mcp_server
    except ImportError:
        mcp_server = types.ModuleType("mcp.server")
        mcp_pkg = types.ModuleType("mcp")
        mcp_pkg.server = mcp_server
        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.server", mcp_server)
    monkeypatch.setattr(mcp_server, "MCPServer", _FakeMCPServer, raising=False)


def _web_search_def():
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }


class TestBuildServer:

    def test_schemas_come_from_model_tools_and_dispatch_is_lazy(self, monkeypatch):
        """Schemas use model_tools.get_tool_definitions (plugin overrides,
        check_fn gates, dynamic schemas, sanitization); handle_function_call
        is resolved on the first tool call, not while the server is built."""
        import sys
        import types

        import agent.transports.hermes_tools_mcp_server as m

        _install_fake_mcp_server(monkeypatch)
        fake_model_tools = types.ModuleType("model_tools")
        fake_model_tools.get_tool_definitions = lambda quiet_mode=False: [_web_search_def()]
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        server = m._build_server()

        assert server.tools == [("web_search", "Search the web.")]
        assert not hasattr(fake_model_tools, "handle_function_call")

        fake_model_tools.handle_function_call = lambda tool_name, args: f"called:{tool_name}:{args}"
        assert server.handlers["web_search"](query="hello") == "called:web_search:{'query': 'hello'}"

    def test_tool_with_failing_check_fn_is_not_listed(self, monkeypatch):
        """An unavailable tool (check_fn False) must not be advertised to the MCP client."""
        import agent.transports.hermes_tools_mcp_server as m
        import model_tools
        from tools.registry import registry

        _install_fake_mcp_server(monkeypatch)
        entry = registry.get_entry("web_search")
        assert entry is not None
        monkeypatch.setattr(entry, "check_fn", lambda: False)
        model_tools._clear_tool_defs_cache()
        try:
            server = m._build_server()
        finally:
            model_tools._clear_tool_defs_cache()

        names = [name for name, _ in server.tools]
        assert "web_search" not in names
        assert set(names) <= set(m.EXPOSED_TOOLS)


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

"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from typing import get_args

from agent.transports.hermes_tools_mcp_server import (
    _make_spill_reader,
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


class TestSpillReader:
    def test_mcp_callback_spills_and_recovers_oversized_result(self, monkeypatch, tmp_path):
        class FakeMCPServer:
            def __init__(self, *_args, **_kwargs):
                self.handlers = {}

            def add_tool(self, handler, *, name, description):
                self.handlers[name] = handler

        fake_server_module = types.ModuleType("mcp.server")
        setattr(fake_server_module, "MCPServer", FakeMCPServer)
        monkeypatch.setitem(sys.modules, "mcp.server", fake_server_module)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_SESSION_ID", "mcp-producer-session")
        monkeypatch.setattr(
            "model_tools.get_tool_definitions",
            lambda **_kwargs: [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        )
        payload = "mcp-result\n" + ("z" * 120_000)
        monkeypatch.setattr("model_tools.registry.dispatch", lambda *_a, **_k: payload)

        from agent.transports.hermes_tools_mcp_server import _build_server
        from tools.tool_result_storage import resolve_spill_capability

        server = _build_server()
        notice = server.handlers["web_search"]()
        uri = next(part for part in notice.split() if part.startswith("hermes-spill://"))
        assert resolve_spill_capability(uri, "mcp-producer-session") == payload

    def test_forwards_capability_with_captured_session(self):
        calls = []

        def dispatch(name, args, **kwargs):
            calls.append((name, args, kwargs))
            return json.dumps({"ok": True})

        uri = f"hermes-spill://v1/{'a' * 64}/{'b' * 64}/{'c' * 32}"
        reader = _make_spill_reader(dispatch, "session-123")

        assert json.loads(reader(path=uri)) == {"ok": True}
        assert calls == [
            ("read_file", {"path": uri}, {"session_id": "session-123"})
        ]

    def test_rejects_ordinary_paths_without_dispatch(self):
        def dispatch(*_args, **_kwargs):
            raise AssertionError("ordinary path was dispatched")

        result = json.loads(
            _make_spill_reader(dispatch, "session-123")(path="/etc/passwd")
        )
        assert "error" in result

    def test_build_server_registers_capability_reader_with_mcp2(self, monkeypatch):
        class FakeMCPServer:
            def __init__(self, *_args, **_kwargs):
                self.handlers = {}

            def add_tool(self, handler, *, name, description):
                self.handlers[name] = handler

        fake_server_module = types.ModuleType("mcp.server")
        setattr(fake_server_module, "MCPServer", FakeMCPServer)
        monkeypatch.setitem(sys.modules, "mcp.server", fake_server_module)
        monkeypatch.setenv("HERMES_SESSION_ID", "mcp-session")
        monkeypatch.setattr(
            "model_tools.get_tool_definitions",
            lambda **_kwargs: [],
        )
        calls = []

        def dispatch(name, args, **kwargs):
            calls.append((name, args, kwargs))
            return json.dumps({"ok": True})

        monkeypatch.setattr("model_tools.handle_function_call", dispatch)
        from agent.transports.hermes_tools_mcp_server import _build_server

        server = _build_server()
        assert set(server.handlers) == {"read_file"}
        uri = f"hermes-spill://v1/{'a' * 64}/{'b' * 64}/{'c' * 32}"
        assert json.loads(server.handlers["read_file"](path=uri)) == {"ok": True}
        assert calls == [
            ("read_file", {"path": uri}, {"session_id": "mcp-session"})
        ]


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

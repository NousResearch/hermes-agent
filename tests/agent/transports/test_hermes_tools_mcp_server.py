"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import inspect
import json
from typing import Literal, get_args, get_origin
from unittest.mock import patch

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


class TestToolHandlerBuilder:
    def test_normalize_tool_args_unwraps_legacy_kwargs(self):
        from agent.transports.hermes_tools_mcp_server import _normalize_tool_args

        assert _normalize_tool_args({"kwargs": {"all_boards": True}}) == {
            "all_boards": True,
        }
        assert _normalize_tool_args({"all_boards": True}) == {"all_boards": True}

    def test_build_tool_handler_exposes_flat_parameters(self):
        from agent.transports.hermes_tools_mcp_server import _build_tool_handler

        captured: list[tuple[str, dict]] = []

        def fake_dispatch(tool_name: str, args: dict) -> str:
            captured.append((tool_name, args))
            return json.dumps({"ok": True})

        schema = {
            "type": "object",
            "properties": {
                "all_boards": {"type": "boolean", "description": "List all boards"},
                "board": {"type": "string", "description": "Board slug"},
            },
        }
        handler = _build_tool_handler(
            "ideas_list",
            schema,
            "List ideas",
            dispatch=fake_dispatch,
        )
        sig = inspect.signature(handler)
        assert list(sig.parameters) == ["all_boards", "board"]
        assert sig.parameters["all_boards"].default is None
        assert sig.parameters["board"].default is None

        handler(all_boards=True)
        assert captured == [("ideas_list", {"all_boards": True})]

    def test_build_tool_handler_preserves_false_boolean(self):
        from agent.transports.hermes_tools_mcp_server import _build_tool_handler

        captured: list[dict] = []

        def fake_dispatch(_tool_name: str, args: dict) -> str:
            captured.append(args)
            return "{}"

        handler = _build_tool_handler(
            "ideas_list",
            {"type": "object", "properties": {"all_boards": {"type": "boolean"}}},
            "List ideas",
            dispatch=fake_dispatch,
        )
        handler(all_boards=False)
        assert captured == [{"all_boards": False}]

    def test_build_tool_handler_enum_becomes_literal(self):
        from typing import Union

        from agent.transports.hermes_tools_mcp_server import _build_tool_handler

        handler = _build_tool_handler(
            "ideas_update",
            {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["draft", "done"]},
                },
            },
            "Update idea",
            dispatch=lambda _n, _a: "{}",
        )
        status_type = handler.__annotations__["status"]
        assert get_origin(status_type) is Union
        inner = get_args(status_type)[0]
        assert get_origin(inner) is type(Literal)
        assert set(get_args(inner)) == {"draft", "done"}


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

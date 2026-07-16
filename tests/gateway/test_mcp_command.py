"""Tests for the ``/mcp`` gateway slash command handler.

``/mcp`` is a read-only companion to ``/reload-mcp``: it lists configured
MCP servers and their live connection status by delegating to
``tools.mcp_tool.get_mcp_status``. This file exercises just the gateway
handler wiring — the underlying status collection is covered by
``tests/tools/test_mcp_tool.py``.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/mcp") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    return runner


class TestHandleMcpCommand:
    def test_no_servers_configured(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        monkeypatch.setattr(mcp_tool, "get_mcp_status", lambda: [])
        runner = _make_runner()
        result = asyncio.run(runner._handle_mcp_command(_make_event()))
        assert "No MCP servers configured" in result
        assert "hermes mcp add" in result

    def test_lists_connected_server(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        monkeypatch.setattr(
            mcp_tool,
            "get_mcp_status",
            lambda: [
                {
                    "name": "codegraph",
                    "transport": "stdio",
                    "tools": 12,
                    "connected": True,
                    "disabled": False,
                    "status": "connected",
                }
            ],
        )
        runner = _make_runner()
        result = asyncio.run(runner._handle_mcp_command(_make_event()))
        assert "codegraph" in result
        assert "connected" in result
        assert "stdio" in result
        assert "12 tool" in result

    def test_shows_disabled_and_failed(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        monkeypatch.setattr(
            mcp_tool,
            "get_mcp_status",
            lambda: [
                {"name": "off-server", "transport": "http", "tools": 0,
                 "connected": False, "disabled": True, "status": "disabled"},
                {"name": "broken", "transport": "stdio", "tools": 0,
                 "connected": False, "disabled": False, "status": "failed",
                 "error": "connection refused"},
            ],
        )
        runner = _make_runner()
        result = asyncio.run(runner._handle_mcp_command(_make_event()))
        assert "off-server" in result
        assert "disabled" in result
        assert "broken" in result
        assert "failed" in result
        assert "connection refused" in result

    def test_survives_status_exception(self, monkeypatch):
        """A crash in get_mcp_status must not take down the gateway thread."""
        import tools.mcp_tool as mcp_tool

        def _boom():
            raise RuntimeError("mcp subsystem exploded")

        monkeypatch.setattr(mcp_tool, "get_mcp_status", _boom)
        runner = _make_runner()
        result = asyncio.run(runner._handle_mcp_command(_make_event()))
        assert "Failed to read MCP status" in result
        assert "exploded" in result


class TestCommandRegistration:
    def test_mcp_is_gateway_known(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "mcp" in GATEWAY_KNOWN_COMMANDS

    def test_mcp_resolves_to_a_command_def(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("mcp")
        assert cmd is not None
        assert cmd.name == "mcp"
        assert not cmd.gateway_only  # available in both CLI and gateway

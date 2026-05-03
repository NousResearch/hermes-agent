"""Regression tests for gateway quick-command aliases."""
from unittest.mock import MagicMock

import pytest


class _Source:
    user_id = "test_user"
    user_name = "Test User"
    chat_type = "dm"
    chat_id = "123"
    thread_id = None
    platform = type("P", (), {"value": "weixin"})()


class _Event:
    def __init__(self, command, args=""):
        self._command = command
        self._args = args
        self.text = f"/{command} {args}".strip()
        self.source = _Source()

    def get_command(self):
        return self._command

    def get_command_args(self):
        return self._args


class TestGatewayQuickCommandAliases:
    @pytest.mark.asyncio
    async def test_alias_to_exec_quick_command_executes_target(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "quick_commands": {
                "tk1": {"type": "exec", "command": "echo switched-tk1"},
                "tokenx24": {"type": "alias", "target": "/tk1"},
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        result = await runner._handle_message(_Event("tokenx24"))
        assert result == "switched-tk1"

    @pytest.mark.asyncio
    async def test_alias_loop_returns_error(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "quick_commands": {
                "a": {"type": "alias", "target": "/b"},
                "b": {"type": "alias", "target": "/a"},
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        result = await runner._handle_message(_Event("a"))
        assert result is not None
        assert "alias loop" in result.lower()

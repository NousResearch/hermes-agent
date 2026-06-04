"""
Tests for post_tool_call hook dispatch.

Tests the has_hook() gate, _emit_post_tool_call_hook() in model_tools.py,
and the terminal tool executor integration in tool_executor.py.

Run: python -m pytest tests/test_post_tool_call_hook.py -v
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch
from typing import Any


class TestHasHook(unittest.TestCase):
    """Test the has_hook() method on PluginManager."""

    def _make_manager(self):
        from hermes_cli.plugins import PluginManager
        return PluginManager()

    def test_no_hooks_registered(self):
        mgr = self._make_manager()
        assert mgr.has_hook("post_tool_call") is False

    def test_hook_registered(self):
        mgr = self._make_manager()
        mgr._hooks["post_tool_call"] = [lambda **kw: None]
        assert mgr.has_hook("post_tool_call") is True

    def test_empty_hooks_list(self):
        mgr = self._make_manager()
        mgr._hooks["post_tool_call"] = []
        assert mgr.has_hook("post_tool_call") is False

    def test_module_level_has_hook(self):
        from hermes_cli.plugins import has_hook
        assert callable(has_hook)


class TestEmitPostToolCallHook(unittest.TestCase):
    """Test _emit_post_tool_call_hook() in model_tools.py."""

    def _import(self):
        import importlib
        import model_tools
        importlib.reload(model_tools)
        return model_tools._emit_post_tool_call_hook

    def test_no_listener_is_noop(self):
        emit = self._import()
        with patch("hermes_cli.plugins.has_hook", return_value=False):
            with patch("hermes_cli.plugins.invoke_hook") as mock_invoke:
                emit(
                    function_name="test_tool",
                    function_args={"key": "value"},
                    result='{"ok": true}',
                    task_id="task-1",
                    session_id="sess-1",
                    tool_call_id="tc-1",
                    turn_id="turn-1",
                    api_request_id="api-1",
                    duration_ms=100,
                )
                mock_invoke.assert_not_called()

    def test_listener_calls_invoke_hook(self):
        emit = self._import()
        with patch("hermes_cli.plugins.has_hook", return_value=True):
            with patch("hermes_cli.plugins.invoke_hook") as mock_invoke:
                emit(
                    function_name="test_tool",
                    function_args={"key": "value"},
                    result='{"ok": true}',
                    task_id="task-1",
                    session_id="sess-1",
                    tool_call_id="tc-1",
                    turn_id="turn-1",
                    api_request_id="api-1",
                    duration_ms=150,
                )
                mock_invoke.assert_called_once()
                args, kwargs = mock_invoke.call_args
                assert args[0] == "post_tool_call"
                assert kwargs["tool_name"] == "test_tool"
                assert kwargs["args"] == {"key": "value"}
                assert kwargs["task_id"] == "task-1"
                assert kwargs["session_id"] == "sess-1"
                assert kwargs["tool_call_id"] == "tc-1"
                assert kwargs["turn_id"] == "turn-1"
                assert kwargs["api_request_id"] == "api-1"
                assert kwargs["duration_ms"] == 150

    def test_error_in_hook_does_not_propagate(self):
        emit = self._import()
        with patch("hermes_cli.plugins.has_hook", return_value=True):
            with patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("hook error")):
                emit(
                    function_name="test_tool",
                    function_args={},
                    result="ok",
                )

    def test_status_derived_from_result_when_not_provided(self):
        emit = self._import()
        with patch("hermes_cli.plugins.has_hook", return_value=True):
            with patch("hermes_cli.plugins.invoke_hook") as mock_invoke:
                emit(
                    function_name="test_tool",
                    function_args={},
                    result='{"data": "value"}',
                )
                _, kwargs = mock_invoke.call_args
                assert kwargs["status"] == "ok"
                assert kwargs["error_type"] is None

    def test_error_result_derived(self):
        emit = self._import()
        with patch("hermes_cli.plugins.has_hook", return_value=True):
            with patch("hermes_cli.plugins.invoke_hook") as mock_invoke:
                emit(
                    function_name="test_tool",
                    function_args={},
                    result='{"error": "something failed"}',
                )
                _, kwargs = mock_invoke.call_args
                assert kwargs["status"] == "error"
                assert kwargs["error_type"] == "tool_error"

    def test_explicit_status_overrides_derived(self):
        emit = self._import()
        with patch("hermes_cli.plugins.has_hook", return_value=True):
            with patch("hermes_cli.plugins.invoke_hook") as mock_invoke:
                emit(
                    function_name="test_tool",
                    function_args={},
                    result='{"data": "value"}',
                    status="blocked",
                    error_type="plugin_block",
                )
                _, kwargs = mock_invoke.call_args
                assert kwargs["status"] == "blocked"
                assert kwargs["error_type"] == "plugin_block"


class TestToolResultObserverFields(unittest.TestCase):
    """Test _tool_result_observer_fields() helper."""

    def _import(self):
        import importlib
        import model_tools
        importlib.reload(model_tools)
        return model_tools._tool_result_observer_fields

    def test_ok_result(self):
        fields = self._import()
        status, error_type, error_message = fields('{"data": "value"}')
        assert status == "ok"
        assert error_type is None
        assert error_message is None

    def test_error_result(self):
        fields = self._import()
        status, error_type, error_message = fields('{"error": "failed"}')
        assert status == "error"
        assert error_type == "tool_error"
        assert "failed" in error_message

    def test_non_json_result(self):
        fields = self._import()
        status, error_type, error_message = fields("plain text result")
        assert status == "ok"
        assert error_type is None

    def test_dict_result(self):
        fields = self._import()
        status, error_type, error_message = fields({"error": "dict error"})
        assert status == "error"
        assert error_type == "tool_error"


class TestTerminalPostToolCall(unittest.TestCase):
    """Test _emit_terminal_post_tool_call() in tool_executor.py."""

    def _import(self):
        import importlib
        import agent.tool_executor as te
        importlib.reload(te)
        return te._emit_terminal_post_tool_call

    def test_emits_hook_with_correct_payload(self):
        emit = self._import()
        agent = MagicMock()
        agent.session_id = "sess-1"
        agent._current_turn_id = "turn-1"
        agent._current_api_request_id = "api-1"

        with patch("model_tools._emit_post_tool_call_hook") as mock_emit:
            emit(
                agent,
                function_name="write_file",
                function_args={"path": "test.py", "content": "x=1"},
                result='{"success": true}',
                effective_task_id="task-1",
                tool_call_id="tc-1",
                duration_ms=200,
                status="ok",
            )
            mock_emit.assert_called_once()
            _, kwargs = mock_emit.call_args
            assert kwargs["function_name"] == "write_file"
            assert kwargs["function_args"] == {"path": "test.py", "content": "x=1"}
            assert kwargs["result"] == '{"success": true}'
            assert kwargs["task_id"] == "task-1"
            assert kwargs["session_id"] == "sess-1"
            assert kwargs["tool_call_id"] == "tc-1"
            assert kwargs["duration_ms"] == 200
            assert kwargs["status"] == "ok"

    def test_error_does_not_propagate(self):
        emit = self._import()
        agent = MagicMock()
        agent.session_id = ""
        agent._current_turn_id = ""
        agent._current_api_request_id = ""

        with patch("model_tools._emit_post_tool_call_hook", side_effect=RuntimeError("boom")):
            emit(
                agent,
                function_name="test",
                function_args={},
                result="ok",
                effective_task_id="",
                tool_call_id="",
            )


class TestCancelledTerminalPostToolCall(unittest.TestCase):
    """Test _emit_cancelled_terminal_post_tool_call() in tool_executor.py."""

    def _import(self):
        import importlib
        import agent.tool_executor as te
        importlib.reload(te)
        return te._emit_cancelled_terminal_post_tool_call

    def test_returns_result_json(self):
        emit = self._import()
        agent = MagicMock()
        agent.session_id = "sess-1"
        agent._current_turn_id = "turn-1"
        agent._current_api_request_id = "api-1"

        with patch("model_tools._emit_post_tool_call_hook"):
            result = emit(
                agent,
                function_name="long_running_tool",
                function_args={},
                effective_task_id="task-1",
                tool_call_id="tc-1",
                start_time=time.time() - 5.0,
                reason="user interrupt",
            )
            parsed = json.loads(result)
            assert "cancelled" in parsed["status"]
            assert "error" in parsed

    def test_emits_hook_with_cancelled_status(self):
        emit = self._import()
        agent = MagicMock()
        agent.session_id = "sess-1"
        agent._current_turn_id = "turn-1"
        agent._current_api_request_id = "api-1"

        with patch("model_tools._emit_post_tool_call_hook") as mock_emit:
            emit(
                agent,
                function_name="test",
                function_args={},
                effective_task_id="task-1",
                tool_call_id="tc-1",
                start_time=time.time(),
            )
            _, kwargs = mock_emit.call_args
            assert kwargs["status"] == "cancelled"
            assert kwargs["error_type"] == "keyboard_interrupt"


if __name__ == "__main__":
    unittest.main()

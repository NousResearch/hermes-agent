"""Tests that handle_function_call forwards session_id into registry.dispatch."""

import json
from unittest.mock import MagicMock, patch


def _make_registry(captured: dict):
    """Return a mock registry whose dispatch records the kwargs it receives."""
    registry = MagicMock()

    def _dispatch(name, args, **kwargs):
        captured.update(kwargs)
        return json.dumps({"result": "ok"})

    registry.dispatch.side_effect = _dispatch
    return registry


class TestSessionIdForwarding:

    def test_standard_path_forwards_session_id(self):
        """registry.dispatch receives session_id on the normal tool path."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="t1",
                session_id="sess-abc",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("session_id") == "sess-abc"

    def test_execute_code_path_forwards_session_id(self):
        """registry.dispatch receives session_id on the execute_code path."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "execute_code",
                {"code": "print(1)"},
                task_id="t1",
                session_id="sess-xyz",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("session_id") == "sess-xyz"

    def test_session_id_default_is_none(self):
        """When session_id is omitted, dispatch receives None."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="t1",
                skip_pre_tool_call_hook=True,
            )
        assert "session_id" in captured
        assert captured["session_id"] is None

    def test_task_id_still_forwarded(self):
        """Existing task_id forwarding is not broken by this change."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="task-999",
                session_id="sess-1",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("task_id") == "task-999"


class TestToolCallIdForwarding:
    """tool_call_id must reach registry.dispatch so tools (e.g. terminal) can
    attribute their output to the originating tool call."""

    def test_standard_path_forwards_tool_call_id(self):
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="t1",
                session_id="sess-abc",
                tool_call_id="call-abc",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("tool_call_id") == "call-abc"

    def test_execute_code_path_forwards_tool_call_id(self):
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "execute_code",
                {"code": "print(1)"},
                task_id="t1",
                session_id="sess-xyz",
                tool_call_id="call-xyz",
                skip_pre_tool_call_hook=True,
            )
        assert captured.get("tool_call_id") == "call-xyz"

    def test_tool_call_id_default_is_none(self):
        """When tool_call_id is omitted, dispatch receives None (the tool's own
        fire site normalizes it to '')."""
        captured = {}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="t1",
                skip_pre_tool_call_hook=True,
            )
        assert "tool_call_id" in captured
        assert captured["tool_call_id"] is None

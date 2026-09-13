"""Tests that handle_function_call forwards session_id into registry.dispatch."""

import json
from types import SimpleNamespace
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

    def test_conversation_origin_is_forwarded_only_when_supplied(self):
        captured = {}
        origin = {"message_row_id": 767, "prompt": "Create a card"}
        with patch("model_tools.registry", _make_registry(captured)):
            from model_tools import handle_function_call
            handle_function_call(
                "kanban_create", {"title": "traceable"}, session_id="sess-1",
                platform="telegram", tool_origin=origin, skip_pre_tool_call_hook=True,
            )
        assert captured["platform"] == "telegram"
        assert captured["tool_origin"] == origin

    def test_kanban_dispatch_captures_persisted_triggering_user_row(self):
        from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef

        messages = [{"role": "user", "content": "Create the launch card"}]

        def flush(current_messages):
            current_messages[0]["_row_id"] = 767

        agent = SimpleNamespace(
            _context_engine_tool_names=set(), _memory_manager=None, quiet_mode=False,
            session_id="sess-1", platform="telegram", valid_tool_names=[],
            _flush_messages_to_session_db=flush, enabled_toolsets=None, disabled_toolsets=None,
        )
        ref = _ToolCallRef("kanban_create", {"title": "launch"}, "task-1", "call-1", [])
        with patch("model_tools.handle_function_call", return_value='{"ok": true}') as execute:
            _resolve_sequential_dispatch(agent, ref, messages).execute(ref.args)

        assert execute.call_args.kwargs["session_id"] == "sess-1"
        assert execute.call_args.kwargs["platform"] == "telegram"
        assert execute.call_args.kwargs["tool_origin"] == {
            "message_row_id": 767,
            "prompt": "Create the launch card",
        }

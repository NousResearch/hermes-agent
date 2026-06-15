"""Unit tests for the DeepAgents tool dispatcher (agent/deep_agents_tool_dispatcher.py).

Covers:
- execute_hermes_tools: single-tool execution via handle_function_call
- run_hermes_engine: full loop execution via conversation_loop.run_conversation
- build_hermes_engine_tool: StructuredTool construction
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_handle_function_call():
    """Mock model_tools.handle_function_call to return a success string."""
    with patch("model_tools.handle_function_call") as mock:
        mock.return_value = json.dumps({"status": "ok"})
        yield mock


@pytest.fixture()
def mock_run_conversation():
    """Mock agent.conversation_loop.run_conversation to return a result dict."""
    with patch("agent.conversation_loop.run_conversation") as mock:
        mock.return_value = {
            "final_response": "done.",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }
        yield mock


@pytest.fixture()
def mock_aiaagent_new():
    """Mock AIAgent.__new__ so the agent stub is a plain MagicMock."""
    with patch("run_agent.AIAgent.__new__") as mock:
        mock.return_value = MagicMock()
        yield mock


# ---------------------------------------------------------------------------
# execute_hermes_tools
# ---------------------------------------------------------------------------


def test_execute_hermes_tools_returns_result(monkeypatch, mock_handle_function_call):
    """execute_hermes_tools returns the handle_function_call result."""
    from agent.deep_agents_tool_dispatcher import execute_hermes_tools

    result = execute_hermes_tools("get_weather", {"city": "nyc"})

    assert result == json.dumps({"status": "ok"})
    mock_handle_function_call.assert_called_once_with(
        function_name="get_weather", function_args={"city": "nyc"}
    )


def test_execute_hermes_tools_handles_exception(monkeypatch):
    """execute_hermes_tools returns an error JSON string on exception."""
    from agent.deep_agents_tool_dispatcher import execute_hermes_tools

    def raise_error(**kwargs):
        raise RuntimeError("tool failed")

    with patch("model_tools.handle_function_call", side_effect=raise_error):
        result = execute_hermes_tools("broken_tool", {"x": 1})

    parsed = json.loads(result)
    assert parsed["error"] == "tool failed"


def test_execute_hermes_tools_error_uses_ensure_ascii_false(monkeypatch):
    """Error output uses ensure_ascii=False so non-ASCII chars survive."""
    from agent.deep_agents_tool_dispatcher import execute_hermes_tools

    def raise_unicode(**kwargs):
        raise ValueError("unicode: \u00e9\u00e0\u00fc")

    with patch("model_tools.handle_function_call", side_effect=raise_unicode):
        result = execute_hermes_tools("to", {})

    parsed = json.loads(result)
    assert "\u00e9\u00e0\u00fc" in parsed["error"]


# ---------------------------------------------------------------------------
# run_hermes_engine
# ---------------------------------------------------------------------------


def test_run_hermes_engine_returns_json_result(
    monkeypatch,
    mock_run_conversation,
    mock_aiaagent_new,
):
    """run_hermes_engine calls run_conversation and returns JSON result."""
    from agent.deep_agents_tool_dispatcher import run_hermes_engine

    session_id = "sess-123"
    task_id = "task-456"
    history = [{"role": "user", "content": "hello"}]

    result = run_hermes_engine(
        user_message="hello",
        conversation_history=json.dumps(history),
        session_id=session_id,
        task_id=task_id,
    )

    # Should return JSON-serialized result dict
    parsed = json.loads(result)
    assert parsed["final_response"] == "done."
    assert parsed["messages"] == []
    assert parsed["api_calls"] == 1
    assert parsed["completed"] is True


def test_run_hermes_engine_parses_conversation_history(monkeypatch):
    """run_hermes_engine parses the conversation_history JSON string into a list."""
    history_input = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    with patch("agent.conversation_loop.run_conversation") as mock_run:
        mock_run.return_value = {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }

        from agent.deep_agents_tool_dispatcher import run_hermes_engine

        run_hermes_engine(
            user_message="test",
            conversation_history=json.dumps(history_input),
            session_id="s1",
            task_id="t1",
        )

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["conversation_history"] == history_input


def test_run_hermes_engine_none_conversation_history(monkeypatch):
    """run_hermes_engine handles conversation_history=None as [] (falsy)."""
    with patch("agent.conversation_loop.run_conversation") as mock_run:
        mock_run.return_value = {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }

        from agent.deep_agents_tool_dispatcher import run_hermes_engine

        run_hermes_engine(
            user_message="hi",
            conversation_history=None,
            session_id="s1",
            task_id="t1",
        )

        _, kwargs = mock_run.call_args
        assert kwargs["conversation_history"] == []


def test_run_hermes_engine_empty_string_history(monkeypatch):
    """run_hermes_engine handles conversation_history='' as [] (falsy)."""
    with patch("agent.conversation_loop.run_conversation") as mock_run:
        mock_run.return_value = {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }

        from agent.deep_agents_tool_dispatcher import run_hermes_engine

        run_hermes_engine(
            user_message="hi",
            conversation_history="",
            session_id="s1",
            task_id="t1",
        )

        _, kwargs = mock_run.call_args
        assert kwargs["conversation_history"] == []


def test_run_hermes_engine_broken_json_defaults_to_empty_list(monkeypatch):
    """run_hermes_engine defaults to [] when conversation_history is invalid JSON."""
    with patch("agent.conversation_loop.run_conversation") as mock_run:
        mock_run.return_value = {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }

        from agent.deep_agents_tool_dispatcher import run_hermes_engine

        run_hermes_engine(
            user_message="hi",
            conversation_history="{not valid json",
            session_id="s1",
            task_id="t1",
        )

        _, kwargs = mock_run.call_args
        assert kwargs["conversation_history"] == []


def test_run_hermes_engine_passes_system_message_none(monkeypatch):
    """run_hermes_engine passes system_message=None to run_conversation."""
    with patch("run_agent.AIAgent.__new__") as mock_new:
        mock_new.return_value = MagicMock()

        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import run_hermes_engine

            run_hermes_engine(
                user_message="test",
                conversation_history="[]",
                session_id="s",
                task_id="t",
            )

            _, kwargs = mock_rc.call_args
            assert kwargs["system_message"] is None


def test_run_hermes_engine_forwards_task_id(monkeypatch):
    """run_hermes_engine forwards the task_id to run_conversation."""
    with patch("run_agent.AIAgent.__new__") as mock_new:
        mock_new.return_value = MagicMock()

        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import run_hermes_engine

            run_hermes_engine(
                user_message="hi",
                conversation_history="[]",
                session_id="sess-1",
                task_id="task-custom-id",
            )

            _, kwargs = mock_rc.call_args
            assert kwargs["task_id"] == "task-custom-id"


def test_run_hermes_engine_stubs_agent_attributes(monkeypatch):
    """run_hermes_engine creates a stub agent with expected attributes."""
    from agent.deep_agents_tool_dispatcher import run_hermes_engine

    with patch("run_agent.AIAgent.__new__") as mock_new:
        mock_agent = MagicMock()
        mock_new.return_value = mock_agent

        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            run_hermes_engine(
                user_message="msg",
                conversation_history="[]",
                session_id="sess-id",
                task_id="task-id",
            )

            # The implementation does direct attribute assignment, which
            # MagicMock handles via __setattr__. Check that all expected
            # attributes were set on the stub.
            assert mock_agent.session_id == "sess-id"
            assert mock_agent.model == ""
            assert mock_agent.provider == ""
            assert mock_agent.base_url == ""
            assert mock_agent.max_iterations == 90
            assert mock_agent.quiet_mode is True
            assert mock_agent.enabled_toolsets == []
            assert mock_agent.disabled_toolsets is None
            assert mock_agent.iteration_budget is None
            assert mock_agent.interrupt_requested is False
            assert mock_agent._interrupt_message is None
            assert mock_agent._stream_callback is None


def test_run_hermes_engine_agent_is_standalone_not_initialized():
    """run_hermes_engine uses __new__ only, never calls __init__."""
    with patch("run_agent.AIAgent.__new__") as mock_new:
        mock_agent = MagicMock()
        mock_new.return_value = mock_agent

        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_rc.return_value = {
                "final_response": "x",
                "messages": [],
                "api_calls": 0,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import run_hermes_engine

            run_hermes_engine("hi", "[]", "s", "t")

            # __new__ should have been called with no args (AIAgent.__new__())
            mock_new.assert_called_once()


# ---------------------------------------------------------------------------
# build_hermes_engine_tool
# ---------------------------------------------------------------------------


def test_build_hermes_engine_tool_returns_structured_tool():
    """build_hermes_engine_tool returns a LangChain StructuredTool instance."""
    from langchain_core.tools import StructuredTool
    from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

    tool = build_hermes_engine_tool()

    assert isinstance(tool, StructuredTool)
    assert tool.name == "hermes_engine"


def test_build_hermes_engine_tool_name():
    """The tool name is 'hermes_engine'."""
    from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

    tool = build_hermes_engine_tool()

    assert tool.name == "hermes_engine"


def test_build_hermes_engine_tool_description_mentions_hermes():
    """The tool description mentions key concepts about the Hermes agent loop."""
    from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

    tool = build_hermes_engine_tool()

    assert tool.description is not None
    assert len(tool.description) > 0
    assert "Hermes" in tool.description
    assert "agent loop" in tool.description.lower()


def test_build_hermes_engine_tool_callable_returns_json(monkeypatch):
    """Calling the tool's underlying func runs the loop and returns JSON."""
    with patch("agent.conversation_loop.run_conversation") as mock_rc:
        with patch("run_agent.AIAgent.__new__") as mock_new:
            mock_new.return_value = MagicMock()
            mock_rc.return_value = {
                "final_response": "tool result.",
                "messages": [{"role": "assistant", "content": "tool result."}],
                "api_calls": 3,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            # StructuredTool from_function wraps the callable as .func
            result = tool.func(
                user_message="What's 2+2?",
                conversation_history=json.dumps([{"role": "user", "content": "2+2?"}]),
                session_id="test-session",
                task_id="test-task",
            )

            # Should return a JSON string
            parsed = json.loads(result)
            assert parsed["final_response"] == "tool result."
            assert parsed["api_calls"] == 3
            assert parsed["completed"] is True


def test_build_hermes_engine_tool_forwards_conversation_history():
    """The tool's func forwards conversation_history to run_conversation."""
    history = [{"role": "user", "content": "first message"}]
    with patch("agent.conversation_loop.run_conversation") as mock_rc:
        with patch("run_agent.AIAgent.__new__") as mock_new:
            mock_new.return_value = MagicMock()
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            tool.func(
                user_message="response",
                conversation_history=json.dumps(history),
                session_id="s",
                task_id="t",
            )

            _, kwargs = mock_rc.call_args
            assert kwargs["conversation_history"] == history


def test_build_hermes_engine_tool_forwards_user_message():
    """The tool's func forwards user_message."""
    with patch("agent.conversation_loop.run_conversation") as mock_rc:
        with patch("run_agent.AIAgent.__new__") as mock_new:
            mock_new.return_value = MagicMock()
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            tool.func(
                user_message="What's the weather in Tokyo?",
                conversation_history="[]",
                session_id="s",
                task_id="t",
            )

            _, kwargs = mock_rc.call_args
            assert kwargs["user_message"] == "What's the weather in Tokyo?"


def test_build_hermes_engine_tool_forwards_session_id():
    """The tool's func forwards session_id to run_conversation."""
    with patch("agent.conversation_loop.run_conversation") as mock_rc:
        with patch("run_agent.AIAgent.__new__") as mock_new:
            mock_agent = MagicMock()
            mock_new.return_value = mock_agent
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            tool.func(
                user_message="hi",
                conversation_history="[]",
                session_id="my-session-id-12345",
                task_id="t",
            )

            # session_id is written onto the stub agent
            assert mock_agent.session_id == "my-session-id-12345"


def test_build_hermes_engine_tool_forwards_task_id():
    """The tool's func forwards task_id to run_conversation."""
    with patch("agent.conversation_loop.run_conversation") as mock_rc:
        with patch("run_agent.AIAgent.__new__") as mock_new:
            mock_new.return_value = MagicMock()
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            tool.func(
                user_message="hi",
                conversation_history="[]",
                session_id="s",
                task_id="task-custom-id",
            )

            # task_id is passed as a keyword argument to run_conversation
            _, kwargs = mock_rc.call_args
            assert kwargs["task_id"] == "task-custom-id"


def test_build_hermes_engine_tool_handles_unicode_in_result():
    """The tool result preserves non-ASCII characters (ensure_ascii=False)."""
    with patch("run_agent.AIAgent.__new__") as mock_new:
        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_new.return_value = MagicMock()
            mock_rc.return_value = {
                "final_response": "caf\u00e9 menu",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            result = tool.func(
                user_message="hello",
                conversation_history="[]",
                session_id="s",
                task_id="t",
            )

            parsed = json.loads(result)
            assert parsed["final_response"] == "caf\u00e9 menu"


def test_build_hermes_engine_stub_agent_has_correct_attributes():
    """The agent stub created inside the tool func has expected attributes."""
    with patch("run_agent.AIAgent.__new__") as mock_new:
        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_agent = MagicMock()
            mock_new.return_value = mock_agent
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import build_hermes_engine_tool

            tool = build_hermes_engine_tool()

            tool.func(
                user_message="hi",
                conversation_history="[]",
                session_id="attr-test-session",
                task_id="t",
            )

            # The implementation does `agent.xxx = value` on the stub.
            # MagicMock auto-creates any accessed attribute,
            # so we check the values were written.
            assert mock_agent.session_id == "attr-test-session"
            assert mock_agent.max_iterations == 90
            assert mock_agent.quiet_mode is True


def test_run_hermes_engine_stub_agent_attributes():
    """run_hermes_engine sets all stub agent attributes before calling run_conversation."""
    with patch("run_agent.AIAgent.__new__") as mock_new:
        with patch("agent.conversation_loop.run_conversation") as mock_rc:
            mock_agent = MagicMock()
            mock_new.return_value = mock_agent
            mock_rc.return_value = {
                "final_response": "done.",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

            from agent.deep_agents_tool_dispatcher import run_hermes_engine

            run_hermes_engine(
                user_message="test msg",
                conversation_history="[]",
                session_id="stub-test-session",
                task_id="stub-test-task",
            )

            # The stub agent gets all these attributes set
            assert mock_agent.session_id == "stub-test-session"
            assert mock_agent.model == ""
            assert mock_agent.provider == ""
            assert mock_agent.base_url == ""
            assert mock_agent.max_iterations == 90
            assert mock_agent.quiet_mode is True
            assert mock_agent.enabled_toolsets == []
            assert mock_agent.disabled_toolsets is None
            assert mock_agent.iteration_budget is None
            assert mock_agent.interrupt_requested is False
            assert mock_agent._interrupt_message is None
            assert mock_agent._stream_callback is None


# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Test that the module can be imported without side-effects."""

    def test_import(self):
        """The module imports cleanly."""
        import agent.deep_agents_tool_dispatcher as mod

        assert mod.execute_hermes_tools is not None
        assert mod.run_hermes_engine is not None
        assert mod.build_hermes_engine_tool is not None

    def test_logger_exists(self):
        """The module has a logger attribute set to __name__."""
        import agent.deep_agents_tool_dispatcher as mod

        assert mod.logger is not None
        assert mod.logger.name == "agent.deep_agents_tool_dispatcher"

    def test_no_public_api_mutation(self):
        """Public API functions are the expected three."""
        import agent.deep_agents_tool_dispatcher as mod

        public = [n for n in dir(mod) if not n.startswith("_")]
        assert "execute_hermes_tools" in public
        assert "run_hermes_engine" in public
        assert "build_hermes_engine_tool" in public

    def test_import_idempotent(self):
        """Re-importing the module returns the same functions."""
        import agent.deep_agents_tool_dispatcher as mod1
        import importlib

        importlib.reload(mod1)

        assert mod1.execute_hermes_tools is not None
        assert mod1.run_hermes_engine is not None

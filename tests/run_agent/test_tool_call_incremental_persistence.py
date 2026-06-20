from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from agent.tool_dispatch_helpers import make_tool_result_message
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_run_conversation_flushes_assistant_tool_call_before_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]
    flush_spy = MagicMock()
    agent._flush_messages_to_session_db = flush_spy

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        assert assistant_message.tool_calls[0].id == "c1"
        assert flush_spy.call_count >= 1
        flushed_messages = flush_spy.call_args_list[-1].args[0]
        assert flushed_messages[-1]["role"] == "assistant"
        assert flushed_messages[-1]["tool_calls"][0]["id"] == "c1"
        messages.append(make_tool_result_message("web_search", "search result", "c1"))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("search something")

    assert result["final_response"] == "done"


def test_execute_tool_calls_sequential_flushes_tool_result_immediately():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    flush_spy = MagicMock()
    agent._flush_messages_to_session_db = flush_spy
    agent._invoke_tool = MagicMock(return_value="search result")
    messages = []
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    with patch(
        "agent.tool_executor.maybe_persist_tool_result",
        side_effect=lambda **kwargs: kwargs["content"],
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "c1"
    assert flush_spy.call_count == 1
    flushed_messages = flush_spy.call_args_list[0].args[0]
    assert flushed_messages[-1]["tool_call_id"] == "c1"


def test_execute_tool_calls_concurrent_flushes_each_tool_result_in_order():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(call_id="c1"),
        _mock_tool_call(call_id="c2"),
    ]
    flushed_tool_ids = []

    def _capture_flush(flush_messages):
        flushed_tool_ids.append(flush_messages[-1]["tool_call_id"])

    flush_spy = MagicMock(side_effect=_capture_flush)
    agent._flush_messages_to_session_db = flush_spy
    agent._invoke_tool = MagicMock(side_effect=["result one", "result two"])
    messages = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    with patch(
        "agent.tool_executor.maybe_persist_tool_result",
        side_effect=lambda **kwargs: kwargs["content"],
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    assert [msg["tool_call_id"] for msg in messages] == ["c1", "c2"]
    assert flushed_tool_ids == ["c1", "c2"]

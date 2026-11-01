"""The background-review / side-question tool whitelist holds on the thread that runs the tool.

Regression: the whitelist was a ``threading.local`` set on the fork's thread, but every tool call
dispatches on a ``DaemonThreadPoolExecutor`` worker (agent/tool_executor.py). The worker saw no
whitelist, so a non-whitelisted tool ran. These tests dispatch through the real sequential and
concurrent tool executors and assert the call is blocked.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist
from run_agent import AIAgent


def _make_agent(tmp_path: Path) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[{
            "type": "function",
            "function": {"name": "web_extract", "description": "test tool",
                         "parameters": {"type": "object", "properties": {}}},
        }]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(side_effect=lambda _n, _a, result, **_k: result)
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(side_effect=lambda _n, result: result)
    return agent


def _calls(*ids):
    return SimpleNamespace(tool_calls=[
        SimpleNamespace(id=i, type="function", function=SimpleNamespace(name="web_extract", arguments="{}"))
        for i in ids])


def _run_whitelisted(execute, agent, assistant):
    """Run the executor with a whitelist that excludes web_extract; return (messages, dispatched)."""
    dispatched: list = []

    def _dispatch(_name, _args, *_a, tool_call_id=None, **_k):
        dispatched.append(tool_call_id)
        return "tool ran"

    messages: list[dict] = []
    set_thread_tool_whitelist({"memory", "skill_manage"}, deny_msg_fmt="denied {tool_name}")
    try:
        with patch("model_tools.handle_function_call", side_effect=_dispatch):
            execute(agent, assistant, messages, "task")
    finally:
        clear_thread_tool_whitelist()
    return messages, dispatched


def test_sequential_dispatch_on_worker_thread_honours_whitelist(tmp_path):
    messages, dispatched = _run_whitelisted(execute_tool_calls_sequential, _make_agent(tmp_path), _calls("c1"))
    assert dispatched == []
    assert [m["tool_call_id"] for m in messages] == ["c1"]
    assert "denied web_extract" in messages[0]["content"]


def test_concurrent_dispatch_on_worker_threads_honours_whitelist(tmp_path):
    messages, dispatched = _run_whitelisted(execute_tool_calls_concurrent, _make_agent(tmp_path), _calls("c1", "c2"))
    assert dispatched == []
    assert sorted(m["tool_call_id"] for m in messages) == ["c1", "c2"]
    assert all("denied web_extract" in m["content"] for m in messages)


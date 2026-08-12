"""Sequential tool calls recover when one dispatch never returns."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tool_executor import execute_tool_calls_sequential
from run_agent import AIAgent


def _make_agent(tmp_path: Path) -> AIAgent:
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_extract",
                        "description": "test tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(
        side_effect=lambda _name, _args, result, **_kwargs: result
    )
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(
        side_effect=lambda _name, result: result
    )
    return agent


def _tool_call(call_id: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="web_extract", arguments="{}"),
    )


def test_sequential_tool_timeout_emits_result_and_continues(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    first_started = threading.Event()
    release_first = threading.Event()
    dispatched: list[str] = []
    terminal_events: list[dict] = []

    def _dispatch(_name, _args, _task_id, *, tool_call_id, **_kwargs):
        dispatched.append(tool_call_id)
        if tool_call_id == "hung":
            first_started.set()
            release_first.wait()
            return "late result"
        return "second result"

    def _capture_terminal_event(*_args, **kwargs):
        terminal_events.append(kwargs)

    calls = [_tool_call("hung"), _tool_call("next")]
    assistant = SimpleNamespace(tool_calls=calls)
    messages: list[dict] = []
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0.05")

    started = time.monotonic()
    try:
        with (
            patch("run_agent.handle_function_call", side_effect=_dispatch),
            patch(
                "agent.tool_executor._emit_terminal_post_tool_call",
                side_effect=_capture_terminal_event,
            ),
        ):
            execute_tool_calls_sequential(agent, assistant, messages, "task")
    finally:
        release_first.set()

    assert first_started.is_set()
    assert time.monotonic() - started < 1.0
    assert dispatched == ["hung", "next"]
    assert [message["tool_call_id"] for message in messages] == ["hung", "next"]
    assert "timed out after 0.1s" in messages[0]["content"]
    assert messages[0]["effect_disposition"] == "unknown"
    assert messages[1]["content"] == "second result"
    timeout_events = [event for event in terminal_events if event.get("error_type") == "tool_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0]["status"] == "timeout"
    agent._flush_messages_to_session_db.assert_called()


def test_sequential_tool_timeout_suppresses_late_terminal_event(tmp_path, monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import model_tools

    agent = _make_agent(tmp_path)
    release_first = threading.Event()
    first_returned = threading.Event()
    dispatch_count = 0
    terminal_events: list[dict] = []

    def _dispatch(_name, _args, **_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            release_first.wait()
            first_returned.set()
            return "late result"
        return "second result"

    calls = [_tool_call("hung"), _tool_call("next")]
    messages: list[dict] = []
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0.05")

    try:
        with (
            patch.object(model_tools.registry, "dispatch", side_effect=_dispatch),
            patch.object(lifecycle, "has_hook", return_value=True),
            patch.object(
                lifecycle,
                "invoke_hook",
                side_effect=lambda hook, **kwargs: (
                    terminal_events.append(kwargs) if hook == "post_tool_call" else []
                ),
            ),
        ):
            execute_tool_calls_sequential(
                agent, SimpleNamespace(tool_calls=calls), messages, "task"
            )
            release_first.set()
            assert first_returned.wait(timeout=1)
    finally:
        release_first.set()

    assert [(event["tool_call_id"], event.get("error_type")) for event in terminal_events] == [
        ("hung", "tool_timeout"),
        ("next", None),
    ]

"""Integration tests for the observe-only run trace hook."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_constants import get_hermes_home
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
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


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_assistant_msg(content="Hello", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = _mock_assistant_msg(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/model",
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


def _enable_run_trace_config():
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "observability:\n  run_trace_enabled: true\n",
        encoding="utf-8",
    )


def _read_trace_entries() -> list[dict]:
    path = get_hermes_home() / "run_traces" / "run_traces.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_conversation_writes_metadata_only_trace_for_final_answer():
    _enable_run_trace_config()
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Final answer",
        finish_reason="stop",
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello with sk-test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result["completed"] is True
    entries = _read_trace_entries()
    assert len(entries) == 1
    entry = entries[0]
    encoded = json.dumps(entry, ensure_ascii=False)
    assert entry["schema_version"] == "hermes_run_trace_v1"
    assert entry["status"] == "completed"
    assert entry["api_call_count"] == 1
    assert entry["model"] == "test/model"
    assert entry["provider"] == "openrouter"
    assert entry["tool_calls"] == []
    assert "hello with" not in encoded
    assert "Final answer" not in encoded
    assert "sk-test_aaaaaaaa" not in encoded


def test_run_conversation_trace_records_tool_names_without_args_or_results():
    _enable_run_trace_config()
    agent = _make_agent()
    tc = _mock_tool_call(
        name="web_search",
        arguments=json.dumps({"query": "private query", "token": "sk-test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}),
        call_id="call_search_1",
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc]),
        _mock_response(content="Done", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="raw tool output sk-test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search something private")

    assert result["completed"] is True
    entries = _read_trace_entries()
    assert len(entries) == 1
    entry = entries[0]
    encoded = json.dumps(entry, ensure_ascii=False)
    assert entry["api_call_count"] == 2
    assert len(entry["tool_calls"]) == 1
    tool_data = entry["tool_calls"][0]
    assert tool_data["name"] == "web_search"
    assert tool_data["tool_call_id"].startswith("sha256:")
    assert tool_data["status"] == "requested"
    assert tool_data["duration_ms"] is None
    assert tool_data["error_type"] == ""
    assert tool_data["error_message"] == ""
    assert "call_search_1" not in encoded
    assert "private query" not in encoded
    assert "raw tool output" not in encoded
    assert "sk-test_aaaaaaaa" not in encoded


def test_run_conversation_trace_hashes_caller_supplied_slug_task_id():
    _enable_run_trace_config()
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Final answer",
        finish_reason="stop",
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "hello",
            task_id="customer-private-acquisition-plan",
        )

    assert result["completed"] is True
    entry = _read_trace_entries()[0]
    encoded = json.dumps(entry, ensure_ascii=False)
    assert entry["task_id"].startswith("sha256:")
    assert entry["turn_id"].startswith("sha256:")
    assert "customer-private-acquisition-plan" not in encoded


def test_run_conversation_trace_finishes_on_direct_return_path():
    _enable_run_trace_config()
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="model refusal detail",
        finish_reason="content_filter",
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is False
    entries = _read_trace_entries()
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert entries[0]["api_call_count"] == 1
    assert "model refusal detail" not in json.dumps(entries[0], ensure_ascii=False)


def test_run_conversation_trace_write_failure_does_not_fail_turn(monkeypatch):
    _enable_run_trace_config()
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Final answer",
        finish_reason="stop",
    )

    def _raise(_trace, *, config=None):
        raise RuntimeError("disk full")

    monkeypatch.setattr("agent.run_trace.append_run_trace", _raise)

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert result["final_response"] == "Final answer"


def test_run_conversation_trace_reflects_post_finalize_cleanup_errors():
    _enable_run_trace_config()
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Final answer",
        finish_reason="stop",
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(
            agent,
            "_cleanup_task_resources",
            side_effect=RuntimeError("cleanup leaked private prompt text"),
        ),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert result["cleanup_errors"]
    entries = _read_trace_entries()
    assert len(entries) == 1
    encoded = json.dumps(entries[0], ensure_ascii=False)
    assert entries[0]["status"] == "failed"
    assert entries[0]["exit_reason"] == "text_response"
    assert "cleanup leaked private" not in encoded


def test_run_conversation_does_not_start_trace_for_codex_app_server_path():
    _enable_run_trace_config()
    agent = _make_agent()
    agent.api_mode = "codex_app_server"
    agent._run_codex_app_server_turn = MagicMock(
        return_value={
            "final_response": "codex result",
            "completed": True,
            "api_calls": 0,
        }
    )

    result = agent.run_conversation("hello")

    assert result["final_response"] == "codex result"
    assert agent._run_codex_app_server_turn.called
    assert not (get_hermes_home() / "run_traces" / "run_traces.jsonl").exists()

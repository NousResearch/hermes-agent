import json
import logging
import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent import conversation_loop
from agent.request_budget import RequestBudget
import run_agent


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _message_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=text,
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model="gpt-5.5",
    )


def test_run_conversation_emits_request_budget_payload(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(
        run_agent,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            close=lambda: None,
            is_closed=lambda: False,
        ),
    )

    agent = run_agent.AIAgent(
        model="gpt-5.5",
        provider="openai",
        api_mode="chat_completions",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=2,
        skip_context_files=True,
        skip_memory=True,
        platform="slack",
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None
    agent._disable_streaming = True
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: _message_response("OK"),
    )

    with caplog.at_level(logging.INFO):
        result = agent.run_conversation("hello")

    assert result["final_response"] == "OK"
    payload = result["request_budget"]
    assert payload["tool_schema_tokens"] > 0
    assert payload["model_ttfb_ms"] is not None
    assert payload["tool_execution_ms"] == 0
    assert payload["reason"].startswith("text_response")

    records = [
        record.getMessage().split("request_budget.v1 ", 1)[1]
        for record in caplog.records
        if "request_budget.v1 " in record.getMessage()
    ]
    assert len(records) == 1
    logged = json.loads(records[0])
    assert logged["session_id"] == agent.session_id
    assert logged["tool_schema_tokens"] == payload["tool_schema_tokens"]


def test_direct_terminal_return_still_finalizes_request_budget(monkeypatch, caplog):
    agent = SimpleNamespace(
        session_id="session-early",
        _user_turn_count=4,
        model="gpt-5.5",
        provider="openai-codex",
        platform="slack",
    )

    monkeypatch.setattr(
        conversation_loop,
        "_run_conversation_impl",
        lambda *args, **kwargs: {
            "final_response": "provider unavailable",
            "api_calls": 0,
            "completed": False,
            "failed": True,
        },
    )

    with caplog.at_level(logging.INFO):
        result = conversation_loop.run_conversation(agent, "hello")

    assert result["request_budget"]["reason"] == "failed"
    assert result["request_budget"]["api_calls"] == 0
    assert sum("request_budget.v1 " in r.getMessage() for r in caplog.records) == 1
    assert agent._request_budget is None


def test_tool_batch_records_each_name_separately(monkeypatch):
    agent = object.__new__(run_agent.AIAgent)
    agent._request_budget = RequestBudget(
        session_id="session-tools",
        turn_id="1",
        model="gpt-5.5",
        provider="openai-codex",
        platform="slack",
    )
    agent._execute_tool_calls_sequential = lambda *args: None
    monkeypatch.setattr(
        run_agent,
        "_should_parallelize_tool_batch",
        lambda tool_calls: False,
    )
    assistant_message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name="read_file")),
            SimpleNamespace(function=SimpleNamespace(name="web_search")),
        ]
    )

    agent._execute_tool_calls(assistant_message, [], "task-1")

    payload = agent._request_budget.snapshot(reason="tool_response", api_calls=1)
    assert payload["tool_call_count"] == 2
    assert payload["tool_names"] == ["read_file", "web_search"]

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.chat_completion_helpers import build_api_kwargs, interruptible_api_call
from agent.cursor_adapter import build_cursor_api_kwargs, run_cursor_turn


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_cursor_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="cursor-composer-2.5",
        provider="cursor",
        api_mode="cursor_agent",
        base_url="https://api2.cursor.sh",
        api_key="cursor-token",
        quiet_mode=True,
        max_iterations=2,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def test_build_api_kwargs_cursor_agent(monkeypatch):
    agent = _build_cursor_agent(monkeypatch)
    kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )

    assert kwargs["model"] == "cursor-composer-2.5"
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["tools"][0]["function"]["name"] == "terminal"

    cursor_kwargs = build_cursor_api_kwargs(agent, kwargs)
    assert cursor_kwargs["model_id"] == "cursor-composer-2.5"
    assert cursor_kwargs["system_prompt"] == ["You are Hermes."]
    assert cursor_kwargs["messages"] == [{"role": "user", "content": "Ping"}]
    assert cursor_kwargs["tools"][0]["name"] == "terminal"
    assert cursor_kwargs["conversation_id"]


def test_interruptible_api_call_uses_cursor_path_not_openai(monkeypatch):
    agent = _build_cursor_agent(monkeypatch)
    openai_called = {"value": False}

    def _fake_create_request_openai_client(**kwargs):
        openai_called["value"] = True
        raise AssertionError("OpenAI client should not be created for cursor_agent")

    monkeypatch.setattr(agent, "_create_request_openai_client", _fake_create_request_openai_client)
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content="Cursor OK", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="cursor-composer-2.5",
    )
    called = {"api_kwargs": None}

    def _fake_run_cursor_turn(agent_arg, api_kwargs, on_first_delta=None, on_text_delta=None):
        called["api_kwargs"] = api_kwargs
        return mock_response

    monkeypatch.setattr("agent.chat_completion_helpers.run_cursor_turn", _fake_run_cursor_turn)

    api_kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )
    response = interruptible_api_call(agent, api_kwargs)

    assert response is mock_response
    assert openai_called["value"] is False
    assert called["api_kwargs"] == api_kwargs


def test_get_transport_returns_cursor_transport(monkeypatch):
    agent = _build_cursor_agent(monkeypatch)
    transport = agent._get_transport()
    assert transport is not None
    assert transport.api_mode == "cursor_agent"


def test_primary_runtime_snapshots_cursor_state(monkeypatch):
    agent = _build_cursor_agent(monkeypatch)
    agent._cursor_blob_store = {"abc": b"123"}
    agent._cursor_conversation_state = MagicMock(name="cursor_state")
    agent._cursor_conversation_id = "cursor-session"
    agent._primary_runtime.update(
        {
            "cursor_blob_store": {"abc": b"123"},
            "cursor_conversation_state": agent._cursor_conversation_state,
            "cursor_conversation_id": "cursor-session",
        }
    )

    agent._fallback_activated = True
    agent.model = "fallback-model"
    agent.provider = "custom"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    result = agent._restore_primary_runtime()

    assert result is True
    assert agent.api_mode == "cursor_agent"
    assert agent.client is None
    assert agent._cursor_blob_store == {"abc": b"123"}
    assert agent._cursor_conversation_state is not None
    assert agent._cursor_conversation_id == "cursor-session"


def test_run_cursor_turn_passes_agent_interrupt_event(monkeypatch):
    interrupt_event = threading.Event()
    agent = SimpleNamespace(
        api_key="cursor-token",
        base_url="https://api2.cursor.sh",
        model="cursor-composer-2.5",
        session_id="session-1",
        _cursor_blob_store=None,
        _cursor_conversation_state=None,
        _interrupt_event=interrupt_event,
        _interrupt_requested=False,
    )
    captured: dict[str, object] = {}

    def _fake_run_cursor_agent_turn(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=kwargs["model_id"],
            blob_store={},
            conversation_state=None,
        )

    monkeypatch.setattr("agent.cursor_adapter.run_cursor_agent_turn", _fake_run_cursor_agent_turn)

    run_cursor_turn(
        agent,
        {
            "model": "cursor-composer-2.5",
            "messages": [{"role": "user", "content": "Ping"}],
            "tools": [],
        },
    )

    assert captured["interrupt_event"] is interrupt_event


def test_cursor_agent_transport_normalize_response():
    from agent.transports import get_transport

    transport = get_transport("cursor_agent")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="CURSOR_OK", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    nr = transport.normalize_response(response)
    assert nr.content == "CURSOR_OK"
    assert nr.finish_reason == "stop"
    assert nr.tool_calls is None

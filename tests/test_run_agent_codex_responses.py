import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


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


def _build_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None
    return agent


def _codex_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        status="completed",
        model="gpt-5-codex",
    )


def _codex_tool_call_response():
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments="{}",
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        status="completed",
        model="gpt-5-codex",
    )


def _codex_incomplete_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="in_progress",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="in_progress",
        model="gpt-5-codex",
    )


def _codex_commentary_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="commentary",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5-codex",
    )


def _codex_ack_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5-codex",
    )


class _FakeResponsesStream:
    def __init__(self, *, final_response=None, final_error=None):
        self._final_response = final_response
        self._final_error = final_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(())

    def get_final_response(self):
        if self._final_error is not None:
            raise self._final_error
        return self._final_response


class _FakeCreateStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self):
        self.closed = True


def _codex_request_kwargs():
    return {
        "model": "gpt-5-codex",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "Ping"}],
        "tools": None,
        "store": False,
    }


def test_api_mode_uses_explicit_provider_when_codex(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://openrouter.ai/api/v1",
        provider="openai-codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "openai-codex"


def test_api_mode_normalizes_provider_case(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://openrouter.ai/api/v1",
        provider="OpenAI-Codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.provider == "openai-codex"
    assert agent.api_mode == "codex_responses"


def test_api_mode_respects_explicit_openrouter_provider_over_codex_url(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openrouter",
        api_key="test-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "chat_completions"
    assert agent.provider == "openrouter"


def test_build_api_kwargs_codex(monkeypatch):
    agent = _build_agent(monkeypatch)
    kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )

    assert kwargs["model"] == "gpt-5-codex"
    assert kwargs["instructions"] == "You are Hermes."
    assert kwargs["store"] is False
    assert isinstance(kwargs["input"], list)
    assert kwargs["input"][0]["role"] == "user"
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["name"] == "terminal"
    assert kwargs["tools"][0]["strict"] is False
    assert "function" not in kwargs["tools"][0]
    assert kwargs["store"] is False
    assert "timeout" not in kwargs
    assert "max_tokens" not in kwargs
    assert "extra_body" not in kwargs


def test_run_codex_stream_retries_when_completed_event_missing(monkeypatch):
    agent = _build_agent(monkeypatch)
    calls = {"stream": 0}

    def _fake_stream(**kwargs):
        calls["stream"] += 1
        if calls["stream"] == 1:
            return _FakeResponsesStream(
                final_error=RuntimeError("Didn't receive a `response.completed` event.")
            )
        return _FakeResponsesStream(final_response=_codex_message_response("stream ok"))

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(
            stream=_fake_stream,
            create=lambda **kwargs: _codex_message_response("fallback"),
        )
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert calls["stream"] == 2
    assert response.output[0].content[0].text == "stream ok"


def test_run_codex_stream_falls_back_to_create_after_stream_completion_error(monkeypatch):
    agent = _build_agent(monkeypatch)
    calls = {"stream": 0, "create": 0}

    def _fake_stream(**kwargs):
        calls["stream"] += 1
        return _FakeResponsesStream(
            final_error=RuntimeError("Didn't receive a `response.completed` event.")
        )

    def _fake_create(**kwargs):
        calls["create"] += 1
        return _codex_message_response("create fallback ok")

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(
            stream=_fake_stream,
            create=_fake_create,
        )
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert calls["stream"] == 2
    assert calls["create"] == 1
    assert response.output[0].content[0].text == "create fallback ok"


def test_run_codex_stream_fallback_parses_create_stream_events(monkeypatch):
    agent = _build_agent(monkeypatch)
    calls = {"stream": 0, "create": 0}
    create_stream = _FakeCreateStream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.in_progress"),
            SimpleNamespace(type="response.completed", response=_codex_message_response("streamed create ok")),
        ]
    )

    def _fake_stream(**kwargs):
        calls["stream"] += 1
        return _FakeResponsesStream(
            final_error=RuntimeError("Didn't receive a `response.completed` event.")
        )

    def _fake_create(**kwargs):
        calls["create"] += 1
        assert kwargs.get("stream") is True
        return create_stream

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(
            stream=_fake_stream,
            create=_fake_create,
        )
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert calls["stream"] == 2
    assert calls["create"] == 1
    assert create_stream.closed is True
    assert response.output[0].content[0].text == "streamed create ok"


def test_run_conversation_codex_plain_text(monkeypatch):
    agent = _build_agent(monkeypatch)
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _codex_message_response("OK"))

    result = agent.run_conversation("Say OK")

    assert result["completed"] is True
    assert result["final_response"] == "OK"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "OK"


def test_run_conversation_codex_refreshes_after_401_and_retries(monkeypatch):
    agent = _build_agent(monkeypatch)
    calls = {"api": 0, "refresh": 0}

    class _UnauthorizedError(RuntimeError):
        def __init__(self):
            super().__init__("Error code: 401 - unauthorized")
            self.status_code = 401

    def _fake_api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            raise _UnauthorizedError()
        return _codex_message_response("Recovered after refresh")

    def _fake_refresh(*, force=True):
        calls["refresh"] += 1
        assert force is True
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)
    monkeypatch.setattr(agent, "_try_refresh_codex_client_credentials", _fake_refresh)

    result = agent.run_conversation("Say OK")

    assert calls["api"] == 2
    assert calls["refresh"] == 1
    assert result["completed"] is True
    assert result["final_response"] == "Recovered after refresh"


def test_try_refresh_codex_client_credentials_rebuilds_client(monkeypatch):
    agent = _build_agent(monkeypatch)
    closed = {"value": False}
    rebuilt = {"kwargs": None}

    class _ExistingClient:
        def close(self):
            closed["value"] = True

    class _RebuiltClient:
        pass

    def _fake_openai(**kwargs):
        rebuilt["kwargs"] = kwargs
        return _RebuiltClient()

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda force_refresh=True: {
            "api_key": "new-codex-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    agent.client = _ExistingClient()
    ok = agent._try_refresh_codex_client_credentials(force=True)

    assert ok is True
    assert closed["value"] is True
    assert rebuilt["kwargs"]["api_key"] == "new-codex-token"
    assert rebuilt["kwargs"]["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert isinstance(agent.client, _RebuiltClient)


def test_run_conversation_codex_tool_round_trip(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [_codex_tool_call_response(), _codex_message_response("done")]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("run a command")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert any(msg.get("tool_calls") for msg in result["messages"] if msg.get("role") == "assistant")
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_chat_messages_to_responses_input_uses_call_id_for_function_call(monkeypatch):
    agent = _build_agent(monkeypatch)
    items = agent._chat_messages_to_responses_input(
        [
            {"role": "user", "content": "Run terminal"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": '{"ok":true}'},
        ]
    )

    function_call = next(item for item in items if item.get("type") == "function_call")
    function_output = next(item for item in items if item.get("type") == "function_call_output")

    assert function_call["call_id"] == "call_abc123"
    assert "id" not in function_call
    assert function_output["call_id"] == "call_abc123"


def test_chat_messages_to_responses_input_accepts_call_pipe_fc_ids(monkeypatch):
    agent = _build_agent(monkeypatch)
    items = agent._chat_messages_to_responses_input(
        [
            {"role": "user", "content": "Run terminal"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_pair123|fc_pair123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_pair123|fc_pair123", "content": '{"ok":true}'},
        ]
    )

    function_call = next(item for item in items if item.get("type") == "function_call")
    function_output = next(item for item in items if item.get("type") == "function_call_output")

    assert function_call["call_id"] == "call_pair123"
    assert "id" not in function_call
    assert function_output["call_id"] == "call_pair123"


def test_preflight_codex_api_kwargs_strips_optional_function_call_id(monkeypatch):
    agent = _build_agent(monkeypatch)
    preflight = agent._preflight_codex_api_kwargs(
        {
            "model": "gpt-5-codex",
            "instructions": "You are Hermes.",
            "input": [
                {"role": "user", "content": "hi"},
                {
                    "type": "function_call",
                    "id": "call_bad",
                    "call_id": "call_good",
                    "name": "terminal",
                    "arguments": "{}",
                },
            ],
            "tools": [],
            "store": False,
        }
    )

    fn_call = next(item for item in preflight["input"] if item.get("type") == "function_call")
    assert fn_call["call_id"] == "call_good"
    assert "id" not in fn_call


def test_preflight_codex_api_kwargs_rejects_function_call_output_without_call_id(monkeypatch):
    agent = _build_agent(monkeypatch)

    with pytest.raises(ValueError, match="function_call_output is missing call_id"):
        agent._preflight_codex_api_kwargs(
            {
                "model": "gpt-5-codex",
                "instructions": "You are Hermes.",
                "input": [{"type": "function_call_output", "output": "{}"}],
                "tools": [],
                "store": False,
            }
        )


def test_preflight_codex_api_kwargs_rejects_unsupported_request_fields(monkeypatch):
    agent = _build_agent(monkeypatch)
    kwargs = _codex_request_kwargs()
    kwargs["some_unknown_field"] = "value"

    with pytest.raises(ValueError, match="unsupported field"):
        agent._preflight_codex_api_kwargs(kwargs)


def test_preflight_codex_api_kwargs_allows_reasoning_and_temperature(monkeypatch):
    agent = _build_agent(monkeypatch)
    kwargs = _codex_request_kwargs()
    kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
    kwargs["include"] = ["reasoning.encrypted_content"]
    kwargs["temperature"] = 0.7
    kwargs["max_output_tokens"] = 4096

    result = agent._preflight_codex_api_kwargs(kwargs)
    assert result["reasoning"] == {"effort": "high", "summary": "auto"}
    assert result["include"] == ["reasoning.encrypted_content"]
    assert result["temperature"] == 0.7
    assert result["max_output_tokens"] == 4096


def test_run_conversation_codex_replay_payload_keeps_call_id(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [_codex_tool_call_response(), _codex_message_response("done")]
    requests = []

    def _fake_api_call(api_kwargs):
        requests.append(api_kwargs)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("run a command")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert len(requests) >= 2

    replay_input = requests[1]["input"]
    function_call = next(item for item in replay_input if item.get("type") == "function_call")
    function_output = next(item for item in replay_input if item.get("type") == "function_call_output")
    assert function_call["call_id"] == "call_1"
    assert "id" not in function_call
    assert function_output["call_id"] == "call_1"


def test_run_conversation_codex_continues_after_incomplete_interim_message(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_incomplete_message_response("I'll inspect the repo structure first."),
        _codex_tool_call_response(),
        _codex_message_response("Architecture summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("analyze repo")

    assert result["completed"] is True
    assert result["final_response"] == "Architecture summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "inspect the repo structure" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_normalize_codex_response_marks_commentary_only_message_as_incomplete(monkeypatch):
    agent = _build_agent(monkeypatch)
    assistant_message, finish_reason = agent._normalize_codex_response(
        _codex_commentary_message_response("I'll inspect the repository first.")
    )

    assert finish_reason == "incomplete"
    assert "inspect the repository" in (assistant_message.content or "")


def test_run_conversation_codex_continues_after_commentary_phase_message(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_commentary_message_response("I'll inspect the repo structure first."),
        _codex_tool_call_response(),
        _codex_message_response("Architecture summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("analyze repo")

    assert result["completed"] is True
    assert result["final_response"] == "Architecture summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "inspect the repo structure" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_run_conversation_codex_continues_after_ack_stop_message(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_ack_message_response(
            "Absolutely — I can do that. I'll inspect ~/openclaw-studio and report back with a walkthrough."
        ),
        _codex_tool_call_response(),
        _codex_message_response("Architecture summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("look into ~/openclaw-studio and tell me how it works")

    assert result["completed"] is True
    assert result["final_response"] == "Architecture summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "inspect ~/openclaw-studio" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(
        msg.get("role") == "user"
        and "Continue now. Execute the required tool calls" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_run_conversation_codex_continues_after_ack_for_directory_listing_prompt(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_ack_message_response(
            "I'll check what's in the current directory and call out 3 notable items."
        ),
        _codex_tool_call_response(),
        _codex_message_response("Directory summary complete."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("look at current directory and list 3 notable things")

    assert result["completed"] is True
    assert result["final_response"] == "Directory summary complete."
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "current directory" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(
        msg.get("role") == "user"
        and "Continue now. Execute the required tool calls" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])

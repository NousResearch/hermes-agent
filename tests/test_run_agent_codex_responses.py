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
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
    assert isinstance(kwargs["prompt_cache_key"], str)
    assert len(kwargs["prompt_cache_key"]) > 0
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

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
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


def test_preflight_codex_api_kwargs_preserves_optional_function_call_id(monkeypatch):
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
    # id field is preserved so Codex can correlate function calls with reasoning
    assert fn_call["id"] == "call_bad"


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
    # id is the original response_item_id ("fc_1") so Codex can correlate
    # the function call with its encrypted reasoning from Turn 1.
    assert function_call["id"] == "fc_1"
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


# ---------------------------------------------------------------------------
# Regression: Codex Responses API streaming → stream_delta_callback
# ---------------------------------------------------------------------------

class TestRegressionCodexStreamingDeltaCallback:
    """
    Guards:
      - _run_codex_stream fires stream_delta_callback for every
        response.output_text.delta event.
      - The main loop routes to _run_codex_stream (streaming) when
        stream_delta_callback is set and api_mode == codex_responses.
    Before the fixes, _run_codex_stream discarded all events via
    `for _ in stream: pass` and the dispatch unconditionally used
    _interruptible_api_call for codex_responses regardless of whether a
    callback was registered.
    """

    def _build_codex_agent(self, monkeypatch, callback=None):
        _patch_agent_bootstrap(monkeypatch)
        agent = run_agent.AIAgent(
            model="gpt-5.3-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="codex-token",
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=callback,
        )
        agent._cleanup_task_resources = lambda task_id: None
        agent._persist_session = lambda messages, history=None: None
        agent._save_trajectory = lambda messages, user_message, completed: None
        agent._save_session_log = lambda messages: None
        return agent

    # ------------------------------------------------------------------
    # 1. _run_codex_stream fires stream_delta_callback for delta events
    # ------------------------------------------------------------------

    def test_run_codex_stream_fires_callback_for_output_text_delta(self, monkeypatch):
        """_run_codex_stream must call stream_delta_callback for each
        response.output_text.delta event — not silently discard them."""
        from types import SimpleNamespace
        import contextlib

        received = []
        agent = self._build_codex_agent(monkeypatch, callback=received.append)

        # Build a fake stream that emits two delta events then a completion event
        delta_events = [
            SimpleNamespace(type="response.output_text.delta", delta="Hello"),
            SimpleNamespace(type="response.output_text.delta", delta=", world!"),
            SimpleNamespace(type="response.done"),           # non-delta, must be ignored
        ]
        final_response = _codex_message_response("Hello, world!")

        class _FakeStream:
            def __iter__(self):
                return iter(delta_events)
            def get_final_response(self):
                return final_response

        @contextlib.contextmanager
        def _fake_stream_ctx(**kwargs):
            yield _FakeStream()

        # Patch client.responses.stream
        agent.client = SimpleNamespace(
            responses=SimpleNamespace(stream=_fake_stream_ctx)
        )

        api_kwargs = agent._preflight_codex_api_kwargs(
            {"model": "gpt-5.3-codex", "instructions": "You are helpful.", "input": [], "tools": []},
            allow_stream=True,
        )
        result = agent._run_codex_stream(api_kwargs)

        assert result is final_response, "Should return the final response object"
        assert received == ["Hello", ", world!"], (
            "_run_codex_stream must fire stream_delta_callback for every "
            "response.output_text.delta event; non-delta events must be skipped. "
            f"Got: {received!r}"
        )

    def test_run_codex_stream_no_callback_does_not_raise(self, monkeypatch):
        """_run_codex_stream must work fine when stream_delta_callback is None."""
        from types import SimpleNamespace
        import contextlib

        agent = self._build_codex_agent(monkeypatch, callback=None)

        delta_events = [
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
        ]
        final_response = _codex_message_response("hi")

        class _FakeStream:
            def __iter__(self):
                return iter(delta_events)
            def get_final_response(self):
                return final_response

        @contextlib.contextmanager
        def _fake_stream_ctx(**kwargs):
            yield _FakeStream()

        agent.client = SimpleNamespace(
            responses=SimpleNamespace(stream=_fake_stream_ctx)
        )

        api_kwargs = agent._preflight_codex_api_kwargs(
            {"model": "gpt-5.3-codex", "instructions": "You are helpful.", "input": [], "tools": []},
            allow_stream=True,
        )
        result = agent._run_codex_stream(api_kwargs)
        assert result is final_response

    # ------------------------------------------------------------------
    # 2. Main loop dispatch: codex_responses + callback → _run_codex_stream
    # ------------------------------------------------------------------

    def test_dispatch_uses_run_codex_stream_when_callback_set(self, monkeypatch):
        """When stream_delta_callback is registered the main API dispatch
        must call _run_codex_stream, not _interruptible_api_call.
        Before the fix, codex_responses always used _interruptible_api_call."""
        received = []
        agent = self._build_codex_agent(monkeypatch, callback=received.append)

        calls = {"codex_stream": 0, "blocking": 0}

        def _fake_codex_stream(api_kwargs):
            calls["codex_stream"] += 1
            return _codex_message_response("streamed")

        def _fake_blocking(api_kwargs):
            calls["blocking"] += 1
            return _codex_message_response("blocked")

        monkeypatch.setattr(agent, "_run_codex_stream", _fake_codex_stream)
        monkeypatch.setattr(agent, "_interruptible_api_call", _fake_blocking)

        result = agent.run_conversation("hello")

        assert calls["codex_stream"] >= 1, (
            "_run_codex_stream must be called when stream_delta_callback is set "
            "and api_mode == codex_responses. "
            f"codex_stream={calls['codex_stream']} blocking={calls['blocking']}"
        )
        assert calls["blocking"] == 0, (
            "_interruptible_api_call must NOT be called when stream_delta_callback "
            "is set and api_mode == codex_responses. "
            f"codex_stream={calls['codex_stream']} blocking={calls['blocking']}"
        )
        assert result["final_response"] == "streamed"

    def test_dispatch_uses_blocking_call_when_no_callback(self, monkeypatch):
        """Without a stream_delta_callback the codex_responses path must
        still use the blocking _interruptible_api_call (unchanged behaviour)."""
        agent = self._build_codex_agent(monkeypatch, callback=None)

        calls = {"codex_stream": 0, "blocking": 0}

        def _fake_codex_stream(api_kwargs):
            calls["codex_stream"] += 1
            return _codex_message_response("streamed")

        def _fake_blocking(api_kwargs):
            calls["blocking"] += 1
            return _codex_message_response("blocked")

        monkeypatch.setattr(agent, "_run_codex_stream", _fake_codex_stream)
        monkeypatch.setattr(agent, "_interruptible_api_call", _fake_blocking)

        result = agent.run_conversation("hello")

        assert calls["blocking"] >= 1, (
            "Without stream_delta_callback, codex_responses must use "
            "_interruptible_api_call. "
            f"codex_stream={calls['codex_stream']} blocking={calls['blocking']}"
        )
        assert result["final_response"] == "blocked"

    # ------------------------------------------------------------------
    # 3. Source: stream_consumer accepts StreamingConfig dataclass
    #    (mirrors TestRegressionStreamingConfigLoading but from run_agent side)
    # ------------------------------------------------------------------

    def test_stream_delta_callback_attribute_exists_on_agent(self, monkeypatch):
        """AIAgent must expose stream_delta_callback as a public attribute
        so GatewayStreamConsumer can pass .on_delta to it."""
        received = []
        agent = self._build_codex_agent(monkeypatch, callback=received.append)
        assert callable(agent.stream_delta_callback), (
            "stream_delta_callback must be a callable attribute on AIAgent"
        )
        agent.stream_delta_callback("test")
        assert received == ["test"]

    def test_stream_delta_callback_none_when_not_set(self, monkeypatch):
        """stream_delta_callback must be None when not provided — the
        dispatch guards rely on truthiness of this attribute."""
        agent = self._build_codex_agent(monkeypatch, callback=None)
        assert agent.stream_delta_callback is None, (
            "stream_delta_callback must be None when no callback is provided"
        )

# ─── Regression tests for Codex streaming delta-collection fix ────────────────
# These go at the end of test_run_agent_codex_responses.py


class TestCodexStreamDeltaCollection:
    """
    Regression suite for the bug where Codex fires output_text.delta streaming
    events but returns a final response.output list with no 'message' item
    (observed with reasoning enabled).  _run_codex_stream must synthesise a
    message item from the collected deltas so _normalize_codex_response always
    produces non-empty content.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build(self, monkeypatch):
        _patch_agent_bootstrap(monkeypatch)
        agent = run_agent.AIAgent(
            model="gpt-5.3-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="codex-token",
            quiet_mode=True,
            max_iterations=2,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cleanup_task_resources = lambda task_id: None
        agent._persist_session = lambda messages, history=None: None
        agent._save_trajectory = lambda messages, user_message, completed: None
        agent._save_session_log = lambda messages: None
        return agent

    def _delta_event(self, text):
        return SimpleNamespace(type="response.output_text.delta", delta=text)

    def _response_no_message(self, reasoning_text="thinking..."):
        """Final response with only a reasoning item — no message item."""
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    status="completed",
                    encrypted_content="enc-abc",
                    summary=[],
                )
            ],
            status="completed",
            model="gpt-5.3-codex",
            output_text="",
        )

    def _response_with_message(self, text):
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text=text)],
                )
            ],
            status="completed",
            model="gpt-5.3-codex",
            output_text=text,
        )

    # ------------------------------------------------------------------
    # stream helper that yields delta events and returns a fixed response
    # ------------------------------------------------------------------

    def _make_stream_with_deltas(self, deltas, final_response):
        events = [self._delta_event(d) for d in deltas]

        class _Stream:
            def __init__(self):
                self._events = events
                self._final = final_response

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return iter(self._events)

            def get_final_response(self):
                return self._final

        return _Stream()

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_deltas_collected_and_message_synthesised_when_output_has_no_message(
        self, monkeypatch
    ):
        """
        Core regression: deltas fire, final response has no message item.
        _run_codex_stream must append a synthetic message item.
        """
        agent = self._build(monkeypatch)
        final = self._response_no_message()
        stream = self._make_stream_with_deltas(["Hello", " world", "!"], final)

        monkeypatch.setattr(agent.client.responses, "stream", lambda **kw: stream)

        response = agent._run_codex_stream(_codex_request_kwargs())

        # Synthetic message item must have been appended
        message_items = [i for i in response.output if getattr(i, "type", None) == "message"]
        assert len(message_items) == 1, "Expected exactly one synthetic message item"
        assert message_items[0].content[0].text == "Hello world!"

    def test_normalize_codex_response_extracts_text_from_synthesised_item(
        self, monkeypatch
    ):
        """
        End-to-end: after _run_codex_stream patches the response,
        _normalize_codex_response must extract non-empty content.
        """
        agent = self._build(monkeypatch)
        final = self._response_no_message()
        stream = self._make_stream_with_deltas(["Hi there!"], final)
        monkeypatch.setattr(agent.client.responses, "stream", lambda **kw: stream)

        response = agent._run_codex_stream(_codex_request_kwargs())
        assistant_msg, finish_reason = agent._normalize_codex_response(response)

        assert assistant_msg.content == "Hi there!"
        assert finish_reason == "stop"

    def test_no_synthesis_when_message_item_already_present(self, monkeypatch):
        """
        When the final response already has a message item, the collected deltas
        must NOT produce a duplicate — the real item wins.
        """
        agent = self._build(monkeypatch)
        final = self._response_with_message("Real answer.")
        stream = self._make_stream_with_deltas(["Real", " answer."], final)
        monkeypatch.setattr(agent.client.responses, "stream", lambda **kw: stream)

        response = agent._run_codex_stream(_codex_request_kwargs())

        message_items = [i for i in response.output if getattr(i, "type", None) == "message"]
        assert len(message_items) == 1, "Should not duplicate message items"
        assert message_items[0].content[0].text == "Real answer."

    def test_no_synthesis_when_no_deltas_collected(self, monkeypatch):
        """
        When no deltas were collected (tool-call-only turn), no synthesis
        should happen — output stays as-is.
        """
        agent = self._build(monkeypatch)
        tool_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="terminal",
                    arguments="{}",
                    status="completed",
                )
            ],
            status="completed",
            model="gpt-5.3-codex",
        )
        # Stream with no delta events
        stream = self._make_stream_with_deltas([], tool_response)
        monkeypatch.setattr(agent.client.responses, "stream", lambda **kw: stream)

        response = agent._run_codex_stream(_codex_request_kwargs())

        message_items = [i for i in response.output if getattr(i, "type", None) == "message"]
        assert len(message_items) == 0, "No message item should be synthesised for tool-call turns"

    def test_stream_delta_callback_still_fires_for_each_token(self, monkeypatch):
        """
        The stream_delta_callback must still receive every delta token even
        in the synthesis path.
        """
        agent = self._build(monkeypatch)
        final = self._response_no_message()
        tokens = ["Token1", " Token2", " Token3"]
        stream = self._make_stream_with_deltas(tokens, final)
        monkeypatch.setattr(agent.client.responses, "stream", lambda **kw: stream)

        received = []
        agent.stream_delta_callback = received.append

        agent._run_codex_stream(_codex_request_kwargs())

        assert received == tokens, f"Expected {tokens}, got {received}"

    def test_synthetic_item_has_correct_structure_for_normaliser(self, monkeypatch):
        """
        The synthesised item must have type='message', status='completed', and
        content as a list with one part of type='output_text'.
        """
        agent = self._build(monkeypatch)
        final = self._response_no_message()
        stream = self._make_stream_with_deltas(["check"], final)
        monkeypatch.setattr(agent.client.responses, "stream", lambda **kw: stream)

        response = agent._run_codex_stream(_codex_request_kwargs())

        synth = next(i for i in response.output if getattr(i, "type", None) == "message")
        assert synth.status == "completed"
        assert len(synth.content) == 1
        assert synth.content[0].type == "output_text"
        assert synth.content[0].text == "check"

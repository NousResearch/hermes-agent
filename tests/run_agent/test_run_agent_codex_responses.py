import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


@pytest.fixture(autouse=True)
def _no_codex_backoff(monkeypatch):
    """Short-circuit retry backoff so Codex retry tests don't block on real
    wall-clock waits (5s jittered_backoff base delay + tight time.sleep loop)."""
    import time as _time
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


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


def _build_copilot_agent(monkeypatch, *, model="gpt-5.4"):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model=model,
        provider="copilot",
        api_mode="codex_responses",
        base_url="https://api.githubcopilot.com",
        api_key="gh-token",
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
    """GPT-5.x models need codex_responses even on OpenRouter.

    OpenRouter rejects GPT-5 models on /v1/chat/completions with
    ``unsupported_api_for_model``.  The model-level check overrides
    the provider default.
    """
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
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "openrouter"


def test_copilot_acp_stays_on_chat_completions_for_gpt_5_models(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5.4-mini",
        base_url="acp://copilot",
        provider="copilot-acp",
        api_key="copilot-acp",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.provider == "copilot-acp"
    assert agent.api_mode == "chat_completions"


def test_copilot_gpt_5_mini_stays_on_chat_completions(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-mini",
        base_url="https://api.githubcopilot.com",
        provider="copilot",
        api_key="gh-token",
        api_mode="chat_completions",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.provider == "copilot"
    assert agent.api_mode == "chat_completions"


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


def test_build_api_kwargs_codex_clamps_minimal_effort(monkeypatch):
    """'minimal' reasoning effort is clamped to 'low' on the Responses API.

    GPT-5.4 supports none/low/medium/high/xhigh but NOT 'minimal'.
    Users may configure 'minimal' via OpenRouter conventions, so the Codex
    Responses path must clamp it to the nearest supported level.
    """
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
        reasoning_config={"enabled": True, "effort": "minimal"},
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None

    kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )

    assert kwargs["reasoning"]["effort"] == "low"


def test_build_api_kwargs_codex_preserves_supported_efforts(monkeypatch):
    """Effort levels natively supported by the Responses API pass through unchanged."""
    _patch_agent_bootstrap(monkeypatch)

    for effort in ("low", "medium", "high", "xhigh"):
        agent = run_agent.AIAgent(
            model="gpt-5-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="codex-token",
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
            reasoning_config={"enabled": True, "effort": effort},
        )
        agent._cleanup_task_resources = lambda task_id: None
        agent._persist_session = lambda messages, history=None: None
        agent._save_trajectory = lambda messages, user_message, completed: None
        agent._save_session_log = lambda messages: None

        kwargs = agent._build_api_kwargs(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert kwargs["reasoning"]["effort"] == effort, f"{effort} should pass through unchanged"


def test_build_api_kwargs_copilot_responses_omits_openai_only_fields(monkeypatch):
    agent = _build_copilot_agent(monkeypatch)
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["store"] is False
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
    assert kwargs["reasoning"] == {"effort": "medium"}
    assert "prompt_cache_key" not in kwargs
    assert "include" not in kwargs


def test_build_api_kwargs_copilot_responses_omits_reasoning_for_non_reasoning_model(monkeypatch):
    agent = _build_copilot_agent(monkeypatch, model="gpt-4.1")
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    assert "reasoning" not in kwargs
    assert "include" not in kwargs
    assert "prompt_cache_key" not in kwargs


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


def test_run_conversation_codex_empty_output_with_output_text(monkeypatch):
    """Regression: empty response.output + valid output_text should succeed,
    not trigger retry/fallback. The validation stage must defer to
    _normalize_codex_response which synthesizes output from output_text."""
    agent = _build_agent(monkeypatch)

    def _empty_output_response(api_kwargs):
        return SimpleNamespace(
            output=[],
            output_text="Hello from Codex",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
            status="completed",
            model="gpt-5-codex",
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", _empty_output_response)

    result = agent.run_conversation("Say hello")

    assert result["completed"] is True
    assert result["final_response"] == "Hello from Codex"


def test_run_conversation_codex_empty_output_no_output_text_retries(monkeypatch):
    """When both output and output_text are empty, validation should
    correctly mark the response as invalid and trigger retry."""
    agent = _build_agent(monkeypatch)
    calls = {"api": 0}

    def _fake_api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            return SimpleNamespace(
                output=[],
                output_text=None,
                usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
                status="completed",
                model="gpt-5-codex",
            )
        return _codex_message_response("Recovered")

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    result = agent.run_conversation("Say hello")

    assert calls["api"] >= 2
    assert result["completed"] is True
    assert result["final_response"] == "Recovered"


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


def test_run_conversation_copilot_refreshes_after_401_and_retries(monkeypatch):
    agent = _build_copilot_agent(monkeypatch)
    calls = {"api": 0, "refresh": 0}

    class _UnauthorizedError(RuntimeError):
        def __init__(self):
            super().__init__("Error code: 401 - unauthorized")
            self.status_code = 401

    def _fake_api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            raise _UnauthorizedError()
        return _codex_message_response("Recovered after copilot refresh")

    def _fake_refresh():
        calls["refresh"] += 1
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)
    monkeypatch.setattr(agent, "_try_refresh_copilot_client_credentials", _fake_refresh)

    result = agent.run_conversation("Say OK")

    assert calls["api"] == 2
    assert calls["refresh"] == 1
    assert result["completed"] is True
    assert result["final_response"] == "Recovered after copilot refresh"


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


def test_try_refresh_copilot_client_credentials_rebuilds_client(monkeypatch):
    agent = _build_copilot_agent(monkeypatch)
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
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_new_token", "GH_TOKEN"),
    )
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    agent.client = _ExistingClient()
    ok = agent._try_refresh_copilot_client_credentials()

    assert ok is True
    assert closed["value"] is True
    assert rebuilt["kwargs"]["api_key"] == "gho_new_token"
    assert rebuilt["kwargs"]["base_url"] == "https://api.githubcopilot.com"
    assert rebuilt["kwargs"]["default_headers"]["Copilot-Integration-Id"] == "vscode-chat"
    assert isinstance(agent.client, _RebuiltClient)


def test_try_refresh_copilot_client_credentials_rebuilds_even_if_token_unchanged(monkeypatch):
    agent = _build_copilot_agent(monkeypatch)
    rebuilt = {"count": 0}

    class _RebuiltClient:
        pass

    def _fake_openai(**kwargs):
        rebuilt["count"] += 1
        return _RebuiltClient()

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gh-token", "gh auth token"),
    )
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    ok = agent._try_refresh_copilot_client_credentials()

    assert ok is True
    assert rebuilt["count"] == 1


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
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    items = _chat_messages_to_responses_input(
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
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    items = _chat_messages_to_responses_input(
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
    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    preflight = _preflight_codex_api_kwargs(
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
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        _preflight_codex_api_kwargs(
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
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        _preflight_codex_api_kwargs(kwargs)


def test_preflight_codex_api_kwargs_allows_reasoning_and_temperature(monkeypatch):
    agent = _build_agent(monkeypatch)
    kwargs = _codex_request_kwargs()
    kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
    kwargs["include"] = ["reasoning.encrypted_content"]
    kwargs["temperature"] = 0.7
    kwargs["max_output_tokens"] = 4096

    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    result = _preflight_codex_api_kwargs(kwargs)
    assert result["reasoning"] == {"effort": "high", "summary": "auto"}
    assert result["include"] == ["reasoning.encrypted_content"]
    assert result["temperature"] == 0.7
    assert result["max_output_tokens"] == 4096


def test_preflight_codex_api_kwargs_allows_service_tier(monkeypatch):
    agent = _build_agent(monkeypatch)
    kwargs = _codex_request_kwargs()
    kwargs["service_tier"] = "priority"

    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    result = _preflight_codex_api_kwargs(kwargs)
    assert result["service_tier"] == "priority"


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
    from agent.codex_responses_adapter import _normalize_codex_response
    assistant_message, finish_reason = _normalize_codex_response(
        _codex_commentary_message_response("I'll inspect the repository first.")
    )

    assert finish_reason == "incomplete"
    assert "inspect the repository" in (assistant_message.content or "")


def test_interim_commentary_is_not_marked_already_streamed_without_callbacks(monkeypatch):
    agent = _build_agent(monkeypatch)
    observed = {}

    agent._fire_stream_delta("short version: yes")
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: observed.update(
        {"text": text, "already_streamed": already_streamed}
    )

    agent._emit_interim_assistant_message({"role": "assistant", "content": "short version: yes"})

    assert observed == {
        "text": "short version: yes",
        "already_streamed": False,
    }


def test_interim_commentary_is_not_marked_already_streamed_when_stream_callback_fails(monkeypatch):
    agent = _build_agent(monkeypatch)
    observed = {}

    def failing_callback(_text):
        raise RuntimeError("display failed")

    agent.stream_delta_callback = failing_callback
    agent._fire_stream_delta("short version: yes")
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: observed.update(
        {"text": text, "already_streamed": already_streamed}
    )

    agent._emit_interim_assistant_message({"role": "assistant", "content": "short version: yes"})

    assert observed == {
        "text": "short version: yes",
        "already_streamed": False,
    }


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


def test_dump_api_request_debug_uses_responses_url(monkeypatch, tmp_path):
    """Debug dumps should show /responses URL when in codex_responses mode."""
    import json
    agent = _build_agent(monkeypatch)
    agent.base_url = "http://127.0.0.1:9208/v1"
    agent.logs_dir = tmp_path

    dump_file = agent._dump_api_request_debug(_codex_request_kwargs(), reason="preflight")

    payload = json.loads(dump_file.read_text())
    assert payload["request"]["url"] == "http://127.0.0.1:9208/v1/responses"


def test_dump_api_request_debug_uses_chat_completions_url(monkeypatch, tmp_path):
    """Debug dumps should show /chat/completions URL for chat_completions mode."""
    import json
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-4o",
        base_url="http://127.0.0.1:9208/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.logs_dir = tmp_path

    dump_file = agent._dump_api_request_debug(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        reason="preflight",
    )

    payload = json.loads(dump_file.read_text())
    assert payload["request"]["url"] == "http://127.0.0.1:9208/v1/chat/completions"


# --- Reasoning-only response tests (fix for empty content retry loop) ---


def _codex_reasoning_only_response(*, encrypted_content="enc_abc123", summary_text="Thinking..."):
    """Codex response containing only reasoning items — no message text, no tool calls."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_001",
                encrypted_content=encrypted_content,
                summary=[SimpleNamespace(type="summary_text", text=summary_text)],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=50, output_tokens=100, total_tokens=150),
        status="completed",
        model="gpt-5-codex",
    )


def test_normalize_codex_response_marks_reasoning_only_as_incomplete(monkeypatch):
    """A response with only reasoning items and no content should be 'incomplete', not 'stop'.

    Without this fix, reasoning-only responses get finish_reason='stop' which
    sends them into the empty-content retry loop (3 retries then failure).
    """
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response
    assistant_message, finish_reason = _normalize_codex_response(
        _codex_reasoning_only_response()
    )

    assert finish_reason == "incomplete"
    assert assistant_message.content == ""
    assert assistant_message.codex_reasoning_items is not None
    assert len(assistant_message.codex_reasoning_items) == 1
    assert assistant_message.codex_reasoning_items[0]["encrypted_content"] == "enc_abc123"


def test_normalize_codex_response_reasoning_with_content_is_stop(monkeypatch):
    """If a response has both reasoning and message content, it should still be 'stop'."""
    agent = _build_agent(monkeypatch)
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_001",
                encrypted_content="enc_xyz",
                summary=[SimpleNamespace(type="summary_text", text="Thinking...")],
                status="completed",
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Here is the answer.")],
                status="completed",
            ),
        ],
        usage=SimpleNamespace(input_tokens=50, output_tokens=100, total_tokens=150),
        status="completed",
        model="gpt-5-codex",
    )
    from agent.codex_responses_adapter import _normalize_codex_response
    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "stop"
    assert "Here is the answer" in assistant_message.content


def test_run_conversation_codex_continues_after_reasoning_only_response(monkeypatch):
    """End-to-end: reasoning-only → final message should succeed, not hit retry loop."""
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_reasoning_only_response(),
        _codex_message_response("The final answer is 42."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("what is the answer?")

    assert result["completed"] is True
    assert result["final_response"] == "The final answer is 42."
    # The reasoning-only turn should be in messages as an incomplete interim
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and msg.get("codex_reasoning_items") is not None
        for msg in result["messages"]
    )


def test_run_conversation_codex_preserves_encrypted_reasoning_in_interim(monkeypatch):
    """Encrypted codex_reasoning_items must be preserved in interim messages
    even when there is no visible reasoning text or content."""
    agent = _build_agent(monkeypatch)
    # Response with encrypted reasoning but no human-readable summary
    reasoning_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_002",
                encrypted_content="enc_opaque_blob",
                summary=[],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=50, output_tokens=100, total_tokens=150),
        status="completed",
        model="gpt-5-codex",
    )
    responses = [
        reasoning_response,
        _codex_message_response("Done thinking."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("think hard")

    assert result["completed"] is True
    assert result["final_response"] == "Done thinking."
    # The interim message must have codex_reasoning_items preserved
    interim_msgs = [
        msg for msg in result["messages"]
        if msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
    ]
    assert len(interim_msgs) >= 1
    assert interim_msgs[0].get("codex_reasoning_items") is not None
    assert interim_msgs[0]["codex_reasoning_items"][0]["encrypted_content"] == "enc_opaque_blob"


def test_chat_messages_to_responses_input_reasoning_only_has_following_item(monkeypatch):
    """When converting a reasoning-only interim message to Responses API input,
    the reasoning items must be followed by an assistant message (even if empty)
    to satisfy the API's 'required following item' constraint."""
    agent = _build_agent(monkeypatch)
    messages = [
        {"role": "user", "content": "think hard"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": None,
            "finish_reason": "incomplete",
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_001", "encrypted_content": "enc_abc", "summary": []},
            ],
        },
    ]
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    items = _chat_messages_to_responses_input(messages)

    # Find the reasoning item
    reasoning_indices = [i for i, it in enumerate(items) if it.get("type") == "reasoning"]
    assert len(reasoning_indices) == 1
    ri_idx = reasoning_indices[0]

    # There must be a following item after the reasoning
    assert ri_idx < len(items) - 1, "Reasoning item must not be the last item (missing_following_item)"
    following = items[ri_idx + 1]
    assert following.get("role") == "assistant"


def test_duplicate_detection_distinguishes_different_codex_reasoning(monkeypatch):
    """Two consecutive reasoning-only responses with different encrypted content
    must NOT be treated as duplicates."""
    agent = _build_agent(monkeypatch)
    responses = [
        # First reasoning-only response
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning", id="rs_001",
                    encrypted_content="enc_first", summary=[], status="completed",
                )
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=100, total_tokens=150),
            status="completed", model="gpt-5-codex",
        ),
        # Second reasoning-only response (different encrypted content)
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning", id="rs_002",
                    encrypted_content="enc_second", summary=[], status="completed",
                )
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=100, total_tokens=150),
            status="completed", model="gpt-5-codex",
        ),
        _codex_message_response("Final answer after thinking."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("think very hard")

    assert result["completed"] is True
    assert result["final_response"] == "Final answer after thinking."
    # Both reasoning-only interim messages should be in history (not collapsed)
    interim_msgs = [
        msg for msg in result["messages"]
        if msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
    ]
    assert len(interim_msgs) == 2
    encrypted_contents = [
        msg["codex_reasoning_items"][0]["encrypted_content"]
        for msg in interim_msgs
    ]
    assert "enc_first" in encrypted_contents
    assert "enc_second" in encrypted_contents


def test_chat_messages_to_responses_input_deduplicates_reasoning_ids(monkeypatch):
    """Duplicate reasoning item IDs across multi-turn incomplete responses
    must be deduplicated so the Responses API doesn't reject with HTTP 400."""
    agent = _build_agent(monkeypatch)
    messages = [
        {"role": "user", "content": "think hard"},
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_aaa", "encrypted_content": "enc_1"},
                {"type": "reasoning", "id": "rs_bbb", "encrypted_content": "enc_2"},
            ],
        },
        {
            "role": "assistant",
            "content": "partial answer",
            "codex_reasoning_items": [
                # rs_aaa is duplicated from the previous turn
                {"type": "reasoning", "id": "rs_aaa", "encrypted_content": "enc_1"},
                {"type": "reasoning", "id": "rs_ccc", "encrypted_content": "enc_3"},
            ],
        },
    ]
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    items = _chat_messages_to_responses_input(messages)

    reasoning_items = [it for it in items if it.get("type") == "reasoning"]
    # Dedup: rs_aaa appears in both turns but should only be emitted once.
    # 3 unique items total: enc_1 (from rs_aaa), enc_2 (rs_bbb), enc_3 (rs_ccc).
    assert len(reasoning_items) == 3
    encrypted = [it["encrypted_content"] for it in reasoning_items]
    assert encrypted.count("enc_1") == 1
    assert "enc_2" in encrypted
    assert "enc_3" in encrypted
    # IDs must be stripped — with store=False the API 404s on id lookups.
    for it in reasoning_items:
        assert "id" not in it


def test_preflight_codex_input_deduplicates_reasoning_ids(monkeypatch):
    """_preflight_codex_input_items should also deduplicate reasoning items by ID."""
    agent = _build_agent(monkeypatch)
    raw_input = [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"type": "reasoning", "id": "rs_xyz", "encrypted_content": "enc_a"},
        {"role": "assistant", "content": "ok"},
        {"type": "reasoning", "id": "rs_xyz", "encrypted_content": "enc_a"},
        {"type": "reasoning", "id": "rs_zzz", "encrypted_content": "enc_b"},
        {"role": "assistant", "content": "done"},
    ]
    from agent.codex_responses_adapter import _preflight_codex_input_items
    normalized = _preflight_codex_input_items(raw_input)

    reasoning_items = [it for it in normalized if it.get("type") == "reasoning"]
    # rs_xyz duplicate should be collapsed to one item; rs_zzz kept.
    assert len(reasoning_items) == 2
    encrypted = [it["encrypted_content"] for it in reasoning_items]
    assert encrypted.count("enc_a") == 1
    assert "enc_b" in encrypted
    # IDs must be stripped — with store=False the API 404s on id lookups.
    for it in reasoning_items:
        assert "id" not in it

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


def _codex_final_answer_with_top_level_incomplete_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="final_answer",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="incomplete",
        model="gpt-5.4",
    )


class _FakeCreateStream:
    """Iterable-only fake for ``responses.create(stream=True)`` outputs.

    The event-driven Codex path expects an iterable that yields SSE events;
    tests use this to drive it through the same code paths the wire does.
    """

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
    # ``timeout`` is now wired from ``_resolved_api_call_timeout`` (default 1800s)
    # so per-provider ``request_timeout_seconds`` actually reaches the SDK.
    assert isinstance(kwargs.get("timeout"), float)
    assert kwargs["timeout"] > 0
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


# ---------------------------------------------------------------------------
# #27907: xAI tool-schema sanitization must NOT mutate ``agent.tools`` in place
#
# ``strip_slash_enum`` and ``strip_pattern_and_format`` are documented to
# mutate their input in place ("Callers that need to preserve the original
# should deep-copy first" — see ``tools/schema_sanitizer.py``).  Until this
# fix, ``chat_completion_helpers.build_api_kwargs`` and ``auxiliary_client``
# passed ``agent.tools`` straight through to the sanitizers.  The first xAI
# request would permanently strip slash-containing enum constraints and the
# ``pattern``/``format`` keywords from the per-agent tool registry — any
# subsequent non-xAI call from the same agent (auxiliary task routed to
# Anthropic, OpenRouter fallback, mid-session model switch) saw the
# already-stripped schema.
#
# Fix: deepcopy ``tools_for_api`` before handing it to the sanitizers.
# ---------------------------------------------------------------------------


def _build_xai_agent_with_slash_enum_tool(monkeypatch):
    """Build an xAI agent whose tool registry has a slash-containing enum.

    Mirrors the Brave Search MCP shape that originally triggered #27907.
    """

    def _fake_get_tool_definitions(**_kwargs):
        return [
            {
                "type": "function",
                "function": {
                    "name": "brave_like",
                    "description": "Tool with slash-containing enum + pattern/format",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "accept": {
                                "type": "string",
                                "enum": ["application/json", "*/*"],
                            },
                            "match": {
                                "type": "string",
                                "pattern": "^[a-z]+$",
                                "format": "regex",
                            },
                        },
                    },
                },
            }
        ]

    monkeypatch.setattr(run_agent, "get_tool_definitions", _fake_get_tool_definitions)
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})

    agent = run_agent.AIAgent(
        model="grok-4.3",
        provider="xai-oauth",
        api_mode="codex_responses",
        base_url="https://api.x.ai/v1",
        api_key="xai-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def test_build_api_kwargs_xai_strips_slash_enum_from_outgoing_request(monkeypatch):
    """The xAI request sent to the API must NOT contain slash-enum values."""
    agent = _build_xai_agent_with_slash_enum_tool(monkeypatch)
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    # ``tools`` comes back in Responses format from the codex transport;
    # find the parameters dict for our function regardless of shape.
    out_tool = kwargs["tools"][0]
    params = out_tool["parameters"]
    assert "enum" not in params["properties"]["accept"], (
        "outgoing xAI request must not carry slash-containing enums — "
        "xAI would 400 with 'Invalid arguments passed to the model'"
    )
    # pattern/format must also be stripped (existing #27197 contract).
    assert "pattern" not in params["properties"]["match"]
    assert "format" not in params["properties"]["match"]


def test_build_api_kwargs_xai_does_not_mutate_agent_tools(monkeypatch):
    """Headline #27907 regression: ``agent.tools`` must survive intact.

    Pre-fix the sanitizers mutated ``agent.tools`` in place, so a subsequent
    non-xAI call from the same agent saw an already-stripped schema —
    silent constraint loss with no way for the user to notice from their
    config.
    """
    agent = _build_xai_agent_with_slash_enum_tool(monkeypatch)

    # Snapshot the schema before the request.
    accept_before = agent.tools[0]["function"]["parameters"]["properties"]["accept"]
    match_before = agent.tools[0]["function"]["parameters"]["properties"]["match"]
    assert accept_before["enum"] == ["application/json", "*/*"]
    assert match_before.get("pattern") == "^[a-z]+$"
    assert match_before.get("format") == "regex"

    # Build the API kwargs (which runs the sanitizers).
    agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    # The agent's tool registry must be UNCHANGED.
    accept_after = agent.tools[0]["function"]["parameters"]["properties"]["accept"]
    match_after = agent.tools[0]["function"]["parameters"]["properties"]["match"]
    assert accept_after.get("enum") == ["application/json", "*/*"], (
        "agent.tools mutated — slash-containing enum was stripped from the "
        "shared per-agent registry, will leak to non-xAI calls"
    )
    assert match_after.get("pattern") == "^[a-z]+$", (
        "agent.tools mutated — pattern stripped from shared registry"
    )
    assert match_after.get("format") == "regex", (
        "agent.tools mutated — format stripped from shared registry"
    )


def test_build_api_kwargs_xai_is_idempotent_across_repeated_calls(monkeypatch):
    """Multiple xAI requests must each produce the same sanitized output
    AND must not progressively erode the source schema."""
    agent = _build_xai_agent_with_slash_enum_tool(monkeypatch)

    kwargs1 = agent._build_api_kwargs([{"role": "user", "content": "first"}])
    kwargs2 = agent._build_api_kwargs([{"role": "user", "content": "second"}])
    kwargs3 = agent._build_api_kwargs([{"role": "user", "content": "third"}])

    for k in (kwargs1, kwargs2, kwargs3):
        params = k["tools"][0]["parameters"]
        assert "enum" not in params["properties"]["accept"]
        assert "pattern" not in params["properties"]["match"]
        assert "format" not in params["properties"]["match"]

    # Source schema still untouched after three rounds.
    assert agent.tools[0]["function"]["parameters"]["properties"]["accept"].get(
        "enum"
    ) == ["application/json", "*/*"]


def test_run_codex_stream_returns_collected_items_when_stream_ends_without_terminal(monkeypatch):
    """The event-driven path tolerates streams that end without a terminal frame.

    Previously the SDK's ``responses.stream(...)`` helper raised
    ``RuntimeError("Didn't receive a `response.completed` event.")`` which the
    primary path caught and retried/fell back through. The new
    ``responses.create(stream=True)`` path consumes events directly and just
    returns whatever it collected — no retry, no separate fallback path.
    """
    agent = _build_agent(monkeypatch)
    output_item = SimpleNamespace(
        type="message",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="no terminal frame")],
    )
    calls = {"create": 0}

    def _fake_create(**kwargs):
        calls["create"] += 1
        assert kwargs.get("stream") is True
        return _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=output_item),
            # stream ends without a response.completed/incomplete/failed frame
        ])

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert calls["create"] == 1
    assert response.status == "completed"
    assert response.output == [output_item]


def test_run_codex_stream_surfaces_failed_status_in_final_response(monkeypatch):
    """A ``response.failed`` terminal event is reflected on the returned object."""
    agent = _build_agent(monkeypatch)
    error_payload = {"message": "model overloaded", "code": "overloaded"}
    failed_event = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            status="failed",
            error=error_payload,
            id="resp_failed_1",
            usage=None,
        ),
    )

    def _fake_create(**kwargs):
        return _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            failed_event,
        ])

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert response.status == "failed"
    assert response.error == error_payload


def test_run_codex_stream_parses_create_stream_events(monkeypatch):
    """The primary path consumes ``responses.create(stream=True)`` events directly."""
    agent = _build_agent(monkeypatch)
    calls = {"create": 0}
    create_stream = _FakeCreateStream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.in_progress"),
            SimpleNamespace(type="response.completed", response=_codex_message_response("streamed create ok")),
        ]
    )

    def _fake_create(**kwargs):
        calls["create"] += 1
        assert kwargs.get("stream") is True
        return create_stream

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert calls["create"] == 1
    assert create_stream.closed is True
    # The wire's response.completed.response.output is a list with the message item,
    # but the event-driven path reconstructs from response.output_item.done.
    # _codex_message_response returns a SimpleNamespace whose .output is a list of
    # items — we don't read those directly, we read the items via output_item.done,
    # but this fixture doesn't emit output_item.done. So the consumer assembles a
    # message from streamed text deltas if present, or returns the items it has.
    # For backward compatibility with the helper that builds _codex_message_response,
    # we just assert status is completed and id propagated.
    assert response.status == "completed"


def test_run_codex_stream_ignores_completed_response_with_null_output(monkeypatch):
    """Regression: Codex may send response.completed.response.output=null.

    The SDK's high-level ``responses.stream(...)`` helper used to reconstruct
    the final Response from that terminal field and raised ``TypeError:
    'NoneType' object is not iterable``. The Hermes runtime consumes raw
    ``response.output_item.done`` events instead, so a null terminal ``output``
    must not affect the returned assistant/function-call items.
    """
    agent = _build_agent(monkeypatch)
    output_item = SimpleNamespace(
        type="message",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="terminal output was null")],
    )
    create_stream = _FakeCreateStream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=output_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="resp_null_output",
                    status="completed",
                    output=None,
                    usage=SimpleNamespace(input_tokens=7, output_tokens=4, total_tokens=11),
                ),
            ),
        ]
    )

    def _fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return create_stream

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )

    response = agent._run_codex_stream(_codex_request_kwargs())
    assert response is not None
    assert create_stream.closed is True
    assert response.id == "resp_null_output"
    assert response.status == "completed"
    assert response.output == [output_item]
    assert response.usage.total_tokens == 11


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


def _build_xai_oauth_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="grok-4.3",
        provider="xai-oauth",
        api_mode="codex_responses",
        base_url="https://api.x.ai/v1",
        api_key="xai-oauth-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def test_build_api_kwargs_xai_oauth_sends_cache_key_via_extra_body(monkeypatch):
    """xai-oauth + codex_responses must route prompt caching via the
    ``prompt_cache_key`` body field on /v1/responses (xAI's documented
    Responses-API cache key — see docs.x.ai prompt-caching/maximizing-
    cache-hits).

    We pass it through ``extra_body`` rather than as a top-level kwarg so
    the body field is serialized into JSON regardless of whether the
    installed openai SDK build still accepts ``prompt_cache_key`` on
    ``Responses.stream()``. Older or trimmed SDK builds drop it from the
    signature and would otherwise raise ``TypeError`` before the request
    reaches api.x.ai. The ``x-grok-conv-id`` header is retained as a
    belt-and-braces fallback for clients/proxies that route on headers."""
    agent = _build_xai_oauth_agent(monkeypatch)
    kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )

    assert kwargs.get("model") == "grok-4.3"
    # Top-level kwarg must NOT be set — that's the openai SDK
    # incompatibility this whole indirection exists to dodge.
    assert "prompt_cache_key" not in kwargs
    extra_body = kwargs.get("extra_body") or {}
    assert extra_body.get("prompt_cache_key"), (
        "xAI prompt-cache routing must travel via extra_body.prompt_cache_key "
        "for /v1/responses — body field is the documented surface."
    )
    headers = kwargs.get("extra_headers") or {}
    assert "x-grok-conv-id" in headers, (
        "x-grok-conv-id header kept as belt-and-braces fallback for clients "
        "that route on headers."
    )


def test_run_conversation_xai_oauth_refreshes_after_401_and_retries(monkeypatch):
    """xai-oauth speaks the Responses API just like codex.  When the access
    token is rejected mid-call (401), the same proactive refresh-and-retry
    handler that fires for openai-codex must also fire for xai-oauth — the
    bug it caught: the gating condition checked only ``provider == "openai-codex"``,
    so xai-oauth 401s leaked straight to non-retryable abort path with no
    chance to swap in a freshly refreshed access token."""
    agent = _build_xai_oauth_agent(monkeypatch)
    calls = {"api": 0, "refresh": 0}

    class _UnauthorizedError(RuntimeError):
        def __init__(self):
            super().__init__("Error code: 401 - unauthorized")
            self.status_code = 401

    def _fake_api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            raise _UnauthorizedError()
        return _codex_message_response("Recovered after xAI refresh")

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
    assert result["final_response"] == "Recovered after xAI refresh"


def test_try_refresh_codex_client_credentials_handles_xai_oauth(monkeypatch):
    """``_try_refresh_codex_client_credentials`` must rebuild the OpenAI
    client with freshly resolved xAI OAuth credentials when the active
    provider is xai-oauth.  The function name is shared between codex and
    xai-oauth (both speak codex_responses) — covering both cases prevents
    silent regressions where the function gets gated to a single provider."""
    agent = _build_xai_oauth_agent(monkeypatch)
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

    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        # The pre-refresh guard reads the singleton with refresh_if_expiring=False
        # to verify that the agent's active key still matches; the actual
        # refresh later passes force_refresh=True.  Both calls must succeed.
        return {
            "api_key": "fresh-xai-token" if force_refresh else agent.api_key,
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        _fake_resolve,
    )
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    agent.client = _ExistingClient()
    ok = agent._try_refresh_codex_client_credentials(force=True)

    assert ok is True
    assert closed["value"] is True
    assert rebuilt["kwargs"]["api_key"] == "fresh-xai-token"
    assert rebuilt["kwargs"]["base_url"] == "https://api.x.ai/v1"
    assert isinstance(agent.client, _RebuiltClient)
    assert agent.api_key == "fresh-xai-token"


def test_try_refresh_codex_client_credentials_skips_xai_oauth_when_singleton_differs(monkeypatch):
    """An xai-oauth agent constructed with a non-singleton credential
    (e.g. a manual pool entry whose tokens belong to a different account
    than the loopback_pkce singleton, or an explicit ``api_key=`` arg)
    MUST NOT silently adopt the singleton's tokens on a 401 reactive
    refresh.  Otherwise a 401 mid-conversation would re-route the rest
    of the conversation onto a different account, with no user feedback.

    The credential pool's reactive recovery is the right channel for
    pool-managed credentials; this fallback path is for the singleton-
    only case and must short-circuit when the active key differs."""
    agent = _build_xai_oauth_agent(monkeypatch)
    # Agent is using "xai-oauth-token" (per the builder); singleton holds
    # a *different* account's token.  No force_refresh should fire.
    refresh_calls = {"count": 0}

    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        if force_refresh:
            refresh_calls["count"] += 1
            return {
                "api_key": "singleton-account-token",
                "base_url": "https://api.x.ai/v1",
            }
        # The pre-refresh guard read — return the singleton's view of the
        # singleton's token, which is NOT what the agent is currently using.
        return {
            "api_key": "singleton-account-token",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        _fake_resolve,
    )

    pre_refresh_key = agent.api_key
    ok = agent._try_refresh_codex_client_credentials(force=True)

    assert ok is False, (
        "must not refresh when the active credential isn't the singleton; "
        "otherwise the conversation silently swaps accounts mid-flight."
    )
    assert refresh_calls["count"] == 0, (
        "force_refresh must not run — that would mutate the singleton's "
        "tokens on disk and consume its single-use refresh_token for an "
        "agent that wasn't even using the singleton."
    )
    assert agent.api_key == pre_refresh_key


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

    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        # Pre-refresh guard reads the singleton (refresh_if_expiring=False).
        # It must report the agent's current api_key so the equality check
        # passes; only then does the actual force_refresh run.
        return {
            "api_key": "new-codex-token" if force_refresh else agent.api_key,
            "base_url": "https://chatgpt.com/backend-api/codex",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        _fake_resolve,
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


def test_preflight_codex_api_kwargs_preserves_positive_timeout(monkeypatch):
    """Positive numeric timeouts survive preflight so the SDK honors them."""
    agent = _build_agent(monkeypatch)
    kwargs = _codex_request_kwargs()
    kwargs["timeout"] = 600.0

    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    result = _preflight_codex_api_kwargs(kwargs)
    assert result["timeout"] == 600.0


def test_preflight_codex_api_kwargs_drops_invalid_timeout(monkeypatch):
    """Zero, negative, inf, and booleans are all dropped — not passed to SDK."""
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _preflight_codex_api_kwargs

    for bad in (0, -1, float("inf"), True, False, "300", None):
        kwargs = _codex_request_kwargs()
        kwargs["timeout"] = bad
        result = _preflight_codex_api_kwargs(kwargs)
        assert "timeout" not in result, f"timeout={bad!r} should be dropped"


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


def test_normalize_codex_response_final_answer_overrides_top_level_incomplete(monkeypatch):
    from agent.codex_responses_adapter import _normalize_codex_response

    assistant_message, finish_reason = _normalize_codex_response(
        _codex_final_answer_with_top_level_incomplete_response(
            "Briefly:\n\n- I'm Ramsay, your assistant."
        )
    )

    assert finish_reason == "stop"
    assert "Ramsay" in (assistant_message.content or "")


def test_normalize_codex_response_top_level_incomplete_without_final_answer_stays_incomplete(monkeypatch):
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="Partial...")],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="incomplete",
        model="gpt-5.4",
    )

    _, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"


@pytest.mark.parametrize("top_level_status", ["queued", "in_progress"])
def test_normalize_codex_response_final_answer_does_not_override_streaming_status(
    monkeypatch, top_level_status
):
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="final_answer",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="Interim answer.")],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status=top_level_status,
        model="gpt-5.4",
    )

    _, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"


def test_normalize_codex_response_final_answer_does_not_override_per_item_in_progress(monkeypatch):
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="final_answer",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="Partial final.")],
            ),
            SimpleNamespace(
                type="message",
                status="in_progress",
                content=[SimpleNamespace(type="output_text", text="")],
            ),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5.4",
    )

    _, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"


def test_normalize_codex_response_preserves_message_status_for_replay(monkeypatch):
    """Incomplete Codex output messages must not be replayed as completed."""
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                id="msg_partial",
                phase="commentary",
                status="in_progress",
                content=[SimpleNamespace(type="output_text", text="Still working...")],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="in_progress",
        model="gpt-5-codex",
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"
    assert assistant_message.codex_message_items[0]["id"] == "msg_partial"
    assert assistant_message.codex_message_items[0]["status"] == "in_progress"


def test_normalize_codex_response_detects_leaked_tool_call_text(monkeypatch):
    """Harmony-style `to=functions.foo` leaked into assistant content with no
    structured function_call items must be treated as incomplete so the
    continuation path can re-elicit a proper tool call. This is the
    Taiwan-embassy-email (Discord bug report) failure mode: child agent
    produces a confident-looking summary, tool_trace is empty because no
    tools actually ran, parent can't audit the claim.
    """
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response

    leaked_content = (
        "I'll check the official page directly.\n"
        "to=functions.exec_command {\"cmd\": \"curl https://example.test\"}\n"
        "assistant to=functions.exec_command {\"stdout\": \"mailto:foo@example.test\"}\n"
        "Extracted: foo@example.test"
    )
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=leaked_content)],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5.4",
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"
    # Content is scrubbed so the parent never surfaces the leaked text as a
    # summary. tool_calls stays empty because no structured function_call
    # item existed.
    assert (assistant_message.content or "") == ""
    assert assistant_message.tool_calls == []


def test_normalize_codex_response_ignores_tool_call_text_when_real_tool_call_present(monkeypatch):
    """If the model emitted BOTH a structured function_call AND some text that
    happens to contain `to=functions.*` (unlikely but possible), trust the
    structured call — don't wipe content that came alongside a real tool use.
    """
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(
                    type="output_text",
                    text="Running the command via to=functions.exec_command now.",
                )],
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments="{}",
            ),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5.4",
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "tool_calls"
    assert assistant_message.tool_calls  # real call preserved
    assert "Running the command" in (assistant_message.content or "")


def test_normalize_codex_response_no_leak_passes_through(monkeypatch):
    """Sanity: normal assistant content that doesn't contain the leak pattern
    is returned verbatim with finish_reason=stop."""
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(
                    type="output_text",
                    text="Here is the answer with no leak.",
                )],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5.4",
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "stop"
    assert assistant_message.content == "Here is the answer with no leak."
    assert assistant_message.tool_calls == []


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


def test_interim_commentary_preserves_assistant_content(monkeypatch):
    """Interim commentary must not silently mutate assistant text containing
    literal <memory-context> markers — that's legitimate model output (docs,
    code).  Streaming-path leak prevention happens delta-by-delta upstream."""
    agent = _build_agent(monkeypatch)
    observed = {}
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: observed.update(
        {"text": text, "already_streamed": already_streamed}
    )

    content = (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n"
        "## Honcho Context\n"
        "stale memory\n"
        "</memory-context>\n\n"
        "I'll inspect the repo structure first."
    )

    agent._emit_interim_assistant_message({"role": "assistant", "content": content})

    assert "<memory-context>" in observed["text"]
    assert "I'll inspect the repo structure first." in observed["text"]


def test_stream_delta_strips_leaked_memory_context(monkeypatch):
    agent = _build_agent(monkeypatch)
    observed = []
    agent.stream_delta_callback = observed.append

    leaked = (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n"
        "## Honcho Context\n"
        "stale memory\n"
        "</memory-context>\n\n"
        "Visible answer"
    )

    agent._fire_stream_delta(leaked)

    assert observed == ["Visible answer"]


def test_stream_delta_strips_leaked_memory_context_across_chunks(monkeypatch):
    """Regression for #5719 — the real streaming case.

    Providers typically emit 1-80 char chunks, so the memory-context open
    tag, system-note line, payload, and close tag each arrive in separate
    deltas.  The per-delta sanitize_context() regex cannot survive that
    — only a stateful scrubber can.  None of the payload, system-note
    text, or "## Honcho Context" header may reach the delta callback.
    """
    agent = _build_agent(monkeypatch)
    observed = []
    agent.stream_delta_callback = observed.append

    deltas = [
        "<memory-context>\n[System note: The following",
        " is recalled memory context, NOT new user input. ",
        "Treat as informational background data.]\n\n",
        "## Honcho Context\n",
        "stale memory about eri\n",
        "</memory-context>\n\n",
        "Visible answer",
    ]
    for d in deltas:
        agent._fire_stream_delta(d)

    combined = "".join(observed)
    assert "Visible answer" in combined
    # None of the leaked payload may surface.
    assert "System note" not in combined
    assert "Honcho Context" not in combined
    assert "stale memory" not in combined
    assert "<memory-context>" not in combined
    assert "</memory-context>" not in combined


def test_stream_delta_scrubber_resets_between_turns(monkeypatch):
    """An unterminated span from a prior turn must not taint the next turn."""
    agent = _build_agent(monkeypatch)

    # Simulate a hung span carried over — directly populate the scrubber.
    agent._stream_context_scrubber.feed("pre <memory-context>leaked")

    # Normally run_conversation() resets the scrubber at turn start.
    agent._stream_context_scrubber.reset()

    observed = []
    agent.stream_delta_callback = observed.append
    agent._fire_stream_delta("clean new turn text")
    assert "".join(observed) == "clean new turn text"


def test_stream_delta_preserves_mid_stream_leading_newlines(monkeypatch):
    """Mid-stream leading newlines must survive — they are legitimate
    markdown (lists, code fences, paragraph breaks).  Stripping them
    based on chunk boundaries silently breaks formatting.

    Only the very first delta of a stream gets leading-newlines stripped
    (so stale provider preamble doesn't leak); after that, deltas are
    emitted verbatim.
    """
    agent = _build_agent(monkeypatch)
    observed = []
    agent.stream_delta_callback = observed.append

    # First delta delivers text — strips its own leading "\n" once.
    agent._fire_stream_delta("\nHere is a list:")
    # Second delta starts with "\n- item" — must NOT be stripped.
    agent._fire_stream_delta("\n- first")
    agent._fire_stream_delta("\n- second")

    combined = "".join(observed)
    assert combined == "Here is a list:\n- first\n- second"


def test_stream_delta_preserves_code_fence_newlines(monkeypatch):
    """Code blocks span multiple deltas.  A "\\n```python\\n" boundary
    is the canonical case where stripping leading newlines corrupts output."""
    agent = _build_agent(monkeypatch)
    observed = []
    agent.stream_delta_callback = observed.append

    agent._fire_stream_delta("Here is the code:")
    agent._fire_stream_delta("\n```python\n")
    agent._fire_stream_delta("print('hi')\n")
    agent._fire_stream_delta("```\n")

    combined = "".join(observed)
    assert "```python\n" in combined
    assert combined.startswith("Here is the code:\n```python\n")


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


def test_dump_api_request_debug_normalizes_responses_alias(monkeypatch, tmp_path):
    """Explicit ``api_mode='responses'`` should still hit the Responses API."""
    import json

    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="deepseek-v4-pro",
        provider="custom",
        api_mode="responses",
        base_url="http://127.0.0.1:9208/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
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


def test_dump_api_request_debug_redacts_request_and_error_secrets(monkeypatch, tmp_path, capsys):
    """Request debug dumps should redact secrets before disk/stdout output."""
    import json

    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setenv("HERMES_DUMP_REQUEST_STDOUT", "1")
    agent = run_agent.AIAgent(
        model="gpt-4o",
        base_url="http://127.0.0.1:9208/v1",
        api_key="sk-ant-providersecret1234567890",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.logs_dir = tmp_path

    notion_token = "ntn_abc123def456ghi789jkl"
    error_secret = "sk-ant-errorsecret1234567890"
    response_secret = "sk-ant-responsesecret1234567890"
    response = SimpleNamespace(status_code=400, text=f"provider echoed {response_secret}")

    class ProviderError(RuntimeError):
        body: object
        response: object

    error = ProviderError(f"bad token {error_secret}")
    error.body = {"message": f"bad token {error_secret}"}
    error.response = response

    dump_file = agent._dump_api_request_debug(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": f"use {notion_token}"}],
            "metadata": {"NOTION_API_KEY": notion_token},
        },
        reason="provider_error",
        error=error,
    )

    assert dump_file is not None
    dumped_text = dump_file.read_text()
    stdout_text = capsys.readouterr().out
    for raw in (notion_token, error_secret, response_secret, "providersecret1234567890"):
        assert raw not in dumped_text
        assert raw not in stdout_text

    payload = json.loads(dumped_text)
    assert payload["request"]["headers"]["Authorization"].startswith("Bearer sk-ant-p...")
    assert "***" in dumped_text or "..." in dumped_text


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


def test_codex_message_item_status_survives_conversion_and_preflight(monkeypatch):
    """Stored Codex assistant message statuses must survive replay normalization."""
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import (
        _chat_messages_to_responses_input,
        _preflight_codex_input_items,
    )

    items = _chat_messages_to_responses_input([
        {
            "role": "assistant",
            "content": "partial",
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "id": "msg_incomplete",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "partial"}],
                }
            ],
        }
    ])
    replay_item = next(item for item in items if item.get("type") == "message")
    assert replay_item["status"] == "incomplete"

    normalized = _preflight_codex_input_items([
        {
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "working"}],
        }
    ])
    assert normalized[0]["status"] == "in_progress"


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


def test_duplicate_detection_distinguishes_different_codex_message_items(monkeypatch):
    """Incomplete turns with new message ids/phases/statuses must not be collapsed."""
    agent = _build_agent(monkeypatch)
    responses = [
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    id="msg_first",
                    phase="commentary",
                    status="in_progress",
                    content=[SimpleNamespace(type="output_text", text="Still working...")],
                )
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=10, total_tokens=60),
            status="in_progress",
            model="gpt-5-codex",
        ),
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    id="msg_second",
                    phase="commentary",
                    status="in_progress",
                    content=[SimpleNamespace(type="output_text", text="Still working...")],
                )
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=10, total_tokens=60),
            status="in_progress",
            model="gpt-5-codex",
        ),
        _codex_message_response("Final answer after progress updates."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("keep going")

    assert result["completed"] is True
    interim_msgs = [
        msg for msg in result["messages"]
        if msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
    ]
    assert len(interim_msgs) == 2
    assert [msg["codex_message_items"][0]["id"] for msg in interim_msgs] == [
        "msg_first",
        "msg_second",
    ]
    assert all(msg["codex_message_items"][0]["status"] == "in_progress" for msg in interim_msgs)


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


def test_run_conversation_codex_disables_reasoning_replay_after_invalid_encrypted_content(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.provider = "custom"
    agent.base_url = "https://api.example.com/v1"

    request_payloads = []

    class _InvalidEncryptedContentError(Exception):
        def __init__(self):
            super().__init__(
                "Error code: 400 - The encrypted content for item rs_001 could not be verified. "
                "Reason: Encrypted content could not be decrypted or parsed."
            )
            self.status_code = 400
            self.body = {
                "error": {
                    "message": (
                        '{"error":{"message":"The encrypted content for item rs_001 could not be verified. '
                        'Reason: Encrypted content could not be decrypted or parsed.",'
                        '"type":"invalid_request_error","param":"","code":"invalid_encrypted_content"}}'
                    ),
                    "type": "400",
                }
            }

    responses = [_InvalidEncryptedContentError(), _codex_message_response("Recovered without replay.")]

    def _fake_api_call(api_kwargs):
        request_payloads.append(api_kwargs)
        current = responses.pop(0)
        if isinstance(current, Exception):
            raise current
        return current

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    history = [
        {
            "role": "assistant",
            "content": "",
            "finish_reason": "incomplete",
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_001", "encrypted_content": "enc_bad", "summary": []},
            ],
        }
    ]

    result = agent.run_conversation("continue", conversation_history=history)

    assert result["completed"] is True
    assert result["final_response"] == "Recovered without replay."
    assert len(request_payloads) == 2
    assert any(item.get("type") == "reasoning" for item in request_payloads[0]["input"])
    assert not any(item.get("type") == "reasoning" for item in request_payloads[1]["input"])
    assert request_payloads[0].get("include") == ["reasoning.encrypted_content"]
    assert request_payloads[1].get("include") == []
    assert result["messages"][0].get("codex_reasoning_items") is None
    assert agent._codex_reasoning_replay_enabled is False


def test_run_conversation_codex_invalid_encrypted_content_without_replay_state_does_not_disable_replay(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.provider = "custom"
    agent.base_url = "https://api.example.com/v1"
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *args, **kwargs: 0)

    request_payloads = []

    class _InvalidEncryptedContentError(Exception):
        def __init__(self):
            super().__init__("Error code: 400 - bad request")
            self.status_code = 400
            self.body = {
                "error": {
                    "code": "INVALID_ENCRYPTED_CONTENT",
                    "message": "Bad request",
                }
            }

    responses = [_InvalidEncryptedContentError(), _codex_message_response("Recovered after generic retry.")]

    def _fake_api_call(api_kwargs):
        request_payloads.append(api_kwargs)
        current = responses.pop(0)
        if isinstance(current, Exception):
            raise current
        return current

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    result = agent.run_conversation(
        "continue",
        conversation_history=[{"role": "assistant", "content": "No replay state here."}],
    )

    assert result["completed"] is True
    assert result["final_response"] == "Recovered after generic retry."
    assert len(request_payloads) == 2
    assert all(payload.get("include") == ["reasoning.encrypted_content"] for payload in request_payloads)
    assert all(not any(item.get("type") == "reasoning" for item in payload["input"]) for payload in request_payloads)
    assert agent._codex_reasoning_replay_enabled is True
    assert result["messages"][0].get("codex_reasoning_items") is None

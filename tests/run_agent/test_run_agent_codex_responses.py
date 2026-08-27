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


AZURE_FOUNDRY_BASE_URL = (
    "https://placeholder.services.ai.azure.com/api/projects/placeholder/openai/v1"
)


def _build_azure_foundry_agent(monkeypatch, *, model="gpt-5.4"):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model=model,
        provider="azure-foundry",
        api_mode="codex_responses",
        base_url=AZURE_FOUNDRY_BASE_URL,
        api_key="foundry-token",
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


def _codex_max_output_incomplete_response(text: str = ""):
    content = []
    if text:
        content.append(SimpleNamespace(type="output_text", text=text))
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="incomplete",
                content=content,
            )
        ],
        usage=SimpleNamespace(input_tokens=270_000, output_tokens=1, total_tokens=270_001),
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
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


def _codex_commentary_final_tool_response(commentary: str, final_answer: str = "Done."):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="commentary",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=commentary)],
            ),
            SimpleNamespace(
                type="message",
                phase="final_answer",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=final_answer)],
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments="{}",
            ),
        ],
        usage=SimpleNamespace(input_tokens=8, output_tokens=5, total_tokens=13),
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
    assert agent._is_codex_backend() is False














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


def test_build_api_kwargs_mantle_sets_extended_prompt_cache_retention(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="openai.gpt-5.5",
        provider="custom",
        api_mode="codex_responses",
        base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
        api_key="test-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Ping"}])

    assert kwargs["prompt_cache_retention"] == "24h"


def _azure_reasoning_item():
    return {"type": "reasoning", "encrypted_content": "sealed", "summary": []}


def _azure_post_tool_messages():
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Create a marker"},
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [_azure_reasoning_item()],
            "tool_calls": [
                {
                    "id": "call_marker",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_marker", "content": "marker written"},
    ]


def test_build_api_kwargs_azure_foundry_post_tool_suppresses_reasoning(monkeypatch):
    """Live agent path reaches Azure Foundry detection and scopes suppression.

    Exercises ``chat_completion_helpers.build_api_kwargs`` end-to-end rather
    than the transport in isolation: the agent must forward ``provider`` and
    ``base_url`` into ``build_kwargs`` for the Foundry detection to fire at
    all. On the post-tool follow-up shape the encrypted reasoning item is
    dropped while function_call / function_call_output continuity is kept.
    """
    agent = _build_azure_foundry_agent(monkeypatch)
    assert agent._codex_reasoning_replay_enabled is True

    kwargs = agent._build_api_kwargs(_azure_post_tool_messages())

    item_types = [item.get("type") for item in kwargs["input"] if isinstance(item, dict)]
    assert "reasoning" not in item_types
    assert "function_call" in item_types
    assert "function_call_output" in item_types
    assert kwargs.get("include") == []


def test_build_api_kwargs_azure_foundry_non_tool_preserves_reasoning(monkeypatch):
    """Ordinary (non-tool) Azure Foundry continuity is unchanged via the live path.

    Without the post-tool follow-up shape there is no evidence Foundry rejects
    the payload, so the encrypted reasoning item must still be replayed even
    though the agent forwards the Foundry identity fields.
    """
    agent = _build_azure_foundry_agent(monkeypatch)

    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Explain recursion"},
        {
            "role": "assistant",
            "content": "Recursion is when a function calls itself.",
            "codex_reasoning_items": [_azure_reasoning_item()],
        },
        {"role": "user", "content": "Give an example"},
    ]

    kwargs = agent._build_api_kwargs(messages)

    item_types = [item.get("type") for item in kwargs["input"] if isinstance(item, dict)]
    assert "reasoning" in item_types
    assert "function_call" not in item_types
    assert "function_call_output" not in item_types
    assert kwargs.get("include") == ["reasoning.encrypted_content"]


def test_build_api_kwargs_azure_foundry_user_turn_after_tool_call_keeps_reasoning(
    monkeypatch,
):
    """Suppression does not stick once the tool call is answered.

    Regression guard for the sticky-history shape: after the assistant has
    replied to the tool result, a plain user follow-up is a payload Foundry
    accepts, so reasoning replay must come back on rather than stay off for
    the remainder of the conversation.
    """
    agent = _build_azure_foundry_agent(monkeypatch)

    messages = _azure_post_tool_messages() + [
        {
            "role": "assistant",
            "content": "Marker created.",
            "codex_reasoning_items": [_azure_reasoning_item()],
        },
        {"role": "user", "content": "Now explain recursion"},
    ]

    kwargs = agent._build_api_kwargs(messages)

    item_types = [item.get("type") for item in kwargs["input"] if isinstance(item, dict)]
    assert "reasoning" in item_types
    assert "function_call" in item_types
    assert "function_call_output" in item_types
    assert kwargs.get("include") == ["reasoning.encrypted_content"]










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








def test_run_codex_stream_strips_relay_added_retention_at_consumer_wire(
    monkeypatch,
    caplog,
):
    """A Relay-added unsupported field cannot reach consumer ChatGPT Codex."""
    from agent import relay_llm

    agent = _build_agent(monkeypatch)
    captured = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeCreateStream([
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed"),
            )
        ])

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )
    original = _codex_request_kwargs()
    relay_request_body = relay_llm._relay_request_body(
        original,
        {"api_mode": "codex_responses"},
    )
    relayed = relay_llm._provider_request(
        original,
        SimpleNamespace(
            content={
                **relay_request_body,
                "prompt_cache_retention": "24h",
            },
            headers={},
        ),
        relay_request_body=relay_request_body,
        codec_baseline_body=dict(relay_request_body),
        metadata={"api_mode": "codex_responses"},
    )
    assert relayed["prompt_cache_retention"] == "24h"

    with caplog.at_level("WARNING", logger="agent.codex_runtime"):
        agent._run_codex_stream(relayed)

    assert "prompt_cache_retention" not in captured
    assert any(
        "Dropped unsupported prompt_cache_retention at consumer Codex wire boundary"
        in record.message
        for record in caplog.records
    )


def test_run_codex_stream_strips_nested_request_override_retention(
    monkeypatch,
    caplog,
):
    """Configured extra_body retention cannot cross the final wire boundary."""
    from agent.transports.codex import ResponsesApiTransport

    agent = _build_agent(monkeypatch)
    captured = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeCreateStream([
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed"),
            )
        ])

    agent.client = SimpleNamespace(
        responses=SimpleNamespace(create=_fake_create),
    )
    request = ResponsesApiTransport().build_kwargs(
        model="gpt-5.6-sol",
        messages=[
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ],
        tools=[],
        is_codex_backend=True,
        base_url="https://chatgpt.com/backend-api/codex",
        request_overrides={
            "extra_body": {"prompt_cache_retention": "24h"},
        },
    )
    assert request["extra_body"] == {"prompt_cache_retention": "24h"}

    with caplog.at_level("WARNING", logger="agent.codex_runtime"):
        agent._run_codex_stream(request)

    # The transform bypass (#93650) re-introduces extra_body to carry the bulk
    # payload fields; the guard's contract is that retention itself never
    # crosses the wire boundary in either shape.
    assert "prompt_cache_retention" not in captured
    assert "prompt_cache_retention" not in captured.get("extra_body", {})
    assert "input" in captured.get("extra_body", {})
    assert request["extra_body"] == {"prompt_cache_retention": "24h"}
    assert any(
        "Dropped unsupported prompt_cache_retention at consumer Codex wire boundary"
        in record.message
        for record in caplog.records
    )


def test_consumer_codex_wire_guard_strips_nested_extra_body_retention(caplog):
    """The SDK merges ``extra_body`` into the JSON body, so a nested
    ``extra_body.prompt_cache_retention`` reaches the endpoint exactly like the
    top-level field — the guard must strip both shapes (#89897)."""
    from agent.codex_runtime import _sanitize_consumer_codex_request

    agent = SimpleNamespace(_is_codex_backend=lambda: True, model="gpt-5.6-sol")
    extra_body = {"prompt_cache_retention": "24h", "unrelated": "keep"}
    request = {
        "model": "gpt-5.6-sol",
        "prompt_cache_key": "cache-key-sentinel",
        "extra_body": extra_body,
    }

    with caplog.at_level("WARNING", logger="agent.codex_runtime"):
        sanitized = _sanitize_consumer_codex_request(agent, request)

    assert "prompt_cache_retention" not in sanitized.get("extra_body", {})
    # Unrelated extra_body entries survive; the caller's mapping is untouched.
    assert sanitized["extra_body"]["unrelated"] == "keep"
    assert extra_body["prompt_cache_retention"] == "24h"
    assert sanitized["prompt_cache_key"] == "cache-key-sentinel"
    assert any(
        "Dropped unsupported prompt_cache_retention" in record.message
        for record in caplog.records
    )


def test_consumer_codex_wire_guard_drops_emptied_extra_body():
    """When retention was extra_body's only entry, the emptied mapping is
    removed rather than sent as ``extra_body={}``."""
    from agent.codex_runtime import _sanitize_consumer_codex_request

    agent = SimpleNamespace(_is_codex_backend=lambda: True, model="gpt-5.6-sol")
    request = {
        "model": "gpt-5.6-sol",
        "extra_body": {"prompt_cache_retention": "24h"},
    }

    sanitized = _sanitize_consumer_codex_request(agent, request)

    assert "extra_body" not in sanitized


def test_consumer_codex_wire_guard_preserves_nested_retention_on_compatible_endpoint():
    """Compatible endpoints keep a nested extra_body retention untouched."""
    from agent.codex_runtime import _sanitize_consumer_codex_request

    agent = SimpleNamespace(_is_codex_backend=lambda: False)
    request = {
        "model": "openai.gpt-5.5",
        "extra_body": {"prompt_cache_retention": "24h"},
    }

    sanitized = _sanitize_consumer_codex_request(agent, request)

    assert sanitized["extra_body"]["prompt_cache_retention"] == "24h"


@pytest.mark.parametrize(
    "base_url,model,expect_dropped",
    [
        # Consumer ChatGPT Codex: the endpoint rejects retention outright.
        ("https://chatgpt.com/backend-api/codex", "gpt-5.6-sol", True),
        ("https://chatgpt.com/backend-api/codex", "gpt-5-codex", True),
        # Hosts that support 24h retention must keep it (#70083, #88601).
        ("https://api.meta.ai/v1", "gpt-5.6-sol", False),
        ("https://bedrock-mantle.us-east-1.api.aws/v1", "openai.gpt-5.5", False),
        # Non-Codex OpenAI and a same-host/different-path backend are both
        # outside the consumer-Codex contract the guard enforces.
        ("https://api.openai.com/v1", "gpt-5.6-sol", False),
        ("https://chatgpt.com/backend-api/other", "gpt-5.6-sol", False),
    ],
)
def test_wire_guard_scopes_retention_drop_by_real_endpoint(
    monkeypatch,
    base_url,
    model,
    expect_dropped,
):
    """The drop is scoped by the real endpoint, not by a stubbed predicate.

    The sibling test above pins the helper's contract against an explicit
    boolean. This one drives ``_is_codex_backend()`` off a real ``AIAgent``
    built on each base URL, so a change to the hostname/path predicate that
    widened the drop onto retention-supporting hosts would fail here.
    """
    from agent.codex_runtime import _sanitize_consumer_codex_request

    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model=model,
        api_mode="codex_responses",
        base_url=base_url,
        api_key="codex-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )

    request = {
        "model": model,
        "prompt_cache_key": "cache-key-sentinel",
        "prompt_cache_retention": "24h",
        "input": [{"role": "user", "content": "hi"}],
    }
    sanitized = _sanitize_consumer_codex_request(agent, request)

    assert ("prompt_cache_retention" not in sanitized) is expect_dropped
    # Cache-key routing is independent of retention: the guard must never
    # disturb the prompt-cache key, on any endpoint.
    assert sanitized["prompt_cache_key"] == "cache-key-sentinel"
    assert request["prompt_cache_retention"] == "24h"
    # The caller mutates the result (stream_kwargs["stream"] = True), so the
    # guard must return a fresh mapping on EVERY path, including no-drop ones.
    assert sanitized is not request


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


def test_consume_codex_stream_routes_commentary_phase_deltas_to_reasoning(monkeypatch):
    from agent.codex_runtime import _consume_codex_event_stream

    commentary_item = SimpleNamespace(
        type="message",
        phase="commentary",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="I’ll call the tool now.")],
    )
    function_item = SimpleNamespace(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="terminal",
        arguments="{}",
    )
    streamed = []
    reasoning_streamed = []

    response = _consume_codex_event_stream(
        _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="message", phase="commentary"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="I’ll call the tool now."),
            SimpleNamespace(type="response.output_item.done", item=commentary_item),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="function_call"),
            ),
            SimpleNamespace(type="response.output_item.done", item=function_item),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
        ]),
        model="gpt-5-codex",
        on_text_delta=streamed.append,
        on_reasoning_delta=reasoning_streamed.append,
    )

    assert streamed == []
    assert reasoning_streamed == ["I’ll call the tool now."]
    assert response.output == [commentary_item, function_item]
    assert response.output_text == ""


def test_consume_codex_stream_separates_commentary_from_analysis(monkeypatch):
    from agent.codex_runtime import _consume_codex_event_stream

    commentary_item = SimpleNamespace(
        type="message",
        phase="commentary",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="I'll inspect the repo first.")],
    )
    streamed = []
    reasoning_streamed = []
    commentary_messages = []

    response = _consume_codex_event_stream(
        _FakeCreateStream([
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="message", phase="commentary"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="I'll inspect "),
            SimpleNamespace(type="response.output_text.delta", delta="the repo first."),
            SimpleNamespace(type="response.output_item.done", item=commentary_item),
            SimpleNamespace(
                type="response.reasoning_text.delta",
                delta="Need inspect files privately.",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed"),
            ),
        ]),
        model="gpt-5-codex",
        on_text_delta=streamed.append,
        on_reasoning_delta=reasoning_streamed.append,
        on_commentary_message=commentary_messages.append,
    )

    assert commentary_messages == ["I'll inspect the repo first."]
    assert reasoning_streamed == ["Need inspect files privately."]
    assert streamed == []
    assert response.output == [commentary_item]




def test_run_codex_stream_delivers_redacted_commentary_once(monkeypatch):
    from agent.codex_responses_adapter import _normalize_codex_response

    agent = _build_agent(monkeypatch)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    delivered = []
    reasoning_streamed = []
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: delivered.append(
            (text, already_streamed)
        )
    )
    agent.reasoning_callback = reasoning_streamed.append
    secret = "sk-" + ("A" * 32)
    commentary_text = f"Using credential {secret}. I'll inspect the repo."
    commentary_item = SimpleNamespace(
        type="message",
        phase="commentary",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=commentary_text)],
    )
    function_item = SimpleNamespace(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="terminal",
        arguments="{}",
    )

    def _fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return _FakeCreateStream([
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="message", phase="commentary"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta=commentary_text),
            SimpleNamespace(type="response.output_item.done", item=commentary_item),
            SimpleNamespace(type="response.reasoning_text.delta", delta="Private scratchpad."),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="function_call"),
            ),
            SimpleNamespace(type="response.output_item.done", item=function_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed"),
            ),
        ])

    agent.client = SimpleNamespace(responses=SimpleNamespace(create=_fake_create))

    response = agent._run_codex_stream(_codex_request_kwargs())

    assert len(delivered) == 1
    assert delivered[0][1] is False
    assert secret not in delivered[0][0]
    assert "Using credential" in delivered[0][0]
    assert reasoning_streamed == ["Private scratchpad."]

    # The completed-response fallback sees the same preserved commentary but
    # must not enqueue it again after live delivery.
    normalized, finish_reason = _normalize_codex_response(response)
    agent._emit_interim_assistant_message(
        agent._build_assistant_message(normalized, finish_reason)
    )
    assert len(delivered) == 1






def test_run_codex_stream_returns_terminal_response_when_post_terminal_drain_fails(
    monkeypatch, caplog
):
    """Regression test for issue #74310.

    A transport error while draining the SSE iterator *after* a valid
    ``response.completed`` has already been observed (and the response
    object fully assembled) must NOT discard that response and retry with
    a brand-new physical request -- that would silently duplicate an
    already-billed inference. Only errors that occur BEFORE a terminal
    event is captured should trigger the retry-with-new-request path.
    """
    import logging

    import httpx

    agent = _build_agent(monkeypatch)

    message_item = SimpleNamespace(
        type="message",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="All done.")],
    )
    usage = SimpleNamespace(input_tokens=10, output_tokens=6, total_tokens=16)

    class _PostTerminalDroppingStream(_FakeCreateStream):
        """Yields events normally, then raises only on the *next* pull --
        i.e. after ``response.completed`` has already been consumed and the
        event-driven parser has broken out of its loop."""

        def __iter__(self):
            yield from super().__iter__()
            raise httpx.RemoteProtocolError("connection dropped during drain")

    events = [
        SimpleNamespace(type="response.output_item.done", item=message_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed",
                usage=usage,
                id="resp_post_terminal_1",
            ),
        ),
    ]

    calls = {"count": 0}

    def _fake_create(**kwargs):
        calls["count"] += 1
        return _PostTerminalDroppingStream(events)

    agent.client = SimpleNamespace(responses=SimpleNamespace(create=_fake_create))

    with caplog.at_level(logging.WARNING, logger="agent.codex_runtime"):
        response = agent._run_codex_stream(_codex_request_kwargs())

    # Only ONE physical request was ever opened -- the drain failure did not
    # trigger a second call to responses.create(stream=True).
    assert calls["count"] == 1
    assert response.status == "completed"
    assert response.usage is usage
    assert response.id == "resp_post_terminal_1"
    assert any(
        "finalization" in record.message for record in caplog.records
    )


def test_run_conversation_codex_plain_text(monkeypatch):
    agent = _build_agent(monkeypatch)
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _codex_message_response("OK"))

    result = agent.run_conversation("Say OK")

    assert result["completed"] is True
    assert result["final_response"] == "OK"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "OK"


def test_codex_preflight_defangs_harmony_tokens_before_and_after_middleware(monkeypatch):
    """Both mutable request boundaries must reject literal Harmony wire tokens."""
    agent = _build_agent(monkeypatch)
    setattr(agent, "_disable_streaming", True)
    token = f"<\x7cstart\x7c>"
    captured = {}

    def _request_middleware(request, **_context):
        # Initial preflight runs before request middleware.
        assert token not in str(request["input"])
        replacement = dict(request)
        replacement["instructions"] = "Inspect source containing " + token
        replacement["input"] = [{
            "type": "function_call_output",
            "call_id": "call_poisoned",
            "output": "source contains " + token,
        }]
        return SimpleNamespace(
            payload=replacement,
            original_payload=request,
            changed=True,
            trace=[],
        )

    def _execution_middleware(request, next_call, **_context):
        # Request middleware can reintroduce a reserved token after initial
        # preflight, so it must still be present before the dispatch chokepoint.
        assert token in str(request)
        return next_call(request)

    def _capture_api_call(api_kwargs):
        captured.update(api_kwargs)
        return _codex_message_response("OK")

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        _request_middleware,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        _execution_middleware,
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", _capture_api_call)

    result = agent.run_conversation("Read " + token)

    assert result["completed"] is True
    assert token not in captured["instructions"]
    assert token not in str(captured["input"])
    assert "<｜start｜>" in captured["instructions"]


def test_copilot_responses_preflight_preserves_harmony_tokens(monkeypatch):
    """Other Responses-compatible providers remain byte-identical."""
    agent = _build_copilot_agent(monkeypatch)
    setattr(agent, "_disable_streaming", True)
    token = f"<\x7cstart\x7c>"
    captured = {}

    def _capture_api_call(api_kwargs):
        captured.update(api_kwargs)
        return _codex_message_response("OK")

    monkeypatch.setattr(agent, "_interruptible_api_call", _capture_api_call)

    result = agent.run_conversation("Read " + token)

    assert result["completed"] is True
    assert token in str(captured["input"])


def test_codex_backend_detection_is_narrow(monkeypatch):
    codex = _build_agent(monkeypatch)
    copilot = _build_copilot_agent(monkeypatch)

    assert codex._is_codex_backend() is True
    assert copilot._is_codex_backend() is False

    # Exact backend URL detection still works for an explicitly custom route.
    setattr(codex, "provider", "custom")
    assert codex._is_codex_backend() is True
    setattr(codex, "api_mode", "chat_completions")
    assert codex._is_codex_backend() is False


def test_copilot_final_preflight_sanitizes_both_middleware_layers(monkeypatch):
    """The dispatch chokepoint must sanitize after every mutable layer."""
    agent = _build_copilot_agent(monkeypatch)
    setattr(agent, "_disable_streaming", True)
    captured = {}

    def _message_item(item_id, *, text, phase, status):
        return {
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": text}],
            "id": item_id,
            "phase": phase,
        }

    def _request_middleware(request, **_context):
        replacement = dict(request)
        replacement["input"] = [
            _message_item(
                "request_middleware_id",
                text="request-layer",
                phase="commentary",
                status="completed",
            )
        ]
        return SimpleNamespace(
            payload=replacement,
            original_payload=request,
            changed=True,
            trace=[],
        )

    def _execution_middleware(request, next_call, **_context):
        # Request middleware runs after the initial preflight, so its ID is
        # still present here. The dispatch chokepoint must remove the ID that
        # this execution middleware introduces immediately before the API call.
        assert request["input"][0]["id"] == "request_middleware_id"
        replacement = dict(request)
        replacement["input"] = [
            _message_item(
                "execution_middleware_id",
                text="execution-layer",
                phase="final_answer",
                status="in_progress",
            )
        ]
        return next_call(replacement)

    def _capture_api_call(api_kwargs):
        captured.update(api_kwargs)
        return _codex_message_response("OK")

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        _request_middleware,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        _execution_middleware,
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", _capture_api_call)

    result = agent.run_conversation("Say OK")

    assert result["completed"] is True
    message_item = captured["input"][0]
    assert "id" not in message_item
    assert message_item["status"] == "in_progress"
    assert message_item["phase"] == "final_answer"
    assert message_item["content"] == [
        {"type": "output_text", "text": "execution-layer"}
    ]


def test_codex_final_preflight_bounds_middleware_cache_key(monkeypatch):
    """Execution middleware cannot reintroduce an over-length provider key."""
    agent = _build_agent(monkeypatch)
    setattr(agent, "_disable_streaming", True)
    captured = {}
    long_key = "paperclip:" + "x" * 130

    def _execution_middleware(request, next_call, **_context):
        replacement = dict(request)
        replacement["prompt_cache_key"] = long_key
        return next_call(replacement)

    def _capture_api_call(api_kwargs):
        captured.update(api_kwargs)
        return _codex_message_response("OK")

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        _execution_middleware,
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", _capture_api_call)

    result = agent.run_conversation("Say OK")

    assert result["completed"] is True
    assert captured["prompt_cache_key"].startswith("pck_")
    assert len(captured["prompt_cache_key"]) <= 64


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

    existing = _ExistingClient()
    agent.client = existing
    retired = {"client": None}
    monkeypatch.setattr(
        agent,
        "_retire_shared_openai_client",
        lambda client, *, reason: retired.__setitem__("client", client),
    )
    ok = agent._try_refresh_codex_client_credentials(force=True)

    assert ok is True
    # #70773: the replaced shared client must NOT be close()d from the
    # refresh path (cross-thread close is the FD-recycle corruption
    # vector) — it is retired (sockets shutdown, FD release via GC).
    assert closed["value"] is False
    assert retired["client"] is existing
    assert rebuilt["kwargs"]["api_key"] == "fresh-xai-token"
    assert rebuilt["kwargs"]["base_url"] == "https://api.x.ai/v1"
    assert isinstance(agent.client, _RebuiltClient)
    assert agent.api_key == "fresh-xai-token"


def test_try_refresh_codex_client_credentials_skips_xai_oauth_when_singleton_differs(monkeypatch):
    """An xai-oauth agent constructed with a non-singleton credential
    (e.g. a manual pool entry whose tokens belong to a different account
    than the device_code singleton, or an explicit ``api_key=`` arg)
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






def test_try_refresh_copilot_client_credentials_rebuilds_client(monkeypatch):
    agent = _build_copilot_agent(monkeypatch)
    rebuilt = {"kwargs": None}

    class _ExistingClient:
        pass

    class _RebuiltClient:
        pass

    def _fake_openai(**kwargs):
        rebuilt["kwargs"] = kwargs
        return _RebuiltClient()

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_new_token", "GH_TOKEN"),
    )
    # The 401 refresh forces a fresh IDE-token exchange; mock it to a valid
    # exchanged token so the test is deterministic and network-free.
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.evict_cached_exchanged_token",
        lambda _raw: None,
    )
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        lambda _raw: ("tid=exchanged-ide-token", None),
    )
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    agent.client = _ExistingClient()
    ok = agent._try_refresh_copilot_client_credentials()

    assert ok is True
    # The old client is retired by _replace_primary_openai_client (release is
    # deferred to GC on current main — no synchronous .close() contract).
    # The freshly EXCHANGED IDE token — not the raw ghu_/gho_ token — goes on
    # the wire, which is what fixes the "401 IDE token expired" recurrence.
    assert rebuilt["kwargs"]["api_key"] == "tid=exchanged-ide-token"
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
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.evict_cached_exchanged_token",
        lambda _raw: None,
    )
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        lambda _raw: ("tid=fresh-exchanged", None),
    )
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    ok = agent._try_refresh_copilot_client_credentials()

    assert ok is True
    assert rebuilt["count"] == 1


def test_try_refresh_copilot_client_credentials_falls_back_when_exchange_unavailable(monkeypatch):
    """If the IDE-token re-exchange itself fails (network blip), the refresh
    still rebuilds the client on the resolved raw token rather than throwing —
    clears stale client state and degrades gracefully."""
    agent = _build_copilot_agent(monkeypatch)
    rebuilt = {"kwargs": None}

    class _RebuiltClient:
        pass

    def _fake_openai(**kwargs):
        rebuilt["kwargs"] = kwargs
        return _RebuiltClient()

    def _boom(_raw):
        raise RuntimeError("exchange endpoint unreachable")

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_raw_token", "GH_TOKEN"),
    )
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.evict_cached_exchanged_token",
        lambda _raw: None,
    )
    monkeypatch.setattr("hermes_cli.copilot_auth.get_copilot_api_token", _boom)
    monkeypatch.setattr(run_agent, "OpenAI", _fake_openai)

    ok = agent._try_refresh_copilot_client_credentials()

    assert ok is True
    # Exchange failed → falls back to the resolved raw token, client still rebuilt.
    assert rebuilt["kwargs"]["api_key"] == "gho_raw_token"


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












def test_run_conversation_codex_replay_payload_keeps_call_id(monkeypatch):
    agent = _build_agent(monkeypatch)
    responses = [_codex_tool_call_response(), _codex_message_response("done")]
    requests = []

    def _fake_api_call(api_kwargs):
        requests.append(api_kwargs)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, *_args):
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






def test_run_conversation_compresses_mid_turn_before_output_budget_exhaustion(monkeypatch):
    """Long tool-heavy turns should compact before the next API request.

    Initial preflight compression only sees the user's first message. A single
    turn can then grow by many tool results and leave almost no output budget
    (the live 271k/272k GPT-5.5 failure). The agent should re-check request
    pressure before every API call and compact before asking the model to
    produce the final answer.
    """
    agent = _build_agent(monkeypatch)
    agent.context_compressor.context_length = 20_000
    agent.context_compressor.threshold_tokens = 20_000

    responses = [
        _codex_tool_call_response(),
        _codex_message_response("Summary after compaction."),
    ]
    requests = []
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: requests.append(api_kwargs) or responses.pop(0),
    )

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "x" * 80_000,
                }
            )

    compress_calls = []

    def _fake_compress_context(messages, system_message, *, approx_tokens=None, task_id="default", focus_topic=None):
        compress_calls.append(approx_tokens)
        return [
            {"role": "user", "content": "[summary of prior tool-heavy work]"},
        ], "You are Hermes."

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)
    monkeypatch.setattr(agent, "_compress_context", _fake_compress_context)

    result = agent.run_conversation("do a tool-heavy task")

    assert result["completed"] is True
    assert result["final_response"] == "Summary after compaction."
    assert len(compress_calls) == 1
    assert compress_calls[0] >= 15_000
    assert len(requests) == 2


def test_mid_turn_compaction_does_not_double_persist_in_place_rows(monkeypatch, tmp_path):
    """Mid-turn pre-API compaction must re-baseline the flush cursor.

    In-place compaction (``compression.in_place: True``, the default) inserts
    the compacted rows into the session DB itself via ``archive_and_compact``
    WITHOUT stamping them with the intrinsic persisted-marker. The loop must
    therefore set ``conversation_history`` to those compacted dicts so the next
    flush skips them by identity. Setting ``conversation_history = None`` here
    (as the original PR did) makes the flush treat the already-persisted
    compacted dicts as new and append them a second time — doubling the active
    context and retriggering compression. This guards that regression with a
    REAL SessionDB and the REAL archive_and_compact path (no persist stubs).
    """
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _build_agent(monkeypatch)
    # _build_agent stubs _persist_session; restore the real one so the flush
    # cursor / double-write behaviour is exercised end to end.
    agent._persist_session = run_agent.AIAgent._persist_session.__get__(agent)
    agent._cleanup_task_resources = lambda task_id: None

    agent.context_compressor.context_length = 20_000
    agent.context_compressor.threshold_tokens = 20_000

    agent._session_db = SessionDB()
    agent._ensure_db_session()

    responses = [
        _codex_tool_call_response(),
        _codex_message_response("Summary after compaction."),
    ]
    monkeypatch.setattr(
        agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0)
    )

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": "x" * 80_000}
            )

    def _fake_compress_context(messages, system_message, *, approx_tokens=None, task_id="default", focus_topic=None):
        # Emulate the real in-place compaction DB side effect: soft-archive the
        # prior rows and insert the compacted set under the SAME session id,
        # then reset the flush identity seed — exactly as archive_and_compact +
        # the in_place branch in conversation_compression.py do.
        agent._last_compaction_in_place = True
        compacted = [{"role": "user", "content": "[summary of prior tool-heavy work]"}]
        agent._session_db.archive_and_compact(agent.session_id, compacted)
        agent._flushed_db_message_ids = set()
        return compacted, "You are Hermes."

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)
    monkeypatch.setattr(agent, "_compress_context", _fake_compress_context)

    result = agent.run_conversation("do a tool-heavy task")
    assert result["completed"] is True

    # The compacted summary row must appear exactly once in the active
    # transcript that a resume would reload.
    active = agent._session_db.get_messages(agent.session_id)
    summary_rows = [
        m for m in active
        if isinstance(m.get("content"), str)
        and "summary of prior tool-heavy work" in m["content"]
    ]
    assert len(summary_rows) == 1, (
        f"compacted summary row double-persisted: {len(summary_rows)} copies "
        "(conversation_history flush cursor not re-baselined for in-place compaction)"
    )


def _codex_incomplete_with_reasoning(text: str, reasoning_id: str = "rs_default"):
    """Incomplete response with a reasoning item whose id/encrypted_content
    can vary independently of the visible message text."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                id=reasoning_id,
                encrypted_content=f"opaque_{reasoning_id}",
                summary=[SimpleNamespace(text="thinking...")],
            ),
            SimpleNamespace(
                type="message",
                status="in_progress",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="in_progress",
        model="gpt-5-codex",
    )


def test_codex_incomplete_visible_dedup_suppresses_duplicate_interims(monkeypatch):
    """Two consecutive incomplete responses with identical visible content
    but different opaque reasoning items should be collapsed — only the first
    interim is emitted to the user (#52711)."""
    agent = _build_agent(monkeypatch)
    # 2 incompletes with same text but different reasoning ids, then a final.
    # (Only 2 to avoid hitting the cap of 3.)
    responses = [
        _codex_incomplete_with_reasoning("Working on it...", "rs_1"),
        _codex_incomplete_with_reasoning("Working on it...", "rs_2"),
        _codex_message_response("Done."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    emitted: list = []
    original_emit = agent._emit_interim_assistant_message
    def _capture_emit(msg):
        emitted.append(msg.get("content"))
        original_emit(msg)
    monkeypatch.setattr(agent, "_emit_interim_assistant_message", _capture_emit)

    result = agent.run_conversation("test dedup")

    assert result["completed"] is True
    # Only ONE interim should have been emitted (the first), not two.
    assert len(emitted) == 1
    assert emitted[0] == "Working on it..."


def test_codex_incomplete_opaque_state_updated_in_place(monkeypatch):
    """When visible content is a duplicate, the last message's opaque state
    (codex_reasoning_items) should be updated in-place without emitting a new
    interim (#52711)."""
    agent = _build_agent(monkeypatch)
    responses = [
        _codex_incomplete_with_reasoning("Partial output...", "rs_1"),
        _codex_incomplete_with_reasoning("Partial output...", "rs_2"),
        _codex_message_response("Final."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("test opaque update")

    assert result["completed"] is True
    # Find the incomplete interim message in the result.
    incompletes = [
        m for m in result["messages"]
        if m.get("role") == "assistant" and m.get("finish_reason") == "incomplete"
    ]
    # Only one incomplete message should exist (the second was deduped).
    assert len(incompletes) == 1
    # The opaque state should reflect the LATEST reasoning item (rs_2),
    # updated in-place on the single message.
    items = incompletes[0].get("codex_reasoning_items")
    if items:
        assert any(
            (i.get("id") if isinstance(i, dict) else getattr(i, "id", None)) == "rs_2"
            for i in items
        )


def test_normalize_codex_response_marks_commentary_only_message_as_incomplete(monkeypatch):
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response
    assistant_message, finish_reason = _normalize_codex_response(
        _codex_commentary_message_response("I'll inspect the repository first.")
    )

    assert finish_reason == "incomplete"
    assert (assistant_message.content or "") == ""
    assert "inspect the repository" in (assistant_message.reasoning or "")
    assert assistant_message.codex_message_items
    assert assistant_message.codex_message_items[0]["phase"] == "commentary"
    assert "inspect the repository" in assistant_message.codex_message_items[0]["content"][0]["text"]


def test_normalize_codex_response_does_not_fallback_to_output_text_for_commentary_only(monkeypatch):
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response

    response = _codex_commentary_message_response("I’ll call the tool now.")
    response.output_text = "I’ll call the tool now."

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"
    assert (assistant_message.content or "") == ""
    assert "call the tool" in (assistant_message.reasoning or "")
    assert assistant_message.codex_message_items[0]["phase"] == "commentary"

















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




def test_interim_content_was_streamed_matches_prefix_not_exact(monkeypatch):
    """_interim_content_was_streamed should return True when the streamed text
    is a PREFIX of the final content (trailing delta added after stream, or
    partial stream before verify nudge).  Exact equality is too strict — it
    fails safe to a benign duplicate bubble instead of settling the interim.
    (#65919 review: prefix-based match like the TUI's finalTail dedup.)"""
    agent = _build_agent(monkeypatch)

    # Exact match still works
    agent._current_streamed_assistant_text = "hello world"
    assert agent._interim_content_was_streamed("hello world") is True

    # Streamed is a prefix of the final (trailing delta) — should match
    agent._current_streamed_assistant_text = "hello"
    assert agent._interim_content_was_streamed("hello world") is True

    # Streamed is empty — should not match
    agent._current_streamed_assistant_text = ""
    assert agent._interim_content_was_streamed("hello world") is False

    # Final is empty — should not match
    agent._current_streamed_assistant_text = "hello"
    assert agent._interim_content_was_streamed("") is False

    # Streamed is LONGER than final (reverse direction) — should NOT match.
    # This is the unsafe direction: it could suppress a needed resend in the
    # gateway path where already_streamed=True calls on_segment_break().
    agent._current_streamed_assistant_text = "hello world extra"
    assert agent._interim_content_was_streamed("hello") is False




def test_interim_commentary_precedes_content_from_real_codex_normalization(monkeypatch):
    """Structured commentary wins over final-answer content on tool turns."""
    agent = _build_agent(monkeypatch)
    from agent.codex_responses_adapter import _normalize_codex_response

    observed = {}
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: observed.update(
        {"text": text, "already_streamed": already_streamed}
    )

    normalized, finish_reason = _normalize_codex_response(
        _codex_commentary_final_tool_response("I'll inspect the repo first.")
    )
    assert finish_reason == "tool_calls"
    assert normalized.content == "Done."
    agent._emit_interim_assistant_message(
        agent._build_assistant_message(normalized, finish_reason)
    )

    assert observed == {
        "text": "I'll inspect the repo first.",
        "already_streamed": False,
    }










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










def test_codex_commentary_emits_before_tool_and_withholds_final_answer(monkeypatch):
    agent = _build_agent(monkeypatch)
    events = []
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: events.append(("interim", text))
    )
    responses = [
        _codex_commentary_message_response("I'll inspect the repo first."),
        _codex_commentary_final_tool_response("I'll inspect the repo first."),
        _codex_message_response("Verified final answer."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, *_args):
        events.append(("tool", assistant_message.tool_calls[0].function.name))
        messages.append({
            "role": "tool",
            "tool_call_id": assistant_message.tool_calls[0].id,
            "content": '{"ok":true}',
        })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("analyze repo")

    assert result["completed"] is True
    assert events == [
        ("interim", "I'll inspect the repo first."),
        ("tool", "terminal"),
    ]
    assert all(text != "Done." for kind, text in events if kind == "interim")






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
    are deduped on visible content — only one interim is kept, but opaque state
    is updated in-place (#52711)."""
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
    # Only one reasoning-only interim should be in history (deduped on
    # visible content — both have empty visible output).
    interim_msgs = [
        msg for msg in result["messages"]
        if msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
    ]
    assert len(interim_msgs) == 1
    # But the opaque state should reflect the LATEST reasoning item.
    items = interim_msgs[0].get("codex_reasoning_items")
    if items:
        assert items[0].get("encrypted_content") == "enc_second"


def test_duplicate_detection_uses_commentary_when_hidden_reasoning_changes(monkeypatch):
    """Identical commentary is emitted once while newer replay state wins."""
    agent = _build_agent(monkeypatch)
    emitted = []
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: emitted.append(text)
    )
    responses = [
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    id="rs_first",
                    encrypted_content="enc_first",
                    summary=[SimpleNamespace(text="hidden first")],
                    status="completed",
                ),
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
                    type="reasoning",
                    id="rs_second",
                    encrypted_content="enc_second",
                    summary=[SimpleNamespace(text="hidden second")],
                    status="completed",
                ),
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
    assert emitted == ["Still working..."]
    interim_msgs = [
        msg for msg in result["messages"]
        if msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
    ]
    # Only one interim — deduped on visible content ("Still working..." == "Still working...").
    assert len(interim_msgs) == 1
    # Opaque state should reflect the latest message item.
    items = interim_msgs[0].get("codex_message_items")
    if items:
        assert items[0].get("id") == "msg_second"
    assert "hidden second" in (interim_msgs[0].get("reasoning") or "")
    reasoning_items = interim_msgs[0].get("codex_reasoning_items")
    if reasoning_items:
        assert reasoning_items[0].get("id") == "rs_second"




def test_consume_codex_stream_separates_reasoning_summary_parts():
    """summary_index is the part boundary; the wire sends no separator itself."""
    from agent.codex_runtime import _consume_codex_event_stream

    reasoning_streamed = []

    _consume_codex_event_stream(
        _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.reasoning_summary_text.delta",
                summary_index=0,
                delta="**Investigating culprit PRs**",
            ),
            SimpleNamespace(
                type="response.reasoning_summary_text.delta",
                summary_index=1,
                delta="**Inspecting message schema**",
            ),
            SimpleNamespace(
                type="response.reasoning_summary_text.delta",
                summary_index=1,
                delta=" and tool_calls content",
            ),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
        ]),
        model="gpt-5-codex",
        on_reasoning_delta=reasoning_streamed.append,
    )

    joined = "".join(reasoning_streamed)
    assert "****" not in joined
    assert joined == (
        "**Investigating culprit PRs**"
        "\n\n**Inspecting message schema** and tool_calls content"
    )


def test_consume_codex_stream_leaves_unindexed_reasoning_untouched():
    """Streams with no summary_index (plain reasoning_text) must not gain breaks."""
    from agent.codex_runtime import _consume_codex_event_stream

    reasoning_streamed = []

    _consume_codex_event_stream(
        _FakeCreateStream([
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="Need to "),
            SimpleNamespace(type="response.reasoning_text.delta", delta="inspect files."),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed")),
        ]),
        model="gpt-5-codex",
        on_reasoning_delta=reasoning_streamed.append,
    )

    assert "".join(reasoning_streamed) == "Need to inspect files."

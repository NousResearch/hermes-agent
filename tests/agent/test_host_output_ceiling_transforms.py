"""Real provider assembly must retain a bounded host's per-request limit."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.anthropic import AnthropicTransport


@pytest.mark.parametrize("model,reasoning,limit,ephemeral,ceiling,rejected", [
    ("claude-sonnet-4-20250514", {"enabled": True, "effort": "medium"}, 128, None, 128, True),
    ("claude-sonnet-4-20250514", {"enabled": True, "effort": "medium"}, 128, 256, 128, True),
    ("claude-sonnet-4-20250514", {"enabled": True, "effort": "medium"}, 16384, 64, 16384, True),
    ("claude-sonnet-4-20250514", {"enabled": True, "effort": "medium"}, 16384, None, 16384, False),
    ("claude-sonnet-4-20250514", {"enabled": False}, 128, 64, 128, False),
    ("claude-sonnet-4-6", {"enabled": True, "effort": "medium"}, 128, 256, 128, False),
    ("claude-haiku-4-5", {"enabled": True, "effort": "medium"}, 128, None, 128, False),
    ("claude-sonnet-4-20250514", {"enabled": True, "effort": "medium"}, 128, None, None, False),
    ("claude-sonnet-4-20250514", {"enabled": True, "effort": "medium"}, 128, 256, None, False),
])
def test_anthropic_assembly_rejects_incompatible_thinking_before_sdk_dispatch(
    model, reasoning, limit, ephemeral, ceiling, rejected,
):
    from anthropic import Anthropic

    transport = AnthropicTransport()
    agent = SimpleNamespace(
        model=model, api_mode="anthropic_messages", provider="anthropic",
        base_url="https://api.anthropic.com", session_id="bounded-fixture",
        max_tokens=limit, reasoning_config=deepcopy(reasoning), tools=[],
        _max_output_tokens_ceiling=ceiling, _ephemeral_max_output_tokens=ephemeral,
        _is_anthropic_oauth=False, _get_transport=lambda: transport,
        _prepare_anthropic_messages_for_api=deepcopy, _anthropic_preserve_dots=lambda: False,
    )
    messages = [{"role": "user", "content": "Reply briefly."}]
    original = deepcopy(messages)
    requests = []

    def capture(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 1},
        })

    with Anthropic(api_key="fixture-only", max_retries=0,
                   http_client=httpx.Client(transport=httpx.MockTransport(capture))) as client:
        if rejected:
            with pytest.raises(ValueError, match="output limit.*thinking"):
                client.messages.create(**build_api_kwargs(agent, messages))
            assert requests == []
        else:
            client.messages.create(**build_api_kwargs(agent, messages))
            assert len(requests) == 1
            if ceiling is not None:
                assert requests[0]["max_tokens"] <= min(ceiling, ephemeral or limit)
            else:
                assert requests[0]["max_tokens"] > limit  # ordinary manual-thinking headroom
    assert agent.reasoning_config == reasoning
    assert messages == original
    assert agent._ephemeral_max_output_tokens is None


@pytest.mark.parametrize("kind", ["gemini", "gemini-native", "responses", "codex", "bedrock", "extra-body"])
def test_other_provider_transforms_cannot_erase_or_expand_output_limit(kind):
    from agent.output_ceiling import validate_output_ceiling
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport
    from agent.transports.bedrock import BedrockTransport

    messages = [{"role": "user", "content": "Reply briefly."}]
    agent = SimpleNamespace(api_mode="chat_completions", base_url="https://example.invalid/v1")
    rejected = kind not in {"responses", "bedrock"}
    if kind in {"gemini", "gemini-native", "extra-body"}:
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="google/gemini-3-flash" if kind == "gemini" else "fixture-model",
            messages=messages, tools=[], max_tokens=128,
            max_tokens_param_fn=lambda n: {"max_tokens": n},
            reasoning_config={"enabled": True, "effort": "medium"} if kind == "gemini" else None,
            request_overrides={"extra_body": {"max_tokens": 256}} if kind == "extra-body" else {},
        )
        if kind == "gemini-native":
            agent.base_url = "https://generativelanguage.googleapis.com/v1beta"
            kwargs["extra_body"] = {"thinking_config": {"includeThoughts": True}}
    elif kind in {"responses", "codex"}:
        agent.api_mode = "codex_responses"
        kwargs = ResponsesApiTransport().build_kwargs(
            model="gpt-5", messages=messages, tools=[], max_tokens=128,
            is_codex_backend=kind == "codex", base_url="https://example.invalid/v1",
        )
    else:
        agent.api_mode = "bedrock_converse"
        kwargs = BedrockTransport().build_kwargs(model="fixture-model", messages=messages, max_tokens=128)
    original = deepcopy(kwargs)
    if rejected:
        with pytest.raises(ValueError, match="output limit"):
            validate_output_ceiling(agent, kwargs, 128)
    else:
        validate_output_ceiling(agent, kwargs, 128)
    assert kwargs == original  # reject rather than silently rewriting reasoning or requests
    validate_output_ceiling(agent, kwargs, None)  # ordinary-agent compatibility

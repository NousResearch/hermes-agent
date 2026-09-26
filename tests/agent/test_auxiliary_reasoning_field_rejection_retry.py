"""Reasoning-field rejection retry in ``agent.auxiliary_client`` (#112781).

The title lane disables reasoning (``reasoning_config={"enabled": False}``); on
``provider=custom`` the profile encodes that as top-level ``reasoning_effort: "none"``
(the deliberate Ollama /v1 / vLLM / GLM thinking-off wire). A chat-only model behind an
OpenAI-compatible relay (gpt-4.1-mini on a one-api style relay) answers
``400 Unrecognized request argument supplied: reasoning_effort`` and, before the fix,
the title was lost with no retry. The recovery is reactive like the temperature and
``response_format`` rungs: strip every reasoning field and retry once. The custom
profile's encoding itself is untouched so Ollama/vLLM/GLM users keep thinking-off.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm

_RELAY_400 = (
    "Error code: 400 - {'error': {'message': 'Unrecognized request argument supplied: "
    "reasoning_effort', 'type': 'invalid_request_error', 'param': '', 'code': None}}"
)
_VENICE_REASONING_400 = "Error code: 400 - {'error': 'reasoning.max_tokens must be positive'}"


def _custom_route_patches(client):
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "gpt-4.1-mini", "https://relay.example/v1", "sk-x", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "gpt-4.1-mini")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
    )


def _call(async_mode, client, **kw):
    common = dict(task="title_generation", messages=[{"role": "user", "content": "hi"}],
                  extra_body={"response_format": {"type": "json_object"}},
                  reasoning_config={"enabled": False}, **kw)
    p1, p2, p3, p4 = _custom_route_patches(client)
    with p1, p2, p3, p4:
        if async_mode:
            return asyncio.run(async_call_llm(**common))
        return call_llm(**common)


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_reasoning_effort_rejection_retries_once_without_reasoning_fields(async_mode):
    """Relay 400 naming ``reasoning_effort`` → one retry with no reasoning field; title lands."""
    client = MagicMock()
    client.base_url = "https://relay.example/v1"
    side_effect = [RuntimeError(_RELAY_400), {"ok": True}]
    client.chat.completions.create = AsyncMock(side_effect=side_effect) if async_mode else MagicMock(side_effect=side_effect)

    assert _call(async_mode, client) == {"ok": True}

    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 2
    first, retry = calls[0].kwargs, calls[1].kwargs
    assert first["reasoning_effort"] == "none"  # the deliberate thinking-off encoding still goes out first
    assert "reasoning_effort" not in retry
    assert "reasoning" not in (retry.get("extra_body") or {})
    assert retry["extra_body"]["response_format"] == {"type": "json_object"}  # unrelated fields survive
    assert retry["model"] == first["model"]


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("provider", ["venice", "custom"])
@pytest.mark.parametrize("reject_retry", [False, True], ids=["recovers", "exhausted"])
def test_venice_reasoning_max_tokens_rejection_retries_without_nested_reasoning(async_mode, provider, reject_retry):
    """Real generic/custom projections both recover once; a second rejection propagates unchanged."""
    client = MagicMock()
    client.base_url = "https://api.venice.ai/api/v1"
    retry_error = RuntimeError(_VENICE_REASONING_400)
    side_effect = [RuntimeError(_VENICE_REASONING_400), retry_error if reject_retry else {"ok": True}]
    client.chat.completions.create = AsyncMock(side_effect=side_effect) if async_mode else MagicMock(side_effect=side_effect)
    _, p2, p3, p4 = _custom_route_patches(client)
    kwargs: dict = dict(task="title_generation", messages=[{"role": "user", "content": "hi"}],
                  extra_body={"response_format": {"type": "json_object"}}, reasoning_config={"enabled": False})
    with p2, p3, p4, patch("agent.auxiliary_client._resolve_task_provider_model", return_value=(
        provider, "relay-model", str(client.base_url), "sk-x", None,
    )):
        if reject_retry:
            with pytest.raises(RuntimeError) as raised:
                asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
            assert raised.value is retry_error
        else:
            result = asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
            assert result == {"ok": True}

    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 2
    first, retry = calls[0].kwargs, calls[1].kwargs
    if provider == "custom":
        assert first["reasoning_effort"] == "none"
        assert "reasoning" not in first.get("extra_body", {})
    else:
        assert first["extra_body"]["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in retry
    assert "reasoning" not in retry.get("extra_body", {})
    assert retry["extra_body"]["response_format"] == {"type": "json_object"}


def test_unrelated_400_does_not_strip_reasoning_fields():
    """A 400 that does not name a reasoning field must not silently drop the thinking-off encoding."""
    client = MagicMock()
    client.base_url = "https://relay.example/v1"
    client.chat.completions.create.side_effect = RuntimeError(
        "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'")

    with pytest.raises(RuntimeError, match="Invalid value"):
        _call(False, client)
    assert client.chat.completions.create.call_count == 1


def test_top_level_max_tokens_value_error_with_reasoning_note_does_not_strip_reasoning_fields():
    """A top-level output-cap 400 mentioning reasoning is not a rejection of the reasoning control."""
    client = MagicMock()
    client.base_url = "https://relay.example/v1"
    client.chat.completions.create.side_effect = RuntimeError(
        "Error code: 400 - max_tokens must be positive (reasoning is enabled for this model)")

    with pytest.raises(RuntimeError, match="max_tokens must be positive"):
        _call(False, client)
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "none"


def test_model_gating_400_naming_a_thinking_model_still_reaches_the_fallback_chain():
    """A route-gating 400 whose text merely contains a reasoning token inside the model id
    ("kimi-k2-thinking is not supported when using this account") is not a field rejection: no
    strip-retry is spent on it and the configured fallback chain is consulted exactly as on main."""
    client = MagicMock()
    client.base_url = "https://relay.example/v1"
    client.chat.completions.create.side_effect = RuntimeError(
        "Error code: 400 - The model kimi-k2-thinking is not supported when using this account")
    fb_client = MagicMock()
    fb_client.chat.completions.create.return_value = {"fb": True}
    p1, p2, p3, _p4 = _custom_route_patches(client)
    with p1, p2, p3, patch("agent.auxiliary_client._try_configured_fallback_chain",
                           return_value=(fb_client, "fallback-model", "fallback")) as fallback:
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}],
                          reasoning_config={"enabled": False})

    assert result == {"fb": True}
    assert client.chat.completions.create.call_count == 1  # no wasted reasoning-strip retry
    assert fallback.called

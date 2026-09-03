"""Exact subscription-OAuth backend adapter for the review runner."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import AUTH_TYPE_API_KEY, AUTH_TYPE_OAUTH, PooledCredential
from agent.review_backend import (
    complete_subscription_review,
    resolve_subscription_review_route,
)


def _entry(auth_type=AUTH_TYPE_OAUTH):
    return PooledCredential(
        provider="openai-codex",
        id="credential-1",
        label="ChatGPT subscription",
        auth_type=auth_type,
        priority=0,
        source="device_code",
        access_token="private-oauth-token",
    )


class _Pool:
    def __init__(self, entry):
        self.entry = entry
        self.calls = []

    def select(self, **kwargs):
        self.calls.append(kwargs)
        return self.entry


def _resolve(pool):
    return resolve_subscription_review_route(
        provider="openai-codex",
        model="gpt-review",
        credential_policy="subscription_oauth_only",
        fallback_policy="none",
        pool_loader=lambda provider: pool,
        profile_name=lambda: "work",
    )


def test_resolver_selects_only_oauth_and_returns_sanitized_handle():
    pool = _Pool(_entry())

    route = _resolve(pool)

    assert pool.calls == [{"auth_type": AUTH_TYPE_OAUTH}]
    assert route.profile == "work"
    assert route.provider == "openai-codex"
    assert route.model == "gpt-review"
    assert route.credential_kind == "subscription_oauth"
    assert "private-oauth-token" not in repr(route.credential_handle)


def test_resolver_rejects_api_key_even_if_pool_violates_filter():
    with pytest.raises(RuntimeError, match="subscription OAuth"):
        _resolve(_Pool(_entry(AUTH_TYPE_API_KEY)))


def test_resolver_does_not_fallback_when_no_oauth_entry_exists():
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve(_Pool(None))


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "openrouter"},
        {"model": ""},
        {"credential_policy": "any"},
        {"fallback_policy": "auto"},
    ],
)
def test_resolver_requires_supported_exact_subscription_route(overrides):
    kwargs = {
        "provider": "openai-codex",
        "model": "gpt-review",
        "credential_policy": "subscription_oauth_only",
        "fallback_policy": "none",
        "pool_loader": lambda provider: _Pool(_entry()),
        "profile_name": lambda: "default",
    }
    kwargs.update(overrides)

    with pytest.raises(RuntimeError):
        resolve_subscription_review_route(**kwargs)


def _response(content, *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
        id="request-1",
    )


class _Completions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _complete(response):
    pool = _Pool(_entry())
    route = _resolve(pool)
    completions = _Completions(response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        close=MagicMock(),
    )
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return client

    call = partial(complete_subscription_review, client_factory=factory)
    return route, completions, factory_calls, client, call


def test_completion_calls_exact_model_once_without_tool_schema():
    route, completions, factory_calls, client, complete = _complete(_response(
        '{"verdict":"PASS","summary":"good","feedback":[]}'
    ))

    result = complete(
        credential_handle=route.credential_handle,
        provider="openai-codex",
        model="gpt-review",
        messages=[{"role": "user", "content": "review"}],
        tools=[],
        tool_choice="none",
        timeout=30,
        idempotency_key="checkpoint-1",
    )

    assert result["verdict"] == "PASS"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert result["request_id"] == "request-1"
    assert len(factory_calls) == 1
    assert factory_calls[0]["provider"] == "openai-codex"
    assert factory_calls[0]["model"] == "gpt-review"
    assert factory_calls[0]["timeout"] == 30
    assert len(completions.calls) == 1
    assert completions.calls[0]["model"] == "gpt-review"
    assert "tools" not in completions.calls[0]
    assert "tool_choice" not in completions.calls[0]
    client.close.assert_called_once_with()


def test_completion_rejects_any_requested_or_returned_tool_call():
    route, completions, _, client, complete = _complete(_response(
        '{"verdict":"PASS","summary":"good","feedback":[]}',
        tool_calls=[SimpleNamespace(id="tool-1")],
    ))

    with pytest.raises(RuntimeError, match="tool"):
        complete(
            credential_handle=route.credential_handle,
            provider="openai-codex",
            model="gpt-review",
            messages=[],
            tools=[],
            tool_choice="none",
            timeout=30,
            idempotency_key="checkpoint-1",
        )
    assert len(completions.calls) == 1
    client.close.assert_called_once_with()

    with pytest.raises(RuntimeError, match="tool"):
        complete(
            credential_handle=route.credential_handle,
            provider="openai-codex",
            model="gpt-review",
            messages=[],
            tools=[{"type": "function"}],
            tool_choice="none",
            timeout=30,
            idempotency_key="checkpoint-2",
        )
    assert len(completions.calls) == 1


def test_completion_rejects_route_swap_before_client_creation():
    route, completions, factory_calls, _, complete = _complete(_response("{}"))

    with pytest.raises(RuntimeError, match="route"):
        complete(
            credential_handle=route.credential_handle,
            provider="anthropic",
            model="gpt-review",
            messages=[],
            tools=[],
            tool_choice="none",
            timeout=30,
            idempotency_key="checkpoint-1",
        )

    assert factory_calls == []
    assert completions.calls == []


def test_completion_accepts_json_fence_but_rejects_invalid_payload():
    route, _, _, _, complete = _complete(_response(
        '```json\n{"verdict":"REVISE","summary":"fix","feedback":["test"]}\n```'
    ))
    result = complete(
        credential_handle=route.credential_handle,
        provider="openai-codex",
        model="gpt-review",
        messages=[],
        tools=[],
        tool_choice="none",
        timeout=30,
        idempotency_key="checkpoint-1",
    )
    assert result["verdict"] == "REVISE"

    route, _, _, client, complete = _complete(_response("not json"))
    with pytest.raises(RuntimeError, match="JSON"):
        complete(
            credential_handle=route.credential_handle,
            provider="openai-codex",
            model="gpt-review",
            messages=[],
            tools=[],
            tool_choice="none",
            timeout=30,
            idempotency_key="checkpoint-2",
        )
    client.close.assert_called_once_with()

"""xKiro Anthropic Messages needs Bearer auth and verbatim catalog ids.

The standalone plugin lives outside this tree. Core only allowlists the
official host so `/v1/messages` does not send `x-api-key` or strip
``vendor/model`` prefixes.
"""

from __future__ import annotations

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.anthropic_endpoints import _is_xkiro_endpoint, _requires_bearer_auth


def test_requires_bearer_auth_recognizes_xkiro():
    assert _requires_bearer_auth("https://api.xkiro.com/v1") is True
    assert _requires_bearer_auth("https://api.xkiro.com/v1/messages") is True
    assert _requires_bearer_auth("https://API.XKIRO.COM/v1") is True


def test_bearer_auth_does_not_match_lookalike_hosts():
    assert _requires_bearer_auth("https://api.anthropic.com") is False
    assert _requires_bearer_auth("https://api.xkiro.com.evil.example/v1") is False
    assert _requires_bearer_auth("https://evil.example/api.xkiro.com/v1") is False
    assert _is_xkiro_endpoint("https://evil.example/api.xkiro.com/v1") is False


def test_build_anthropic_kwargs_keeps_xkiro_vendor_model_id():
    kwargs = build_anthropic_kwargs(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config=None,
        base_url="https://api.xkiro.com/v1",
    )
    assert kwargs["model"] == "anthropic/claude-sonnet-4.6"


def test_build_anthropic_kwargs_still_normalizes_unrelated_hosts():
    kwargs = build_anthropic_kwargs(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config=None,
        base_url="https://api.anthropic.com",
    )
    assert kwargs["model"] == "claude-sonnet-4-6"

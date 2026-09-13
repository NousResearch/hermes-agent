"""xKiro Anthropic Messages needs Bearer auth and verbatim catalog ids.

The standalone plugin lives outside this tree. Core allowlists the official
host for Bearer auth; model-id preservation is a generic provider-profile
capability so custom xKiro endpoints work without another hostname exception.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.anthropic_endpoints import _is_xkiro_endpoint, _requires_bearer_auth
from agent.chat_completion_helpers import _build_anthropic_kwargs
from providers.base import ProviderProfile


def test_requires_bearer_auth_recognizes_xkiro():
    assert _requires_bearer_auth("https://api.xkiro.com/v1") is True
    assert _requires_bearer_auth("https://api.xkiro.com/v1/messages") is True
    assert _requires_bearer_auth("https://API.XKIRO.COM/v1") is True


def test_bearer_auth_does_not_match_lookalike_hosts():
    assert _requires_bearer_auth("https://api.anthropic.com") is False
    assert _requires_bearer_auth("https://api.xkiro.com.evil.example/v1") is False
    assert _requires_bearer_auth("https://evil.example/api.xkiro.com/v1") is False
    assert _is_xkiro_endpoint("https://evil.example/api.xkiro.com/v1") is False


def test_build_anthropic_kwargs_keeps_provider_declared_vendor_model_id():
    kwargs = build_anthropic_kwargs(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config=None,
        base_url="https://api.xkiro.com/v1",
        preserve_anthropic_model_id=True,
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


def test_xkiro_hostname_alone_does_not_control_model_normalization():
    kwargs = build_anthropic_kwargs(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config=None,
        base_url="https://api.xkiro.com/v1",
    )
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_anthropic_request_uses_provider_model_id_contract():
    profile = ProviderProfile(name="relay", preserve_anthropic_model_id=True)
    transport = SimpleNamespace(build_kwargs=lambda **kwargs: kwargs)
    agent = SimpleNamespace(
        provider="relay",
        model="vendor/claude-sonnet-4.6",
        context_compressor=None,
        _ephemeral_max_output_tokens=None,
        max_tokens=1024,
        _is_anthropic_oauth=False,
        _anthropic_base_url="https://relay.example/v1",
        _oauth_1m_beta_disabled=False,
        _get_transport=lambda: transport,
        _prepare_anthropic_messages_for_api=lambda messages: messages,
        _anthropic_preserve_dots=lambda: False,
    )

    with patch("providers.get_provider_profile", return_value=profile):
        kwargs = _build_anthropic_kwargs(
            agent,
            [{"role": "user", "content": "hi"}],
            None,
            None,
            {},
        )

    assert kwargs["preserve_anthropic_model_id"] is True

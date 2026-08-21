"""Fail-closed endpoint verification for ``resolve_fast_mode_overrides``.

A model *name* (``gpt-5.4``, ``grok-4.6``) does not prove the request will
hit a first-party endpoint that honours ``service_tier: priority`` — an
OpenAI-compatible proxy echoes the same id. When a caller passes runtime
identity (``provider`` / ``api_mode`` / ``base_url``) the resolver must fail
*closed* and only allow-list first-party origins. Zero-arg calls (the legacy
static ``fast`` path) stay byte-identical. This file is that matrix.
"""

import pytest

from hermes_cli.models import resolve_fast_mode_overrides

PRIORITY = {"service_tier": "priority"}


# Zero-arg legacy path: identity omitted → unchanged behavior.
@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("gpt-5.4", PRIORITY),
        ("gpt-4.1", PRIORITY),
        ("grok-4.6", PRIORITY),
        ("x-ai/grok-4.6-latest", PRIORITY),
        ("claude-opus-4-6", {"speed": "fast"}),
        ("anthropic/claude-opus-4.6", {"speed": "fast"}),
        ("gpt-5-codex", None),  # codex-series excluded upstream
        ("grok-4.5", None),  # not the 4.6 family
        ("some-random-model", None),
    ],
)
def test_zero_arg_calls_unchanged(model_id, expected):
    assert resolve_fast_mode_overrides(model_id) == expected


# ALLOW: first-party origins (native OpenAI, ChatGPT-Codex, xAI) earn priority.
# The grok-4.6 @ api.x.ai rows are the blocking-rework guard: the upstream
# whitelist predated Grok fast-mode support and would fail closed without them.
@pytest.mark.parametrize(
    "model_id, provider, api_mode, base_url",
    [
        ("gpt-5.4", "openai", "chat_completions", "https://api.openai.com/v1"),
        ("gpt-5.4", "openai-api", "chat_completions", "https://api.openai.com/v1"),
        ("gpt-5.4", "openai", "codex_responses", "https://api.openai.com/v1"),
        ("gpt-5.4", "openai-codex", "codex_responses", "https://chatgpt.com/backend-api/codex"),
        ("grok-4.6", "xai", "codex_responses", "https://api.x.ai/v1"),
        ("grok-4.6", "xai", "chat_completions", "https://api.x.ai/v1"),
        ("grok-4.6", "xai-oauth", "codex_responses", "https://api.x.ai/v1"),
        ("x-ai/grok-4.6-latest", "grok", "codex_responses", "https://api.x.ai/v1"),
    ],
)
def test_allowlisted_endpoints_get_priority(model_id, provider, api_mode, base_url):
    assert (
        resolve_fast_mode_overrides(
            model_id, provider=provider, api_mode=api_mode, base_url=base_url
        )
        == PRIORITY
    )


# DENY: a model name on an unverified endpoint fails closed to None.
@pytest.mark.parametrize(
    "model_id, provider, api_mode, base_url",
    [
        # gpt-shaped model on a non-OpenAI proxy
        ("gpt-5.4", "openrouter", "chat_completions", "https://openrouter.ai/api/v1"),
        # right provider name, spoofed host
        ("gpt-5.4", "openai", "chat_completions", "https://evil.example/v1"),
        # grok-shaped model on a proxy
        ("grok-4.6", "openrouter", "chat_completions", "https://openrouter.ai/api/v1"),
        # xAI provider but non-first-party host
        ("grok-4.6", "xai", "codex_responses", "https://proxy.example/v1"),
        # openai-codex provider but wrong host
        ("gpt-5.4", "openai-codex", "codex_responses", "https://api.openai.com/v1"),
        # only base_url supplied (provider→openrouter default)
        ("gpt-5.4", None, None, "https://openrouter.ai/api/v1"),
        # only api_mode supplied, no matching host
        ("gpt-5.4", None, "chat_completions", None),
    ],
)
def test_unverified_endpoints_fail_closed(model_id, provider, api_mode, base_url):
    assert (
        resolve_fast_mode_overrides(
            model_id, provider=provider, api_mode=api_mode, base_url=base_url
        )
        is None
    )


# Anthropic speed=fast is transport-gated to the native Messages API.
def test_anthropic_speed_fast_only_on_messages_transport():
    assert resolve_fast_mode_overrides(
        "claude-opus-4-6", api_mode="anthropic_messages"
    ) == {"speed": "fast"}
    assert (
        resolve_fast_mode_overrides("claude-opus-4-6", api_mode="chat_completions")
        is None
    )

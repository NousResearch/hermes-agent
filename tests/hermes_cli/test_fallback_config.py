"""Focused tests for shared fallback-provider config resolution."""

import pytest

from agent import secret_scope
from hermes_cli.fallback_config import (
    fallback_entry_hints,
    resolve_fallback_transport,
)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.anthropic.com.attacker.test/v1",
        "https://api.openai.com.attacker.test/v1",
    ],
)
def test_fallback_transport_rejects_provider_host_lookalikes(base_url):
    assert resolve_fallback_transport(
        validated_api_mode=None,
        provider="openrouter",
        model_requires_responses=False,
        base_url=base_url,
        is_azure=False,
    ) == "chat_completions"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.anthropic.com/v1", "anthropic_messages"),
        ("https://api.openai.com/v1", "codex_responses"),
        (
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "bedrock_converse",
        ),
    ],
)
def test_fallback_transport_keeps_exact_endpoint_detection(base_url, expected):
    assert resolve_fallback_transport(
        validated_api_mode=None,
        provider="openrouter",
        model_requires_responses=False,
        base_url=base_url,
        is_azure=False,
    ) == expected


@pytest.mark.parametrize(
    ("entry", "scope"),
    [
        (
            {
                "provider": "custom:scoped",
                "model": "test-model",
                "key_env": "FALLBACK_TEST_API_KEY",
            },
            {"FALLBACK_TEST_API_KEY": "profile-key"},
        ),
        (
            {
                "provider": "ollama",
                "model": "test-model",
                "base_url": "https://ollama.com/v1",
            },
            {"OLLAMA_API_KEY": "profile-key"},
        ),
    ],
)
def test_fallback_entry_hints_read_credentials_from_active_scope(
    monkeypatch,
    entry,
    scope,
):
    for name in scope:
        monkeypatch.setenv(name, "another-profile-key")

    token = secret_scope.set_secret_scope(scope)
    try:
        assert fallback_entry_hints(entry)["api_key"] == "profile-key"
    finally:
        secret_scope.reset_secret_scope(token)

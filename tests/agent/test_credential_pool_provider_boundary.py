"""Credential pools must never cross provider or custom-endpoint boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.credential_pool import credential_pool_matches_provider
from hermes_cli import runtime_provider as rp
from run_agent import AIAgent


def test_provider_match_requires_exact_non_custom_identity():
    assert credential_pool_matches_provider("deepseek", "deepseek")
    assert not credential_pool_matches_provider("openai-codex", "deepseek")
    assert not credential_pool_matches_provider("", "deepseek")


def test_custom_pool_match_is_scoped_by_endpoint():
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:lab",
    ):
        assert credential_pool_matches_provider(
            "custom:lab", "custom", base_url="https://lab.example/v1"
        )
        assert not credential_pool_matches_provider(
            "custom:other", "custom", base_url="https://lab.example/v1"
        )


def test_runtime_ignores_pool_loaded_for_different_provider(monkeypatch):
    entry = SimpleNamespace(
        provider="openai-codex",
        access_token="wrong-token",
        runtime_api_key="wrong-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(
        provider="openai-codex",
        has_credentials=lambda: True,
        select=lambda: entry,
    )
    monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: "deepseek")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "deepseek", "default": "deepseek-chat"},
    )
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "deepseek",
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "source": "env",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="deepseek")

    assert resolved["provider"] == "deepseek"
    assert resolved["api_key"] == "deepseek-key"
    assert resolved["base_url"] == "https://api.deepseek.com/v1"


@pytest.mark.parametrize(
    ("base_url", "pool_provider", "expected_provider", "expected_api_mode"),
    [
        (
            "https://chatgpt.com/backend-api/codex",
            "openai-codex",
            "openai-codex",
            "codex_responses",
        ),
        ("https://api.x.ai/v1", "xai", "xai", "codex_responses"),
        (
            "https://api.anthropic.com",
            "anthropic",
            "anthropic",
            "anthropic_messages",
        ),
    ],
)
def test_url_inferred_provider_keeps_matching_credential_pool(
    base_url, pool_provider, expected_provider, expected_api_mode
):
    pool = SimpleNamespace(provider=pool_provider)

    agent = AIAgent(
        provider=None,
        base_url=base_url,
        api_key="test-key",
        model="test-model",
        credential_pool=pool,
        skip_context_files=True,
        skip_memory=True,
        quiet_mode=True,
    )

    assert agent.provider == expected_provider
    assert agent.api_mode == expected_api_mode
    assert agent._credential_pool is pool


def test_provider_inference_is_independent_of_explicit_api_mode():
    pool = SimpleNamespace(provider="anthropic")

    agent = AIAgent(
        provider=None,
        base_url="https://api.anthropic.com",
        api_mode="chat_completions",
        api_key="test-key",
        model="test-model",
        credential_pool=pool,
        skip_context_files=True,
        skip_memory=True,
        quiet_mode=True,
    )

    assert agent.provider == "anthropic"
    assert agent.api_mode == "chat_completions"
    assert agent._credential_pool is pool


def test_url_inference_still_rejects_mismatched_credential_pool():
    pool = SimpleNamespace(provider="openai-codex")

    agent = AIAgent(
        provider=None,
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model="test-model",
        credential_pool=pool,
        skip_context_files=True,
        skip_memory=True,
        quiet_mode=True,
    )

    assert agent.provider == "anthropic"
    assert agent._credential_pool is None

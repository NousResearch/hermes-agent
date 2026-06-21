from types import SimpleNamespace

from agent.agent_init import (
    _merge_custom_provider_disable_tools,
    _merge_custom_provider_extra_body,
)


def test_custom_provider_extra_body_merges_into_request_overrides():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={"service_tier": "priority"},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1/",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides == {
        "service_tier": "priority",
        "extra_body": {
            "enable_thinking": True,
            "reasoning_effort": "high",
        },
    }


def test_custom_provider_extra_body_preserves_caller_override():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={
            "extra_body": {
                "reasoning_effort": "low",
                "caller_only": True,
            }
        },
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "caller_only": True,
    }


def test_custom_provider_extra_body_ignores_other_custom_models():
    agent = SimpleNamespace(
        provider="custom",
        model="other-model",
        base_url="https://example.test/v1",
        request_overrides={},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {"enable_thinking": True},
            }
        ],
    )

    assert agent.request_overrides == {}


def test_custom_provider_disable_tools_sets_request_override():
    agent = SimpleNamespace(
        provider="custom",
        model="sonar",
        base_url="https://api.perplexity.ai",
        request_overrides={},
    )

    _merge_custom_provider_disable_tools(
        agent,
        [
            {
                "name": "perplexity",
                "base_url": "https://api.perplexity.ai/",
                "model": "sonar",
                "disable_tools": True,
            }
        ],
    )

    assert agent.request_overrides == {"disable_tools": True}


def test_custom_provider_disable_tools_ignores_other_models():
    agent = SimpleNamespace(
        provider="custom",
        model="sonar-pro",
        base_url="https://api.perplexity.ai",
        request_overrides={},
    )

    _merge_custom_provider_disable_tools(
        agent,
        [
            {
                "name": "perplexity",
                "base_url": "https://api.perplexity.ai",
                "model": "sonar",
                "disable_tools": True,
            }
        ],
    )

    assert agent.request_overrides == {}

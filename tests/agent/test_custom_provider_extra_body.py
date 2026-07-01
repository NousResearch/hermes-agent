from types import SimpleNamespace

from agent.agent_init import (
    _configure_custom_provider_reasoning_replay,
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


def test_named_custom_provider_extra_body_matches_provider_key():
    agent = SimpleNamespace(
        provider="custom:zai-coding-plan",
        model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4",
        request_overrides={},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "provider_key": "other-provider",
                "name": "Other Provider",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": True},
            },
            {
                "provider_key": "zai-coding-plan",
                "name": "Z.AI Coding Plan",
                "base_url": "https://api.z.ai/api/coding/paas/v4/",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": False},
            },
        ],
    )

    assert agent.request_overrides == {"extra_body": {"enable_thinking": False}}


def test_custom_provider_preserve_thinking_enables_reasoning_replay():
    agent = SimpleNamespace(
        provider="custom:my-vllm",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        base_url="http://localhost:8000/v1",
        request_overrides={},
    )

    _configure_custom_provider_reasoning_replay(
        agent,
        [
            {
                "provider_key": "my-vllm",
                "name": "my-vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen3.6-35B-A3B-FP8",
                "extra_body": {
                    "chat_template_kwargs": {"preserve_thinking": True},
                },
            }
        ],
    )

    assert agent._reasoning_replay_field == "reasoning"


def test_custom_provider_explicit_reasoning_replay_field_wins():
    agent = SimpleNamespace(
        provider="custom",
        model="qwen3.6-local",
        base_url="http://localhost:8000/v1",
        request_overrides={},
    )

    _configure_custom_provider_reasoning_replay(
        agent,
        [
            {
                "name": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "qwen3.6-local",
                "preserve_thinking": True,
                "reasoning_replay_field": "reasoning_content",
            }
        ],
    )

    assert agent._reasoning_replay_field == "reasoning_content"

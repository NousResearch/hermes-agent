from types import SimpleNamespace

import pytest

from agent.agent_init import (
    _configure_custom_provider_reasoning_replay,
    _merge_custom_provider_extra_body,
)




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


def test_named_custom_provider_reasoning_replay_matches_provider_key_and_model():
    agent = SimpleNamespace(
        provider="custom:qwen-vllm",
        model="Qwen/Qwen3.8-27B",
        base_url="http://127.0.0.1:8000/v1",
        _reasoning_replay_field=None,
    )

    _configure_custom_provider_reasoning_replay(
        agent,
        [
            {
                "provider_key": "other-provider",
                "name": "Other Provider",
                "base_url": "http://127.0.0.1:8000/v1",
                "models": {"Qwen/Qwen3.8-27B": {}},
                "reasoning_replay_field": "reasoning_content",
            },
            {
                "provider_key": "qwen-vllm",
                "name": "Qwen vLLM",
                "base_url": "http://127.0.0.1:8000/v1/",
                "models": {"Qwen/Qwen3.8-27B": {}},
                "reasoning_replay_field": "reasoning",
            },
        ],
    )

    assert agent._reasoning_replay_field == "reasoning"


@pytest.mark.parametrize("mode", ["auto", "none"])
def test_custom_provider_preserves_non_carrier_replay_modes(mode):
    agent = SimpleNamespace(
        provider="custom:qwen-local",
        model="model",
        base_url="http://localhost:8080/v1",
        _reasoning_replay_field=None,
    )

    _configure_custom_provider_reasoning_replay(
        agent,
        [
            {
                "provider_key": "qwen-local",
                "base_url": "http://localhost:8080/v1",
                "model": "model",
                "reasoning_replay_field": mode,
            }
        ],
    )

    assert agent._reasoning_replay_field == mode

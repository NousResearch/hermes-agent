from types import SimpleNamespace

from agent.agent_init import _merge_custom_provider_extra_body


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


def test_custom_provider_openai_extra_body_drops_static_reasoning_effort():
    agent = SimpleNamespace(
        provider="custom",
        model="gpt-5.5",
        base_url="https://example.test/v1",
        request_overrides={"service_tier": "priority"},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "openai-compatible",
                "base_url": "https://example.test/v1/",
                "model": "gpt-5.5",
                "extra_body": {
                    "reasoning_effort": "xhigh",
                },
            }
        ],
    )

    assert agent.request_overrides == {"service_tier": "priority"}


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


def test_custom_provider_openai_semantic_model_emits_dynamic_reasoning_effort():
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "xhigh"},
        model="gpt-5.5",
    )

    assert extra_body == {}
    assert top_level == {"reasoning_effort": "xhigh"}


def test_custom_provider_unknown_model_does_not_emit_reasoning_effort():
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "xhigh"},
        model="my-local-model",
    )

    assert extra_body == {}
    assert top_level == {}


def test_custom_provider_openai_semantic_model_maps_max_to_xhigh():
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "max"},
        model="gpt-5.5",
    )

    assert extra_body == {}
    assert top_level == {"reasoning_effort": "xhigh"}

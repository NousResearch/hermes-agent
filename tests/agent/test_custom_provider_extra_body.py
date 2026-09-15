from types import SimpleNamespace

from agent.agent_init import _merge_custom_provider_extra_body




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


def test_send_session_metadata_resolves_for_matching_url():
    from agent.agent_init import _custom_provider_send_session_metadata

    assert _custom_provider_send_session_metadata(
        provider="custom",
        model="qwen3.8",
        base_url="https://gateway.internal.example.com/v1",
        custom_providers=[
            {
                "name": "my-gateway",
                "base_url": "https://gateway.internal.example.com/v1",
                "send_session_metadata": True,
            }
        ],
    ) is True


def test_send_session_metadata_off_for_other_url():
    from agent.agent_init import _custom_provider_send_session_metadata

    assert _custom_provider_send_session_metadata(
        provider="custom",
        model="qwen3.8",
        base_url="https://other.internal.example.com/v1",
        custom_providers=[
            {
                "name": "my-gateway",
                "base_url": "https://gateway.internal.example.com/v1",
                "send_session_metadata": True,
            }
        ],
    ) is False


def test_send_session_metadata_off_for_non_custom_provider():
    from agent.agent_init import _custom_provider_send_session_metadata

    assert _custom_provider_send_session_metadata(
        provider="openai",
        model="gpt-4o",
        base_url="https://gateway.internal.example.com/v1",
        custom_providers=[
            {
                "name": "my-gateway",
                "base_url": "https://gateway.internal.example.com/v1",
                "send_session_metadata": True,
            }
        ],
    ) is False


def test_send_session_metadata_named_provider_key_scopes_match():
    from agent.agent_init import _custom_provider_send_session_metadata

    assert _custom_provider_send_session_metadata(
        provider="custom:my-gateway",
        model="qwen3.8",
        base_url="https://gateway.internal.example.com/v1",
        custom_providers=[
            {
                "provider_key": "other-provider",
                "name": "Other Provider",
                "base_url": "https://gateway.internal.example.com/v1",
                "send_session_metadata": True,
            },
            {
                "provider_key": "my-gateway",
                "name": "My Gateway",
                "base_url": "https://gateway.internal.example.com/v1",
                "send_session_metadata": True,
            },
        ],
    ) is True

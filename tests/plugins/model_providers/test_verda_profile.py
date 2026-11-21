"""Verda reasoning controls reach the Chat Completions request."""

import pytest


@pytest.fixture
def verda_profile():
    import model_tools
    import providers

    profile = providers.get_provider_profile("verda")
    assert profile is not None, "Verda provider must be discovered"
    return profile


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"effort": "low"}, "low"),
        ({"effort": "high"}, "high"),
        ({"effort": "max"}, "max"),
        ({"effort": "minimal"}, "low"),
        ({"effort": "medium"}, "low"),
        ({"effort": "xhigh"}, "high"),
        ({"effort": "ultra"}, "max"),
        (None, None),
        ({}, None),
        ({"enabled": True}, None),
        ({"effort": "none"}, None),
        ({"enabled": False, "effort": "high"}, None),
        ({"effort": "future-tier"}, None),
    ],
)
def test_reasoning_effort_reaches_request(verda_profile, config, expected):
    from agent.transports.chat_completions import ChatCompletionsTransport

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="kimi-k3-1-pro",
        messages=[{"role": "user", "content": "ping"}],
        tools=None,
        provider_profile=verda_profile,
        provider_name="verda",
        reasoning_config=config,
        supports_reasoning=False,
    )

    if expected is None:
        assert "reasoning_effort" not in kwargs
    else:
        assert kwargs["reasoning_effort"] == expected

    extra_body = kwargs.get("extra_body") or {}
    assert "reasoning" not in extra_body
    assert "thinking" not in extra_body
    assert "reasoning_effort" not in extra_body


def test_unknown_model_does_not_inherit_kimi_reasoning(verda_profile):
    assert verda_profile.supported_reasoning_efforts("unknown-model") is None

    extra_body, top_level = verda_profile.build_api_kwargs_extras(
        model="unknown-model",
        reasoning_config={"enabled": True, "effort": "high"},
    )

    assert extra_body == {}
    assert top_level == {}
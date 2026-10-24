"""Target-specific reasoning is a wire-only policy, not a model/config change."""

from copy import deepcopy

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


VERTEX = "https://aiplatform.googleapis.com/v1/projects/fixture/locations/global/endpoints/openapi"
MESSAGES = [{"role": "user", "content": "fixture"}]


@pytest.mark.parametrize("provider", ["custom", "custom:fixture"])
@pytest.mark.parametrize("model", [
    "gemini-4-argon", "google/gemini-4-argon", "google/gemini-3.1-pro-preview-20261001",
    "publishers/google/models/gemini-4-argon", "google/gemini-4-argon@001",
])
@pytest.mark.parametrize("effort", ["xhigh", "ultra"])
def test_custom_gemini_request_uses_highest_accepted_effort(provider, model, effort):
    config = {"enabled": True, "effort": effort, "fixture": {"keep": True}}
    original = deepcopy(config)
    transport = ChatCompletionsTransport()
    request = transport.build_kwargs(
        model, MESSAGES, provider_profile=get_provider_profile(provider),
        base_url=VERTEX, reasoning_config=config,
    )
    assert request["reasoning_effort"] in {"minimal", "low", "medium", "high", "max"}
    assert request["reasoning_effort"] == "max"
    assert request["model"] == model
    assert config == original
    # The same stored intent remains available on the next OpenAI attempt.
    next_request = transport.build_kwargs(
        "gpt-6.1-sol", MESSAGES, provider_profile=get_provider_profile("custom"),
        base_url="https://api.openai.com/v1", reasoning_config=config,
    )
    assert next_request["reasoning_effort"] == ("xhigh" if effort == "xhigh" else "max")


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "max"])
def test_custom_gemini_preserves_supported_efforts(effort):
    request = ChatCompletionsTransport().build_kwargs(
        "google/gemini-4-argon", MESSAGES, provider_profile=get_provider_profile("custom"),
        base_url=VERTEX, reasoning_config={"enabled": True, "effort": effort},
    )
    assert request["reasoning_effort"] == effort


@pytest.mark.parametrize("model,url", [
    ("gpt-6.1-sol", VERTEX), ("fixture-model", VERTEX),
    ("gpt-6.1-sol", "https://api.openai.com/v1"),
    ("gemma-3", VERTEX), ("not-gemini-fixture", VERTEX),
])
def test_non_gemini_custom_requests_preserve_xhigh(model, url):
    request = ChatCompletionsTransport().build_kwargs(
        model, MESSAGES, provider_profile=get_provider_profile("custom"),
        base_url=url, reasoning_config={"effort": "xhigh"},
    )
    assert request["reasoning_effort"] == "xhigh"


def test_unknown_gemini_route_preserves_wire_vocabulary():
    request = ChatCompletionsTransport().build_kwargs(
        "google/gemini-4-argon", MESSAGES, provider_name="fixture-unknown",
        base_url="https://fixture.invalid/v1", supports_reasoning=True,
        reasoning_config={"effort": "xhigh"},
    )
    assert request["extra_body"]["reasoning"]["effort"] == "xhigh"


def test_legacy_vertex_gemini_request_clamps_without_a_profile():
    request = ChatCompletionsTransport().build_kwargs(
        "google/gemini-4-argon", MESSAGES, provider_name="custom:fixture",
        base_url=VERTEX, supports_reasoning=True, reasoning_config={"effort": "xhigh"},
    )
    assert request["extra_body"]["reasoning"]["effort"] == "max"


def test_builtin_gemini_retains_native_thinking_and_xhigh_decode_budget():
    config = {"enabled": True, "effort": "xhigh"}
    request = ChatCompletionsTransport().build_kwargs(
        "gemini-4-argon", MESSAGES, provider_profile=get_provider_profile("gemini"),
        base_url="https://generativelanguage.googleapis.com/v1beta", reasoning_config=config,
    )
    assert request["extra_body"]["thinking_config"] == {"includeThoughts": True, "thinkingLevel": "high"}
    assert request["extra_body"]["max_output_tokens"] == 262_144
    assert "reasoning_effort" not in request
    assert config == {"enabled": True, "effort": "xhigh"}


@pytest.mark.parametrize("provider", ["openai", "openai-codex"])
def test_openai_responses_preserves_xhigh(provider):
    from agent.transports import get_transport
    import agent.transports.codex

    transport = get_transport("codex_responses")
    assert transport is not None
    config = {"enabled": True, "effort": "xhigh"}
    request = transport.build_kwargs(
        "gpt-6.1-sol", MESSAGES, provider=provider,
        base_url="https://api.openai.com/v1", reasoning_config=config,
    )
    assert request["reasoning"]["effort"] == "xhigh"
    assert config == {"enabled": True, "effort": "xhigh"}


@pytest.mark.parametrize("config,expected", [
    (None, None), ({}, None), ({"enabled": False, "effort": "xhigh"}, "none"),
    ({"effort": "none"}, "none"), ({"effort": "fixture-bespoke"}, "fixture-bespoke"),
])
def test_custom_gemini_preserves_unset_disable_and_unknown_intent(config, expected):
    original = deepcopy(config)
    request = ChatCompletionsTransport().build_kwargs(
        "google/gemini-4-argon", MESSAGES, provider_profile=get_provider_profile("custom"),
        base_url=VERTEX, reasoning_config=config,
    )
    assert request.get("reasoning_effort") == expected
    assert config == original

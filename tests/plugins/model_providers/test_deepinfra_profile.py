"""DeepInfra profile puts the reasoning switch on the wire (#111872).

DeepInfra's OpenAI-compatible endpoint reads one top-level ``reasoning_effort`` field validated
against a gateway-wide enum (``none``..``max``; Hermes-internal ``ultra`` is rejected). The
profile is the ONLY source of that field on the transport's profile path, and the core
``_supports_reasoning_extra_body`` allowlist passes ``supports_reasoning=False`` for this host,
so the profile must emit without gating on it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def deepinfra_profile():
    import model_tools  # noqa: F401  (plugin discovery registers the profile)
    import providers

    profile = providers.get_provider_profile("deepinfra")
    assert profile is not None, "deepinfra provider profile must be registered"
    return profile


@pytest.mark.parametrize(
    "reasoning_config, expected_top_level",
    [
        ({"enabled": True, "effort": "high"}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "xhigh"}, {"reasoning_effort": "xhigh"}),  # native, never folded into max
        ({"enabled": True, "effort": "ultra"}, {"reasoning_effort": "max"}),  # Hermes-internal tier clamps
        ({"enabled": False}, {"reasoning_effort": "none"}),  # the only off switch for default-on models
        ({"enabled": True, "effort": "none"}, {"reasoning_effort": "none"}),
        (None, {}),  # nothing requested → keep DeepInfra's per-model default
        ({"enabled": True}, {}),
        ({"enabled": True, "effort": "future-tier"}, {}),  # unknown level omitted rather than 422
    ],
)
def test_profile_translates_reasoning_config_to_top_level_effort(deepinfra_profile, reasoning_config, expected_top_level):
    extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config, supports_reasoning=False, model="deepseek-ai/DeepSeek-V4.1-Flash",
    )
    assert extra_body == {}
    assert top_level == expected_top_level


def test_transport_main_turn_carries_reasoning_effort_without_capability_gate(deepinfra_profile):
    """The main turn builds through ``_build_kwargs_from_profile`` with ``supports_reasoning=False``
    (core allowlist excludes this host) — the field must still reach the request."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    build = ChatCompletionsTransport().build_kwargs
    on = build(
        model="deepseek-ai/DeepSeek-V4.1-Flash", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=deepinfra_profile, provider_name="deepinfra",
        reasoning_config={"enabled": True, "effort": "high"}, supports_reasoning=False,
    )
    off = build(
        model="zai-org/GLM-4.6", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=deepinfra_profile, provider_name="deepinfra",
        reasoning_config={"enabled": False}, supports_reasoning=False,
    )
    assert on["reasoning_effort"] == "high"
    assert off["reasoning_effort"] == "none"
    assert "reasoning" not in (on.get("extra_body") or {})

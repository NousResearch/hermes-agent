"""Xiaomi MiMo profile puts the thinking toggle and effort on the wire (#27325, #17314).

MiMo's chat-completions endpoint reads ``extra_body.thinking`` (on/off) and a top-level
``reasoning_effort`` validated against exactly none/low/medium/high — live-verified
2026-10 on api.xiaomimimo.com; any other level 400s "Param Incorrect". The profile is
the ONLY source of those fields on the transport's profile path, so without it every
Hermes effort level was a silent no-op and thinking could never be disabled.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def xiaomi_profile():
    import model_tools  # discovery side effect registers bundled provider plugins
    import providers

    profile = providers.get_provider_profile("xiaomi")
    assert profile is not None, "xiaomi provider profile must be registered"
    return profile


@pytest.mark.parametrize(
    "reasoning_config, expected_extras, expected_top_level",
    [
        ({"enabled": True, "effort": "high"}, {}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "medium"}, {}, {"reasoning_effort": "medium"}),
        ({"enabled": True, "effort": "low"}, {}, {"reasoning_effort": "low"}),
        # Out-of-vocabulary Hermes levels clamp to MiMo's three (nearest weaker level).
        ({"enabled": True, "effort": "minimal"}, {}, {"reasoning_effort": "low"}),
        ({"enabled": True, "effort": "xhigh"}, {}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "max"}, {}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "ultra"}, {}, {"reasoning_effort": "high"}),
        # Off switches: the documented thinking toggle (#27325), incl. the effort-field form.
        ({"enabled": False}, {"thinking": {"type": "disabled"}}, {}),
        ({"enabled": True, "effort": "none"}, {"thinking": {"type": "disabled"}}, {}),
        ({"enabled": True, "effort": "disabled"}, {"thinking": {"type": "disabled"}}, {}),
        # Bare effort-field forms with no ``enabled`` key at all (same #107238 family).
        ({"effort": "none"}, {"thinking": {"type": "disabled"}}, {}),
        ({"effort": "false"}, {"thinking": {"type": "disabled"}}, {}),
        ({"effort": "disabled"}, {"thinking": {"type": "disabled"}}, {}),
        # Thinking off wins over any effort level riding along.
        ({"enabled": False, "effort": "high"}, {"thinking": {"type": "disabled"}}, {}),
        # No expressible effort: the shared toggle pins the documented default (thinking on)
        # instead of leaving the wire silent; unknown levels likewise never reach the wire.
        (None, {"thinking": {"type": "enabled"}}, {}),
        ({"enabled": True}, {"thinking": {"type": "enabled"}}, {}),
        ({"enabled": True, "effort": "future-tier"}, {"thinking": {"type": "enabled"}}, {}),
    ],
)
def test_profile_translates_reasoning_config_to_thinking_and_effort(
    xiaomi_profile, reasoning_config, expected_extras, expected_top_level,
):
    extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config, supports_reasoning=False, model="mimo-v2.6-pro",
    )
    assert extra_body == expected_extras
    assert top_level == expected_top_level


@pytest.mark.parametrize("model", ["mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-omni"])
def test_retired_models_get_no_reasoning_fields(xiaomi_profile, model):
    """v2.5 and earlier are retired on the endpoint (400 "Unsupported model") and predate
    the knob — the profile must not add fields they never accepted."""
    extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}, supports_reasoning=False, model=model,
    )
    assert extra_body == {}
    assert top_level == {}


def test_transport_main_turn_carries_reasoning_without_capability_gate(xiaomi_profile):
    """The main turn builds through ``_build_kwargs_from_profile`` with ``supports_reasoning=False``
    (the core allowlist excludes this host) — the fields must still reach the request."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    build = ChatCompletionsTransport().build_kwargs
    on = build(
        model="mimo-v2.6-pro", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=xiaomi_profile, provider_name="xiaomi",
        reasoning_config={"enabled": True, "effort": "high"}, supports_reasoning=False,
    )
    off = build(
        model="mimo-v2.6-pro", messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=xiaomi_profile, provider_name="xiaomi",
        reasoning_config={"enabled": False}, supports_reasoning=False,
    )
    assert on["reasoning_effort"] == "high"
    assert "thinking" not in (on.get("extra_body") or {})
    assert off.get("extra_body", {}).get("thinking") == {"type": "disabled"}
    assert "reasoning_effort" not in off


def test_reasoning_effort_none_from_config_reaches_wire(xiaomi_profile):
    """``agent.reasoning_effort: none`` through ``parse_reasoning_effort`` and the bare
    UI effort-field spelling both land disabled on the wire (the #27363 review ask)."""
    from agent.transports.chat_completions import ChatCompletionsTransport
    from hermes_constants import parse_reasoning_effort

    build = ChatCompletionsTransport().build_kwargs
    assert parse_reasoning_effort("none") == {"enabled": False}
    for rc in (parse_reasoning_effort("none"), {"effort": "none"}):
        kw = build(
            model="mimo-v2.6-pro", messages=[{"role": "user", "content": "ping"}], tools=None,
            provider_profile=xiaomi_profile, provider_name="xiaomi",
            reasoning_config=rc, supports_reasoning=False,
        )
        assert kw.get("extra_body", {}).get("thinking") == {"type": "disabled"}
        assert "reasoning_effort" not in kw


def test_setup_picker_offers_only_mimos_three_levels(xiaomi_profile):
    """The ``hermes model`` effort step must offer exactly MiMo's levels (plus the shared
    Disable row the prompt appends), never the seven-step Hermes ladder."""
    assert xiaomi_profile.setup_reasoning_efforts is not None
    assert xiaomi_profile.setup_reasoning_efforts("mimo-v2.6-pro") == ("low", "medium", "high")
    assert xiaomi_profile.setup_reasoning_efforts("mimo-v2.6-flash") == ("low", "medium", "high")
    assert xiaomi_profile.setup_reasoning_efforts("mimo-v2.5-pro") == ()

"""Tests for agent.reasoning_effort — the canonical effort ladder + clamp.

This module is the single source of truth every transport and provider
profile uses to translate Hermes' internal effort ladder onto a wire's
supported vocabulary. The policy under test:

- supported levels pass through verbatim
- unsupported levels clamp to the nearest WEAKER supported level (never
  escalate above the ask, never invert the ladder)
- nothing weaker → weakest supported level
- "none" is never a degradation target (would silently disable thinking)
- declared overrides (vendor-documented roundings) win over nearest-weaker
- unknown supported-sets and bespoke level names pass through unchanged
"""

import pytest

from agent.reasoning_effort import (
    CODEX_GPT56_EFFORTS,
    EFFORT_LADDER,
    GLM52_EFFORTS,
    GLM52_OVERRIDES,
    KIMI_K2_EFFORTS,
    KIMI_K3_EFFORTS,
    KIMI_K3_OVERRIDES,
    OPENAI_COMPAT_WIRE_EFFORTS,
    clamp_effort,
    kimi_supported_efforts,
    requested_effort,
)
from hermes_constants import VALID_REASONING_EFFORTS


class TestLadderContract:
    def test_ladder_is_superset_of_valid_efforts(self):
        """The canonical ladder must cover every level users can configure —
        an internal level missing from the ladder would pass through clamps
        unchanged and leak to the wire (the #89503 class)."""
        for level in VALID_REASONING_EFFORTS:
            assert level in EFFORT_LADDER, level

    def test_no_declared_wire_set_contains_ultra(self):
        """ultra is internal vocabulary; every wire set must exclude it so it
        always clamps down."""
        import agent.reasoning_effort as mod

        for name in dir(mod):
            if name.endswith("_EFFORTS"):
                assert "ultra" not in getattr(mod, name), name


class TestClampEffort:
    def test_supported_levels_pass_through(self):
        for level in OPENAI_COMPAT_WIRE_EFFORTS:
            assert clamp_effort(level, OPENAI_COMPAT_WIRE_EFFORTS) == level

    def test_ultra_clamps_to_max_on_openai_wire(self):
        assert clamp_effort("ultra", OPENAI_COMPAT_WIRE_EFFORTS) == "max"

    def test_nearest_weaker_never_escalates(self):
        # xhigh against low..high → high (weaker), never max.
        assert clamp_effort("xhigh", ("low", "medium", "high", "max")) == "high"

    def test_floor_when_nothing_weaker(self):
        assert clamp_effort("minimal", ("low", "medium")) == "low"

    def test_none_is_never_a_degradation_target(self):
        # minimal against {none, low}: clamping to none would silently
        # disable thinking — must take low.
        assert clamp_effort("minimal", ("none", "low", "high")) == "low"

    def test_none_still_passes_through_when_requested(self):
        assert clamp_effort("none", ("none", "low", "high")) == "none"

    def test_unknown_supported_set_passes_through(self):
        assert clamp_effort("ultra", None) == "ultra"
        assert clamp_effort("ultra", []) == "ultra"

    def test_bespoke_level_passes_through(self):
        assert clamp_effort("turbo-9000", ("low", "high")) == "turbo-9000"

    def test_empty_effort_passes_through(self):
        assert clamp_effort(None, ("low",)) is None
        assert clamp_effort("", ("low",)) == ""

    def test_overrides_win(self):
        assert clamp_effort("medium", KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES) == "high"
        assert clamp_effort("xhigh", KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES) == "max"

    def test_override_ignored_when_target_unsupported(self):
        # An override pointing outside the supported set falls back to the
        # ladder walk instead of emitting an invalid level.
        assert clamp_effort("medium", ("low", "high"), {"medium": "max"}) == "low"

    def test_monotonic_over_full_ladder(self):
        """A stronger ask never resolves weaker than a weaker ask — for every
        declared wire set."""
        import agent.reasoning_effort as mod

        sets = [
            getattr(mod, name) for name in dir(mod) if name.endswith("_EFFORTS")
        ]
        enabled_ladder = [l for l in EFFORT_LADDER if l != "none"]
        for supported in sets:
            prev_rank = -1
            for level in enabled_ladder:
                out = clamp_effort(level, supported)
                rank = EFFORT_LADDER.index(out)
                assert rank >= prev_rank, (supported, level, out)
                prev_rank = rank


class TestKimiVocabulary:
    @pytest.mark.parametrize(
        "model",
        ["k3", "kimi-k3", "kimi-k3-cot", "moonshotai/kimi-k3", "k3-256k"],
    )
    def test_k3_slugs(self, model):
        assert kimi_supported_efforts(model) is KIMI_K3_EFFORTS

    @pytest.mark.parametrize(
        "model",
        ["kimi-k2.6", "moonshotai/kimi-k2-0905", "kimi-latest", "mk3000", None],
    )
    def test_k2_era_slugs(self, model):
        assert kimi_supported_efforts(model) is KIMI_K2_EFFORTS


class TestGlm52Vocabulary:
    def test_two_level_set(self):
        assert clamp_effort("ultra", GLM52_EFFORTS, GLM52_OVERRIDES) == "max"
        assert clamp_effort("xhigh", GLM52_EFFORTS, GLM52_OVERRIDES) == "max"
        # GLM's floor is high — weaker asks land there.
        assert clamp_effort("low", GLM52_EFFORTS, GLM52_OVERRIDES) == "high"
        assert clamp_effort("medium", GLM52_EFFORTS, GLM52_OVERRIDES) == "high"


class TestCodexVocabulary:
    def test_minimal_and_ultra(self):
        assert clamp_effort("minimal", CODEX_GPT56_EFFORTS) == "low"
        assert clamp_effort("ultra", CODEX_GPT56_EFFORTS) == "max"

    def test_per_model_max_support(self):
        """Live-verified (Aug 2026, #68365): 'max' is gpt-5.6-only — gpt-5.5
        rejects it ("Supported values are: 'none','low','medium','high',
        'xhigh'"); 'minimal' is rejected by both generations."""
        from agent.reasoning_effort import (
            CODEX_GPT56_EFFORTS,
            CODEX_LEGACY_EFFORTS,
            codex_supported_efforts,
        )

        assert codex_supported_efforts("gpt-5.6") is CODEX_GPT56_EFFORTS
        assert codex_supported_efforts("gpt-5.6-codex") is CODEX_GPT56_EFFORTS
        assert codex_supported_efforts("gpt-5.5") is CODEX_LEGACY_EFFORTS
        assert codex_supported_efforts("o5-pro") is CODEX_LEGACY_EFFORTS
        # The consequential clamps:
        assert clamp_effort("max", CODEX_GPT56_EFFORTS) == "max"
        assert clamp_effort("max", CODEX_LEGACY_EFFORTS) == "xhigh"
        assert clamp_effort("ultra", CODEX_LEGACY_EFFORTS) == "xhigh"
        assert clamp_effort("minimal", CODEX_LEGACY_EFFORTS) == "low"

    @pytest.mark.parametrize(
        "model",
        ["gpt-daybreak-blue-latest", "gpt-daybreak-blue-latest-900k"],
    )
    def test_daybreak_alias_uses_gpt56_max_support(self, model):
        """Daybreak Blue is a GPT-5.6 Sol alias, so its base and 900k picker
        slugs must preserve Sol's max reasoning level."""
        from agent.reasoning_effort import codex_supported_efforts

        supported = codex_supported_efforts(model)
        assert clamp_effort("max", supported) == "max"
        assert clamp_effort("ultra", supported) == "max"


class TestRequestedEffort:
    def test_extracts_effort(self):
        assert requested_effort({"enabled": True, "effort": "High"}) == "high"

    def test_none_when_absent_or_disabled(self):
        assert requested_effort(None) is None
        assert requested_effort({}) is None
        assert requested_effort({"enabled": False, "effort": "high"}) is None
        assert requested_effort("not-a-dict") is None
        assert requested_effort({"effort": ""}) is None


class TestRouteEffortClampTarget:
    """The level a route really sends: the provider profile's final vocabulary and declared remap."""

    @pytest.mark.parametrize("provider, model, effort, sent", [
        ("tencent-tokenhub", "hunyuan-t2", "xhigh", "high"),
        ("nebius-token-factory", "any", "max", "high"),
        ("upstage", "solar-pro3", "xhigh", "high"),
        ("kimi-coding", "k3", "medium", "high"),       # KIMI_K3_OVERRIDES
        ("kimi-coding", "k3", "xhigh", "max"),
        ("kimi-coding", "kimi-k2.6", "xhigh", "high"),
        ("deepseek", "deepseek-v4", "xhigh", "max"),
        ("zai", "glm-5.2", "medium", "high"),          # GLM-5.2 floor is high
        ("zai", "glm-5.3", "xhigh", "max"),
        ("ollama-cloud", "gpt-oss:120b", "minimal", "low"),
        ("meta-ai", "muse-spark", "max", "xhigh"),
        ("actual", "relay", "xhigh", "high"),
        ("xai", "grok-4.6", "max", "xhigh"),
        ("xai", "grok-4", "xhigh", "high"),
        ("gemini", "gemini-4-argon", "xhigh", "high"),
        ("xai-oauth", "grok-4.6", "max", "xhigh"),  # same xAI Responses wire as xai
        ("xai-oauth", "grok-4", "xhigh", "high"),
        ("google-gemini", "gemini-4-argon", "xhigh", "high"),  # gemini profile aliases
        ("google-ai-studio", "gemini-4-argon", "xhigh", "high"),
        ("opencode-zen", "x-preview-f-free", "medium", "low"),
        ("opencode-zen", "x-preview-f-free", "xhigh", "max"),
        ("openrouter", "anthropic/claude-sonnet-4", "xhigh", "xhigh"),
        ("openai-codex", "gpt-5.5", "max", "xhigh"),
    ])
    def test_sends(self, provider, model, effort, sent):
        from agent.reasoning_effort import route_effort_clamp_target
        assert clamp_effort(effort, *route_effort_clamp_target(provider, model)) == sent

    def test_every_registered_xai_id_gets_the_xai_wire(self):
        """Any provider id the registry resolves to an xAI route (API key or OAuth) clamps like xAI."""
        from agent.reasoning_effort import XAI_GROK46_EFFORTS, route_wire_efforts
        from hermes_cli.providers import ALIASES
        ids = {"xai", "xai-oauth"} | {alias for alias, canon in ALIASES.items() if canon in ("xai", "xai-oauth")}
        assert {pid: route_wire_efforts(pid, "grok-4.6") for pid in ids} == dict.fromkeys(ids, XAI_GROK46_EFFORTS)

    def test_entry_clamp_set_is_unchanged(self):
        """route_supported_efforts stays the ENTRY clamp (session info / TUI wire field still use it)."""
        from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, route_supported_efforts
        assert route_supported_efforts("tencent-tokenhub", "hunyuan-t2") == OPENAI_COMPAT_WIRE_EFFORTS

    def test_overrides_only_where_declared(self):
        from agent.reasoning_effort import KIMI_K3_OVERRIDES, route_effort_overrides
        assert route_effort_overrides("kimi", "kimi-k3") is KIMI_K3_OVERRIDES
        assert route_effort_overrides("kimi", "kimi-k2.6") is None
        assert route_effort_overrides("tencent-tokenhub", "x") is None
        assert route_effort_overrides(None, None) is None

    def test_display_label_names_the_provider_clamp(self):
        from agent.reasoning_effort import effort_display_label
        assert effort_display_label("xhigh", "tencent-tokenhub", "hunyuan-t2") == "xhigh (sends high on this route)"
        assert effort_display_label("medium", "kimi-coding", "k3") == "medium (sends high on this route)"
        assert effort_display_label("high", "kimi-coding", "k3") == "high"

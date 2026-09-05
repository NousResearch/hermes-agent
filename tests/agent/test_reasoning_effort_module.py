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
    codex_supported_efforts,
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
    def test_astra_supports_max(self):
        assert "max" in codex_supported_efforts("gpt-6-astra")

    def test_astra_rejects_none(self):
        # Astra speaks low..max (#103016) but, unlike gpt-5.6, rejects ``none``:
        # the set must not offer it, and a configured none/minimal clamps to low
        # instead of leaking to the wire and 400ing.
        from agent.reasoning_effort import CODEX_ASTRA_EFFORTS

        assert codex_supported_efforts("gpt-6-astra") is CODEX_ASTRA_EFFORTS
        assert "none" not in CODEX_ASTRA_EFFORTS
        assert clamp_effort("none", CODEX_ASTRA_EFFORTS) == "low"
        assert clamp_effort("minimal", CODEX_ASTRA_EFFORTS) == "low"

    def test_astra_whole_catalog_family_matches(self):
        # The substring must cover the pro/fast/flex combinations the catalog
        # ships (models_catalog_static) without matching on a bare "astra".
        for model in (
            "gpt-6-astra",
            "gpt-6-astra-pro",
            "openai/gpt-6-astra-pro-fast",
            "openai/gpt-6-astra-flex",
        ):
            assert "max" in codex_supported_efforts(model), model

    def test_astra_max_is_not_clamped(self):
        assert clamp_effort("max", codex_supported_efforts("gpt-6-astra")) == "max"

    def test_legacy_model_excludes_max(self):
        assert "max" not in codex_supported_efforts("gpt-5.5")

    def test_gpt_56_still_accepts_none(self):
        assert "none" in codex_supported_efforts("gpt-5.6")

    def test_minimal_and_ultra(self):
        assert clamp_effort("minimal", CODEX_GPT56_EFFORTS) == "low"
        assert clamp_effort("ultra", CODEX_GPT56_EFFORTS) == "max"

    def test_per_model_max_support(self):
        """Live-verified (Aug 2026, #68365): gpt-5.6 accepts 'max', but gpt-5.5
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


class TestRequestedEffort:
    def test_extracts_effort(self):
        assert requested_effort({"enabled": True, "effort": "High"}) == "high"

    def test_none_when_absent_or_disabled(self):
        assert requested_effort(None) is None
        assert requested_effort({}) is None
        assert requested_effort({"enabled": False, "effort": "high"}) is None
        assert requested_effort("not-a-dict") is None
        assert requested_effort({"effort": ""}) is None


class TestCodexTransportResolveReasoning:
    """End-to-end through the Codex transport's real decision point (#103556).

    ``_resolve_reasoning`` is where a stored session effort meets the wire on the
    openai-codex route: it must keep astra's ``max`` (the reported bug) and demote
    astra's unsupported ``none`` to ``low`` instead of leaking it to the wire.
    """

    def test_astra_max_reaches_the_wire(self):
        from agent.transports.codex import _resolve_reasoning

        effort, enabled = _resolve_reasoning(
            "gpt-6-astra", {"reasoning_config": {"enabled": True, "effort": "max"}})
        assert (effort, enabled) == ("max", True)

    def test_astra_none_clamps_to_low_on_the_wire(self):
        from agent.transports.codex import _resolve_reasoning

        effort, enabled = _resolve_reasoning(
            "gpt-6-astra", {"reasoning_config": {"enabled": True, "effort": "none"}})
        assert (effort, enabled) == ("low", True)

    def test_legacy_clamp_unchanged_through_transport(self):
        from agent.transports.codex import _resolve_reasoning

        effort, enabled = _resolve_reasoning(
            "gpt-5.5", {"reasoning_config": {"enabled": True, "effort": "max"}})
        assert (effort, enabled) == ("xhigh", True)

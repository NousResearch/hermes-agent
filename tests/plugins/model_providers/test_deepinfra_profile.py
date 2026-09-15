"""Unit tests for the DeepInfra provider profile's reasoning-effort wiring.

DeepInfra's ``/v1/openai/chat/completions`` endpoint takes top-level
``reasoning_effort``, validated against one gateway-wide enum:
``none``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``, ``max``.
That is ``hermes_constants.VALID_REASONING_EFFORTS`` minus ``ultra``.

The profile is the *only* thing that can put a reasoning field on the wire
for DeepInfra: the core ``_supports_reasoning_extra_body()`` allowlist in
``run_agent.py`` has no ``api.deepinfra.com`` branch, and the registered
profile means the transport never reaches its legacy ``extra_body.reasoning``
block. These tests pin that contract.
"""

from __future__ import annotations

import pytest

from hermes_constants import VALID_REASONING_EFFORTS


@pytest.fixture
def deepinfra_profile():
    """Resolve the registered DeepInfra profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone replaces the registered class with a plain ``ProviderProfile``,
    every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the DeepInfra profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("deepinfra")
    assert profile is not None, "deepinfra provider profile must be registered"
    return profile


# Values DeepInfra's request schema accepts for ``reasoning_effort``.
ACCEPTED_ON_WIRE = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


class TestDeepInfraReasoningEffort:
    """``build_api_kwargs_extras`` emits correct top-level ``reasoning_effort``."""

    # ── no request → no field (preserve the per-model default) ─────

    @pytest.mark.parametrize("cfg", [None, {}, "high", 42, []])
    def test_absent_or_non_dict_config_emits_nothing(self, deepinfra_profile, cfg):
        """DeepInfra's default thinking mode is per-model, so when the user
        asked for nothing we must send nothing rather than pick a side."""
        extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config=cfg,
        )
        assert extra_body == {}
        assert top_level == {}

    # ── standard efforts pass through unchanged ────────────────────

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_standard_efforts_pass_through(self, deepinfra_profile, effort):
        extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": effort}

    def test_xhigh_is_not_folded_into_max(self, deepinfra_profile):
        """``xhigh`` is native on DeepInfra — unlike Ollama Cloud, where it is
        mapped to ``max``. Guards against someone 'helpfully' copying that."""
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )
        assert top_level == {"reasoning_effort": "xhigh"}

    # ── ultra is the one Hermes value DeepInfra rejects ────────────

    @pytest.mark.parametrize("effort", ["ultra", "ULTRA", "  Ultra  "])
    def test_ultra_degrades_to_max(self, deepinfra_profile, effort):
        """``ultra`` 422s on DeepInfra's request schema; degrade rather than
        fail the turn."""
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert top_level == {"reasoning_effort": "max"}

    def test_ultra_is_the_only_rewritten_effort(self, deepinfra_profile):
        """Every other Hermes effort reaches the wire verbatim."""
        for effort in VALID_REASONING_EFFORTS:
            _, top_level = deepinfra_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
            )
            emitted = top_level.get("reasoning_effort")
            if effort == "ultra":
                assert emitted == "max"
            else:
                assert emitted == effort, f"{effort} was rewritten to {emitted}"

    # ── the off switch ─────────────────────────────────────────────

    def test_disabled_emits_explicit_none(self, deepinfra_profile):
        """GLM-4.6 and Qwen3-*-Thinking default to thinking ON, so omitting
        the field leaves it on. Only ``reasoning_effort: "none"`` turns it
        off — this is the user-visible off switch."""
        extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "none"}

    @pytest.mark.parametrize("effort", ["none", "false", "disabled", "NONE"])
    def test_off_effort_strings_emit_none(self, deepinfra_profile, effort):
        """``parse_reasoning_effort`` normally converts these to
        ``{"enabled": False}`` before the profile sees them; handled
        defensively for the paths that bypass it."""
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert top_level == {"reasoning_effort": "none"}

    # ── normalization ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        "raw,expected",
        [("  High  ", "high"), ("MEDIUM", "medium"), ("Low", "low"), ("XHigh", "xhigh")],
    )
    def test_case_and_whitespace_normalized(self, deepinfra_profile, raw, expected):
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": raw},
        )
        assert top_level == {"reasoning_effort": expected}

    # ── unknown values fail soft ───────────────────────────────────

    @pytest.mark.parametrize("effort", ["", "   ", "banana", "extreme", None])
    def test_unrecognized_effort_omits_field(self, deepinfra_profile, effort):
        """Sending an out-of-enum value is a hard HTTP 422 that kills the whole
        turn, so omit and let the model default instead."""
        extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert extra_body == {}
        assert top_level == {}


class TestDeepInfraNeverSendsAnInvalidEffort:
    """The drift guard: whatever we emit must be in DeepInfra's enum."""

    def test_every_hermes_effort_maps_into_the_accepted_set(self, deepinfra_profile):
        """If Hermes gains a new effort level, this fails loudly here rather
        than as a 422 on every DeepInfra turn in production."""
        for effort in VALID_REASONING_EFFORTS:
            _, top_level = deepinfra_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
            )
            if "reasoning_effort" in top_level:
                assert top_level["reasoning_effort"] in ACCEPTED_ON_WIRE, (
                    f"effort {effort!r} produced "
                    f"{top_level['reasoning_effort']!r}, which DeepInfra 422s"
                )


class TestDeepInfraIgnoresSupportsReasoning:
    """The regression that would silently reintroduce the original bug."""

    def test_emits_even_when_supports_reasoning_is_false(self, deepinfra_profile):
        """``run_agent.py``'s ``_supports_reasoning_extra_body()`` allowlist has
        no ``api.deepinfra.com`` branch, so the transport always passes
        ``supports_reasoning=False`` here. Early-returning on it — as the
        sibling ``ollama-cloud`` profile does — would make this method a
        permanent no-op."""
        extra_body, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=False,
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    def test_off_switch_works_when_supports_reasoning_is_false(self, deepinfra_profile):
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=False,
        )
        assert top_level == {"reasoning_effort": "none"}

    def test_accepts_full_transport_context_without_error(self, deepinfra_profile):
        """The transport passes several kwargs the profile doesn't read; they
        must be absorbed by ``**ctx`` rather than raising ``TypeError``."""
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            supports_reasoning=False,
            qwen_session_metadata=None,
            model="deepseek-ai/DeepSeek-V4.1-Flash",
            base_url="https://api.deepinfra.com/v1/openai",
            ollama_num_ctx=None,
            session_id="abc123",
        )
        assert top_level == {"reasoning_effort": "medium"}


class TestDeepInfraIsModelAgnostic:
    """No model-name matching and no catalog-tag gating, so new DeepInfra
    model releases work on day one without a Hermes change."""

    @pytest.mark.parametrize(
        "model",
        [
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "deepseek-ai/DeepSeek-V4-Pro",  # produces reasoning with no `reasoning` tag
            "zai-org/GLM-4.6",
            "Qwen/Qwen3-235B-A22B-Thinking-2507",  # absent from the filtered catalog
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # no reasoning at all; accepts the field
            "some-vendor/Model-Released-Next-Month",
            "",
            None,
        ],
    )
    def test_same_output_for_every_model(self, deepinfra_profile, model):
        _, top_level = deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "high"}


class TestDeepInfraExtraBodyStaysEmpty:
    """DeepInfra reads top-level ``reasoning_effort`` only.

    The direct-DeepSeek profile also emits ``extra_body["thinking"]``; that is
    inert here (verified live) and must not be copied over.
    """

    @pytest.mark.parametrize(
        "cfg",
        [
            None,
            {},
            {"enabled": False},
            {"enabled": True, "effort": "high"},
            {"enabled": True, "effort": "ultra"},
            {"enabled": True, "effort": "banana"},
        ],
    )
    def test_extra_body_is_always_empty(self, deepinfra_profile, cfg):
        extra_body, _ = deepinfra_profile.build_api_kwargs_extras(reasoning_config=cfg)
        assert extra_body == {}

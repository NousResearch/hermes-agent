"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
  - disabled            → extra_body.think = False
  - enabled + effort    → top-level reasoning_effort (native OpenAI-compat
                          format GLM/ARK expect), passed through verbatim
                          including ``max``/``xhigh``
  - enabled + no effort → nothing emitted (endpoint's server default applies)
  - ollama_num_ctx      → extra_body.options.num_ctx, orthogonal to reasoning
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom profile via the global registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    ``custom`` profile. Going through ``get_provider_profile`` keeps the test
    honest — if the registered class is ever downgraded to a plain
    ``ProviderProfile``, the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """Unset reasoning → omit everything so the endpoint's default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_think_false(self, custom_profile):
        """enabled=False → extra_body.think = False (Ollama thinking-off flag)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → think=False, no effort."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        """enabled + effort → TOP-LEVEL reasoning_effort, passed through verbatim.

        GLM-5.2/ARK and OpenAI-compatible reasoning APIs read reasoning_effort
        as a top-level string, not nested in extra_body. ``max`` is GLM's
        native deep-reasoning level and must survive.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb

    def test_enabled_without_effort_emits_nothing(self, custom_profile):
        """enabled but no effort → omit; do NOT force a level the user didn't pick."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_does_not_force_think_true_on_enable(self, custom_profile):
        """We must never send think=True on enable — it's Ollama-only and
        would 400 on GLM/vLLM endpoints that don't recognize it."""
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert eb.get("think") is not True


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}

    def test_num_ctx_with_effort(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_num_ctx=8192,
            model="qwen3",
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {"reasoning_effort": "high"}


# ---------------------------------------------------------------------------
# Model-aware reasoning gating (#59660)
# ---------------------------------------------------------------------------


class TestCustomReasoningModelAware:
    """``reasoning_effort`` must not be forwarded to non-reasoning models.

    Before #59660, ``CustomProfile.build_api_kwargs_extras`` unconditionally
    set ``top_level['reasoning_effort']`` whenever the user had a non-empty
    ``reasoning_config['effort']``, regardless of whether the actual model
    being called supports reasoning/thinking. This caused silent HTTP 400s
    on cross-provider fallback to local Ollama plain models (e.g. llama3.1
    8b) whose API rejects the field outright.

    Fix: gate the ``reasoning_effort`` emission on a per-model allowlist
    (mirroring ``ZaiProfile._model_supports_thinking``). Models NOT in the
    allowlist must produce an empty ``top_level`` dict so the endpoint's
    server default applies, rather than crashing on the unknown field.
    """

    @pytest.mark.parametrize(
        "non_reasoning_model",
        [
            "llama3.1-8b-64k",      # the canonical repro from #59660
            "llama3.2-3b",         # plain non-reasoning llama
            "mistral-7b",          # plain non-reasoning mistral
            "qwen2.5-7b",          # plain Qwen 2.5 (NOT Qwen3, which is reasoning)
            "phi-3-mini-4k",       # plain non-reasoning phi
            "gemma-2-9b",          # plain non-reasoning gemma
        ],
    )
    def test_non_reasoning_model_omits_reasoning_effort(
        self, custom_profile, non_reasoning_model
    ):
        """Non-reasoning models must not receive ``reasoning_effort`` in top_level.

        The user's primary/interactive model is usually a cloud reasoning-
        capable model via OpenRouter or Nous, where ``reasoning_effort`` is
        accepted. On cross-provider fallback to a local Ollama plain model
        (e.g. ``llama3.1-8b-64k``), the field 400s the request. The
        CustomProfile must check the actual model and only forward the
        field when the model is known to support it.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=non_reasoning_model,
        )
        assert "reasoning_effort" not in tl, (
            f"Non-reasoning model {non_reasoning_model!r} got reasoning_effort "
            f"in top_level ({tl!r}) — will 400 the request (#59660)"
        )
        # ``think=False`` on disable is still allowed since it's the
        # explicit "turn off thinking" hint, not an effort level. So we
        # only assert the top-level ``reasoning_effort`` is absent.

    def test_known_reasoning_model_still_passes_through(self, custom_profile):
        """Regression guard: GLM-5.2 (a known reasoning model) must still
        receive the ``reasoning_effort`` field as before.

        Models in the per-profile allowlist keep their previous wire
        shape. This test would have caught a fix that over-corrects by
        dropping the field for every model.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="glm-5.2",
        )
        assert tl == {"reasoning_effort": "max"}

    @pytest.mark.parametrize(
        "reasoning_model",
        [
            "gemma-3-9b",               # Google Gemma 3 (reasoning-capable)
            "gemma-4-26b-a4b-it",       # Gemma 4 (per user-reported repro)
            "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M",  # full HF repo id w/ tag
            "ollama:gemma-3-27b",        # ollama-prefixed variant
        ],
    )
    def test_gemma_3_and_gemma_4_are_reasoning_capable(
        self, custom_profile, reasoning_model
    ):
        """Gemma 3 and Gemma 4 must be allowed through.

        Added per PR #59678 review comment from @nitinthewiz who reported
        Gemma-3 and Gemma-4 supporting thinking via ollama. Without these
        entries the user's reasoning-effort preference would be silently
        dropped on those models — a regression vs. the pre-fix behavior
        (where the field was unconditionally sent, which 400'd instead).
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model=reasoning_model,
        )
        assert tl == {"reasoning_effort": "medium"}

    def test_unknown_model_defaults_to_omitting_reasoning_effort(self, custom_profile):
        """An unknown model name (custom OpenAI-compatible endpoint) must
        default to NOT sending ``reasoning_effort`` — the safe choice when
        we don't know whether the model supports it.

        Better to silently omit the field than to crash on a 400. The
        user's endpoint will just not receive the effort hint, which is a
        graceful degradation.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="some-unknown-custom-model-7b",
        )
        assert "reasoning_effort" not in tl

    def test_non_reasoning_model_disable_still_sends_think_false(self, custom_profile):
        """``reasoning_config={'enabled': False}`` is the explicit turn-off
        signal. It must still emit ``extra_body.think = False`` even for
        non-reasoning models — the Ollama convention treats ``think=False``
        as a no-op for plain models (they have no thinking to disable) but
        it's also a valid query parameter that the server accepts.

        This guards against an over-correction where the fix accidentally
        drops the disable case for non-reasoning models.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="llama3.1-8b-64k",
        )
        assert eb == {"think": False}
        assert "reasoning_effort" not in tl

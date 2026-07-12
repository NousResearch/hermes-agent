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
                          including ``max``/``xhigh`` — except ``xhigh`` is
                          clamped to ``high`` for the Ollama lane (port 11434),
                          which rejects the superset level (TestOllamaXhighClamp)
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


class TestOllamaXhighClamp:
    """``xhigh`` is clamped to ``high`` for the Ollama lane only (port 11434).

    Ollama's OpenAI-compatible endpoint validates reasoning_effort against
    {high, medium, low, max, none} and 400s on Hermes' ``xhigh`` superset
    level (agent.log.1 2026-07-10). The clamp is keyed on Ollama's signature
    port so GLM-5.2/ARK, vLLM, and llama.cpp keep the verbatim passthrough
    pinned by TestCustomReasoningWireShape — the free local last-resort
    fallback survives instead of dying on a total-outage day.
    """

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://127.0.0.1:11434/v1",
            "http://localhost:11434/v1",
            "http://100.85.75.123:11434/v1",  # ollama reached over Tailscale
        ],
    )
    def test_xhigh_clamped_to_high_for_ollama(self, custom_profile, base_url):
        """xhigh + ollama endpoint (:11434) → high, regardless of host form."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            base_url=base_url,
            model="qwen3:14b",
        )
        assert tl == {"reasoning_effort": "high"}
        assert "reasoning_effort" not in eb

    def test_xhigh_passthrough_for_non_ollama_custom(self, custom_profile):
        """xhigh on a non-ollama custom endpoint (GLM/ARK) is NOT clamped —
        the verbatim-passthrough contract for those backends is preserved."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="glm-5.2",
        )
        assert tl == {"reasoning_effort": "xhigh"}

    def test_xhigh_passthrough_when_base_url_absent(self, custom_profile):
        """No base_url (ollama can't be identified) → passthrough, not clamp."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="glm-5.2",
        )
        assert tl == {"reasoning_effort": "xhigh"}

    def test_max_not_clamped_for_ollama(self, custom_profile):
        """``max`` is a valid ollama level — only xhigh is clamped."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3:14b",
        )
        assert tl == {"reasoning_effort": "max"}

    def test_high_unchanged_for_ollama(self, custom_profile):
        """high + ollama is a no-op (already a valid level)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3:14b",
        )
        assert tl == {"reasoning_effort": "high"}

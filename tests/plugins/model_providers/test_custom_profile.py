"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at -- local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
  - disabled            -> extra_body.think = False (default), or a
                           per-model override via thinking_field/thinking_subkey
  - enabled + effort    -> top-level reasoning_effort (native OpenAI-compat
                           format GLM/ARK expect), passed through verbatim
                           including ``max``/``xhigh``
  - enabled + no effort -> nothing emitted (endpoint's server default applies)
  - ollama_num_ctx      -> extra_body.options.num_ctx, orthogonal to reasoning
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
        """enabled=False → reasoning_effort='none' top-level + think=False.

        Both fields are required: Ollama's /v1/chat/completions silently
        ignores extra_body.think (only /api/chat honours it — ollama#14820)
        but respects top-level reasoning_effort (#25758). think=False stays
        for proxies and the native /api/chat path.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → same dual emission."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

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


class TestCustomThinkingFieldOverride:
    """Per-model thinking_field/thinking_subkey config overrides."""

    def test_thinking_field_with_subkey(self, custom_profile, monkeypatch):
        """Model with thinking_field + thinking_subkey emits nested structure."""
        monkeypatch.setattr(
            "hermes_cli.config.get_custom_provider_thinking_field",
            lambda model, base_url, **kw: {"field": "chat_template_kwargs", "subkey": "enable_thinking"},
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="Qwen3.6-35B-A3B-bf16",
            base_url="http://127.0.0.1:8000/v1",
        )
        assert eb == {"chat_template_kwargs": {"enable_thinking": False}}
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    def test_thinking_field_without_subkey(self, custom_profile, monkeypatch):
        """Model with thinking_field but no subkey emits a flat boolean."""
        monkeypatch.setattr(
            "hermes_cli.config.get_custom_provider_thinking_field",
            lambda model, base_url, **kw: {"field": "disable_reasoning"},
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"},
            model="some-model",
            base_url="http://localhost:11434/v1",
        )
        assert eb == {"disable_reasoning": False}
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    def test_no_override_falls_back_to_think(self, custom_profile, monkeypatch):
        """When no per-model override exists, the default think=False is used."""
        monkeypatch.setattr(
            "hermes_cli.config.get_custom_provider_thinking_field",
            lambda model, base_url, **kw: None,
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url="http://localhost:11434/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_missing_model_or_base_url_falls_back(self, custom_profile):
        """Without model/base_url the lookup is skipped, default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}


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

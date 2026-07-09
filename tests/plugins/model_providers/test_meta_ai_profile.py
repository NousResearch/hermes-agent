"""Unit tests for the Meta Model API (Muse Spark) provider profile.

Meta's OpenAI-compatible ``/v1/chat/completions`` endpoint accepts
top-level ``reasoning_effort`` with values ``minimal``, ``low``,
``medium``, ``high``, and ``xhigh``.  Hermes maps ``max`` → ``xhigh``
and never emits ``none`` (Meta returns HTTP 400 for that value).

When reasoning is disabled (``enabled: false``) or effort is ``none``,
the profile emits ``reasoning_effort=minimal`` so the request stays
valid while staying at the lowest thinking tier.  For Responses API
mode (if user forces api_mode override), also emits nested
``reasoning.effort`` in extra_body.

Empty / missing effort omits the field so the model uses its default.
Vision: Muse Spark supports image, video, PDF in user messages.

Ref: https://dev.meta.ai/docs/features/reasoning/
     https://dev.meta.ai/docs/getting-started/models/
"""

from __future__ import annotations

import pytest


@pytest.fixture
def meta_ai_profile():
    """Resolve the registered Meta Model API profile.

    Going through ``providers.get_provider_profile`` keeps the test
    honest — if someone replaces the registered class with a plain
    ``ProviderProfile``, every reasoning assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Meta AI profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("meta-ai")
    assert profile is not None, "meta-ai provider profile must be registered"
    return profile


class TestMetaAiProfileBasics:
    """Identity / catalog fields on the Meta Model API profile."""

    def test_name_and_aliases(self, meta_ai_profile):
        assert meta_ai_profile.name == "meta-ai"
        for alias in ("meta", "muse", "llama-api", "model-api"):
            assert alias in meta_ai_profile.aliases

    def test_display_metadata(self, meta_ai_profile):
        assert meta_ai_profile.display_name == "Meta Model API"
        assert "Muse Spark" in meta_ai_profile.description
        assert meta_ai_profile.signup_url == "https://dev.meta.ai/"
        assert meta_ai_profile.base_url == "https://api.meta.ai/v1"
        assert meta_ai_profile.auth_type == "api_key"
        assert meta_ai_profile.api_mode == "chat_completions"

    def test_env_vars_priority(self, meta_ai_profile):
        assert meta_ai_profile.env_vars == (
            "MODEL_API_KEY",
            "META_API_KEY",
            "META_MODEL_API_KEY",
            "META_BASE_URL",
        )

    def test_fallback_and_aux_model(self, meta_ai_profile):
        assert meta_ai_profile.default_aux_model == "muse-spark-1.1"
        assert "muse-spark-1.1" in meta_ai_profile.fallback_models

    def test_hostname(self, meta_ai_profile):
        assert meta_ai_profile.get_hostname() == "api.meta.ai"

    def test_supports_vision(self, meta_ai_profile):
        # Per https://dev.meta.ai/docs/getting-started/models/ :
        # Input modalities: Text, image, video, PDF
        assert meta_ai_profile.supports_vision is True


class TestMetaAiReasoningEffort:
    """``build_api_kwargs_extras`` emits correct reasoning effort.

    Hermes uses Chat Completions by default (top-level reasoning_effort).
    When Responses API mode is forced via api_mode context, extra_body also
    carries reasoning.effort.
    """

    # ── standard levels pass through ──────────────────────────

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
    def test_standard_efforts_pass_through(self, meta_ai_profile, effort):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        # Chat Completions default: top-level only, extra_body empty
        assert top_level == {"reasoning_effort": effort}
        assert extra_body == {}

    # ── max → xhigh ───────────────────────────────────────────

    @pytest.mark.parametrize("effort", ["max", "MAX", "  Max  "])
    def test_max_normalizes_to_xhigh(self, meta_ai_profile, effort):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert top_level == {"reasoning_effort": "xhigh"}
        assert extra_body == {}

    # ── disabled / none → minimal (never emit none) ───────────

    def test_explicitly_disabled_emits_minimal(self, meta_ai_profile):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
        )
        assert top_level == {"reasoning_effort": "minimal"}
        # Disabled also populates extra_body for Responses fallback
        assert extra_body == {"reasoning": {"effort": "minimal"}}

    def test_disabled_ignores_effort_field(self, meta_ai_profile):
        """Even if effort is high, disabled forces minimal."""
        _, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
        )
        assert top_level == {"reasoning_effort": "minimal"}

    def test_none_effort_emits_minimal(self, meta_ai_profile):
        """Meta returns HTTP 400 for reasoning_effort=none — never emit it."""
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
        )
        assert top_level == {"reasoning_effort": "minimal"}
        assert top_level.get("reasoning_effort") != "none"
        assert extra_body == {}

    # ── empty / missing effort → omit ─────────────────────────

    def test_no_reasoning_config_emits_nothing(self, meta_ai_profile):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config=None,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_empty_effort_emits_nothing(self, meta_ai_profile):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""},
        )
        assert extra_body == {}
        assert top_level == {}

    def test_no_effort_key_emits_nothing(self, meta_ai_profile):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True},
        )
        assert extra_body == {}
        assert top_level == {}

    # ── case / whitespace normalization ───────────────────────

    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("  HIGH  ", "high"),
            ("XHigh", "xhigh"),
            ("Minimal", "minimal"),
        ],
    )
    def test_case_and_whitespace_normalized(self, meta_ai_profile, effort, expected):
        _, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert top_level == {"reasoning_effort": expected}

    # ── Responses API mode also populates extra_body ──────────

    def test_responses_api_mode_populates_extra_body(self, meta_ai_profile):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            api_mode="codex_responses",
        )
        assert top_level == {"reasoning_effort": "high"}
        assert extra_body == {"reasoning": {"effort": "high"}}

    def test_chat_completions_mode_leaves_extra_body_empty(self, meta_ai_profile):
        extra_body, top_level = meta_ai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            api_mode="chat_completions",
        )
        assert top_level == {"reasoning_effort": "high"}
        assert extra_body == {}


class TestMetaAiFullKwargsIntegration:
    """End-to-end: transport kwargs include Meta's reasoning_effort contract."""

    def test_full_kwargs_with_max(self, meta_ai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="muse-spark-1.1",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=meta_ai_profile,
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://api.meta.ai/v1",
            provider_name="meta-ai",
        )
        assert kwargs["model"] == "muse-spark-1.1"
        assert kwargs["reasoning_effort"] == "xhigh"
        # Chat completions path should NOT have extra_body.reasoning
        assert "reasoning" not in kwargs.get("extra_body", {})

    def test_full_kwargs_with_disabled(self, meta_ai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="muse-spark-1.1",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=meta_ai_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.meta.ai/v1",
            provider_name="meta-ai",
        )
        assert kwargs["reasoning_effort"] == "minimal"

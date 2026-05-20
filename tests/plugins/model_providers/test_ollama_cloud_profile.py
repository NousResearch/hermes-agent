"""Unit tests for the Ollama Cloud provider profile's reasoning-effort wiring.

Ollama Cloud's ``/v1/chat/completions`` endpoint supports top-level
``reasoning_effort`` with values ``none``, ``low``, ``medium``, ``high``,
and (undocumented but empirically confirmed) ``max``.  The profile maps
Hermes's ``xhigh`` → ``max`` to unlock DeepSeek V4's "Max thinking" tier
and passes the standard levels through unchanged.

These tests pin the profile's wire-shape contract so Ollama Cloud
requests carry the correct ``reasoning_effort`` field.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def ollama_cloud_profile():
    """Resolve the registered Ollama Cloud profile.

    Going through ``providers.get_provider_profile`` keeps the test
    honest — if someone replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Ollama Cloud profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("ollama-cloud")
    assert profile is not None, "ollama-cloud provider profile must be registered"
    return profile


class TestOllamaCloudReasoningEffort:
    """``build_api_kwargs_extras`` emits correct top-level ``reasoning_effort``."""

    # ── xhigh / max → max ──────────────────────────────────────────

    @pytest.mark.parametrize("effort", ["xhigh", "max", "MAX", "  Max  "])
    def test_xhigh_and_max_normalize_to_max(self, ollama_cloud_profile, effort):
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "max"}

    # ── low / medium / high pass through ───────────────────────────

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_standard_efforts_pass_through(self, ollama_cloud_profile, effort):
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert top_level == {"reasoning_effort": effort}

    # ── disabled → no reasoning_effort emitted ─────────────────────

    def test_explicitly_disabled_emits_nothing(self, ollama_cloud_profile):
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
        )
        assert extra_body == {}
        assert top_level == {}

    def test_disabled_ignores_effort_field(self, ollama_cloud_profile):
        """Effort silently dropped when thinking is off."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
        )
        assert top_level == {}

    # ── none effort → no reasoning_effort ──────────────────────────

    def test_none_effort_emits_nothing(self, ollama_cloud_profile):
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
        )
        assert extra_body == {}
        assert top_level == {}

    # ── missing / empty effort → let model default ─────────────────

    def test_no_reasoning_config_emits_nothing(self, ollama_cloud_profile):
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config=None,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_empty_effort_emits_nothing(self, ollama_cloud_profile):
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""},
        )
        assert top_level == {}

    def test_no_effort_key_emits_nothing(self, ollama_cloud_profile):
        """When effort key is absent, let the model use its default."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True},
        )
        assert top_level == {}

    # ── unknown effort → forwarded as-is ───────────────────────────

    def test_unknown_effort_forwarded(self, ollama_cloud_profile):
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
        )
        assert top_level == {"reasoning_effort": "ultra"}


class TestOllamaCloudFullKwargsIntegration:
    """End-to-end: the transport's full kwargs include reasoning_effort."""

    def test_full_kwargs_with_xhigh(self, ollama_cloud_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-v4-pro:cloud",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=ollama_cloud_profile,
            reasoning_config={"enabled": True, "effort": "xhigh"},
            base_url="https://ollama.com/v1",
            provider_name="ollama-cloud",
        )
        assert kwargs["model"] == "deepseek-v4-pro:cloud"
        assert kwargs["reasoning_effort"] == "max"
        # No extra_body — Ollama Cloud uses top-level reasoning_effort
        assert "extra_body" not in kwargs or "reasoning" not in kwargs.get("extra_body", {})

    def test_full_kwargs_with_disabled(self, ollama_cloud_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-v4-pro:cloud",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=ollama_cloud_profile,
            reasoning_config={"enabled": False},
            base_url="https://ollama.com/v1",
            provider_name="ollama-cloud",
        )
        assert "reasoning_effort" not in kwargs


class TestOllamaCloudAuxModel:
    """Ollama Cloud aux model is set on the profile."""

    def test_profile_advertises_aux_model(self, ollama_cloud_profile):
        assert ollama_cloud_profile.default_aux_model == "nemotron-3-nano:30b"

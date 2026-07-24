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
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "max"}

    # ── low / medium / high pass through ───────────────────────────

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_standard_efforts_pass_through(self, ollama_cloud_profile, effort):
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert top_level == {"reasoning_effort": effort}

    # ── disabled → reasoning_effort:"none" (the only working off switch) ──

    def test_explicitly_disabled_sends_none(self, ollama_cloud_profile):
        """Ollama Cloud defaults to thinking ON and ignores extra_body.thinking,
        so disabling requires top-level reasoning_effort:"none" (verified live);
        omitting the field would leave thinking on."""
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": False},
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "none"}

    def test_disabled_ignores_effort_field(self, ollama_cloud_profile):
        """Effort is overridden by the disable off switch when thinking is off."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": False, "effort": "high"},
        )
        assert top_level == {"reasoning_effort": "none"}

    # ── none effort → reasoning_effort:"none" ──────────────────────

    def test_none_effort_sends_none(self, ollama_cloud_profile):
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "none"},
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "none"}

    # ── missing / empty effort → let model default ─────────────────

    def test_no_reasoning_config_emits_nothing(self, ollama_cloud_profile):
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config=None,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_empty_effort_emits_nothing(self, ollama_cloud_profile):
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": ""},
        )
        assert top_level == {}

    def test_no_effort_key_emits_nothing(self, ollama_cloud_profile):
        """When effort key is absent, let the model use its default."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True},
        )
        assert top_level == {}

    # ── unknown / minimal effort → omitted (server default) ────────

    def test_unknown_effort_omitted(self, ollama_cloud_profile):
        """Unrecognized effort is omitted, not forwarded verbatim, so the
        model applies its own default. Matches the sibling deepseek profile,
        which targets the same backend."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "future-tier"},
        )
        assert top_level == {}

    def test_minimal_effort_omitted(self, ollama_cloud_profile):
        """``minimal`` is a real Hermes effort level but is not documented for
        Ollama Cloud's /v1/chat/completions, so it is omitted rather than sent
        verbatim (which could trigger a 400)."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "minimal"},
        )
        assert top_level == {}


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
            supports_reasoning=True,
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
            supports_reasoning=True,
        )
        # Disabling requires the explicit off switch — Ollama Cloud defaults to
        # thinking ON, so omitting reasoning_effort would NOT disable it.
        assert kwargs["reasoning_effort"] == "none"


class TestOllamaCloudCapabilityGating:
    """reasoning_effort is gated on the model's thinking capability."""

    def test_non_thinking_model_emits_nothing(self, ollama_cloud_profile):
        """A model that doesn't support thinking (supports_reasoning=False)
        gets no reasoning_effort, even when an effort is requested — Ollama
        resolves thinking capability from /api/show, and we don't send a
        meaningless field to e.g. gemma3 / qwen3-coder."""
        extra_body, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            supports_reasoning=False,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_non_thinking_model_ignores_disable(self, ollama_cloud_profile):
        """Even a disable request is a no-op for a non-thinking model."""
        _, top_level = ollama_cloud_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=False,
        )
        assert top_level == {}


class TestOllamaModelSupportsThinking:
    """The /api/show capability probe used to resolve supports_reasoning."""

    def _patch_show(self, monkeypatch, *, status=200, capabilities=None, raise_exc=None):
        import httpx

        class _Resp:
            status_code = status

            def json(self):
                return {"capabilities": capabilities} if capabilities is not None else {}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, *a, **k):
                if raise_exc:
                    raise raise_exc
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)

    def test_thinking_capability_true(self, monkeypatch):
        from hermes_cli.models import ollama_model_supports_thinking

        self._patch_show(monkeypatch, capabilities=["completion", "tools", "thinking"])
        assert (
            ollama_model_supports_thinking(
                "deepseek-v4-pro", "https://ollama.com/v1", "key"
            )
            is True
        )

    def test_no_thinking_capability_false(self, monkeypatch):
        from hermes_cli.models import ollama_model_supports_thinking

        self._patch_show(monkeypatch, capabilities=["completion", "vision"])
        assert (
            ollama_model_supports_thinking("gemma3:27b", "https://ollama.com/v1", "key")
            is False
        )

    def test_probe_failure_returns_none(self, monkeypatch):
        from hermes_cli.models import ollama_model_supports_thinking

        self._patch_show(monkeypatch, status=404)
        assert (
            ollama_model_supports_thinking("x", "https://ollama.com/v1", "key") is None
        )

    def test_exception_returns_none(self, monkeypatch):
        from hermes_cli.models import ollama_model_supports_thinking

        self._patch_show(monkeypatch, raise_exc=RuntimeError("boom"))
        assert (
            ollama_model_supports_thinking("x", "https://ollama.com/v1", "key") is None
        )


class TestOllamaCloudAuxModel:
    """Ollama Cloud aux model is set on the profile."""

    def test_profile_advertises_aux_model(self, ollama_cloud_profile):
        assert ollama_cloud_profile.default_aux_model == "nemotron-3-nano:30b"

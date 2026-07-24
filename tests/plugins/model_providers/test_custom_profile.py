"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama as well as arbitrary hosted or self-hosted relays.
Ollama's reasoning controls must never be forwarded to those other endpoints.

These tests pin the wire-shape contract:
  - disabled verified Ollama → extra_body.think = False + top-level
                                reasoning_effort="none"
  - every generic relay (including Groq) → omit all Ollama-only controls
  - enabled verified Ollama + effort → top-level reasoning_effort
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

    def test_disabled_verified_ollama_sends_disabled_controls(self, custom_profile, monkeypatch):
        """enabled=False → reasoning_effort='none' top-level + think=False.

        Both fields are required: Ollama's /v1/chat/completions silently
        ignores extra_body.think (only /api/chat honours it — ollama#14820)
        but respects top-level reasoning_effort (#25758). think=False stays
        for proxies and the native /api/chat path.
        """
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", lambda *_args, **_kwargs: "ollama"
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2", base_url="http://127.0.0.1:11434/v1"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.groq.com/openai/v1",
            "https://not-ollama.example/v1",
            "http://127.0.0.1:11434/v1",
        ],
        ids=["groq", "url-containing-ollama", "non-ollama-port-11434"],
    )
    def test_non_ollama_endpoint_omits_all_reasoning_controls(
        self, custom_profile, monkeypatch, base_url
    ):
        """Only a positive Ollama probe may enable any Ollama control."""
        probe_calls = []

        def _not_ollama(url, *, api_key=""):
            probe_calls.append((url, api_key))
            return None

        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", _not_ollama
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            base_url=base_url,
            api_key="custom-key",
        )
        assert "think" not in eb
        assert "reasoning_effort" not in tl
        assert probe_calls == [(base_url, "custom-key")]

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_verified_ollama_enabled_effort_goes_top_level(
        self, custom_profile, monkeypatch, effort
    ):
        """A positively identified Ollama endpoint receives the chosen effort."""
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", lambda *_args, **_kwargs: "ollama"
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="qwen3",
            base_url="http://127.0.0.1:11434/v1",
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb

    def test_verified_ollama_enabled_without_effort_emits_nothing(self, custom_profile, monkeypatch):
        """enabled but no effort → omit; do NOT force a level the user didn't pick."""
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", lambda *_args, **_kwargs: "ollama"
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True},
            model="qwen3",
            base_url="http://127.0.0.1:11434/v1",
        )
        assert eb == {}
        assert tl == {}

    def test_does_not_force_think_true_on_enable(self, custom_profile, monkeypatch):
        """We must never send think=True on enable — it's Ollama-only and
        the server already defaults to thinking on."""
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", lambda *_args, **_kwargs: "ollama"
        )
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="qwen3",
            base_url="http://127.0.0.1:11434/v1",
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

    def test_num_ctx_with_verified_ollama_effort(self, custom_profile, monkeypatch):
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", lambda *_args, **_kwargs: "ollama"
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_num_ctx=8192,
            model="qwen3",
            base_url="http://127.0.0.1:11434/v1",
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {"reasoning_effort": "high"}

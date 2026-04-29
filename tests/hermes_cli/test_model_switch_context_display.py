"""Regression test for /model context-length display on provider-capped models.

Bug (April 2026): `/model gpt-5.5` on openai-codex (ChatGPT OAuth) showed
"Context: 1,050,000 tokens" because the display code used the raw models.dev
``ModelInfo.context_window`` (which reports the direct-OpenAI API value) instead
of the provider-aware resolver. The agent was actually running at 272K — Codex
OAuth's enforced cap — so the display was lying to the user.

Fix: ``resolve_display_context_length()`` prefers
``agent.model_metadata.get_model_context_length`` (which knows about Codex OAuth,
Copilot, Nous, etc.) and falls back to models.dev only if that returns nothing.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import resolve_display_context_length


class _FakeModelInfo:
    def __init__(self, ctx):
        self.context_window = ctx


class TestResolveDisplayContextLength:
    def test_codex_oauth_overrides_models_dev(self):
        """gpt-5.5 on openai-codex must show Codex's 272K cap, not models.dev's 1.05M."""
        fake_mi = _FakeModelInfo(1_050_000)  # what models.dev reports
        with patch(
            "agent.model_metadata.get_model_context_length",
            return_value=272_000,  # what Codex OAuth actually enforces
        ):
            ctx = resolve_display_context_length(
                "gpt-5.5",
                "openai-codex",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="",
                model_info=fake_mi,
            )
        assert ctx == 272_000, (
            "Codex OAuth's 272K cap must win over models.dev's 1.05M for gpt-5.5"
        )

    def test_falls_back_to_model_info_when_resolver_returns_none(self):
        fake_mi = _FakeModelInfo(1_048_576)
        with patch(
            "agent.model_metadata.get_model_context_length", return_value=None
        ):
            ctx = resolve_display_context_length(
                "some-model",
                "some-provider",
                model_info=fake_mi,
            )
        assert ctx == 1_048_576

    def test_returns_none_when_both_sources_empty(self):
        with patch(
            "agent.model_metadata.get_model_context_length", return_value=None
        ):
            ctx = resolve_display_context_length(
                "unknown-model",
                "unknown-provider",
                model_info=None,
            )
        assert ctx is None

    def test_resolver_exception_falls_back_to_model_info(self):
        fake_mi = _FakeModelInfo(200_000)
        with patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=RuntimeError("network down"),
        ):
            ctx = resolve_display_context_length(
                "x", "y", model_info=fake_mi
            )
        assert ctx == 200_000

    def test_prefers_resolver_even_when_model_info_has_larger_value(self):
        """Invariant: provider-aware resolver is authoritative, even if models.dev
        reports a bigger window."""
        fake_mi = _FakeModelInfo(2_000_000)
        with patch(
            "agent.model_metadata.get_model_context_length", return_value=128_000
        ):
            ctx = resolve_display_context_length(
                "capped-model",
                "capped-provider",
                model_info=fake_mi,
            )
        assert ctx == 128_000

    def test_custom_providers_override_honored(self):
        """Regression for #15779: /model switch onto a custom provider must
        surface the configured per-model context_length, not the 128K/256K
        fallback.
        """
        custom_provs = [
            {
                "name": "my-custom-endpoint",
                "base_url": "https://example.invalid/v1",
                "models": {"gpt-5.5": {"context_length": 1_050_000}},
            }
        ]
        # Real resolver call — no mock — so the override path is exercised
        # through agent.model_metadata.get_model_context_length.
        from unittest.mock import patch as _p
        from agent import model_metadata as _mm
        with _p.object(_mm, "get_cached_context_length", return_value=None), \
             _p.object(_mm, "fetch_endpoint_model_metadata", return_value={}), \
             _p.object(_mm, "fetch_model_metadata", return_value={}), \
             _p.object(_mm, "is_local_endpoint", return_value=False), \
             _p.object(_mm, "_is_known_provider_base_url", return_value=False):
            ctx = resolve_display_context_length(
                "gpt-5.5",
                "custom",
                base_url="https://example.invalid/v1",
                api_key="k",
                custom_providers=custom_provs,
            )
        assert ctx == 1_050_000, (
            "custom_providers[].models.gpt-5.5.context_length=1.05M must win "
            "over probe-down fallback"
        )

    def test_custom_providers_trailing_slash_insensitive(self):
        """Base URL comparison must tolerate trailing-slash differences
        between config.yaml and the runtime value.
        """
        custom_provs = [
            {
                "base_url": "https://example.invalid/v1/",
                "models": {"m": {"context_length": 400_000}},
            }
        ]
        from unittest.mock import patch as _p
        from agent import model_metadata as _mm
        with _p.object(_mm, "get_cached_context_length", return_value=None), \
             _p.object(_mm, "fetch_endpoint_model_metadata", return_value={}), \
             _p.object(_mm, "fetch_model_metadata", return_value={}), \
             _p.object(_mm, "is_local_endpoint", return_value=False), \
             _p.object(_mm, "_is_known_provider_base_url", return_value=False):
            ctx = resolve_display_context_length(
                "m",
                "custom",
                base_url="https://example.invalid/v1",  # no trailing slash
                custom_providers=custom_provs,
            )
        assert ctx == 400_000

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

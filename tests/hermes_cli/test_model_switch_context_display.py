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


class TestMoAPresetContextDisplay:
    """Regression test for /model context-length display when switching to a
    MoA preset name.

    Bug (June 2026): ``/model alex-max`` (or any MoA preset) showed
    ``Context: 256,000 tokens`` even though the underlying aggregator
    (e.g. ``minimax/minimax-m3``) exposes 1M tokens.  The display path
    passed the preset name ``alex-max`` straight into the context-length
    resolver, which never found a match and fell through to the
    256K fallback.  The fix in ``agent/model_metadata.get_model_context_length``
    step 0a resolves a MoA preset name to its aggregator slug before the
    normal resolution chain runs.
    """

    _FAKE_MOA_CONFIG = {
        "moa": {
            "default_preset": "alex-max",
            "active_preset": "alex-max",
            "presets": {
                "alex-max": {
                    "enabled": True,
                    "reference_models": [
                        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                        {"provider": "openrouter", "model": "xiaomi/mimo-v2.5-pro"},
                    ],
                    "aggregator": {
                        "provider": "openrouter",
                        "model": "minimax/minimax-m3",
                    },
                },
            },
        },
    }

    def test_moa_preset_name_resolves_to_aggregator_context(self):
        """``alex-max`` with ``provider="moa"`` must surface the aggregator's
        context window (1M for ``minimax/minimax-m3``), not the 256K fallback.
        """
        from unittest.mock import patch as _p
        from agent import model_metadata as _mm
        import hermes_cli.config as _hcfg

        with _p.object(_hcfg, "load_config", return_value=self._FAKE_MOA_CONFIG), \
             _p.object(_mm, "get_cached_context_length", return_value=None), \
             _p.object(_mm, "fetch_endpoint_model_metadata", return_value={}), \
             _p.object(_mm, "fetch_model_metadata", return_value={
                 "minimax/minimax-m3": {"context_length": 1_048_576},
             }), \
             _p.object(_mm, "is_local_endpoint", return_value=False), \
             _p.object(_mm, "_is_known_provider_base_url", return_value=False):
            ctx = resolve_display_context_length(
                "alex-max",
                "moa",
                base_url="",
                api_key="",
            )
        # Aggregator (minimax/minimax-m3) is 1M tokens.  Must not be the 256K fallback.
        assert ctx is not None
        assert ctx >= 1_000_000, (
            f"MoA preset 'alex-max' must resolve to aggregator context (>=1M), "
            f"got {ctx:,}.  Pre-fix bug displayed 256,000."
        )

    def test_unknown_moa_preset_falls_back(self):
        """An unknown MoA preset name must still fall back gracefully (256K)
        rather than raising or returning ``None``.
        """
        from unittest.mock import patch as _p
        from agent import model_metadata as _mm
        import hermes_cli.config as _hcfg

        with _p.object(_hcfg, "load_config", return_value=self._FAKE_MOA_CONFIG), \
             _p.object(_mm, "get_cached_context_length", return_value=None), \
             _p.object(_mm, "fetch_endpoint_model_metadata", return_value={}), \
             _p.object(_mm, "fetch_model_metadata", return_value={}), \
             _p.object(_mm, "is_local_endpoint", return_value=False), \
             _p.object(_mm, "_is_known_provider_base_url", return_value=False):
            ctx = resolve_display_context_length(
                "does-not-exist",
                "moa",
                base_url="",
                api_key="",
            )
        assert ctx == 256_000  # documented fallback

    def test_non_moa_provider_does_not_rewrite_preset_name(self):
        """A bare preset name passed without ``provider="moa"`` must NOT be
        rewritten — it must hit the regular resolution chain (and fall back
        to 256K because ``alex-max`` is not a real provider slug).
        """
        from unittest.mock import patch as _p
        from agent import model_metadata as _mm
        import hermes_cli.config as _hcfg

        with _p.object(_hcfg, "load_config", return_value=self._FAKE_MOA_CONFIG), \
             _p.object(_mm, "get_cached_context_length", return_value=None), \
             _p.object(_mm, "fetch_endpoint_model_metadata", return_value={}), \
             _p.object(_mm, "fetch_model_metadata", return_value={}), \
             _p.object(_mm, "is_local_endpoint", return_value=False), \
             _p.object(_mm, "_is_known_provider_base_url", return_value=False):
            ctx = resolve_display_context_length(
                "alex-max",
                "openrouter",  # not moa
                base_url="",
                api_key="",
            )
        assert ctx == 256_000  # no rewriting happens

"""Tests for the ``recommended:free`` sentinel in fallback chain resolution."""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.fallback_config import (
    RECOMMENDED_FREE_SENTINEL,
    get_fallback_chain,
)


def _cfg(model: str, provider: str = "nous") -> dict:
    return {"fallback_providers": [{"provider": provider, "model": model}]}


class TestRecommendedFreeSentinel:
    def test_sentinel_resolves_to_portal_free_model(self):
        with patch(
            "hermes_cli.models.get_nous_recommended_free_model",
            return_value="stepfun/step-3.7-flash:free",
        ):
            chain = get_fallback_chain(_cfg(RECOMMENDED_FREE_SENTINEL))
        assert chain == [
            {"provider": "nous", "model": "stepfun/step-3.7-flash:free"}
        ]

    def test_sentinel_entry_dropped_when_no_recommendation(self):
        """No cache, no network → entry skipped; rest of chain unaffected."""
        cfg = {
            "fallback_providers": [
                {"provider": "nous", "model": RECOMMENDED_FREE_SENTINEL},
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            ]
        }
        with patch(
            "hermes_cli.models.get_nous_recommended_free_model",
            return_value=None,
        ):
            chain = get_fallback_chain(cfg)
        assert chain == [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
        ]

    def test_sentinel_on_other_provider_is_literal(self):
        """The sentinel is nous-namespaced; elsewhere it's just a model name."""
        with patch(
            "hermes_cli.models.get_nous_recommended_free_model",
        ) as resolver:
            chain = get_fallback_chain(_cfg(RECOMMENDED_FREE_SENTINEL, "openrouter"))
        resolver.assert_not_called()
        assert chain == [
            {"provider": "openrouter", "model": RECOMMENDED_FREE_SENTINEL}
        ]

    def test_resolved_duplicate_deduped_against_literal_entry(self):
        """Sentinel resolving to a model already in the chain is deduped."""
        cfg = {
            "fallback_providers": [
                {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
                {"provider": "nous", "model": RECOMMENDED_FREE_SENTINEL},
            ]
        }
        with patch(
            "hermes_cli.models.get_nous_recommended_free_model",
            return_value="stepfun/step-3.7-flash:free",
        ):
            chain = get_fallback_chain(cfg)
        assert chain == [
            {"provider": "nous", "model": "stepfun/step-3.7-flash:free"}
        ]

    def test_legacy_fallback_model_key_supports_sentinel(self):
        cfg = {
            "fallback_model": {"provider": "nous", "model": RECOMMENDED_FREE_SENTINEL}
        }
        with patch(
            "hermes_cli.models.get_nous_recommended_free_model",
            return_value="stepfun/step-3.7-flash:free",
        ):
            chain = get_fallback_chain(cfg)
        assert chain == [
            {"provider": "nous", "model": "stepfun/step-3.7-flash:free"}
        ]

    def test_resolver_exception_drops_entry_not_chain(self):
        cfg = {
            "fallback_providers": [
                {"provider": "nous", "model": RECOMMENDED_FREE_SENTINEL},
                {"provider": "nous", "model": "Hermes-4-Llama-3.1-405B"},
            ]
        }
        with patch(
            "hermes_cli.models.get_nous_recommended_free_model",
            side_effect=RuntimeError("portal exploded"),
        ):
            chain = get_fallback_chain(cfg)
        assert chain == [{"provider": "nous", "model": "Hermes-4-Llama-3.1-405B"}]

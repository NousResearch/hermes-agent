"""Dashboard recommended-default model: the paid-tier Nous cost-safe policy (#51491)."""

import pytest


class TestNousRecommendedDefaultCostSafePolicy:
    """Paid-tier Nous recommended default must never land on an Anthropic frontier
    tier (Opus / Fable) — the user gets no opt-out before it pins their main model
    (#51491). Regression tests from PR #51493."""

    def test_recommended_default_nous_paid_uses_curated_default(self, monkeypatch):
        """A paid Nous user gets the cost-safe silent default from the list.

        With no preferred catalog label present in the curated list and no
        Anthropic frontier entries, the first curated entry is selected.
        """
        import hermes_cli.models as models_mod
        from hermes_cli.web_routers.models import get_recommended_default_model

        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda: ["top/model", "other/model"])
        import hermes_cli.models_pricing as mp
        monkeypatch.setattr(mp, "get_pricing_for_provider", lambda provider: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda *, force_fresh=False: False)
        monkeypatch.setattr(
            models_mod, "union_with_portal_paid_recommendations",
            lambda ids, pricing, url: (ids, pricing),
        )
        # Keep the catalog preferred out of this list so we exercise the
        # non-frontier fallback rather than the preferred-hit branch.
        monkeypatch.setattr(
            models_mod, "get_preferred_silent_default_model",
            lambda provider="openrouter": "z-ai/glm-5.2",
        )

        result = get_recommended_default_model(provider="nous")
        assert result["provider"] == "nous"
        assert result["model"] == "top/model"
        assert result["free_tier"] is False

    @pytest.mark.parametrize(
        "model_ordering, expected_model",
        [
            # Opus-first ordering (historical PR-branch catalog shape).
            (
                [
                    "anthropic/claude-opus-4.8",
                    "anthropic/claude-sonnet-5",
                    "anthropic/claude-haiku-4.5",
                ],
                "anthropic/claude-sonnet-5",
            ),
            # Fable-first ordering (current origin/main catalog shape).
            (
                [
                    "anthropic/claude-fable-5",
                    "anthropic/claude-opus-4.8",
                    "anthropic/claude-sonnet-5",
                    "anthropic/claude-haiku-4.5",
                ],
                "anthropic/claude-sonnet-5",
            ),
            # Preferred silent default present in the list always wins,
            # independent of relative ordering / frontiers.
            (
                [
                    "anthropic/claude-fable-5",
                    "anthropic/claude-opus-4.8",
                    "z-ai/glm-5.2",
                    "anthropic/claude-sonnet-5",
                ],
                "z-ai/glm-5.2",
            ),
        ],
        ids=["opus_first", "fable_first_main", "override_beats_ordering"],
    )
    def test_recommended_default_nous_paid_cost_safe_policy(
        self, monkeypatch, model_ordering, expected_model,
    ):
        """Regression for PR #51493 maintainer feedback (Teknium + DavidMetcalfe).

        The interactive Nous recommended default must use the shared
        cost-safe silent policy: preferred catalog label when present,
        else first non-frontier (Opus / Fable) entry. Covers both the
        current Fable-first catalog and a future Opus-first catalog.
        """
        import hermes_cli.models as models_mod
        from hermes_cli.web_routers.models import get_recommended_default_model

        monkeypatch.setattr(
            models_mod, "get_curated_nous_model_ids", lambda: list(model_ordering),
        )
        import hermes_cli.models_pricing as mp
        monkeypatch.setattr(mp, "get_pricing_for_provider", lambda provider: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda *, force_fresh=False: False)
        monkeypatch.setattr(
            models_mod, "union_with_portal_paid_recommendations",
            lambda ids, pricing, url: (ids, pricing),
        )
        monkeypatch.setattr(
            models_mod, "get_preferred_silent_default_model",
            lambda provider="openrouter": "z-ai/glm-5.2",
        )

        result = get_recommended_default_model(provider="nous")
        assert result["provider"] == "nous"
        assert result["model"] == expected_model
        assert result["free_tier"] is False

    def test_recommended_default_nous_paid_falls_back_when_all_frontier(self, monkeypatch):
        """If every curated entry is a frontier tier (Opus / Fable), fall
        back to the head of the list so the picker is never empty."""
        import hermes_cli.models as models_mod
        from hermes_cli.web_routers.models import get_recommended_default_model

        monkeypatch.setattr(
            models_mod, "get_curated_nous_model_ids",
            lambda: ["anthropic/claude-fable-5", "anthropic/claude-opus-4.8"],
        )
        import hermes_cli.models_pricing as mp
        monkeypatch.setattr(mp, "get_pricing_for_provider", lambda provider: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda *, force_fresh=False: False)
        monkeypatch.setattr(
            models_mod, "union_with_portal_paid_recommendations",
            lambda ids, pricing, url: (ids, pricing),
        )
        monkeypatch.setattr(
            models_mod, "get_preferred_silent_default_model",
            lambda provider="openrouter": "z-ai/glm-5.2",
        )

        result = get_recommended_default_model(provider="nous")
        assert result["model"] == "anthropic/claude-fable-5"

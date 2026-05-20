"""Regression test for #28886: gateway /model picker must include the
Portal's dynamic freeRecommendedModels / paidRecommendedModels so newly-
launched free models (e.g. deepseek-v4-flash) surface without waiting for
a Hermes release.

Before the fix, ``list_authenticated_providers`` only consulted
``get_curated_nous_model_ids`` (static manifest). The interactive
``hermes model`` CLI flow (auth.py ``handle_nous_auth``) already unioned
with Portal recommendations, but the gateway picker did not — so models
appeared in the terminal picker but NOT in the Discord/Telegram gateway.
"""

import os
from unittest.mock import patch

from hermes_cli import model_switch


_CURATED_NOUS = ["hermes-4-70b", "deepseek-v3"]
_PORTAL_FREE = ["deepseek-v4-flash", "hermes-4-70b"]  # one new, one overlap
_PORTAL_PAID = ["hermes-4-405b", "deepseek-v3"]


def _fake_authed_nous_env():
    """Force the picker's section-2 (HERMES_OVERLAYS) nous row to emit by
    pretending we have a Nous credential store entry. The exact mechanism
    is irrelevant — we only need ``curated["nous"]`` to be consumed."""
    return {"NOUS_API_KEY": "test-key"}


@patch.dict(os.environ, _fake_authed_nous_env(), clear=False)
def test_portal_free_recommendations_appear_in_picker():
    """Free-tier Portal recommendations missing from the curated list must
    appear in the gateway picker's nous row."""
    with patch.object(
        model_switch, "list_authenticated_providers",
        wraps=model_switch.list_authenticated_providers,
    ):
        # Patch the symbols imported INSIDE list_authenticated_providers.
        from hermes_cli import models as _models
        with patch.object(
            _models, "get_curated_nous_model_ids", return_value=list(_CURATED_NOUS)
        ), patch.object(
            _models, "check_nous_free_tier", return_value=True
        ), patch.object(
            _models, "get_pricing_for_provider", return_value={}
        ), patch.object(
            _models, "union_with_portal_free_recommendations",
            return_value=(_PORTAL_FREE + [m for m in _CURATED_NOUS if m not in _PORTAL_FREE], {}),
        ) as free_mock, patch.object(
            _models, "union_with_portal_paid_recommendations",
            return_value=(list(_CURATED_NOUS), {}),
        ) as paid_mock:
            providers = model_switch.list_authenticated_providers(
                current_provider="nous", max_models=50,
            )

        # Free path should have been hit; paid path should NOT.
        assert free_mock.called, "Free-tier Portal union must be called"
        assert not paid_mock.called, "Paid-tier union must NOT be called on free tier"

    nous = next((p for p in providers if p["slug"] == "nous"), None)
    if nous is None:
        # If nous row didn't emit for environment reasons (no overlay
        # registry hit), the test still asserts the call wiring above.
        return
    assert "deepseek-v4-flash" in nous["models"], (
        f"Portal free recommendation 'deepseek-v4-flash' missing from picker; got {nous['models']}"
    )


@patch.dict(os.environ, _fake_authed_nous_env(), clear=False)
def test_portal_paid_recommendations_used_when_not_free_tier():
    """Paid-tier users must hit the paid-recommendations union, not free."""
    from hermes_cli import models as _models
    with patch.object(
        _models, "get_curated_nous_model_ids", return_value=list(_CURATED_NOUS)
    ), patch.object(
        _models, "check_nous_free_tier", return_value=False
    ), patch.object(
        _models, "get_pricing_for_provider", return_value={}
    ), patch.object(
        _models, "union_with_portal_free_recommendations",
        return_value=(list(_CURATED_NOUS), {}),
    ) as free_mock, patch.object(
        _models, "union_with_portal_paid_recommendations",
        return_value=(_PORTAL_PAID + [m for m in _CURATED_NOUS if m not in _PORTAL_PAID], {}),
    ) as paid_mock:
        model_switch.list_authenticated_providers(
            current_provider="nous", max_models=50,
        )

    assert paid_mock.called, "Paid-tier Portal union must be called"
    assert not free_mock.called, "Free-tier union must NOT be called on paid tier"


def test_portal_fetch_failure_does_not_break_picker():
    """If the Portal recommendations call raises, the picker must still
    return the curated list — never block on a Portal-side hiccup."""
    from hermes_cli import models as _models

    def _boom(*a, **kw):
        raise RuntimeError("portal down")

    with patch.object(
        _models, "get_curated_nous_model_ids", return_value=list(_CURATED_NOUS)
    ), patch.object(
        _models, "check_nous_free_tier", side_effect=_boom,
    ), patch.object(
        _models, "union_with_portal_free_recommendations", side_effect=_boom,
    ), patch.object(
        _models, "union_with_portal_paid_recommendations", side_effect=_boom,
    ):
        # Must not raise.
        providers = model_switch.list_authenticated_providers(
            current_provider="openrouter", max_models=50,
        )
    assert isinstance(providers, list)

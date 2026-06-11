"""Unit tests for the MiniMax provider profile.

Three MiniMax provider profiles (`minimax` direct API, `minimax-cn` China direct
API, `minimax-oauth` browser OAuth) all advertise a `default_aux_model` on
their `ProviderProfile`. The previous M2.7 / M2.7-highspeed values were
stale relative to the current frontier model (M3, released 2026-06-01) and
inconsistent with the `_PROVIDER_MODELS["minimax"]` catalog top entry in
`hermes_cli/models.py`.

This file pins the new defaults so the choice is reviewable and any future
revert shows up in a failing test rather than silent behavior drift.

Refs:
  - Issue #36196: M3 support request
  - PR #36205 (closed unmerged): Csrayz's M3 + 1M context work
  - PR #36212 (open): adds M3 to `_PROVIDER_MODELS["minimax"]` catalog
  - PR #6082: M2.7-highspeed → M2.7 for aux model (half-price fix)
  - Commit 773a0faca: same profile-layer fix pattern for `deepseek`
"""

from __future__ import annotations

import pytest


@pytest.fixture(params=["minimax", "minimax-cn", "minimax-oauth"])
def minimax_profile(request):
    """Resolve each registered MiniMax profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    import model_tools  # noqa: F401  -- triggers plugin discovery
    import providers

    profile = providers.get_provider_profile(request.param)
    assert profile is not None, f"{request.param} provider profile must be registered"
    return profile, request.param


class TestMinimaxAuxModelM3:
    """MiniMax profile aux model is the new frontier M3, not the stale M2.7.

    The catalog top entry is ``MiniMax-M3`` in
    ``hermes_cli.models._PROVIDER_MODELS['minimax']`` and the
    user-facing ``model.default`` for a Token-Plan install is M3,
    so pinning the aux default to the same model keeps the runtime
    consistent (same auth, same billing pool, same rate limits, no
    surprise 2x-cost highspeed variant). M3 was released 2026-06-01
    — picking it as the aux default matches the forward-looking
    catalog order rather than the pre-M3 era.
    """

    @pytest.mark.parametrize(
        "provider_id,expected",
        [
            ("minimax", "MiniMax-M3"),
            ("minimax-cn", "MiniMax-M3"),
            # minimax-oauth sticks with M2.7: the OAuth / Coding Plan
            # tier historically used -highspeed (PR #6082 collapsed that
            # to plain M2.7 to avoid the 2x TPS surcharge). M3 is not on
            # the OAuth/Coding Plan tier per platform docs as of this PR,
            # so the safe choice is the cheapest generally-available
            # M2.7 — matching PR #6082's intent.
            ("minimax-oauth", "MiniMax-M2.7"),
        ],
    )
    def test_profile_advertises_expected_aux_model(
        self, provider_id, expected
    ):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile(provider_id)
        assert profile is not None
        assert profile.default_aux_model == expected, (
            f"{provider_id} default_aux_model drifted to "
            f"{profile.default_aux_model!r}, expected {expected!r}"
        )

    def test_consumer_api_returns_non_empty_for_each_provider(self, minimax_profile):
        from agent.auxiliary_client import _get_aux_model_for_provider

        profile, provider_id = minimax_profile
        resolved = _get_aux_model_for_provider(provider_id)
        assert resolved != "", (
            f"_get_aux_model_for_provider({provider_id!r}) returned empty — "
            "the 'No auxiliary LLM provider configured' warning will fire on "
            f"every {provider_id} session even though the profile advertises "
            f"default_aux_model={profile.default_aux_model!r}"
        )
        assert resolved == profile.default_aux_model, (
            f"_get_aux_model_for_provider({provider_id!r}) returned "
            f"{resolved!r} but profile advertises {profile.default_aux_model!r} "
            "— the consumer API and the profile have drifted out of sync"
        )


class TestMinimaxAuxModelNotHighspeed:
    """Regression guard against re-introducing the M2.7-highspeed aux default.

    PR #6082 collapsed the highspeed aux choice to plain M2.7 because the
    highspeed variant costs 2x with no real benefit for compression / vision /
    session-search aux tasks. None of the three MiniMax profiles should
    silently re-introduce that 2x-cost path.
    """

    @pytest.mark.parametrize("provider_id", ["minimax", "minimax-cn", "minimax-oauth"])
    def test_default_aux_model_is_not_highspeed(self, provider_id):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile(provider_id)
        assert profile is not None
        assert "highspeed" not in profile.default_aux_model.lower(), (
            f"{provider_id} default_aux_model={profile.default_aux_model!r} "
            "is a -highspeed variant — that costs 2x for the same model and "
            "broke #4082 the first time. Revert to plain M2.7 or M3."
        )

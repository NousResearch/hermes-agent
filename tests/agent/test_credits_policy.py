"""Tests for evaluate_credits_notices — pure threshold reconciliation policy (L4.1).

All tests use fresh latch = {"active": set(), "seen_below_90": False, "usage_band": None} per scenario.
CreditsState is constructed directly (not parsed from headers).
"""

from __future__ import annotations

import pytest

from agent.credits_tracker import (
    CREDITS_NOTICE_KIND,
    CREDITS_RESTORED_TTL_MS,
    AgentNotice,
    CreditsState,
    evaluate_credits_notices,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def fresh_latch() -> dict:
    return {"active": set(), "seen_below_90": False, "usage_band": None}


def state_with_fraction(
    uf: float | None,
    *,
    paid_access: bool = True,
    denominator_kind: str = "subscription_cap",
    purchased_micros: int = 0,
    purchased_usd: str = "0.00",
    subscription_limit_usd: str | None = "20.00",
) -> CreditsState:
    """Build a minimal CreditsState that yields the desired used_fraction.

    used_fraction = (limit - subscription_micros) / limit

    When uf is None, we set limit to None so used_fraction returns None.
    """
    if uf is None:
        return CreditsState(
            subscription_limit_micros=None,
            subscription_limit_usd=None,
            subscription_micros=0,
            denominator_kind="none",
            paid_access=paid_access,
            purchased_micros=purchased_micros,
            purchased_usd=purchased_usd,
        )
    # We want (limit - sub) / limit == uf  →  sub = limit * (1 - uf)
    limit = 20_000_000  # $20 in micros
    sub = int(limit * (1.0 - uf))
    return CreditsState(
        subscription_limit_micros=limit,
        subscription_limit_usd=subscription_limit_usd,
        subscription_micros=sub,
        denominator_kind=denominator_kind,
        paid_access=paid_access,
        purchased_micros=purchased_micros,
        purchased_usd=purchased_usd,
    )


# ── Scenario 1: crossing 90% threshold ───────────────────────────────────────


class TestWarn90Crossing:
    def test_below_lowest_band_no_notice_but_latch_set(self):
        latch = fresh_latch()
        s = state_with_fraction(0.10)  # below the 50% band
        to_show, to_clear = evaluate_credits_notices(s, latch)
        assert all(n.key != "credits.usage" for n in to_show)
        assert "credits.usage" not in to_clear
        assert latch["seen_below_90"] is True

    def test_crossing_to_90_fires_once(self):
        latch = fresh_latch()
        # First call: uf < 0.5 — sets seen_below_90 (below lowest band)
        s1 = state_with_fraction(0.10)
        evaluate_credits_notices(s1, latch)
        # Second call: uf >= 0.9 — should fire the usage band at 90
        s2 = state_with_fraction(0.95)
        to_show, to_clear = evaluate_credits_notices(s2, latch)
        keys = [n.key for n in to_show]
        assert "credits.usage" in keys
        assert "credits.usage" not in to_clear

    def test_no_refire_on_repeated_over_90(self):
        latch = fresh_latch()
        s_below = state_with_fraction(0.10)
        evaluate_credits_notices(s_below, latch)
        s_over = state_with_fraction(0.95)
        evaluate_credits_notices(s_over, latch)
        # Third call: still ≥ 0.9 — must NOT re-fire
        to_show, to_clear = evaluate_credits_notices(s_over, latch)
        assert all(n.key != "credits.usage" for n in to_show)
        assert "credits.usage" not in to_clear


# ── Scenario 2: recovery + re-cross ──────────────────────────────────────────


class TestWarn90RecoveryReCross:
    def test_recovery_clears_warn90(self):
        latch = fresh_latch()
        # Cross below → above
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(state_with_fraction(0.95), latch)
        # Recovery: uf drops back below ALL bands → usage notice clears entirely
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.10), latch)
        assert "credits.usage" in to_clear
        assert "credits.usage" not in latch["active"]

    def test_recross_after_recovery_fires_again(self):
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(state_with_fraction(0.95), latch)
        evaluate_credits_notices(state_with_fraction(0.10), latch)  # recovery
        # Re-cross: uf >= 0.9 again — should fire again because the band is clearable
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.95), latch)
        keys = [n.key for n in to_show]
        assert "credits.usage" in keys


# ── Scenario 3: open-already-over (hybrid Q3 gate) ───────────────────────────


class TestOpenAlreadyOver:
    def test_warn90_does_not_fire_without_seen_below_90(self):
        """First call uf≥0.9 with seen_below_90=False — warn90 must NOT fire."""
        latch = fresh_latch()
        assert latch["seen_below_90"] is False
        s = state_with_fraction(0.95)
        to_show, to_clear = evaluate_credits_notices(s, latch)
        assert all(n.key != "credits.usage" for n in to_show)
        assert "credits.usage" not in to_clear


# ── Scenario 3b: boundary — exact 0.9 and just-below-1.0 ────────────────────


class TestBoundaryFractions:
    def test_exact_0_9_fires_warn90(self):
        """used_fraction == 0.9 exactly must fire warn90 (threshold is inclusive)."""
        latch = fresh_latch()
        # First: prime seen_below_90 with a sub-50% observation
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        # Now construct a state where used_fraction is EXACTLY 0.9:
        # subscription_limit_micros=20_000_000, subscription_micros=2_000_000
        # → used = 18_000_000 / 20_000_000 = 0.9 exactly
        s = CreditsState(
            subscription_limit_micros=20_000_000,
            subscription_limit_usd="20.00",
            subscription_micros=2_000_000,
            denominator_kind="subscription_cap",
            paid_access=True,
        )
        assert s.used_fraction == 0.9
        to_show, to_clear = evaluate_credits_notices(s, latch)
        keys = [n.key for n in to_show]
        assert "credits.usage" in keys
        assert "credits.usage" not in to_clear

    def test_just_below_1_0_does_not_fire_grant_spent(self):
        """subscription_micros = limit - 1 (used_fraction just under 1.0) must NOT fire grant_spent.

        Locks the boundary so a future used_fraction clamp refactor cannot fire
        grant_spent a micro early.
        """
        latch = fresh_latch()
        limit = 20_000_000
        s = CreditsState(
            subscription_limit_micros=limit,
            subscription_limit_usd="20.00",
            subscription_micros=1,           # limit - 1 → used_fraction < 1.0
            denominator_kind="subscription_cap",
            purchased_micros=5_000_000,
            purchased_usd="5.00",
            paid_access=True,
        )
        assert s.used_fraction is not None and s.used_fraction < 1.0
        to_show, to_clear = evaluate_credits_notices(s, latch)
        assert all(n.key != "credits.grant_spent" for n in to_show)
        assert "credits.grant_spent" not in to_clear


# ── Scenario 4: grant_spent ───────────────────────────────────────────────────


class TestGrantSpent:
    def _grant_state(self, purchased_micros: int = 12_340_000) -> CreditsState:
        return state_with_fraction(
            1.0,
            denominator_kind="subscription_cap",
            purchased_micros=purchased_micros,
            purchased_usd="12.34",
        )

    def test_grant_spent_fires_on_first_obs(self):
        """No crossing gate for grant_spent — fires immediately on first obs."""
        latch = fresh_latch()
        to_show, to_clear = evaluate_credits_notices(self._grant_state(), latch)
        keys = [n.key for n in to_show]
        assert "credits.grant_spent" in keys

    def test_grant_spent_no_refire(self):
        latch = fresh_latch()
        evaluate_credits_notices(self._grant_state(), latch)
        to_show, to_clear = evaluate_credits_notices(self._grant_state(), latch)
        assert all(n.key != "credits.grant_spent" for n in to_show)
        assert "credits.grant_spent" not in to_clear

    def test_grant_spent_clears_when_purchased_zero(self):
        latch = fresh_latch()
        evaluate_credits_notices(self._grant_state(), latch)
        # Now purchased → 0: grant_cond becomes False
        s_no_purchase = state_with_fraction(
            1.0,
            denominator_kind="subscription_cap",
            purchased_micros=0,
            purchased_usd="0.00",
        )
        to_show, to_clear = evaluate_credits_notices(s_no_purchase, latch)
        assert "credits.grant_spent" in to_clear
        assert all(n.key != "credits.grant_spent" for n in to_show)


# ── Scenario 5: depleted + recovery ──────────────────────────────────────────


class TestDepleted:
    def test_depleted_fires_level_error_kind_sticky(self):
        latch = fresh_latch()
        s = CreditsState(paid_access=False)
        to_show, to_clear = evaluate_credits_notices(s, latch)
        depleted_notices = [n for n in to_show if n.key == "credits.depleted"]
        assert len(depleted_notices) == 1
        n = depleted_notices[0]
        assert n.level == "error"
        assert n.kind == CREDITS_NOTICE_KIND

    def test_recovery_emits_clear_and_restored(self):
        latch = fresh_latch()
        # Fire depleted
        evaluate_credits_notices(CreditsState(paid_access=False), latch)
        # Now recovered
        to_show, to_clear = evaluate_credits_notices(CreditsState(paid_access=True), latch)
        assert "credits.depleted" in to_clear
        restored = [n for n in to_show if n.key == "credits.restored"]
        assert len(restored) == 1
        r = restored[0]
        assert r.level == "success"
        assert r.kind == "ttl"
        assert r.ttl_ms == CREDITS_RESTORED_TTL_MS

    def test_depleted_refires_after_recovery(self):
        latch = fresh_latch()
        evaluate_credits_notices(CreditsState(paid_access=False), latch)
        evaluate_credits_notices(CreditsState(paid_access=True), latch)
        # Goes depleted again
        to_show, to_clear = evaluate_credits_notices(CreditsState(paid_access=False), latch)
        keys = [n.key for n in to_show]
        assert "credits.depleted" in keys


# ── Scenario 5b: free-model suppression of the depleted notice ───────────────


class TestDepletedFreeModelSuppression:
    def test_depleted_suppressed_when_model_is_free(self):
        latch = fresh_latch()
        s = CreditsState(paid_access=False)
        to_show, to_clear = evaluate_credits_notices(s, latch, model_is_free=True)
        assert all(n.key != "credits.depleted" for n in to_show)
        assert "credits.depleted" not in latch["active"]
        assert to_clear == []

    def test_switch_to_free_model_clears_without_restored(self):
        latch = fresh_latch()
        # Depleted on a paid model → notice fires
        evaluate_credits_notices(CreditsState(paid_access=False), latch)
        assert "credits.depleted" in latch["active"]
        # Same depleted account, but now on a free model → clear, NO "restored"
        to_show, to_clear = evaluate_credits_notices(
            CreditsState(paid_access=False), latch, model_is_free=True
        )
        assert "credits.depleted" in to_clear
        assert "credits.depleted" not in latch["active"]
        assert all(n.key != "credits.restored" for n in to_show)

    def test_switch_back_to_paid_model_while_depleted_reshows(self):
        latch = fresh_latch()
        evaluate_credits_notices(CreditsState(paid_access=False), latch)
        evaluate_credits_notices(CreditsState(paid_access=False), latch, model_is_free=True)
        # Back on a paid model, still depleted → notice re-fires
        to_show, to_clear = evaluate_credits_notices(CreditsState(paid_access=False), latch)
        keys = [n.key for n in to_show]
        assert "credits.depleted" in keys
        assert "credits.depleted" in latch["active"]

    def test_genuine_recovery_on_free_model_no_spurious_restored(self):
        """Recovery observed while suppressed (notice never shown) → nothing to
        clear, no 'restored' (there was no visible depleted state to restore)."""
        latch = fresh_latch()
        evaluate_credits_notices(CreditsState(paid_access=False), latch, model_is_free=True)
        to_show, to_clear = evaluate_credits_notices(
            CreditsState(paid_access=True), latch, model_is_free=True
        )
        assert to_clear == []
        assert all(n.key != "credits.restored" for n in to_show)

    def test_genuine_recovery_still_emits_restored_when_notice_active(self):
        """paid_access flip back to True with the notice showing → clear + restored
        (unchanged behaviour, regardless of the model-free flag)."""
        latch = fresh_latch()
        evaluate_credits_notices(CreditsState(paid_access=False), latch)
        to_show, to_clear = evaluate_credits_notices(
            CreditsState(paid_access=True), latch, model_is_free=True
        )
        assert "credits.depleted" in to_clear
        restored = [n for n in to_show if n.key == "credits.restored"]
        assert len(restored) == 1

    def test_free_flag_does_not_affect_other_notices(self):
        """Usage-band and grant notices are independent of the model-free gate."""
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch, model_is_free=True)
        to_show, _ = evaluate_credits_notices(
            state_with_fraction(0.95, paid_access=False), latch, model_is_free=True
        )
        keys = [n.key for n in to_show]
        assert "credits.usage" in keys
        assert "credits.depleted" not in keys


# ── Scenario 5c: is_free_tier_model (local-data-only check) ──────────────────


class TestIsFreeTierModel:
    def test_free_suffix_is_free(self):
        from agent.credits_tracker import is_free_tier_model

        assert is_free_tier_model("nvidia/nemotron-3-ultra:free") is True
        assert is_free_tier_model("Hermes-4-70B:free", "https://inference-api.nousresearch.com") is True

    def test_empty_or_paid_model_is_not_free(self):
        from agent.credits_tracker import is_free_tier_model

        assert is_free_tier_model("") is False
        assert is_free_tier_model("Hermes-4-405B") is False

    def test_pricing_cache_peek_zero_priced_model(self, monkeypatch):
        from agent.credits_tracker import is_free_tier_model
        import hermes_cli.models as models_mod

        # The picker keys the cache on the pre-/v1 root (get_pricing_for_provider
        # strips a trailing /v1 before fetch_models_with_pricing).
        monkeypatch.setattr(
            models_mod,
            "_pricing_cache",
            {
                "https://inference-api.nousresearch.com": {
                    "some/zero-priced": {"prompt": "0", "completion": "0"},
                    "some/paid": {"prompt": "0.000001", "completion": "0.000002"},
                }
            },
        )
        # The agent holds the /v1-suffixed URL (DEFAULT_NOUS_INFERENCE_URL) —
        # the helper must normalize it down to the picker's cache key.
        base = "https://inference-api.nousresearch.com/v1"
        assert is_free_tier_model("some/zero-priced", base) is True
        assert is_free_tier_model("some/paid", base) is False
        # Pre-stripped and trailing-slash variants resolve to the same key.
        assert is_free_tier_model("some/zero-priced", "https://inference-api.nousresearch.com/") is True
        assert is_free_tier_model("some/zero-priced", "https://inference-api.nousresearch.com/v1/") is True

    def test_cache_miss_is_not_free_and_no_fetch(self, monkeypatch):
        from agent.credits_tracker import is_free_tier_model
        import hermes_cli.models as models_mod

        monkeypatch.setattr(models_mod, "_pricing_cache", {})

        def _boom(*args, **kwargs):  # any network attempt fails the test
            raise AssertionError("is_free_tier_model must never hit the network")

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        assert is_free_tier_model("some/model", "https://inference-api.nousresearch.com/v1") is False

    def test_exception_fails_open_to_false(self, monkeypatch):
        from agent.credits_tracker import is_free_tier_model
        import hermes_cli.models as models_mod

        class _Exploding:
            def get(self, *_a, **_kw):
                raise RuntimeError("boom")

        monkeypatch.setattr(models_mod, "_pricing_cache", _Exploding())
        assert is_free_tier_model("some/model", "https://inference-api.nousresearch.com") is False


# ── Scenario 6: denominator none (uf is None) ────────────────────────────────


class TestDenominatorNone:
    def test_no_warn90_when_uf_none(self):
        latch = fresh_latch()
        s = state_with_fraction(None)
        to_show, to_clear = evaluate_credits_notices(s, latch)
        assert all(n.key != "credits.usage" for n in to_show)
        assert "credits.usage" not in to_clear

    def test_no_grant_spent_when_uf_none(self):
        latch = fresh_latch()
        s = CreditsState(
            subscription_limit_micros=None,
            denominator_kind="none",
            purchased_micros=5_000_000,
            purchased_usd="5.00",
        )
        to_show, to_clear = evaluate_credits_notices(s, latch)
        assert all(n.key != "credits.grant_spent" for n in to_show)

    def test_warn90_clears_when_uf_becomes_none(self):
        """If warn90 was active and uf becomes None, it should clear."""
        latch = fresh_latch()
        # Establish usage notice active: cross below → above
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(state_with_fraction(0.95), latch)
        assert "credits.usage" in latch["active"]
        # Now uf becomes None (denominator changed to "none")
        s_none = state_with_fraction(None)
        to_show, to_clear = evaluate_credits_notices(s_none, latch)
        assert "credits.usage" in to_clear
        assert "credits.usage" not in latch["active"]


# ── Scenario 7: copy / verbatim USD strings ──────────────────────────────────


class TestNoticeCopy:
    def test_warn90_contains_verbatim_subscription_limit_usd(self):
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        s = state_with_fraction(0.95, subscription_limit_usd="20.00")
        to_show, _ = evaluate_credits_notices(s, latch)
        warn_notice = next(n for n in to_show if n.key == "credits.usage")
        assert "$20.00" in warn_notice.text
        assert "cap" in warn_notice.text

    def test_grant_spent_contains_verbatim_purchased_usd(self):
        latch = fresh_latch()
        s = state_with_fraction(
            1.0,
            denominator_kind="subscription_cap",
            purchased_micros=12_340_000,
            purchased_usd="12.34",
        )
        to_show, _ = evaluate_credits_notices(s, latch)
        grant_notice = next(n for n in to_show if n.key == "credits.grant_spent")
        assert "$12.34" in grant_notice.text
        assert "top-up left" in grant_notice.text

    def test_depleted_mentions_usage_command(self):
        latch = fresh_latch()
        s = CreditsState(paid_access=False)
        to_show, _ = evaluate_credits_notices(s, latch)
        depleted_notice = next(n for n in to_show if n.key == "credits.depleted")
        assert "/usage" in depleted_notice.text


# ── Scenario 8: severity order in a single call ──────────────────────────────


class TestSeverityOrder:
    def test_multiple_new_notices_ordered_ascending_severity(self):
        """grant_spent < depleted in to_show when both fire in one call.

        (usage is suppressed here: purchased>0 — see TestTopUpSuppression.
        usage + grant_spent are now mutually exclusive by design.)
        """
        latch = {"active": set(), "seen_below_90": True, "usage_band": None}

        # Build state: subscription_cap, uf >= 1.0, purchased_micros > 0, NOT paid_access
        # grant_cond: subscription_cap + uf >= 1.0 + purchased > 0 ✓
        # depleted_cond: not paid_access ✓
        # usage band: suppressed (purchased > 0)
        s = CreditsState(
            subscription_limit_micros=20_000_000,
            subscription_limit_usd="20.00",
            subscription_micros=0,  # uf = 1.0
            denominator_kind="subscription_cap",
            purchased_micros=5_000_000,
            purchased_usd="5.00",
            paid_access=False,
        )
        to_show, _ = evaluate_credits_notices(s, latch)
        keys = [n.key for n in to_show]
        assert "credits.usage" not in keys
        assert "credits.grant_spent" in keys
        assert "credits.depleted" in keys
        # Ascending severity: grant_spent before depleted
        assert keys.index("credits.grant_spent") < keys.index("credits.depleted")

    def test_usage_before_depleted_without_topup(self):
        """With no top-up funds, usage fires and precedes depleted."""
        latch = {"active": set(), "seen_below_90": True, "usage_band": None}
        s = CreditsState(
            subscription_limit_micros=20_000_000,
            subscription_limit_usd="20.00",
            subscription_micros=0,  # uf = 1.0
            denominator_kind="subscription_cap",
            purchased_micros=0,
            purchased_usd="0.00",
            paid_access=False,
        )
        to_show, _ = evaluate_credits_notices(s, latch)
        keys = [n.key for n in to_show]
        assert "credits.usage" in keys
        assert "credits.depleted" in keys
        assert keys.index("credits.usage") < keys.index("credits.depleted")


# ── Scenario 8b: top-up suppression of the usage gauge ───────────────────────


class TestTopUpSuppression:
    """purchased_micros > 0 suppresses the sub-cap usage gauge: the cap is the
    wrong denominator for an account that can keep spending top-up funds."""

    def test_no_usage_band_with_topup_at_90pct(self):
        latch = fresh_latch()
        evaluate_credits_notices(
            state_with_fraction(0.10, purchased_micros=5_000_000, purchased_usd="5.00"),
            latch,
        )
        to_show, to_clear = evaluate_credits_notices(
            state_with_fraction(0.95, purchased_micros=5_000_000, purchased_usd="5.00"),
            latch,
        )
        assert all(n.key != "credits.usage" for n in to_show)
        assert latch["usage_band"] is None

    def test_topup_landing_mid_session_clears_active_band(self):
        """A showing 90% warn must clear when a top-up lands (purchased 0 → >0)."""
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(state_with_fraction(0.95), latch)
        assert latch["usage_band"] == 90
        to_show, to_clear = evaluate_credits_notices(
            state_with_fraction(0.95, purchased_micros=10_000_000, purchased_usd="10.00"),
            latch,
        )
        assert "credits.usage" in to_clear
        assert latch["usage_band"] is None
        assert all(n.key != "credits.usage" for n in to_show)

    def test_band_resumes_after_topup_spent(self):
        """purchased back to 0 with usage still in-band → gauge resumes."""
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(
            state_with_fraction(0.95, purchased_micros=10_000_000, purchased_usd="10.00"),
            latch,
        )
        assert latch["usage_band"] is None
        to_show, _ = evaluate_credits_notices(state_with_fraction(0.95), latch)
        n = next(n for n in to_show if n.key == "credits.usage")
        assert "90%" in n.text
        assert latch["usage_band"] == 90

    def test_grant_spent_still_fires_with_topup(self):
        """Suppression only affects the gauge — grant_spent (which NEEDS purchased>0)
        is untouched."""
        latch = fresh_latch()
        s = state_with_fraction(
            1.0,
            denominator_kind="subscription_cap",
            purchased_micros=12_340_000,
            purchased_usd="12.34",
        )
        to_show, _ = evaluate_credits_notices(s, latch)
        keys = [n.key for n in to_show]
        assert "credits.grant_spent" in keys
        assert "credits.usage" not in keys

    def test_depleted_unaffected_by_topup_suppression(self):
        latch = fresh_latch()
        s = CreditsState(paid_access=False, purchased_micros=5_000_000, purchased_usd="5.00")
        to_show, _ = evaluate_credits_notices(s, latch)
        assert any(n.key == "credits.depleted" for n in to_show)


# ── Invariant: never fire + clear same key in one call ────────────────────────


class TestNoFireAndClearSameKey:
    def test_usage_never_both_fired_and_cleared(self):
        latch = fresh_latch()
        # Run many state transitions; across each, assert no key is in both lists
        states = [
            state_with_fraction(0.10),
            state_with_fraction(0.95),
            state_with_fraction(0.10),
            state_with_fraction(0.95),
            state_with_fraction(None),
        ]
        for s in states:
            to_show, to_clear = evaluate_credits_notices(s, latch)
            fired_keys = {n.key for n in to_show}
            cleared_keys = set(to_clear)
            overlap = fired_keys & cleared_keys
            assert not overlap, f"Key(s) both fired and cleared: {overlap}"

    def test_depleted_never_both_fired_and_cleared(self):
        latch = fresh_latch()
        states = [
            CreditsState(paid_access=False),
            CreditsState(paid_access=True),
            CreditsState(paid_access=False),
        ]
        for s in states:
            to_show, to_clear = evaluate_credits_notices(s, latch)
            fired_keys = {n.key for n in to_show}
            cleared_keys = set(to_clear)
            overlap = fired_keys & cleared_keys
            assert not overlap, f"Key(s) both fired and cleared: {overlap}"


# ── Scenario 9: escalating usage bands (50 → 75 → 90) ────────────────────────


class TestUsageBands:
    """The usage notice shows the HIGHEST crossed band as a single escalating line."""

    def _band_text(self, to_show):
        n = next((n for n in to_show if n.key == "credits.usage"), None)
        return n.text if n else None

    def test_50_band_fires_info(self):
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)  # prime
        to_show, _ = evaluate_credits_notices(state_with_fraction(0.55), latch)
        n = next(n for n in to_show if n.key == "credits.usage")
        assert "50%" in n.text and n.level == "info"
        assert latch["usage_band"] == 50

    def test_75_band_fires_warn(self):
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        to_show, _ = evaluate_credits_notices(state_with_fraction(0.80), latch)
        n = next(n for n in to_show if n.key == "credits.usage")
        assert "75%" in n.text and n.level == "warn"
        assert latch["usage_band"] == 75

    def test_climb_replaces_band(self):
        """Climbing 50→75→90 replaces the single line (clear old + show new)."""
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        # 55% → 50 band
        evaluate_credits_notices(state_with_fraction(0.55), latch)
        assert latch["usage_band"] == 50
        # 80% → climbs to 75, clearing the 50 line
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.80), latch)
        assert "credits.usage" in to_clear
        assert "75%" in self._band_text(to_show)
        assert latch["usage_band"] == 75
        # 95% → climbs to 90
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.95), latch)
        assert "credits.usage" in to_clear
        assert "90%" in self._band_text(to_show)
        assert latch["usage_band"] == 90

    def test_step_down_on_recovery(self):
        """Recovering steps the band back down, then clears below the lowest band."""
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(state_with_fraction(0.95), latch)
        assert latch["usage_band"] == 90
        # drop to 80% → steps down to 75
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.80), latch)
        assert "credits.usage" in to_clear
        assert "75%" in self._band_text(to_show)
        # drop to 55% → steps down to 50
        to_show, _ = evaluate_credits_notices(state_with_fraction(0.55), latch)
        assert "50%" in self._band_text(to_show)
        # drop below 50% → clears entirely
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.10), latch)
        assert "credits.usage" in to_clear
        assert latch["usage_band"] is None

    def test_no_refire_same_band(self):
        latch = fresh_latch()
        evaluate_credits_notices(state_with_fraction(0.10), latch)
        evaluate_credits_notices(state_with_fraction(0.80), latch)  # fires 75
        # still 80% → same band, no re-emit, no clear
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.80), latch)
        assert all(n.key != "credits.usage" for n in to_show)
        assert "credits.usage" not in to_clear

    def test_exact_band_boundaries_inclusive(self):
        """Thresholds are inclusive: exactly 0.50 / 0.75 / 0.90 land in their band."""
        for uf, want in [(0.50, 50), (0.75, 75), (0.90, 90)]:
            latch = fresh_latch()
            latch["seen_below_90"] = True  # allow firing
            evaluate_credits_notices(state_with_fraction(uf), latch)
            assert latch["usage_band"] == want, (uf, latch["usage_band"])

    def test_open_below_lowest_band_no_notice(self):
        latch = fresh_latch()
        to_show, to_clear = evaluate_credits_notices(state_with_fraction(0.30), latch)
        assert all(n.key != "credits.usage" for n in to_show)
        assert latch["usage_band"] is None

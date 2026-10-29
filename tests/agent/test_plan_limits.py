"""agent/plan_limits.py: the one rule (what a full allowance blocks), the suggestions after a usage-window
refusal (the plan's recommended models that still work), and the copy for surfaces without buttons.

Hermes never switches model on its own; these only describe the limit and what to offer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import plan_limits as pl
from agent.usage_gauges import SCOPE_ALL, SCOPE_TOP_TIER, WindowGauge

RESET = 4_000_000_000_000
_REAL_FITS_CONTEXT = pl._fits_context
_REAL_PORTAL_FREE_MODELS = pl._portal_free_models

PRICING = {
    "example/model-a": {"prompt": "0.00000001", "completion": "0.0000001", "name": "Model A"},
    "example/model-b": {"prompt": "0.0000004", "completion": "0.000004", "name": "Model B"},
    "example/model-c": {"prompt": "0.000005", "completion": "0.000025", "top_tier": True, "name": "Model C"},
    "example/model-free:free": {"prompt": "0", "completion": "0", "name": "Model Free"},
}
FEATURED = ("example/model-a", "example/model-b", "example/model-c", "example/model-free:free")


def _g(gid: str, used_bp: int, *, scope: str = SCOPE_ALL, resets: int = RESET) -> WindowGauge:
    return WindowGauge(id=gid, scope=scope, label=gid, used_bp=used_bp, resets_at_ms=resets)


DAY_FULL = (_g("day", 10_000), _g("week", 4_000), _g("top_tier", 2_000, scope=SCOPE_TOP_TIER))
TOP_FULL = (_g("day", 3_000), _g("week", 2_000), _g("top_tier", 10_000, scope=SCOPE_TOP_TIER))
HEALTHY = (_g("day", 3_000), _g("week", 2_000), _g("top_tier", 2_000, scope=SCOPE_TOP_TIER))


@pytest.fixture(autouse=True)
def _no_caches(monkeypatch):
    """Every test names its own catalog and plan; nothing reads the user's disk caches."""
    monkeypatch.setattr(pl, "_portal_free_models", list)
    monkeypatch.setattr(pl, "_plan_usage", lambda: SimpleNamespace(featured_model_ids=FEATURED))
    monkeypatch.setattr(pl, "cached_pricing", lambda base_url: PRICING)
    monkeypatch.setattr(pl, "_fits_context", lambda model, needed, base_url: True)


# ── the one rule ─────────────────────────────────────────────────────────────────────────────────

class TestTheRule:
    def test_free_models_are_never_blocked(self):
        for model in ("example/model-free:free", "any/x:free"):
            assert pl.is_free(model, PRICING)
            assert not pl.is_blocked(model, DAY_FULL, PRICING)
            assert not pl.is_blocked(model, TOP_FULL, PRICING)

    def test_a_full_all_scope_gauge_blocks_every_paid_model(self):
        for model in ("example/model-a", "example/model-b", "example/model-c"):
            assert pl.is_blocked(model, DAY_FULL, PRICING)

    def test_a_full_top_tier_gauge_blocks_only_marked_models(self):
        assert pl.is_blocked("example/model-c", TOP_FULL, PRICING)
        assert not pl.is_blocked("example/model-b", TOP_FULL, PRICING)
        assert not pl.is_blocked("example/model-a", TOP_FULL, PRICING)

    def test_an_unmarked_or_unpriced_model_is_not_top_tier_fail_open(self):
        assert not pl.is_top_tier("example/model-b", PRICING)
        assert not pl.is_top_tier("unknown/model", PRICING)
        assert not pl.is_top_tier("example/model-c", {})

    def test_the_usage_level_word_marks_top_tier_too(self):
        """Today's gateway sends both marks; either one alone is enough, and a free row never is."""
        level_only = {"example/model-d": {"prompt": "0.000005", "completion": "0.000025", "usage_level": "top_tier"}}
        assert pl.is_top_tier("example/model-d", level_only)
        free_marked = {"example/model-e:free": {"prompt": "0", "completion": "0", "usage_level": "top_tier"}}
        assert not pl.is_top_tier("example/model-e:free", free_marked)

    def test_nothing_full_blocks_nothing(self):
        assert not any(pl.is_blocked(m, HEALTHY, PRICING) for m in PRICING)

    def test_of_several_full_gauges_the_one_that_frees_up_last_binds(self):
        gauges = (_g("day", 10_000, resets=1_000), _g("week", 10_000, resets=RESET))
        assert pl.blocking_gauge("example/model-b", gauges, free=False, top_tier=False).id == "week"

    def test_labels_come_from_the_catalog_name(self):
        assert pl.model_label("example/model-a", PRICING) == "Model A"
        assert pl.model_label("no/name", PRICING) == "no/name"


# ── suggestions ──────────────────────────────────────────────────────────────────────────────────

class TestSuggestions:
    def test_top_tier_limit_offers_included_recommended_models_in_plan_order(self):
        out = pl.suggest_models("example/model-c", TOP_FULL, pricing=PRICING, featured=FEATURED)
        assert [(s.model, s.label, s.note) for s in out] == [
            ("example/model-a", "Model A", pl.NOTE_INCLUDED),
            ("example/model-b", "Model B", pl.NOTE_INCLUDED),
        ]
        assert all(s.provider == "nous" for s in out)

    def test_window_limit_offers_only_free_models(self):
        out = pl.suggest_models("example/model-b", DAY_FULL, pricing=PRICING, featured=FEATURED)
        assert [(s.model, s.note) for s in out] == [("example/model-free:free", pl.NOTE_FREE)]

    def test_the_portal_free_picks_follow_the_plan_list(self, monkeypatch):
        monkeypatch.setattr(pl, "_portal_free_models", lambda: ["example/model-free:free", "other/y:free"])
        out = pl.suggest_models("example/model-b", DAY_FULL, pricing=PRICING, featured=FEATURED)
        assert [s.model for s in out] == ["example/model-free:free", "other/y:free"]

    def test_never_the_current_model(self):
        featured = ("example/model-free:free",)
        assert pl.suggest_models("example/model-free:free", DAY_FULL, pricing=PRICING, featured=featured) == ()

    def test_at_most_two(self):
        featured = ("a/1:free", "a/2:free", "a/3:free")
        assert len(pl.suggest_models("example/model-b", DAY_FULL, pricing=PRICING, featured=featured)) == 2

    def test_a_model_whose_context_cannot_hold_the_conversation_is_skipped(self, monkeypatch):
        monkeypatch.setattr(pl, "_fits_context", lambda model, needed, base_url: model != "example/model-a")
        out = pl.suggest_models("example/model-c", TOP_FULL, pricing=PRICING, featured=FEATURED, needed_tokens=500_000)
        # The large model is still blocked; the free pick fills the second slot.
        assert [s.model for s in out] == ["example/model-b", "example/model-free:free"]

    def test_fits_context_leaves_headroom_for_the_reply(self, monkeypatch):
        import agent.model_metadata as mm
        monkeypatch.setattr(mm, "get_model_context_length", lambda *a, **k: 100_000)
        assert _REAL_FITS_CONTEXT("m", 90_000, "")
        assert not _REAL_FITS_CONTEXT("m", 95_000, "")
        assert _REAL_FITS_CONTEXT("m", None, "")


class TestCreditsOption:
    def test_a_balance_with_keep_going_off_offers_use_credits(self):
        state = SimpleNamespace(paused_reason=None, purchased_micros=5_000_000, keep_going_on_credits=False)
        assert pl.credits_option(state).action == "use_credits"

    def test_no_balance_or_setting_on_offers_add_credits(self):
        for state in (SimpleNamespace(paused_reason=None, purchased_micros=5_000_000, keep_going_on_credits=True),
                      SimpleNamespace(paused_reason=None, purchased_micros=0, keep_going_on_credits=None)):
            assert pl.credits_option(state).action == "add_credits"

    def test_keep_going_off_and_no_balance_needs_both_steps(self):
        """The gateway refuses with Keep going off whatever the balance: a top-up alone would still be refused."""
        state = SimpleNamespace(paused_reason=None, purchased_micros=0, keep_going_on_credits=False)
        assert pl.credits_option(state).action == "add_credits_and_keep_going"
        assert pl.credits_action(False, None) == "add_credits_and_keep_going"  # unknown balance counts as none

    def test_paused_offers_nothing(self):
        state = SimpleNamespace(paused_reason="payment_past_due", purchased_micros=5_000_000, keep_going_on_credits=False)
        assert pl.credits_option(state) is None


# ── limit_hit ────────────────────────────────────────────────────────────────────────────────────

def _agent(model: str, gauges, **state_kw):
    state = SimpleNamespace(is_usage_window=True, gauges=gauges, paused_reason=None, purchased_micros=0,
                            keep_going_on_credits=None, **state_kw)
    return SimpleNamespace(_credits_state=state, model=model, base_url="https://inference-api.nousresearch.com/v1")


class TestLimitHit:
    def test_a_top_tier_refusal(self):
        hit = pl.limit_hit(_agent("example/model-c", TOP_FULL), message="You've hit your monthly large-model limit")
        assert hit is not None and hit.gauge_id == "top_tier" and hit.scope == SCOPE_TOP_TIER
        assert hit.limit_name == "monthly large-model" and hit.resets_at_ms == RESET
        assert [s.model for s in hit.suggestions] == ["example/model-a", "example/model-b"]
        payload = hit.to_payload()
        assert payload["message"] == "You've hit your monthly large-model limit"
        assert payload["suggestions"][0] == {"model": "example/model-a", "label": "Model A",
                                             "note": "included", "provider": "nous"}
        assert payload["credits"] == {"action": "add_credits"} and payload["current_model"] == "example/model-c"

    def test_a_daily_refusal(self):
        hit = pl.limit_hit(_agent("example/model-b", DAY_FULL), message="x")
        assert hit.gauge_id == "day" and hit.limit_name == "daily"
        assert [s.model for s in hit.suggestions] == ["example/model-free:free"]

    def test_an_unmarked_model_refused_with_the_top_tier_spent_still_explains_it(self, monkeypatch):
        """The 402 says this model was refused; with no mark in the cache, the full top-tier gauge is the
        only explanation."""
        monkeypatch.setattr(pl, "cached_pricing", lambda base_url: {})
        hit = pl.limit_hit(_agent("example/model-c", TOP_FULL), message="x")
        assert hit is not None and hit.gauge_id == "top_tier"

    def test_no_full_gauge_or_no_window_plan_is_no_hit(self):
        assert pl.limit_hit(_agent("example/model-b", HEALTHY), message="x") is None
        not_window = SimpleNamespace(_credits_state=SimpleNamespace(is_usage_window=False), model="m", base_url="")
        assert pl.limit_hit(not_window, message="x") is None
        assert pl.limit_hit(SimpleNamespace(_credits_state=None), message="x") is None

    def test_never_raises(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("cache broke")
        monkeypatch.setattr(pl, "suggest_models", boom)
        assert pl.limit_hit(_agent("example/model-b", DAY_FULL), message="x") is None


# ── copy ─────────────────────────────────────────────────────────────────────────────────────────

class TestNextSteps:
    def _payload(self, **kw):
        base = {"gauge_id": "day", "scope": "all", "limit_name": "daily", "resets_at_ms": RESET, "message": "m",
                "suggestions": [{"model": "example/model-free:free", "label": "Model Free", "note": "free", "provider": "nous"}],
                "credits": {"action": "add_credits"}}
        base.update(kw)
        return base

    def test_terminal_copy_names_the_model_and_topup(self):
        assert pl.limit_next_steps_from_payload(self._payload()) == (
            "Switch to Model Free (free) with /model, or add credits with /topup.")

    def test_use_credits_names_the_setting(self):
        text = pl.limit_next_steps_from_payload(self._payload(credits={"action": "use_credits"}))
        assert text.endswith("or turn on Keep going in /topup to use credits.")

    def test_keep_going_off_with_no_balance_names_both_steps(self):
        both = self._payload(credits={"action": "add_credits_and_keep_going"})
        assert pl.limit_next_steps_from_payload(both).endswith("or add credits and turn on Keep going in /topup.")
        assert pl.limit_next_steps_from_payload(both, topup_command=None, portal=True).endswith(
            "or add credits and turn on Keep going in the Portal.")

    def test_messaging_points_at_the_portal_and_uses_the_platform_prefix(self):
        text = pl.limit_next_steps_from_payload(self._payload(), model_command="!model", topup_command=None, portal=True)
        assert text == "Switch to Model Free (free) with !model, or add credits in the Portal."
        assert "/topup" not in text

    def test_a_chat_refusal_points_at_the_portal_and_reads_the_reset_as_a_duration(self):
        """A gateway's clock is not the reader's time zone: the reset reads "resets in 3h" in a chat."""
        import time

        from agent.credits_tracker import format_window_reset
        resets_at = time.time() + 3 * 3600 + 30
        text = f"You've hit your daily limit · resets {format_window_reset(int(resets_at * 1000))} · /topup to add credits"
        assert pl.chat_refusal_wording(text, resets_at) == "You've hit your daily limit · resets in 3h · add credits in the Portal"
        assert pl.chat_refusal_wording("Limit · /topup to add credits") == "Limit · add credits in the Portal"

    def test_no_suggestion_still_says_how_to_change_model(self):
        text = pl.limit_next_steps_from_payload(self._payload(suggestions=[], credits=None))
        assert text == "Pick another model with /model."

    def test_markup_bolds_the_commands(self):
        text = pl.limit_next_steps_from_payload(self._payload(), markup=True)
        assert "[bold]/model[/]" in text and "[bold]/topup[/]" in text

    def test_a_malformed_payload_degrades_to_the_generic_line(self):
        text = pl.limit_next_steps_from_payload({"suggestions": [{"label": "no model"}, "junk"], "credits": "junk"})
        assert text == "Pick another model with /model."


def test_free_picks_come_from_this_profiles_own_portal_entry(monkeypatch):
    """One process holds several profiles' recommended lists side by side: a limit card offers only the
    calling profile's own Portal picks, else its own disk copy; another profile's entry never answers."""
    from hermes_cli import models as m
    portal = "https://portal.example.test"
    monkeypatch.setattr(m, "_resolve_nous_portal_url", lambda: portal)
    other = {"freeRecommendedModels": [{"modelName": "other/profile:free"}]}
    mine = {"freeRecommendedModels": [{"modelName": "mine/model:free"}]}
    monkeypatch.setattr(m, "_nous_recommended_cache", {})
    m._nous_recommended_cache[("other-profile", portal)] = (other, 0.0)
    assert _REAL_PORTAL_FREE_MODELS() == []  # this profile holds nothing, in memory or on disk
    m._nous_recommended_cache[(m._pricing_profile_key(), portal)] = (mine, 0.0)
    m._nous_recommended_cache[("other-profile", "https://other-portal.test")] = (other, 0.0)
    assert _REAL_PORTAL_FREE_MODELS() == ["mine/model:free"]

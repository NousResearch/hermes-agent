"""Unit tests for the Auto Router pure selection core (agent/auto_router.py).

No network — the classifier is an injected callable. Covers capability-vs-cost
separation, lenient score parsing, image hard-zeroing, per-task cache, the
known-family capability cards (incl. the gemini/minimax 'mini' substring trap),
and every safe-fallback path.
"""
import json

import pytest

from agent import auto_router as r


def _cands():
    return [
        r.Candidate(id="cheap", cost=0.3, supports_images=False, card="cheap fast"),
        r.Candidate(id="mid", cost=1.0, supports_images=False, card="mid"),
        r.Candidate(id="strong", cost=5.0, supports_images=True, card="frontier + vision"),
    ]


def _classify(scores):
    def _c(system_prompt, user_content):
        return "prose before " + json.dumps({"scores": scores, "reasoning": "x"}) + " after"
    return _c


@pytest.fixture(autouse=True)
def _clear_cache():
    r.reset_cache()
    yield
    r.reset_cache()


# ---- pure helpers ---------------------------------------------------------

def test_clamp01():
    assert r.clamp01(-3) == 0.0
    assert r.clamp01(9) == 1.0
    assert r.clamp01(0.42) == 0.42
    assert r.clamp01("nope") == 0.0


@pytest.mark.parametrize("text,expect", [
    ('{"scores": {"cheap": 0.9, "mid": 0.5, "strong": 0.2}}', {"cheap": 0.9, "mid": 0.5, "strong": 0.2}),
    ('```json\n{"scores": {"cheap": 1, "mid": 0, "strong": 0}}\n```', {"cheap": 1.0, "mid": 0.0, "strong": 0.0}),
    ('Sure! {"scores": {"cheap": 0.7, "mid": 0.7, "strong": 0.7}} done', {"cheap": 0.7, "mid": 0.7, "strong": 0.7}),
    ('{"scores": {"cheap": 0.9}}', {"cheap": 0.9, "mid": 0.0, "strong": 0.0}),  # missing -> 0
    ('not json at all', {}),
    ('', {}),
])
def test_parse_scores(text, expect):
    got = r.parse_scores(text, ["cheap", "mid", "strong"])
    assert got == expect


def test_router_pick_cheapest_among_viable():
    cands = _cands()
    pick, score, why = r.router_pick({"cheap": 0.9, "mid": 0.92, "strong": 0.95}, cands, 0.7, False)
    assert pick == "cheap"  # all viable -> cheapest


def test_router_pick_escalates_when_cheap_below_bar():
    cands = _cands()
    pick, _, _ = r.router_pick({"cheap": 0.4, "mid": 0.55, "strong": 0.95}, cands, 0.7, False)
    assert pick == "strong"  # only strong clears the bar


def test_router_pick_below_bar_takes_highest():
    cands = _cands()
    pick, _, why = r.router_pick({"cheap": 0.3, "mid": 0.5, "strong": 0.6}, cands, 0.7, False)
    assert pick == "strong" and "below bar" in why


def test_router_pick_image_hard_zeroes_incapable():
    cands = _cands()
    pick, _, _ = r.router_pick({"cheap": 0.99, "mid": 0.99, "strong": 0.8}, cands, 0.7, has_images=True)
    assert pick == "strong"  # only vision-capable can win on image tasks


def test_router_pick_no_usable_score():
    cands = _cands()
    pick, score, _ = r.router_pick({"cheap": 0.0, "mid": 0.0, "strong": 0.0}, cands, 0.7, False)
    assert pick is None and score == 0.0


def test_fallback_id():
    cands = _cands()
    assert r.fallback_id(cands) == "cheap"           # cheapest
    assert r.fallback_id(cands, default="mid") == "mid"
    assert r.fallback_id(cands, default="ghost") == "cheap"  # bad default -> cheapest
    assert r.fallback_id([]) is None


# ---- capability cards -----------------------------------------------------

@pytest.mark.parametrize("model_id,needle", [
    ("MiniMax-M3", "MiniMax"),
    ("minimax-m3", "MiniMax"),
    ("xiaomi/mimo", "MiMo"),
    ("gpt-5.5", "frontier"),
    ("gpt-5.5-codex", "frontier"),
    ("anthropic/claude-opus-4.6", "Opus"),
    ("anthropic/claude-3-7-sonnet", "Sonnet"),
    ("google/gemini-3-pro", "Gemini Pro"),
    ("google/gemini-3-flash-preview", "Small"),     # flash -> small tier
    ("gpt-4o-mini", "Small"),                        # mini token -> small (NOT 'gemini')
    ("anthropic/claude-3-5-haiku", "Small"),
    ("meta-llama/llama-3.1-8b-instruct", "Small"),   # 8b -> small
    ("meta-llama/llama-3.1-70b-instruct", "Llama"),
    ("totally-unknown-xyz", "unknown"),
])
def test_default_card_for(model_id, needle):
    assert needle.lower() in r.default_card_for(model_id).lower()


def test_candidates_from_config_autofills_cards():
    cfg = {"candidates": [
        {"model": "google/gemini-3-flash-preview", "provider": "openrouter", "cost": 0.3},
        {"model": "gpt-5.5", "provider": "codex", "cost": 5.0, "supports_images": True},
        {"model": "custom", "cost": 2.0, "card": "my own card"},
    ]}
    cands = r.candidates_from_config(cfg)
    assert len(cands) == 3
    by = {c.id: c for c in cands}
    assert "Small" in by["google/gemini-3-flash-preview"].card
    assert by["gpt-5.5"].supports_images and "frontier" in by["gpt-5.5"].card.lower()
    assert by["custom"].card == "my own card"  # explicit card preserved


# ---- select() orchestration ----------------------------------------------

def test_select_single_candidate_skips_classifier():
    def boom(s, u):
        raise AssertionError("classifier must not run for a single candidate")
    assert r.select("t", [r.Candidate(id="only")], classify=boom) == "only"


def test_select_no_classifier_uses_fallback():
    assert r.select("t", _cands(), classify=None) == "cheap"


def test_select_routes_easy_to_cheap_hard_to_strong():
    assert r.select("easy", _cands(), classify=_classify({"cheap": 0.9, "mid": 0.9, "strong": 0.9}), tier="a") == "cheap"
    assert r.select("hard", _cands(), classify=_classify({"cheap": 0.3, "mid": 0.5, "strong": 0.95}), tier="b") == "strong"


def test_select_cache_hit_does_not_recall_classifier():
    calls = {"n": 0}
    def counting(s, u):
        calls["n"] += 1
        return json.dumps({"scores": {"cheap": 0.9, "mid": 0.9, "strong": 0.9}})
    a = r.select("same", _cands(), classify=counting, tier="t")
    b = r.select("same", _cands(), classify=counting, tier="t")
    assert a == b == "cheap" and calls["n"] == 1


def test_select_garbage_output_falls_back():
    assert r.select("t", _cands(), classify=lambda s, u: "???", tier="g") == "cheap"


def test_select_classifier_exception_is_safe():
    def raises(s, u):
        raise RuntimeError("boom")
    assert r.select("t", _cands(), classify=raises, tier="e") == "cheap"


def test_select_empty_candidates_returns_none():
    assert r.select("t", [], classify=None) is None

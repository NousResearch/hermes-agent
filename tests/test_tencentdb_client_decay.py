"""Tests for tencentdb_client decay & consolidation helpers.

Covers the pure functions (no gateway dependency). Run:
  python3 -m pytest tests/test_tencentdb_client_decay.py -q
"""
import sys
import time as _t
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import tencentdb_client as t  # noqa: E402


# ── decayed_score ────────────────────────────────────────────────────────

def test_zero_elapsed_equals_importance():
    assert t.decayed_score(0.7, 0, now=0, half_life_hours=24.0) == 0.7


def test_one_half_life_halves():
    assert abs(t.decayed_score(0.8, 0, now=24 * 3600, half_life_hours=24.0) - 0.4) < 1e-9


def test_two_half_lives_quarters():
    assert abs(t.decayed_score(1.0, 0, now=48 * 3600, half_life_hours=24.0) - 0.25) < 1e-9


def test_zero_half_life_disables_decay():
    assert t.decayed_score(0.5, 0, now=1e9, half_life_hours=0.0) == 0.5


def test_missing_timestamp_returns_importance():
    assert t.decayed_score(0.5, None) == 0.5


def test_iso_timestamp_parses():
    iso = "2026-08-05T07:27:16.651Z"
    epoch = t._parse_ts(iso)
    assert epoch is not None
    assert abs(t.decayed_score(1.0, iso, now=epoch + 7 * 24 * 3600, half_life_hours=168.0) - 0.5) < 1e-6


def test_negative_elapsed_clamps_to_zero():
    # timestamp in the future relative to now -> elapsed clamped to 0, no boost
    assert t.decayed_score(0.5, _t.time() + 99999, now=_t.time(), half_life_hours=24.0) == 0.5


# ── apply_decay ──────────────────────────────────────────────────────────

def test_decay_reorders_importance_and_age():
    now = _t.time()
    fake = [
        {"id": "a", "content": "old high", "score": 0.9, "timestamp": now - 20 * 24 * 3600},
        {"id": "b", "content": "fresh low", "score": 0.3, "timestamp": now - 3600},
        {"id": "c", "content": "mid", "score": 0.5, "timestamp": now - 3 * 24 * 3600},
    ]
    ranked = t.apply_decay(fake, half_life_hours=168.0, min_score=0.0)
    # c=0.5*2^-0.43=0.371 > b=0.3*2^-0.006=0.299 > a=0.9*2^-2.86=0.124
    assert [r["id"] for r in ranked] == ["c", "b", "a"]


def test_apply_decay_annotates_and_does_not_mutate():
    fake = [{"id": "x", "score": 0.5, "timestamp": _t.time()}]
    out = t.apply_decay(fake, min_score=0.0)
    assert "decayed_score" in out[0]
    assert "decayed_score" not in fake[0]  # not mutated by default


def test_min_score_filters_low():
    now = _t.time()
    fake = [
        {"id": "high", "score": 0.9, "timestamp": now - 30 * 24 * 3600},  # ~0.06 after 30d
        {"id": "fresh", "score": 0.9, "timestamp": now},                  # 0.9
    ]
    ranked = t.apply_decay(fake, min_score=0.1)
    assert [r["id"] for r in ranked] == ["fresh"]


# ── consolidate_hits ─────────────────────────────────────────────────────

def test_consolidate_folds_near_duplicates():
    dupes = [
        {"content": "pool should recycle below wait_timeout", "score": 0.5},
        {"content": "pool should recycle below wait_timeout yes", "score": 0.4},
        {"content": "totally different topic about indexing", "score": 0.3},
    ]
    kept = t.consolidate_hits(dupes, threshold=0.5)
    assert len(kept) == 2
    assert len(kept[0].get("consolidated_with", [])) == 1


def test_consolidate_no_match_when_distinct():
    items = [{"content": "alpha beta gamma delta"}, {"content": "w x y z"}]
    kept = t.consolidate_hits(items, threshold=0.5)
    assert len(kept) == 2


# ── hardening: never raises on malformed / hostile input ─────────────────

def test_decayed_score_garbage_does_not_raise():
    assert t.decayed_score("not-a-number", 0, now=0) == 0.0
    assert t.decayed_score(0.5, 0, half_life_hours="garbage") == 0.5
    assert t.decayed_score(0.5, 0, now="bad") == 0.5
    assert t.decayed_score(object(), object(), half_life_hours=object()) == 0.0


def test_apply_decay_malformed_rows_skipped():
    now = _t.time()
    rows = [
        None,
        "a string row",
        42,
        {"id": "ok", "score": 0.9, "timestamp": now},
        {"id": "badscore", "score": "garbage", "timestamp": now},  # non-numeric score
        {"id": "notimestamp", "score": 0.5},                       # missing timestamp
    ]
    out = t.apply_decay(rows, min_score=0.0)
    assert len(out) == 3
    assert {r.get("id") for r in out} == {"ok", "badscore", "notimestamp"}


def test_apply_decay_none_and_bad_args_no_raise():
    assert t.apply_decay(None) == []
    assert t.apply_decay([]) == []
    assert t.apply_decay(42) == []
    assert t.apply_decay([{"id": "x", "score": 0.5, "timestamp": _t.time()}],
                          half_life_hours="bad", min_score="bad")[0]["decayed_score"] == 0.5


def test_apply_decay_clamps_importance():
    now = _t.time()
    # importance > 1.0 (corrupt) should be clamped to <= 1.0, not inflate rank
    out = t.apply_decay([{"id": "a", "score": 99.0, "timestamp": now}], min_score=0.0)
    assert out[0]["decayed_score"] <= 1.0
    assert out[0]["decayed_score"] > 0.99  # clamped to ~1.0, not left at 99


def test_consolidate_malformed_does_not_raise():
    assert t.consolidate_hits(None) == []
    assert t.consolidate_hits([None, "junk", 7]) == []
    assert t.consolidate_hits([{"content": None}], threshold="bad") == [{"content": None}]


# ── write_core overwrite guard ───────────────────────────────────────────

class _FakeConfig:
    api_key = "x"


def test_write_core_refuses_when_content_exists(monkeypatch):
    """Without overwrite=True, write_core must not clobber existing content."""
    calls = []
    def fake_post(config, path, body):
        calls.append((path, body))
        if path == "/v2/core/read":
            return {"code": 0, "data": {"content": "EXISTING PERSONA"}}
        if path == "/v2/core/write":
            return {"code": 0, "data": {"accepted": True}}
        return {}
    monkeypatch.setattr(t, "_post_json", fake_post)
    res = t.write_core("researcher", "NEW PERSONA", config=_FakeConfig())
    assert res["skipped"] is True
    assert "overwrite=True" in res["reason"]
    assert res["existing"] == "EXISTING PERSONA"
    # must NOT have issued the write
    assert not any(p == "/v2/core/write" for p, _ in calls)


def test_write_core_proceeds_when_empty(monkeypatch):
    def fake_post(config, path, body):
        if path == "/v2/core/read":
            return {"code": 0, "data": {"content": None}}
        if path == "/v2/core/write":
            return {"code": 0, "data": {"accepted": True}}
        return {}
    monkeypatch.setattr(t, "_post_json", fake_post)
    res = t.write_core("researcher", "FIRST PERSONA", config=_FakeConfig())
    assert res.get("code") == 0  # proceeded to write


def test_write_core_overwrites_when_explicit(monkeypatch):
    def fake_post(config, path, body):
        if path == "/v2/core/read":
            return {"code": 0, "data": {"content": "EXISTING PERSONA"}}
        if path == "/v2/core/write":
            return {"code": 0, "data": {"accepted": True}}
        return {}
    monkeypatch.setattr(t, "_post_json", fake_post)
    res = t.write_core("researcher", "REPLACEMENT", overwrite=True, config=_FakeConfig())
    assert res.get("code") == 0  # explicit overwrite allowed


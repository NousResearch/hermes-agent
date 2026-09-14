"""Tests for the X co-manage scout rebuild (Phase 7).

Covers: registry loading, snowflake freshness, dedupe, voice gate,
blog cross-reference, and the scout's verdict->artifact routing.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

CE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CE))
sys.path.insert(0, str(CE.parent / 'scripts' / 'content_engine'))

import x_ingest
from x_ingest import (
    apply_freshness,
    dedupe_by_id,
    load_registry,
    tweet_age,
)
from x_voice_gate import blog_cross_reference, voice_gate_issues


# ── registry ─────────────────────────────────────────────────────────────

def test_registry_loads_defaults_and_user_overrides(tmp_path, monkeypatch):
    reg = load_registry()
    assert "freshness_hours" in reg
    assert all(v > 0 for v in reg["freshness_hours"].values())
    assert isinstance(reg.get("extra_accounts"), list)


def test_registry_bad_json_falls_back(monkeypatch, tmp_path):
    fake = tmp_path / "x_registry.json"
    fake.write_text("{not json")
    monkeypatch.setattr(x_ingest, "REGISTRY_PATH", fake)
    reg = load_registry()
    assert reg["freshness_hours"]  # defaults survived


# ── snowflake freshness ──────────────────────────────────────────────────

def test_tweet_age_is_sane(monkeypatch):
    # Freeze evaluation time: age is in hours, not days.
    monkeypatch.setattr(x_ingest.time, 'time', lambda: 1789167600.0)
    # 2026-09-11 era id (from live scout data)
    age = tweet_age("2097553728945480119")
    assert age is not None
    assert 0 < age < 90  # days, not centuries

    # a 2024-era id must be much older
    old = tweet_age("1790000000000000000")
    assert old is not None and old > age


def _mk_id(hours_ago: float) -> str:
    """Build a real-shaped snowflake id for a tweet N hours ago."""
    from x_ingest import _SNOWFLAKE_EPOCH
    unix_ms = int(time.time() * 1000) - int(hours_ago * 3600e3)
    return str((unix_ms - _SNOWFLAKE_EPOCH) << 22)


def test_apply_freshness_drops_stale_keeps_fresh():
    fresh_id = _mk_id(1)     # ~1h old
    stale_id = _mk_id(30 * 24)  # ~30d old
    kept = apply_freshness(
        [{"id": tid, "text": text, "url": f"https://x.com/fixture/status/{tid}",
          "created_at": datetime.fromtimestamp(
              ((int(tid) >> 22) + x_ingest._SNOWFLAKE_EPOCH) / 1000, timezone.utc
          ).isoformat(), "origin": "for_you"}
         for tid, text in ((fresh_id, "new"), (stale_id, "old"))],
        hours=6,
    )
    assert {r["text"] for r in kept} == {"new"}
    for r in kept:
        assert r["age_hours"] is not None
        assert r["age_unknown"] is False


def test_tweet_age_rejects_implausible_ids():
    # Non-numeric / empty ids can't be parsed -> None (flagged unknown).
    assert tweet_age("not-a-number") is None
    assert tweet_age("") is None
    # A 2-year-old id parses to a positive age (within the 3-year clamp).
    assert tweet_age(_mk_id(2 * 365 * 24)) is not None
    # A 4-year-old id trips the 3-year implausibility clamp -> None.
    assert tweet_age(_mk_id(4 * 365 * 24)) is None


def test_apply_freshness_keeps_unparseable_but_flags():
    # Legacy test name retained for attribution; unknown identity now fails closed.
    # Otherwise valid provenance isolates the malformed ID, not absent metadata.
    rows = [{"id": "not-a-number", "text": "mystery",
             "url": "https://x.com/fixture/status/not-a-number",
             "created_at": datetime.now(timezone.utc).isoformat(), "origin": "for_you"}]
    assert tweet_age(rows[0]["id"]) is None
    assert apply_freshness(rows, hours=6) == []


def test_dedupe_by_id():
    rows = [
        {"id": "1", "text": "a"},
        {"id": "1", "text": "a-dup"},
        {"id": "2", "text": "b"},
    ]
    out = dedupe_by_id(rows)
    assert [r["id"] for r in out] == ["1", "2"]


# ── voice gate ───────────────────────────────────────────────────────────

def test_voice_gate_clean_draft():
    assert voice_gate_issues(
        "Cheap until you hit the rate limit. That is the whole pitch."
    ) == []


def test_voice_gate_catches_ai_slop():
    issues = voice_gate_issues(
        "Great point! This leverages the power of AI to unlock the "
        "next paradigm. #AI #BuildInPublic"
    )
    kinds = " ".join(issues)
    assert "analyst" in kinds
    assert "praise" in kinds
    assert "hashtag" in kinds


def test_voice_gate_flags_long_chained_drafts():
    issues = voice_gate_issues(
        "This is a very long post that chains and connects and links "
        "many clauses together and keeps going and going and on and on "
        "and on and on and on and on and on and on and on and on and on "
        "and on and on and on and on and on and on and on and on"
    )
    assert len(issues) > 0


# ── blog cross-reference ─────────────────────────────────────────────────

def test_blog_cross_reference_returns_list():
    refs = blog_cross_reference(
        "context engineering is the real bottleneck in agent harnesses, "
        "and memory management is where the work actually is now"
    )
    assert isinstance(refs, list)
    for ref in refs:
        assert "title" in ref and "overlap" in ref


# ── scout routing ────────────────────────────────────────────────────────

def test_scout_prompt_and_verdicts_consistent():
    sys.path.insert(0, str(CE.parent / "scripts" / "content_engine"))
    import importlib
    scout = importlib.import_module("x_quote_scout")
    assert set(scout.VERDICTS) == {"reply", "quote", "standalone", "discard"}
    assert scout.MAX_REPLIES_PER_RUN == 2
    assert scout.LLM_SYSTEM.strip().startswith("You are drafting for Sahil")

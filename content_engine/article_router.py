"""Article router — decide what (if anything) becomes today's article.

Two modes:
  * deep_dive on the single best signal whose priority meets
    ARTICLE_DEEP_DIVE_THRESHOLD (default 7; collect_harness_changes emits 8).
  * digest roll-up of minor signals, only when the count meets
    ARTICLE_DIGEST_MIN_SIGNALS (default 4).
  * skip when nothing qualifies (no filler days).

The chosen signal ids are written to the in-memory state['used'] ledger
so the short-post track cannot re-use the same signal in the same run.
Cross-day dedup is handled by the pipeline, which seeds state['used'] with
the signal ids already used as articles inside the recency window (read from
topic_usage_log) and writes the day's choice back after a successful run. The
router stays pure: state['used'] is its single exclusion set whatever its
source.
"""
from __future__ import annotations
from typing import Optional

import activity_collector as ac
from config import (
    ARTICLE_ENABLED,
    ARTICLE_DEEP_DIVE_THRESHOLD,
    ARTICLE_DIGEST_MIN_SIGNALS,
)


def collect_signals(state: dict) -> list[dict]:
    """Run the existing collectors and return fresh signals.

    The router applies the in-memory `state['used']` ledger filter so the
    dedup semantics live in one place. We deliberately do not filter here
    so the test stub can return the raw list and the router's filter is
    the single source of truth.
    """
    out = ac.collect_all()
    return out.get("signals", [])


def choose(state: dict) -> Optional[dict]:
    """Decide the day's article plan. Returns the plan or None (skip).

    A plan is {mode, signals, pillar, title_hint} where mode is one of
    "deep_dive" or "digest".
    """
    if not ARTICLE_ENABLED:
        return None
    used = set(state.get("used", []) or [])
    sigs = [s for s in collect_signals(state) if s.get("signal_id") not in used]
    if not sigs:
        return None

    # Highest priority first; stable order on ties keeps tests deterministic.
    sigs.sort(key=lambda s: (-s.get("priority", 0), s.get("signal_id", "")))

    top = sigs[0]
    if top.get("priority", 0) >= ARTICLE_DEEP_DIVE_THRESHOLD:
        state.setdefault("used", []).append(top["signal_id"])
        return {
            "mode": "deep_dive",
            "signals": [top],
            "pillar": top.get("pillar", "agent_build_notes"),
            "title_hint": top.get("summary", ""),
        }

    # Accumulate minor signals that fail the deep-dive bar.
    minor = [s for s in sigs if s.get("priority", 0) < ARTICLE_DEEP_DIVE_THRESHOLD]
    if len(minor) >= ARTICLE_DIGEST_MIN_SIGNALS:
        for s in minor:
            state.setdefault("used", []).append(s["signal_id"])
        return {
            "mode": "digest",
            "signals": minor,
            "pillar": "edu_visual",
            "title_hint": "This fortnight in building Hermes",
        }

    return None

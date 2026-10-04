"""Compression outcome classification (item Q hardening).

Each case here is a failure actually observed in production on 2026-09-06,
not a hypothetical.
"""
import pytest

from agent.conversation_compression import (
    MIN_RECLAIM_TOKENS,
    classify_compression_outcome as classify,
)

OK_SPLIT = "in_place_committed"


def _c(**kw):
    base = dict(pre_tokens=80_000, post_tokens=50_000, pre_messages=79,
                post_messages=39, made_progress=True, split_status=OK_SPLIT)
    base.update(kw)
    return classify(**base)


# ── the healthy case: the real event of 2026-09-06 14:53 ────────────────
def test_a_real_compaction_commits_cleanly():
    """79->39 messages, ~73,920 -> ~53,970 tokens, one call, 54.3s."""
    status, failure, reclaimed, bloated, below = classify(
        pre_tokens=73_920, post_tokens=53_970, pre_messages=79,
        post_messages=39, made_progress=True, split_status=OK_SPLIT)
    assert status == "committed"
    assert failure is None
    assert reclaimed == 19_950
    assert not bloated and not below


# ── 1. no-op reported as committed ──────────────────────────────────────
def test_no_op_is_skipped_not_committed():
    """Observed: 50ms and 57ms attempts, no model call, 78->78 and 68->68,
    all reporting commit_status=committed with no failure_class."""
    status, failure, _, _, _ = _c(pre_messages=78, post_messages=78,
                                  made_progress=False,
                                  pre_tokens=87_578, post_tokens=87_578)
    assert status == "skipped"
    assert failure == "no_op"


def test_no_op_detected_even_if_token_estimate_wobbles():
    status, failure, _, _, _ = _c(pre_messages=68, post_messages=68,
                                  made_progress=False,
                                  pre_tokens=83_873, post_tokens=83_000)
    assert (status, failure) == ("skipped", "no_op")


# ── 2. summary larger than what it replaced ─────────────────────────────
def test_bloated_summary_is_rejected():
    """With thinking enabled the summariser measured out:in 1.48 — it returned
    more than it was given, so 'compression' grew the transcript."""
    status, failure, _, bloated, _ = _c(pre_tokens=60_000, post_tokens=70_000,
                                        post_messages=60)
    assert status == "skipped"
    assert failure == "summary_larger_than_source"
    assert bloated is True


def test_equal_size_counts_as_bloated():
    """No shrink is not a compaction."""
    status, failure, _, _, _ = _c(pre_tokens=60_000, post_tokens=60_000,
                                  post_messages=60)
    assert (status, failure) == ("skipped", "summary_larger_than_source")


# ── 3. reclaim below the minimum-worth-it threshold ─────────────────────
def test_below_min_reclaim_commits_but_is_flagged():
    """Every compaction invalidates the whole prompt cache on this model, so a
    tiny reclaim costs more than it saves — commit it, but flag it so the
    anti-thrash breaker can stop a loop."""
    status, failure, reclaimed, _, below = _c(
        pre_tokens=60_000, post_tokens=60_000 - (MIN_RECLAIM_TOKENS - 1),
        post_messages=70)
    assert status == "committed"          # the work is done and valid
    assert failure == "below_min_reclaim"
    assert below is True
    assert reclaimed == MIN_RECLAIM_TOKENS - 1


def test_reclaim_at_threshold_is_not_flagged():
    _, failure, _, _, below = _c(pre_tokens=60_000,
                                 post_tokens=60_000 - MIN_RECLAIM_TOKENS,
                                 post_messages=70)
    assert failure is None
    assert below is False


# ── precedence and regressions ──────────────────────────────────────────
def test_split_failure_takes_precedence_over_below_min_reclaim():
    _, failure, _, _, _ = _c(pre_tokens=60_000, post_tokens=59_999,
                             post_messages=70, split_status="failed_not_indexed")
    assert failure == "session_split_failed"


def test_no_op_takes_precedence_over_bloat():
    status, failure, _, _, _ = _c(pre_messages=50, post_messages=50,
                                  made_progress=False,
                                  pre_tokens=10_000, post_tokens=20_000)
    assert (status, failure) == ("skipped", "no_op")


def test_missing_token_estimate_does_not_crash_or_misclassify():
    """approx_tokens can be None; a real compaction must still commit."""
    status, failure, reclaimed, bloated, below = _c(pre_tokens=None)
    assert status == "committed"
    assert failure is None
    assert reclaimed is None
    assert not bloated and not below


def test_aborted_split_still_aborts():
    status, _, _, _, _ = _c(split_status="aborted")
    assert status == "aborted"

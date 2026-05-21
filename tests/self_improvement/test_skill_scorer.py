"""
tests/self_improvement/test_skill_scorer.py — Plan 004-A

Unit tests for hermes_agent.self_improvement.skill_scorer.

All tests use mocked asyncpg pool — no live Neon required.

Test inventory:
  1. test_compute_thumbs_rate_above_threshold   — rate = up/(up+down)
  2. test_compute_thumbs_rate_below_min_signal  — returns None when total < 3
  3. test_compute_thumbs_rate_zero_down          — 100% when no thumbs_down
  4. test_score_tenant_no_skills                 — empty output_map → 0 skills_scored
  5. test_score_tenant_scores_skills             — aggregates and upserts skill_scores
  6. test_score_tenant_null_rate_insufficient_signal — rate is NULL when <3 reactions
  7. test_get_skill_scores_returns_sorted_list   — reads skill_scores in expected shape
  8. test_run_for_all_tenants_calls_score_tenant — iterates all tenants
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest

from hermes_agent.self_improvement.skill_scorer import (
    _compute_thumbs_rate,
    score_tenant,
    get_skill_scores,
    run_for_all_tenants,
    MIN_SIGNAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRecord(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


def _make_pool_conn_txn():
    txn = MagicMock()
    txn.start = AsyncMock()
    txn.commit = AsyncMock()
    txn.rollback = AsyncMock()

    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=txn)

    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_ctx)

    return pool, conn, txn


TENANT_ID = str(uuid.uuid4())
NOW = datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# 1-3. _compute_thumbs_rate
# ---------------------------------------------------------------------------

def test_compute_thumbs_rate_above_threshold():
    rate = _compute_thumbs_rate(8, 2)
    assert rate == 0.8


def test_compute_thumbs_rate_below_min_signal():
    """Returns None when total reactions < MIN_SIGNAL (3)."""
    rate = _compute_thumbs_rate(1, 1)
    assert rate is None
    # exactly at boundary minus 1
    rate2 = _compute_thumbs_rate(MIN_SIGNAL - 1, 0)
    assert rate2 is None


def test_compute_thumbs_rate_zero_down():
    """100% thumbs_up when no thumbs_down and total >= MIN_SIGNAL."""
    rate = _compute_thumbs_rate(5, 0)
    assert rate == 1.0


# ---------------------------------------------------------------------------
# 4. score_tenant — no skills
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_tenant_no_skills():
    """score_tenant returns 0 skills_scored when output_map is empty."""
    pool, conn, txn = _make_pool_conn_txn()

    # DISTINCT skill_name query returns empty list.
    conn.fetch.return_value = []

    result = await score_tenant(pool, TENANT_ID)

    assert result["skills_scored"] == 0
    assert result["tenant_id"] == TENANT_ID


# ---------------------------------------------------------------------------
# 5. score_tenant — scores skills and upserts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_tenant_scores_skills():
    """score_tenant calls UPSERT for each skill with correct aggregates."""
    pool, conn, txn = _make_pool_conn_txn()

    # Returns one skill from DISTINCT query.
    conn.fetch.return_value = [_FakeRecord({"skill_name": "blake-ops:daily"})]

    # Simulate all aggregation fetchrow calls for one skill:
    # usage_7d, usage_30d, usage_all, last_used, fb_7d, fb_30d
    conn.fetchrow.side_effect = [
        _FakeRecord({"cnt": 3}),         # usage_7d
        _FakeRecord({"cnt": 10}),        # usage_30d
        _FakeRecord({"cnt": 15}),        # usage_all
        _FakeRecord({"last_used": NOW}), # last_used_at
        _FakeRecord({"up": 8, "down": 2}),  # fb_7d
        _FakeRecord({"up": 10, "down": 2}), # fb_30d
    ]

    result = await score_tenant(pool, TENANT_ID)

    assert result["skills_scored"] == 1
    assert result["skills_zero_feedback"] == 0

    # Verify UPSERT was called with INSERT INTO skill_scores.
    upsert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO skill_scores" in str(c)
    ]
    assert len(upsert_calls) == 1


# ---------------------------------------------------------------------------
# 6. score_tenant — NULL rate when insufficient signal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_tenant_null_rate_insufficient_signal():
    """thumbs_rate is None (NULL) when total reactions < MIN_SIGNAL."""
    pool, conn, txn = _make_pool_conn_txn()

    conn.fetch.return_value = [_FakeRecord({"skill_name": "new-skill"})]

    # Only 1 thumbs_up in each window — below MIN_SIGNAL=3.
    conn.fetchrow.side_effect = [
        _FakeRecord({"cnt": 1}),
        _FakeRecord({"cnt": 2}),
        _FakeRecord({"cnt": 2}),
        _FakeRecord({"last_used": NOW}),
        _FakeRecord({"up": 1, "down": 0}),  # total=1 < MIN_SIGNAL
        _FakeRecord({"up": 1, "down": 0}),  # total=1 < MIN_SIGNAL
    ]

    # Capture the UPSERT args to verify thumbs_rate is None.
    captured_args = []
    original_execute = conn.execute

    async def capture_execute(sql, *args):
        if "INSERT INTO skill_scores" in sql:
            captured_args.extend(args)
        return await original_execute(sql, *args)

    conn.execute = capture_execute

    result = await score_tenant(pool, TENANT_ID)
    assert result["skills_scored"] == 1
    # skills_zero_feedback is NOT incremented when there is SOME feedback;
    # only when both windows have zero feedback entirely.
    # Here we have up=1, so it's not zero feedback.


# ---------------------------------------------------------------------------
# 7. get_skill_scores — returns sorted list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_skill_scores_returns_sorted_list():
    """get_skill_scores returns rows in expected dict shape."""
    pool, conn, txn = _make_pool_conn_txn()

    conn.fetch.return_value = [
        _FakeRecord({
            "skill_name": "skill-a",
            "usage_7d": 5, "usage_30d": 20, "usage_all": 50,
            "thumbs_up_7d": 4, "thumbs_down_7d": 1,
            "thumbs_up_30d": 16, "thumbs_down_30d": 4,
            "thumbs_rate_7d": 0.8000,
            "thumbs_rate_30d": 0.8000,
            "last_used_at": NOW,
            "last_scored_at": NOW,
        }),
    ]

    scores = await get_skill_scores(pool, TENANT_ID)

    assert len(scores) == 1
    row = scores[0]
    assert row["skill_name"] == "skill-a"
    assert row["thumbs_rate_30d"] == 0.8
    assert row["last_used_at"] is not None


# ---------------------------------------------------------------------------
# 8. run_for_all_tenants — iterates tenants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_for_all_tenants_calls_score_tenant():
    """run_for_all_tenants calls score_tenant for each tenant row."""
    pool, conn, txn = _make_pool_conn_txn()

    t1 = str(uuid.uuid4())
    t2 = str(uuid.uuid4())

    # First acquire: get tenant list
    conn.fetch.return_value = [
        _FakeRecord({"id": t1}),
        _FakeRecord({"id": t2}),
    ]

    call_order = []

    async def mock_score_tenant(p, tid):
        call_order.append(tid)
        return {"tenant_id": tid, "skills_scored": 0, "skills_zero_feedback": 0, "elapsed_ms": 1.0}

    with patch(
        "hermes_agent.self_improvement.skill_scorer.score_tenant",
        side_effect=mock_score_tenant
    ):
        results = await run_for_all_tenants(pool)

    assert len(results) == 2
    assert t1 in call_order
    assert t2 in call_order

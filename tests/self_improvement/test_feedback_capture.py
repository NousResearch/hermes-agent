"""
tests/self_improvement/test_feedback_capture.py — Plan 004-A

Unit tests for hermes_agent.self_improvement.feedback_capture.

All tests mock the asyncpg pool — no live Neon required.
Pattern mirrors tests/test_storage_neon.py mock helpers.

Test inventory:
  1. test_classify_emoji_thumbs_up            — +1 maps to thumbs_up
  2. test_classify_emoji_thumbs_down          — -1 maps to thumbs_down
  3. test_classify_emoji_unknown_returns_none — arbitrary emoji is ignored
  4. test_reaction_added_writes_feedback_row  — handle_reaction_added inserts to skill_feedback
  5. test_reaction_removed_deletes_row        — handle_reaction_removed deletes the row
  6. test_reaction_added_unknown_ts_skipped   — no skill_output_map entry → no write
  7. test_register_output_writes_map_row      — register_output inserts to skill_output_map
  8. test_register_output_empty_skill_no_write — empty skill_name → False, no write
  9. test_reaction_added_unknown_emoji_skipped — non-quality emoji → False
  10. test_resolve_tenant_id_returns_none_for_unknown — unknown tenant → None
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.self_improvement.feedback_capture import (
    _classify_emoji,
    handle_reaction_added,
    handle_reaction_removed,
    register_output,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_storage_neon.py)
# ---------------------------------------------------------------------------

class _FakeRecord(dict):
    """Dict subclass supporting asyncpg Record-style access."""
    def __getitem__(self, key):
        return super().__getitem__(key)


def _make_mock_pool_and_conn():
    """Return (pool, conn, txn) suitable for feedback_capture tests."""
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
    pool.close = AsyncMock()

    return pool, conn, txn


TENANT_ID = str(uuid.uuid4())
TEAM_ID = "TTEST001"
PLATFORM = "slack"
CHANNEL_ID = "CCHAN001"
SLACK_TS = "1716000000.123456"
REACTOR_ID = "UUSER001"
SKILL_NAME = "blake-ops:daily"


# ---------------------------------------------------------------------------
# 1. _classify_emoji
# ---------------------------------------------------------------------------

def test_classify_emoji_thumbs_up():
    assert _classify_emoji("+1") == "thumbs_up"
    assert _classify_emoji("thumbsup") == "thumbs_up"
    assert _classify_emoji("THUMBSUP") == "thumbs_up"  # case-insensitive
    assert _classify_emoji("thumbs_up") == "thumbs_up"


def test_classify_emoji_thumbs_down():
    assert _classify_emoji("-1") == "thumbs_down"
    assert _classify_emoji("thumbsdown") == "thumbs_down"
    assert _classify_emoji("thumbs_down") == "thumbs_down"


def test_classify_emoji_unknown_returns_none():
    assert _classify_emoji("tada") is None
    assert _classify_emoji("rocket") is None
    assert _classify_emoji("eyes") is None
    assert _classify_emoji("") is None


# ---------------------------------------------------------------------------
# 4. handle_reaction_added — writes row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reaction_added_writes_feedback_row():
    """handle_reaction_added writes a row to skill_feedback for a known ts."""
    pool, conn, txn = _make_mock_pool_and_conn()

    # tenant resolution → found
    # skill_output_map lookup → found
    conn.fetchrow.side_effect = [
        _FakeRecord({"id": TENANT_ID}),              # _resolve_tenant_id
        _FakeRecord({"skill_name": SKILL_NAME}),     # skill_output_map
    ]

    result = await handle_reaction_added(
        pool, PLATFORM, TEAM_ID, REACTOR_ID, CHANNEL_ID, SLACK_TS, "+1"
    )

    assert result is True
    # Verify INSERT INTO skill_feedback was called.
    insert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO skill_feedback" in str(c)
    ]
    assert len(insert_calls) == 1


# ---------------------------------------------------------------------------
# 5. handle_reaction_removed — deletes row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reaction_removed_deletes_row():
    """handle_reaction_removed deletes the skill_feedback row for the reaction."""
    pool, conn, txn = _make_mock_pool_and_conn()

    conn.fetchrow.side_effect = [
        _FakeRecord({"id": TENANT_ID}),
        _FakeRecord({"skill_name": SKILL_NAME}),
    ]
    # Simulate DELETE 1 row returned.
    conn.execute.return_value = "DELETE 1"

    result = await handle_reaction_removed(
        pool, PLATFORM, TEAM_ID, REACTOR_ID, CHANNEL_ID, SLACK_TS, "+1"
    )

    assert result is True
    delete_calls = [
        c for c in conn.execute.await_args_list
        if "DELETE FROM skill_feedback" in str(c)
    ]
    assert len(delete_calls) == 1


# ---------------------------------------------------------------------------
# 6. handle_reaction_added — unknown ts is skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reaction_added_unknown_ts_skipped():
    """handle_reaction_added returns False when ts not in skill_output_map."""
    pool, conn, txn = _make_mock_pool_and_conn()

    conn.fetchrow.side_effect = [
        _FakeRecord({"id": TENANT_ID}),  # tenant found
        None,                            # skill_output_map → not found
    ]

    result = await handle_reaction_added(
        pool, PLATFORM, TEAM_ID, REACTOR_ID, CHANNEL_ID, SLACK_TS, "+1"
    )

    assert result is False
    # No INSERT should have occurred.
    insert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO skill_feedback" in str(c)
    ]
    assert len(insert_calls) == 0


# ---------------------------------------------------------------------------
# 7. register_output — writes map row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_output_writes_map_row():
    """register_output inserts a row into skill_output_map."""
    pool, conn, txn = _make_mock_pool_and_conn()

    conn.fetchrow.return_value = _FakeRecord({"id": TENANT_ID})

    result = await register_output(
        pool, PLATFORM, TEAM_ID, CHANNEL_ID, SLACK_TS, SKILL_NAME
    )

    assert result is True
    insert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO skill_output_map" in str(c)
    ]
    assert len(insert_calls) == 1


# ---------------------------------------------------------------------------
# 8. register_output — empty skill_name → no write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_output_empty_skill_no_write():
    """register_output returns False immediately when skill_name is empty."""
    pool, conn, txn = _make_mock_pool_and_conn()

    result = await register_output(
        pool, PLATFORM, TEAM_ID, CHANNEL_ID, SLACK_TS, ""
    )

    assert result is False
    # No DB calls should have been made.
    conn.fetchrow.assert_not_called()


# ---------------------------------------------------------------------------
# 9. handle_reaction_added — non-quality emoji → False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reaction_added_unknown_emoji_skipped():
    """Unrecognised emoji is ignored — returns False without touching the DB."""
    pool, conn, txn = _make_mock_pool_and_conn()

    result = await handle_reaction_added(
        pool, PLATFORM, TEAM_ID, REACTOR_ID, CHANNEL_ID, SLACK_TS, "rocket"
    )

    assert result is False
    # No DB calls.
    pool.acquire.assert_not_called()


# ---------------------------------------------------------------------------
# 10. _resolve_tenant_id for unknown workspace → None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reaction_added_unknown_tenant_skipped():
    """handle_reaction_added returns False when tenant not found."""
    pool, conn, txn = _make_mock_pool_and_conn()

    # tenant resolution → None (unknown workspace)
    conn.fetchrow.return_value = None

    result = await handle_reaction_added(
        pool, PLATFORM, "TUNKNOWN", REACTOR_ID, CHANNEL_ID, SLACK_TS, "+1"
    )

    assert result is False

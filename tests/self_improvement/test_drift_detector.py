"""
tests/self_improvement/test_drift_detector.py — Plan 004-C

Unit tests for hermes_agent.self_improvement.drift_detector.

Test inventory:
  1. test_run_detects_new_regression_alert     — low thumbs_rate → alert row inserted
  2. test_run_skips_existing_alerted_skill     — dedup: already alerted → no re-insert
  3. test_run_auto_resolves_recovered_skill    — recovered rate → status=resolved
  4. test_dismiss_alert_updates_status         — dismiss action → dismissed
  5. test_mark_iterate_updates_status          — iterate action → iterate
  6. test_get_active_alerts_returns_list       — dashboard read path
  7. test_insufficient_signal_skips_detection  — <MIN_SIGNAL reactions → no alert
  8. test_null_thumbs_rate_skips_detection     — NULL thumbs_rate → no alert
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

import pytest

from hermes_agent.self_improvement.drift_detector import (
    run_drift_detection,
    dismiss_alert,
    mark_iterate,
    get_active_alerts,
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
# 1. Detect new regression alert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_detects_new_regression_alert(monkeypatch):
    """A skill below alert_threshold with enough signal → new alert row inserted."""
    pool, conn, txn = _make_pool_conn_txn()
    monkeypatch.setenv("HERMES_DRIFT_ALERT_THRESHOLD", "0.50")
    monkeypatch.setenv("HERMES_DRIFT_MIN_SIGNAL", "5")

    # Pass 1 (auto-resolution check): no alerted skills.
    # Pass 2 (new alert detection): one at-risk skill.
    conn.fetch.side_effect = [
        [],   # alerted_skills (pass 1)
        [_FakeRecord({   # at_risk (pass 2)
            "skill_name": "bad-skill",
            "thumbs_rate_30d": 0.4000,
            "thumbs_up_30d": 4,
            "thumbs_down_30d": 6,
            "total_reactions": 10,
        })],
    ]

    result = await run_drift_detection(pool, TENANT_ID, slack_client=None)

    assert result["new_alerts"] == 1
    assert result["auto_resolved"] == 0

    insert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO skill_drift_alerts" in str(c)
    ]
    assert len(insert_calls) == 1


# ---------------------------------------------------------------------------
# 2. Dedup: existing alerted skill not re-inserted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_skips_existing_alerted_skill(monkeypatch):
    """Skills already in 'alerted' state do not get a new alert row."""
    pool, conn, txn = _make_pool_conn_txn()
    monkeypatch.setenv("HERMES_DRIFT_ALERT_THRESHOLD", "0.50")
    monkeypatch.setenv("HERMES_DRIFT_RESOLUTION_RATE", "0.65")

    existing_alert_id = uuid.uuid4()
    # Pass 1: this skill is already alerted but still below resolution threshold.
    conn.fetch.side_effect = [
        [_FakeRecord({   # alerted_skills
            "id": existing_alert_id,
            "skill_name": "bad-skill",
            "thumbs_rate_30d": 0.40,
            "total_reactions": 10,
        })],
        [],  # at_risk (SQL NOT EXISTS filters out already-alerted skills)
    ]

    result = await run_drift_detection(pool, TENANT_ID, slack_client=None)

    # No new alerts — the skill was already alerted.
    assert result["new_alerts"] == 0
    insert_calls = [
        c for c in conn.execute.await_args_list
        if "INSERT INTO skill_drift_alerts" in str(c)
    ]
    assert len(insert_calls) == 0


# ---------------------------------------------------------------------------
# 3. Auto-resolve recovered skill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_auto_resolves_recovered_skill(monkeypatch):
    """A skill with current_rate >= RESOLUTION_RATE gets auto-resolved."""
    pool, conn, txn = _make_pool_conn_txn()
    monkeypatch.setenv("HERMES_DRIFT_RESOLUTION_RATE", "0.65")

    existing_alert_id = uuid.uuid4()
    # Pass 1: alerted skill with RECOVERED rate.
    conn.fetch.side_effect = [
        [_FakeRecord({
            "id": existing_alert_id,
            "skill_name": "recovered-skill",
            "thumbs_rate_30d": 0.75,  # above resolution_rate=0.65
            "total_reactions": 8,
        })],
        [],  # at_risk
    ]

    result = await run_drift_detection(pool, TENANT_ID, slack_client=None)

    assert result["auto_resolved"] == 1
    update_calls = [
        c for c in conn.execute.await_args_list
        if "status = 'resolved'" in str(c)
    ]
    assert len(update_calls) == 1


# ---------------------------------------------------------------------------
# 4. dismiss_alert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dismiss_alert_updates_status():
    """dismiss_alert updates the alert row to 'dismissed'."""
    pool, conn, txn = _make_pool_conn_txn()

    result = await dismiss_alert(
        pool, TENANT_ID, "bad-skill", actioned_by="UUSER001"
    )

    assert result["status"] == "dismissed"
    update_calls = [
        c for c in conn.execute.await_args_list
        if "dismissed" in str(c)
    ]
    assert len(update_calls) == 1


# ---------------------------------------------------------------------------
# 5. mark_iterate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_iterate_updates_status():
    """mark_iterate updates the alert row to 'iterate'."""
    pool, conn, txn = _make_pool_conn_txn()

    result = await mark_iterate(
        pool, TENANT_ID, "bad-skill", actioned_by="UUSER001"
    )

    assert result["status"] == "iterate"
    update_calls = [
        c for c in conn.execute.await_args_list
        if "iterate" in str(c)
    ]
    assert len(update_calls) == 1


# ---------------------------------------------------------------------------
# 6. get_active_alerts — dashboard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_active_alerts_returns_list():
    """get_active_alerts returns alerted rows in expected shape."""
    pool, conn, txn = _make_pool_conn_txn()

    alert_id = uuid.uuid4()
    conn.fetch.return_value = [
        _FakeRecord({
            "id": alert_id,
            "skill_name": "bad-skill",
            "status": "alerted",
            "baseline_rate": 0.8500,
            "alert_rate": 0.4000,
            "alerted_at": NOW,
            "notes": None,
        })
    ]

    alerts = await get_active_alerts(pool, TENANT_ID)

    assert len(alerts) == 1
    assert alerts[0]["skill_name"] == "bad-skill"
    assert alerts[0]["status"] == "alerted"
    assert alerts[0]["baseline_rate"] == 0.85
    assert alerts[0]["alert_rate"] == 0.40


# ---------------------------------------------------------------------------
# 7. Insufficient signal → no alert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_insufficient_signal_skips_detection(monkeypatch):
    """Skills with < MIN_SIGNAL reactions are not alerted (SQL filters them)."""
    pool, conn, txn = _make_pool_conn_txn()
    monkeypatch.setenv("HERMES_DRIFT_MIN_SIGNAL", "5")

    # SQL filters out low-signal skills; simulate by returning empty at_risk list.
    conn.fetch.side_effect = [
        [],  # alerted_skills
        [],  # at_risk (MIN_SIGNAL filtered out the skill)
    ]

    result = await run_drift_detection(pool, TENANT_ID, slack_client=None)

    assert result["new_alerts"] == 0


# ---------------------------------------------------------------------------
# 8. NULL thumbs_rate → no alert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_null_thumbs_rate_skips_detection():
    """Skills with NULL thumbs_rate are excluded by the SQL WHERE clause."""
    pool, conn, txn = _make_pool_conn_txn()

    # SQL requires thumbs_rate_30d IS NOT NULL; simulate by returning empty.
    conn.fetch.side_effect = [[], []]

    result = await run_drift_detection(pool, TENANT_ID, slack_client=None)

    assert result["new_alerts"] == 0

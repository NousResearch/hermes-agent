"""Tests for content_engine/publish_tracker.py (attempt counting + DLQ).

Covers the guarantees ported from the scratch tracker:
- claim increments attempts atomically with the drafts enqueue claim
- failed attempts release the claim and schedule retry
- exhausted attempts dead-letter with an observable DLQ row
- stuck claims reset to failed/pending for retry
"""

import sqlite3

import pytest

from publish_tracker import DeliveryError, PublishTracker


@pytest.fixture()
def tracker(tmp_path):
    db = tmp_path / "content_engine.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE drafts (
            id TEXT PRIMARY KEY, brand TEXT, platform TEXT,
            status TEXT NOT NULL DEFAULT 'draft', enqueue_state TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO drafts (id, brand, platform, status) VALUES (?, ?, ?, 'approved')",
        ("draft-1", "sahil_twitter", "twitter"),
    )
    conn.commit()
    conn.close()
    return PublishTracker(db_path=str(db))


def _enqueue_state(db_path: str, draft_id: str) -> str | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT enqueue_state FROM drafts WHERE id = ?", (draft_id,)
    ).fetchone()
    conn.close()
    return row[0]


def test_claim_wins_and_increments_attempts(tracker):
    tracker.register("draft-1", "twitter", max_attempts=3)
    assert tracker.claim("draft-1", "twitter") is True
    assert _enqueue_state(tracker.db_path, "draft-1") == "claiming"
    assert tracker.status("draft-1")["attempt_count"] == 1
    # Second claim while holding fails.
    assert tracker.claim("draft-1", "twitter") is False
    assert tracker.status("draft-1")["attempt_count"] == 1


def test_failure_releases_claim_and_schedules_retry(tracker):
    tracker.register("draft-1", "twitter", max_attempts=3)
    tracker.claim("draft-1", "twitter")
    tracker.mark_failed("draft-1", "twitter", error="postiz timeout", backoff_base_minutes=30)
    assert _enqueue_state(tracker.db_path, "draft-1") == "pending"
    st = tracker.status("draft-1")
    assert st["status"] == "failed"
    assert st["last_error"] == "postiz timeout"
    assert st["next_retry_at"]


def test_exhausted_attempts_dead_letter(tracker):
    tracker.register("draft-1", "twitter", max_attempts=1)
    tracker.claim("draft-1", "twitter")
    tracker.mark_failed("draft-1", "twitter", error="boom")
    st = tracker.status("draft-1")
    assert st["status"] == "dead_letter"
    assert st["next_retry_at"] is None
    dls = tracker.list_dead_letters(acknowledged=0)
    assert len(dls) == 1
    assert dls[0]["draft_id"] == "draft-1"
    assert dls[0]["attempts"] == 1
    assert dls[0]["last_error"] == "boom"


def test_stuck_claim_resets(tracker):
    tracker.register("draft-1", "twitter", max_attempts=3)
    tracker.claim("draft-1", "twitter")
    # Simulate a stale claim by rewriting updated_at into the past.
    conn = sqlite3.connect(tracker.db_path)
    conn.execute(
        "UPDATE publish_delivery SET updated_at = '2020-01-01T00:00:00+00:00' WHERE draft_id = ?",
        ("draft-1",),
    )
    conn.commit()
    conn.close()
    assert tracker.reset_stuck_claims(max_age_minutes=30) == 1
    assert _enqueue_state(tracker.db_path, "draft-1") == "pending"
    assert tracker.status("draft-1")["status"] == "failed"
    assert "crashed or stalled" in tracker.status("draft-1")["last_error"]


def test_mark_failed_without_registration_raises(tracker):
    with pytest.raises(DeliveryError, match="no delivery row"):
        tracker.mark_failed("draft-1", "twitter", error="x")

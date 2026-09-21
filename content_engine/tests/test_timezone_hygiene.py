"""Regression contracts for UTC-aware timestamps and retention boundaries."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import database
from blog import blog_router
from rotation_tracker import RotationTracker


def _is_utc_timestamp(value: str) -> bool:
    parsed = datetime.fromisoformat(value)
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def test_database_persists_utc_aware_timestamps(tmp_path, monkeypatch):
    db_path = tmp_path / "content.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()

    database.insert_draft(
        "draft-1", "brand", "x", "pillar", "topic", None, "body"
    )
    database.log_topic_usage("topic-1", "brand", "topic", "x")

    with sqlite3.connect(db_path) as conn:
        created_at = conn.execute(
            "SELECT created_at FROM drafts WHERE id = 'draft-1'"
        ).fetchone()[0]
        used_at = conn.execute(
            "SELECT used_at FROM topic_usage_log WHERE topic_id = 'topic-1'"
        ).fetchone()[0]

    assert _is_utc_timestamp(created_at)
    assert _is_utc_timestamp(used_at)


def test_rotation_tracker_prune_respects_retention_and_uses_utc(tmp_path):
    tracker = RotationTracker(tmp_path / "rotation.db")
    tracker.record(article_id="old", brand="brand", studio="studio")
    tracker.record(article_id="new", brand="brand", studio="studio")

    old_timestamp = (datetime.now(UTC) - timedelta(days=91)).isoformat()
    with sqlite3.connect(tracker.db_path) as conn:
        conn.execute(
            "UPDATE image_publications SET published_at = ? WHERE article_id = 'old'",
            (old_timestamp,),
        )
        conn.commit()

    assert tracker.prune(retention_days=90) == 1
    rows = tracker.recent_for_brand("brand")
    assert [row["article_id"] for row in rows] == ["new"]
    assert _is_utc_timestamp(rows[0]["published_at"])


def test_topic_reservations_accept_legacy_utc_and_write_aware_utc(tmp_path, monkeypatch):
    reservation_path = tmp_path / "reservations.jsonl"
    legacy_created = datetime.now(UTC).replace(tzinfo=None).isoformat()
    reservation_path.write_text(
        json.dumps({"token": "legacy", "topic_id": "legacy-topic", "created_at": legacy_created}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(blog_router, "TOPIC_RESERVATIONS_PATH", reservation_path)

    assert blog_router._reserved_topic_ids() == {"legacy-topic"}
    token = blog_router.reserve("ai", "fresh-topic")
    rows = [json.loads(line) for line in reservation_path.read_text(encoding="utf-8").splitlines()]
    fresh = next(row for row in rows if row["token"] == token)
    assert _is_utc_timestamp(fresh["created_at"])

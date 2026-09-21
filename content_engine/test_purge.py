"""Tests for purge_stale_drafts — isolated 48-hour retention behaviour."""
from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import config
import database


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Give every test its own schema, independent of collection order."""
    test_db = tmp_path / "purge.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    monkeypatch.setattr(config, "DB_PATH", test_db)
    database.init_db()


def _exec(sql: str, *params):
    """Run SQL with params, commit, close. One-shot only."""
    conn = sqlite3.connect(str(database.DB_PATH))
    cur = conn.execute(sql, params)
    conn.commit()
    conn.close()
    return cur


def _query(sql: str, *params):
    conn = sqlite3.connect(str(database.DB_PATH))
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def _count_all():
    return _query("SELECT COUNT(*) FROM drafts")[0][0]


def _insert_with_age(draft_id: str, brand: str, hours_ago: int):
    timestamp = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    _exec(
        "INSERT INTO drafts (id, brand, platform, pillar, topic, body_text, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        draft_id,
        brand,
        "twitter",
        "test",
        "test_topic",
        "Test body",
        timestamp,
    )


def test_fresh_draft_survives_purge():
    _insert_with_age("fresh-1", "plenishd", hours_ago=1)
    _insert_with_age("fresh-2", "plenishd", hours_ago=12)
    _insert_with_age("fresh-3", "matchdaymaestro", hours_ago=0)

    deleted = database.purge_stale_drafts(retention_hours=48)

    assert deleted == 0
    assert _count_all() == 3


def test_stale_drafts_are_purged():
    _insert_with_age("stale-1", "plenishd", hours_ago=49)
    _insert_with_age("stale-2", "matchdaymaestro", hours_ago=72)
    _insert_with_age("stale-3", "coachos", hours_ago=100)

    assert _count_all() == 3
    assert database.purge_stale_drafts(retention_hours=48) == 3
    assert _count_all() == 0


def test_brand_filter_respected():
    _insert_with_age("s-plen", "plenishd", hours_ago=72)
    _insert_with_age("s-match", "matchdaymaestro", hours_ago=72)
    _insert_with_age("f-plen", "plenishd", hours_ago=1)

    assert database.purge_stale_drafts(retention_hours=48, brand="plenishd") == 1
    assert _count_all() == 2


def test_dry_run_counting():
    _insert_with_age("a", "plenishd", hours_ago=50)
    _insert_with_age("b", "plenishd", hours_ago=50)
    _insert_with_age("c", "matchdaymaestro", hours_ago=50)
    _insert_with_age("d", "matchdaymaestro", hours_ago=2)

    assert database.count_drafts_older_than(retention_hours=48) == 3
    assert database.count_drafts_older_than(retention_hours=48, brand="plenishd") == 2
    assert database.count_drafts_older_than(retention_hours=48, brand="coachos") == 0

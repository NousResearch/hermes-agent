"""Tests for state.db health/stats collection (hermes doctor section).

Covers:
- ``hermes_state.collect_state_db_stats``: read-only, best-effort stats
  (page_count, freelist, WAL size, journal mode, row counts, FTS presence,
  pending v23 FTS-rebuild bookkeeping).
- ``hermes_state.count_db_holders``: /proc-based best-effort probe for how
  many processes hold the DB file open (Linux only; None elsewhere/on error).
- ``hermes_cli.doctor._render_state_db_stats``: formatting/threshold helper
  the doctor state.db section prints from.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_state import SessionDB, collect_state_db_stats, count_db_holders


@pytest.fixture()
def populated_db(tmp_path):
    """A real state.db built through SessionDB's public API, then closed."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("sess-alpha", "cli")
    db.create_session("sess-beta", "cli")
    db.append_message("sess-alpha", "user", "hello world one")
    db.append_message("sess-alpha", "assistant", "hi back")
    db.append_message("sess-beta", "user", "second session message")
    db.close()
    return db_path


# ── collect_state_db_stats ──────────────────────────────────────────────


def test_collect_stats_sane_values(populated_db):
    stats = collect_state_db_stats(populated_db)

    assert isinstance(stats, dict)
    assert stats["page_count"] and stats["page_count"] > 0
    assert stats["page_size"] and stats["page_size"] > 0
    assert stats["logical_size_bytes"] == stats["page_count"] * stats["page_size"]
    assert isinstance(stats["freelist_count"], int) and stats["freelist_count"] >= 0
    assert isinstance(stats["wal_size_bytes"], int) and stats["wal_size_bytes"] >= 0
    assert isinstance(stats["journal_mode"], str) and stats["journal_mode"]
    assert stats["messages"] >= 3
    assert stats["sessions"] >= 2

    fts = stats["fts_tables"]
    assert set(fts) == {"messages_fts", "messages_fts_trigram", "messages_fts_cjk"}
    for name, present in fts.items():
        assert isinstance(present, bool), name
    # Layout marker is absent (None) on legacy DBs, an int once stamped.
    assert stats["fts_storage_version"] is None or isinstance(
        stats["fts_storage_version"], int
    )

    # No rebuild pending on a fresh DB.
    assert stats["fts_rebuild_pending"] in (False, None)


def test_collect_stats_is_read_only(populated_db):
    before_bytes = populated_db.read_bytes()
    before_mtime = populated_db.stat().st_mtime_ns

    collect_state_db_stats(populated_db)

    assert populated_db.read_bytes() == before_bytes
    assert populated_db.stat().st_mtime_ns == before_mtime
    # No stray -wal/-shm growth from the probe either: a subsequent normal
    # open must still work.
    db = SessionDB(db_path=populated_db)
    assert db.get_session("sess-alpha") is not None
    db.close()


def test_collect_and_render_stale_fts_holder_deferral(populated_db):
    conn = sqlite3.connect(str(populated_db))
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        (
            "fts_rebuild_deferral",
            json.dumps({"attempts": 4, "holder_pids": [4242], "first_seen": 1.0}),
        ),
    )
    conn.commit()
    conn.close()

    stats = collect_state_db_stats(populated_db)
    assert stats["fts_rebuild_deferral"]["attempts"] == 4
    assert stats["fts_rebuild_deferral"]["holder_pids"] == [4242]

    from hermes_cli.doctor import _render_state_db_stats

    rendered = _render_state_db_stats(stats)
    warnings = [
        " ".join((text, detail))
        for kind, text, detail in rendered
        if kind == "warn"
    ]
    assert any("4242" in warning and "optimize-storage" in warning for warning in warnings)


def test_collect_stats_missing_file_never_raises(tmp_path):
    stats = collect_state_db_stats(tmp_path / "nope" / "state.db")
    assert isinstance(stats, dict)
    assert stats["page_count"] is None
    assert stats["messages"] is None
    assert stats["logical_size_bytes"] is None


def test_collect_stats_rebuild_pending_flag(populated_db):
    # Simulate the v23 deferred-rebuild bookkeeping directly in state_meta.
    conn = sqlite3.connect(str(populated_db))
    conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES ('fts_rebuild_high_water', '100')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES ('fts_rebuild_progress', '40')"
    )
    conn.commit()
    conn.close()

    stats = collect_state_db_stats(populated_db)
    assert stats["fts_rebuild_pending"] is True
    assert stats["fts_rebuild_high_water"] == 100
    assert stats["fts_rebuild_progress"] == 40


# ── count_db_holders ────────────────────────────────────────────────────


def test_count_db_holders_sees_open_connection(populated_db):
    conn = sqlite3.connect(str(populated_db))
    try:
        holders = count_db_holders(populated_db)
        if sys.platform.startswith("linux"):
            assert isinstance(holders, int)
            assert holders >= 1
        else:
            assert holders is None
    finally:
        conn.close()


def test_count_db_holders_missing_path_no_raise(tmp_path):
    holders = count_db_holders(tmp_path / "absent.db")
    assert holders is None or isinstance(holders, int)


# ── doctor rendering helper ─────────────────────────────────────────────


def _base_stats(**overrides):
    stats = {
        "page_count": 100,
        "page_size": 4096,
        "freelist_count": 2,
        "logical_size_bytes": 100 * 4096,
        "wal_size_bytes": 1024,
        "journal_mode": "wal",
        "messages": 42,
        "sessions": 7,
        "fts_tables": {
            "messages_fts": True,
            "messages_fts_trigram": True,
            "messages_fts_cjk": False,
        },
        "fts_storage_version": 1,
        "fts_rebuild_pending": False,
        "fts_rebuild_high_water": None,
        "fts_rebuild_progress": None,
    }
    stats.update(overrides)
    return stats


def test_render_healthy_stats_no_warnings():
    from hermes_cli.doctor import _render_state_db_stats

    lines = _render_state_db_stats(_base_stats(), holders=2)
    kinds = [k for k, *_ in lines]
    assert "warn" not in kinds
    joined = " | ".join(text for _, text, *_ in lines)
    assert "42" in joined  # messages
    assert "7" in joined  # sessions
    assert "wal" in joined.lower()
    assert "2" in joined  # holders


def test_render_warns_on_large_db():
    from hermes_cli.doctor import (
        STATE_DB_SIZE_WARN_BYTES,
        _render_state_db_stats,
    )

    big = STATE_DB_SIZE_WARN_BYTES + 1
    lines = _render_state_db_stats(
        _base_stats(logical_size_bytes=big, page_count=big // 4096, page_size=4096),
        holders=None,
    )
    warns = [t for k, t, *rest in lines if k == "warn"] + [
        " ".join(rest) for k, t, *rest in lines if k == "warn"
    ]
    blob = " ".join(str(x) for x in warns)
    assert "auto_prune" in blob
    assert "config.yaml" in blob


def test_render_large_db_with_pending_rebuild_suggests_optimize():
    from hermes_cli.doctor import STATE_DB_SIZE_WARN_BYTES, _render_state_db_stats

    big = STATE_DB_SIZE_WARN_BYTES + 1
    lines = _render_state_db_stats(
        _base_stats(logical_size_bytes=big, fts_rebuild_pending=True),
        holders=None,
    )
    blob = " ".join(" ".join(str(p) for p in line) for line in lines)
    assert "optimize-storage" in blob


def test_render_large_db_legacy_trigram_suggests_optimize():
    from hermes_cli.doctor import STATE_DB_SIZE_WARN_BYTES, _render_state_db_stats

    big = STATE_DB_SIZE_WARN_BYTES + 1
    lines = _render_state_db_stats(
        _base_stats(logical_size_bytes=big, fts_storage_version=None),
        holders=None,
    )
    blob = " ".join(" ".join(str(p) for p in line) for line in lines)
    assert "optimize-storage" in blob


def test_render_does_not_duplicate_legacy_wal_warning():
    """A large WAL must NOT warn here: doctor's pre-existing WAL check
    (50 MB threshold, with a --fix checkpoint) already covers it, and a
    second warning at a higher threshold would duplicate the output."""
    from hermes_cli.doctor import _render_state_db_stats

    lines = _render_state_db_stats(
        _base_stats(wal_size_bytes=256 * 1024 * 1024 + 1), holders=None
    )
    warns = [line for line in lines if line[0] == "warn"]
    blob = " ".join(" ".join(str(p) for p in line) for line in warns).lower()
    assert "wal" not in blob


def test_render_handles_all_none_stats():
    from hermes_cli.doctor import _render_state_db_stats

    empty = {k: None for k in _base_stats()}
    empty["fts_tables"] = None
    lines = _render_state_db_stats(empty, holders=None)
    assert isinstance(lines, list)  # must not raise

"""Tests for the optional LivingMemory-style Hindsight decay ledger."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from plugins.memory.hindsight.decay import (
    DecayPolicy,
    HindsightDecayStore,
    policy_from_config,
    result_identity,
)


UTC = timezone.utc


def _result(memory_id, *, created_at=None, tags=None, metadata=None, text="memory"):
    return SimpleNamespace(
        id=memory_id,
        text=text,
        mentioned_at=created_at,
        occurred_start=None,
        document_id=None,
        tags=tags or [],
        metadata=metadata or {},
    )


def _importance(path):
    conn = sqlite3.connect(path)
    value = conn.execute("SELECT importance FROM memory_decay").fetchone()[0]
    conn.close()
    return value


def test_old_low_importance_results_are_soft_filtered(tmp_path):
    now = datetime(2026, 8, 19, tzinfo=UTC)
    store = HindsightDecayStore(tmp_path / "decay.sqlite3", "default", DecayPolicy())

    result = _result("old", created_at=now - timedelta(days=100))
    assert store.filter_results([result], now=now) == []
    assert _importance(tmp_path / "decay.sqlite3") == 0.0


def test_recent_access_halves_elapsed_decay(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    store = HindsightDecayStore(tmp_path / "decay.sqlite3", "default", DecayPolicy())
    result = _result("recent", created_at=start)

    assert store.filter_results([result], now=start) == [result]
    store.apply_decay(now=start + timedelta(days=10))

    # 10 days * 0.01/day, halved because the result was accessed recently.
    assert _importance(tmp_path / "decay.sqlite3") == 0.45


def test_decay_claims_write_lock_before_reading_timestamp(tmp_path, monkeypatch):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    path = tmp_path / "decay.sqlite3"
    store = HindsightDecayStore(path, "default", DecayPolicy())
    result = _result("shared", created_at=start)
    assert store.filter_results([result], now=start) == [result]

    lock_holder = sqlite3.connect(path, timeout=5.0)
    lock_holder.execute("PRAGMA journal_mode=WAL")
    lock_holder.execute("BEGIN IMMEDIATE")

    begin_attempted = threading.Event()
    timestamp_read = threading.Event()
    real_connect = store._connect

    def traced_connect():
        conn = real_connect()

        def trace(statement):
            normalized = " ".join(statement.upper().split())
            if normalized == "BEGIN IMMEDIATE":
                begin_attempted.set()
            if "SELECT LAST_DECAY_AT FROM MEMORY_DECAY_META" in normalized:
                timestamp_read.set()

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(store, "_connect", traced_connect)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(store.apply_decay, start + timedelta(days=10))
            assert begin_attempted.wait(timeout=2.0)
            assert not timestamp_read.wait(timeout=0.2)
            lock_holder.commit()
            assert future.result(timeout=5.0) == 1
    finally:
        lock_holder.close()

    assert timestamp_read.is_set()
    assert _importance(path) == 0.45


def test_permanent_tag_bypasses_decay(tmp_path):
    now = datetime(2026, 8, 19, tzinfo=UTC)
    store = HindsightDecayStore(tmp_path / "decay.sqlite3", "default", DecayPolicy())
    result = _result(
        "permanent",
        created_at=now - timedelta(days=365),
        tags=["hindsight:permanent"],
    )

    assert store.filter_results([result], now=now) == [result]
    assert _importance(tmp_path / "decay.sqlite3") == 0.5


def test_same_memory_id_isolated_by_bank(tmp_path):
    now = datetime(2026, 8, 19, tzinfo=UTC)
    path = tmp_path / "decay.sqlite3"
    old = _result("same-id", created_at=now - timedelta(days=100))
    default_store = HindsightDecayStore(path, "default", DecayPolicy())
    technical_store = HindsightDecayStore(path, "technical", DecayPolicy())

    assert default_store.filter_results([old], now=now) == []
    assert technical_store.filter_results([old], now=now) == []

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM memory_decay").fetchone()[0] == 2
    conn.close()


def test_content_hash_fallback_is_stable():
    first = _result(None, text="same content")
    second = _result(None, text="same content")
    assert result_identity(first) == result_identity(second)
    assert result_identity(first).startswith("content:")


def test_policy_from_config_bounds_invalid_values():
    policy = policy_from_config({
        "decay_rate_per_day": "bad",
        "decay_initial_importance": 2,
        "decay_min_importance": -1,
        "decay_cleanup_age_days": -5,
        "decay_exempt_tags": " permanent, custom ",
    })
    assert policy.rate_per_day == 0.01
    assert policy.initial_importance == 1.0
    assert policy.min_importance == 0.2
    assert policy.cleanup_age_days == 60
    assert policy.exempt_tags == ("permanent", "custom")

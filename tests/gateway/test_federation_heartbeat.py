"""Regression tests for gateway.federation_heartbeat.

Covers the three P2P phases:
1. Heartbeat (idempotent upsert)
2. Offline detection (BEGIN IMMEDIATE lock, peer marking)
3. Task claim (atomic抢领, double-check after lock)

All tests use a real SQLite database in a temp directory — no mocks for the
database layer, so we exercise the actual SQL paths.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def fed_db(tmp_path: Path) -> Path:
    """Return a path to an empty federation database."""
    return tmp_path / "federation.db"


def _connect(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _seed_schema(conn: sqlite3.Connection) -> None:
    """Create the federation tables (same DDL as the module)."""
    from gateway.federation_heartbeat import _ensure_schema
    _ensure_schema(conn)


def _insert_device(
    conn: sqlite3.Connection,
    device_id: str,
    last_seen: float | None = None,
    status: str = "online",
    current_task_id: str | None = None,
) -> None:
    now = last_seen or time.time()
    conn.execute(
        "INSERT OR REPLACE INTO device_heartbeats "
        "(device_id, hostname, status, cpu_cores, memory_gb, load_avg, "
        "current_task_id, last_seen, ip_address, created_at) "
        "VALUES (?, ?, ?, 0, 0, 0, ?, ?, '', ?)",
        (device_id, "host", status, current_task_id, now, now),
    )
    conn.commit()


def _insert_task(
    conn: sqlite3.Connection,
    task_id: str,
    status: str = "pending_reassign",
    assigned_device: str | None = None,
    fail_count: int = 0,
    max_retries: int = 3,
) -> None:
    now = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO federation_tasks "
        "(task_id, title, description, status, priority, assigned_device, "
        "source_device, created_at, started_at, heartbeat_at, completed_at, "
        "context_snapshot, result_data, error_info, fail_count, max_retries) "
        "VALUES (?, 'Test task', '', ?, 3, ?, ?, ?, NULL, NULL, NULL, '{}', '{}', '', ?, ?)",
        (task_id, status, assigned_device or '', now, now, fail_count, max_retries),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 1: Heartbeat (idempotent upsert)
# ---------------------------------------------------------------------------


class TestHeartbeat:
    """Regression: heartbeat must upsert correctly without locks."""

    def test_first_heartbeat_inserts(self, fed_db):
        from gateway.federation_heartbeat import _heartbeat, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            with patch(
                "gateway.federation_heartbeat._device_id",
                return_value="test-device",
            ):
                _heartbeat(conn, "test-device")

            row = conn.execute(
                "SELECT device_id, status, last_seen FROM device_heartbeats "
                "WHERE device_id='test-device'"
            ).fetchone()
            assert row is not None
            assert row["status"] == "online"
            assert row["last_seen"] > 0
        finally:
            conn.close()

    def test_subsequent_heartbeat_updates(self, fed_db):
        from gateway.federation_heartbeat import _heartbeat, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "test-device", last_seen=1000.0)

            with patch(
                "gateway.federation_heartbeat._device_id",
                return_value="test-device",
            ):
                _heartbeat(conn, "test-device")

            row = conn.execute(
                "SELECT device_id, status, last_seen FROM device_heartbeats "
                "WHERE device_id='test-device'"
            ).fetchone()
            assert row["status"] == "online"
            assert row["last_seen"] > 1000.0
            # Only one row should exist (UPDATE, not duplicate INSERT)
            count = conn.execute(
                "SELECT COUNT(*) as c FROM device_heartbeats"
            ).fetchone()["c"]
            assert count == 1
        finally:
            conn.close()

    def test_heartbeat_bring_back_offline(self, fed_db):
        """An offline device that comes back should be marked online."""
        from gateway.federation_heartbeat import _heartbeat, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "test-device", status="offline", last_seen=1000.0)

            _heartbeat(conn, "test-device")

            row = conn.execute(
                "SELECT status FROM device_heartbeats WHERE device_id='test-device'"
            ).fetchone()
            assert row["status"] == "online"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Phase 2: Offline detection
# ---------------------------------------------------------------------------


class TestOfflineDetection:
    """Regression: offline detection must use BEGIN IMMEDIATE and mark
    peers + their tasks correctly."""

    def test_no_offline_peers_returns_empty(self, fed_db):
        from gateway.federation_heartbeat import _detect_offline, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "device-a", last_seen=time.time())

            relays = _detect_offline(conn, "device-a", threshold=30)
            assert relays == []
        finally:
            conn.close()

    def test_offline_peer_marked_and_task_reassigned(self, fed_db):
        from gateway.federation_heartbeat import _detect_offline, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            now = time.time()
            _insert_device(conn, "device-a", last_seen=now)
            _insert_device(conn, "device-b", last_seen=now - 120, current_task_id="T-001")
            _insert_task(conn, "T-001", status="in_progress", assigned_device="device-b")

            relays = _detect_offline(conn, "device-a", threshold=30)

            assert len(relays) == 1
            assert relays[0]["device"] == "device-b"
            assert relays[0]["task_id"] == "T-001"

            # Peer should be marked offline
            peer = conn.execute(
                "SELECT status FROM device_heartbeats WHERE device_id='device-b'"
            ).fetchone()
            assert peer["status"] == "offline"

            # Task should be pending_reassign
            task = conn.execute(
                "SELECT status, fail_count FROM federation_tasks WHERE task_id='T-001'"
            ).fetchone()
            assert task["status"] == "pending_reassign"
            assert task["fail_count"] == 1

            # Peer's current_task_id should be cleared
            peer2 = conn.execute(
                "SELECT current_task_id FROM device_heartbeats WHERE device_id='device-b'"
            ).fetchone()
            assert peer2["current_task_id"] is None
        finally:
            conn.close()

    def test_skips_self_device(self, fed_db):
        """A device should never mark itself offline."""
        from gateway.federation_heartbeat import _detect_offline, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            old_time = time.time() - 120
            _insert_device(conn, "device-a", last_seen=old_time)

            relays = _detect_offline(conn, "device-a", threshold=30)
            assert relays == []

            # device-a should still be online (not self-marked)
            row = conn.execute(
                "SELECT status FROM device_heartbeats WHERE device_id='device-a'"
            ).fetchone()
            assert row["status"] == "online"
        finally:
            conn.close()

    def test_task_without_task_id_skipped(self, fed_db):
        """Offline peer with no current task should be marked offline but
        produce no relay entries."""
        from gateway.federation_heartbeat import _detect_offline, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            now = time.time()
            _insert_device(conn, "device-a", last_seen=now)
            _insert_device(conn, "device-b", last_seen=now - 120, current_task_id=None)

            relays = _detect_offline(conn, "device-a", threshold=30)
            assert relays == []

            # But device-b should still be marked offline
            peer = conn.execute(
                "SELECT status FROM device_heartbeats WHERE device_id='device-b'"
            ).fetchone()
            assert peer["status"] == "offline"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Phase 3: Task claim
# ---------------------------------------------------------------------------


class TestTaskClaim:
    """Regression: task claim must be atomic and handle double-check."""

    def test_claim_idle_device(self, fed_db):
        from gateway.federation_heartbeat import _claim_task, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "device-a", current_task_id=None)
            _insert_task(conn, "T-001", status="pending_reassign")

            result = _claim_task(conn, "device-a")
            assert result is not None
            assert result["task_id"] == "T-001"

            # Task should now be assigned
            task = conn.execute(
                "SELECT status, assigned_device FROM federation_tasks WHERE task_id='T-001'"
            ).fetchone()
            assert task["status"] == "assigned"
            assert task["assigned_device"] == "device-a"

            # Device should have the task
            dev = conn.execute(
                "SELECT current_task_id FROM device_heartbeats WHERE device_id='device-a'"
            ).fetchone()
            assert dev["current_task_id"] == "T-001"
        finally:
            conn.close()

    def test_busy_device_returns_none(self, fed_db):
        """A device with a current task should not claim another."""
        from gateway.federation_heartbeat import _claim_task, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "device-a", current_task_id="T-999")
            _insert_task(conn, "T-001", status="pending_reassign")

            result = _claim_task(conn, "device-a")
            assert result is None
        finally:
            conn.close()

    def test_no_pending_tasks_returns_none(self, fed_db):
        from gateway.federation_heartbeat import _claim_task, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "device-a", current_task_id=None)
            _insert_task(conn, "T-001", status="completed")

            result = _claim_task(conn, "device-a")
            assert result is None
        finally:
            conn.close()

    def test_max_retries_respected(self, fed_db):
        """A task that has exceeded max_retries should not be claimable."""
        from gateway.federation_heartbeat import _claim_task, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "device-a", current_task_id=None)
            _insert_task(conn, "T-001", status="pending_reassign", fail_count=3, max_retries=3)

            result = _claim_task(conn, "device-a")
            assert result is None
        finally:
            conn.close()

    def test_priority_ordering(self, fed_db):
        """Lower priority number should be claimed first."""
        from gateway.federation_heartbeat import _claim_task, _ensure_schema

        conn = _connect(fed_db)
        try:
            _ensure_schema(conn)
            _insert_device(conn, "device-a", current_task_id=None)
            # Insert two tasks with different priorities
            now = time.time()
            conn.execute(
                "INSERT INTO federation_tasks "
                "(task_id, title, status, priority, created_at, context_snapshot, "
                "result_data, error_info, fail_count, max_retries, description, "
                "assigned_device, source_device, started_at, heartbeat_at, completed_at) "
                "VALUES (?, ?, 'pending_reassign', 5, ?, '{}', '{}', '', 0, 3, '', NULL, '', NULL, NULL, NULL)",
                ("T-high", "High priority", now - 100),
            )
            conn.execute(
                "INSERT INTO federation_tasks "
                "(task_id, title, status, priority, created_at, context_snapshot, "
                "result_data, error_info, fail_count, max_retries, description, "
                "assigned_device, source_device, started_at, heartbeat_at, completed_at) "
                "VALUES (?, ?, 'pending_reassign', 1, ?, '{}', '{}', '', 0, 3, '', NULL, '', NULL, NULL, NULL)",
                ("T-low", "Low priority (higher precedence)", now - 50),
            )
            conn.commit()

            result = _claim_task(conn, "device-a")
            assert result is not None
            # Priority 1 (lower number = higher priority) should win
            assert result["task_id"] == "T-low"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# FederationConfig
# ---------------------------------------------------------------------------


class TestFederationConfig:
    """Regression: config must parse correctly from dict."""

    def test_default_config_disabled(self):
        from gateway.config import FederationConfig

        cfg = FederationConfig()
        assert cfg.enabled is False
        assert cfg.offline_threshold_s == 30
        assert cfg.heartbeat_interval_s == 60

    def test_from_dict_parses_all_fields(self):
        from gateway.config import FederationConfig

        data = {
            "enabled": True,
            "db_path": "/tmp/fed.db",
            "offline_threshold_s": 60,
            "heartbeat_interval_s": 120,
        }
        cfg = FederationConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.db_path == "/tmp/fed.db"
        assert cfg.offline_threshold_s == 60
        assert cfg.heartbeat_interval_s == 120

    def test_from_dict_defaults_on_empty(self):
        from gateway.config import FederationConfig

        cfg = FederationConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.offline_threshold_s == 30
        assert cfg.heartbeat_interval_s == 60


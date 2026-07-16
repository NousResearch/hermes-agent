"""Tests for issue #65853 — delegation cleanup can delete results before
delivery.

The 50-record history limit was counting ALL terminal records (both
delivered and pending/undelivered), so cleanup could delete completed
task results before the parent agent received them. The fix limits the
50-record cap to delivered records only, leaving undelivered results
under their separate 1,000-record limit.
"""

from __future__ import annotations

import time
import sqlite3
from unittest.mock import patch
from pathlib import Path

import pytest

from tools.async_delegation import (
    _MAX_RETAINED_COMPLETED,
    _MAX_DURABLE_PENDING,
    _prune_durable_records,
    _connect,
)


# --------------------------------------------------------------------------- #
# _prune_durable_records only deletes delivered records for the 50-limit
# --------------------------------------------------------------------------- #


def test_prune_does_not_delete_pending_records_under_50_delivered():
    """Cleanup must not delete pending (undelivered) records when there are
    fewer than 50 delivered records. See issue #65853.
    """
    # Use a temporary in-memory DB
    with patch("tools.async_delegation._connect") as mock_connect:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE async_delegations (
                delegation_id TEXT PRIMARY KEY,
                state TEXT,
                delivery_state TEXT,
                updated_at REAL,
                completed_at REAL,
                event_json TEXT,
                result_json TEXT,
                delivery_attempts INTEGER DEFAULT 0
            )
        """)
        mock_connect.return_value.__enter__ = lambda self: conn
        mock_connect.return_value.__exit__ = lambda *a: None

        # Insert 10 delivered + 10 pending = 20 total terminal
        for i in range(10):
            conn.execute(
                "INSERT INTO async_delegations (delegation_id, state, delivery_state, updated_at) VALUES (?, ?, ?, ?)",
                (f"del-{i}", "completed", "delivered", time.time() - i),
            )
        for i in range(10):
            conn.execute(
                "INSERT INTO async_delegations (delegation_id, state, delivery_state, updated_at) VALUES (?, ?, ?, ?)",
                (f"pend-{i}", "completed", "pending", time.time() - i),
            )
        conn.commit()

        _prune_durable_records()

        # All 10 pending should survive
        pending_after = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE delivery_state='pending'"
        ).fetchone()[0]
        assert pending_after == 10, (
            f"Expected 10 pending records after prune, got {pending_after} — "
            f"cleanup must not delete undelivered results (issue #65853)"
        )

        # All 10 delivered should also survive (under the 50 limit)
        delivered_after = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE delivery_state='delivered'"
        ).fetchone()[0]
        assert delivered_after == 10


def test_prune_deletes_excess_delivered_records():
    """When there are >50 delivered records, excess (oldest) should be deleted."""
    with patch("tools.async_delegation._connect") as mock_connect:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE async_delegations (
                delegation_id TEXT PRIMARY KEY,
                state TEXT,
                delivery_state TEXT,
                updated_at REAL,
                completed_at REAL,
                event_json TEXT,
                result_json TEXT,
                delivery_attempts INTEGER DEFAULT 0
            )
        """)
        mock_connect.return_value.__enter__ = lambda self: conn
        mock_connect.return_value.__exit__ = lambda *a: None

        # Insert 55 delivered (5 over limit)
        for i in range(55):
            conn.execute(
                "INSERT INTO async_delegations (delegation_id, state, delivery_state, updated_at) VALUES (?, ?, ?, ?)",
                (f"del-{i}", "completed", "delivered", time.time() - 100 + i),
            )
        conn.commit()

        _prune_durable_records()

        delivered_after = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE delivery_state='delivered'"
        ).fetchone()[0]
        assert delivered_after == _MAX_RETAINED_COMPLETED, (
            f"Expected {_MAX_RETAINED_COMPLETED} delivered after prune, got {delivered_after}"
        )


def test_prune_preserves_pending_even_with_many_delivered():
    """Even with >50 delivered records, pending records must not be touched."""
    with patch("tools.async_delegation._connect") as mock_connect:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE async_delegations (
                delegation_id TEXT PRIMARY KEY,
                state TEXT,
                delivery_state TEXT,
                updated_at REAL,
                completed_at REAL,
                event_json TEXT,
                result_json TEXT,
                delivery_attempts INTEGER DEFAULT 0
            )
        """)
        mock_connect.return_value.__enter__ = lambda self: conn
        mock_connect.return_value.__exit__ = lambda *a: None

        # Insert 60 delivered (excess) + 5 pending
        for i in range(60):
            conn.execute(
                "INSERT INTO async_delegations (delegation_id, state, delivery_state, updated_at) VALUES (?, ?, ?, ?)",
                (f"del-{i}", "completed", "delivered", time.time() - 100 + i),
            )
        for i in range(5):
            conn.execute(
                "INSERT INTO async_delegations (delegation_id, state, delivery_state, updated_at) VALUES (?, ?, ?, ?)",
                (f"pend-{i}", "completed", "pending", time.time() - i),
            )
        conn.commit()

        _prune_durable_records()

        pending_after = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE delivery_state='pending'"
        ).fetchone()[0]
        assert pending_after == 5, (
            f"Expected 5 pending records preserved, got {pending_after}"
        )

        delivered_after = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE delivery_state='delivered'"
        ).fetchone()[0]
        assert delivered_after == _MAX_RETAINED_COMPLETED
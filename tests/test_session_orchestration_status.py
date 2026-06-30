"""
Unit tests for session_orchestration/status.py.

Verifies the snapshot builder against a seeded in-memory/temp registry:
- Empty registry produces a "no active tasks" message.
- Per-task lines contain state, age, task_id.
- Summary counts reflect actual states.
- Oldest WAITING_USER task is correctly identified and highlighted.
- Age falls back to created_at when last_output_ts is absent.
- Rows are ordered WAITING_USER first, then by age descending.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.status import (
    _age_seconds,
    _format_age,
    build_registry_snapshot,
    build_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


# ---------------------------------------------------------------------------
# _format_age
# ---------------------------------------------------------------------------


class TestFormatAge:
    def test_seconds_only(self) -> None:
        assert _format_age(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_age(3 * 60 + 12) == "3m 12s"

    def test_hours_and_minutes(self) -> None:
        assert _format_age(65 * 60 + 3) == "1h 5m"

    def test_zero(self) -> None:
        assert _format_age(0) == "0s"

    def test_negative_clamped(self) -> None:
        # Negative ages (clock skew) clamp to zero seconds.
        assert _format_age(-10) == "0s"


# ---------------------------------------------------------------------------
# _age_seconds
# ---------------------------------------------------------------------------


class TestAgeSeconds:
    def test_uses_last_output_ts_when_present(self) -> None:
        now = time.time()
        row: Dict[str, Any] = {"last_output_ts": now - 120.0}
        age = _age_seconds(row, now)
        assert 119.0 <= age <= 121.0

    def test_falls_back_to_created_at(self) -> None:
        now = time.time()
        # Use a created_at 5 minutes ago as an ISO UTC string (SQLite format).
        created = datetime.fromtimestamp(now - 300.0, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        row: Dict[str, Any] = {"created_at": created}
        age = _age_seconds(row, now)
        assert 299.0 <= age <= 301.0

    def test_bad_timestamps_return_zero(self) -> None:
        row: Dict[str, Any] = {"last_output_ts": "not-a-number", "created_at": "bad"}
        assert _age_seconds(row, time.time()) == 0.0


# ---------------------------------------------------------------------------
# build_snapshot — empty registry
# ---------------------------------------------------------------------------


class TestBuildSnapshotEmpty:
    def test_empty_registry_no_active_tasks(self, registry: SessionOrchestrationRegistry) -> None:
        rows = registry.list()
        snapshot = build_snapshot(rows)
        assert "No active tasks" in snapshot

    def test_empty_list_directly(self) -> None:
        snapshot = build_snapshot([])
        assert "No active tasks" in snapshot


# ---------------------------------------------------------------------------
# build_snapshot — seeded registry
# ---------------------------------------------------------------------------


class TestBuildSnapshotSeeded:
    def _seed(self, registry: SessionOrchestrationRegistry) -> float:
        """Insert 3 rows: one RUNNING, two WAITING_USER with different ages."""
        now = time.time()

        # RUNNING task — moderate age
        registry.upsert(
            "task-running-1",
            agent="claude-code",
            state="RUNNING",
            project="/home/user/proj-a",
            last_output_ts=now - 60.0,
        )
        # WAITING_USER — older (90 s)
        registry.upsert(
            "task-waiting-old",
            agent="omp",
            state="WAITING_USER",
            project="/home/user/proj-b",
            last_output_ts=now - 90.0,
        )
        # WAITING_USER — newer (30 s)
        registry.upsert(
            "task-waiting-new",
            agent="omp",
            state="WAITING_USER",
            project="/home/user/proj-c",
            last_output_ts=now - 30.0,
        )
        return now

    def test_snapshot_contains_all_task_ids(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = self._seed(registry)
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)
        assert "task-running-1" in snapshot
        assert "task-waiting-old" in snapshot
        assert "task-waiting-new" in snapshot

    def test_snapshot_contains_states(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = self._seed(registry)
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)
        assert "RUNNING" in snapshot
        assert "WAITING_USER" in snapshot

    def test_summary_counts_correct(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = self._seed(registry)
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)
        # 1 RUNNING + 2 WAITING_USER
        assert "1 RUNNING" in snapshot
        assert "2 WAITING_USER" in snapshot

    def test_oldest_waiting_user_highlighted(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = self._seed(registry)
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)
        # The "Oldest WAITING_USER" line must name the older task.
        assert "Oldest WAITING_USER" in snapshot
        # task-waiting-old is 90s old; task-waiting-new is 30s old.
        # The highlight must reference task-waiting-old.
        oldest_section = snapshot.split("Oldest WAITING_USER")[-1]
        assert "task-waiting-old" in oldest_section

    def test_no_oldest_waiting_when_none_present(self) -> None:
        rows: List[Dict[str, Any]] = [
            {
                "task_id": "t1",
                "agent": "claude-code",
                "state": "RUNNING",
                "last_output_ts": time.time() - 10,
            }
        ]
        snapshot = build_snapshot(rows)
        assert "Oldest WAITING_USER" not in snapshot

    def test_task_count_in_header(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = self._seed(registry)
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)
        assert "3 active task(s)" in snapshot

    def test_age_appears_in_output(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = self._seed(registry)
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)
        # Age lines include "age " prefix
        assert "age " in snapshot


# ---------------------------------------------------------------------------
# build_snapshot — WAITING_USER ordering
# ---------------------------------------------------------------------------


class TestBuildSnapshotOrdering:
    def test_waiting_user_appears_before_running(self) -> None:
        now = time.time()
        rows: List[Dict[str, Any]] = [
            {"task_id": "t-running", "agent": "a", "state": "RUNNING", "last_output_ts": now - 5},
            {"task_id": "t-waiting", "agent": "a", "state": "WAITING_USER", "last_output_ts": now - 5},
        ]
        snapshot = build_snapshot(rows, now=now)
        pos_waiting = snapshot.find("t-waiting")
        pos_running = snapshot.find("t-running")
        # WAITING_USER rows sort before RUNNING rows
        assert pos_waiting < pos_running

# ---------------------------------------------------------------------------
# build_snapshot — unresolved attention items
# ---------------------------------------------------------------------------


class TestBuildSnapshotAttentionItems:
    def test_attention_items_include_reason_priority_and_age(self) -> None:
        now_dt = datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc)
        now = now_dt.timestamp()
        rows: List[Dict[str, Any]] = [
            {
                "task_id": "task-1",
                "agent": "claude",
                "state": "RUNNING",
                "project": "hermes-agent",
                "last_output_ts": now - 60,
            }
        ]
        attention_items: List[Dict[str, Any]] = [
            {
                "id": 1,
                "task_id": "task-1",
                "reason": "FROZEN_STALE",
                "priority": 9,
                "detail": "pane hash unchanged",
                "opened_at": "2026-06-29T01:30:00+00:00",
            }
        ]

        snapshot = build_snapshot(rows, now=now, attention_items=attention_items)

        assert "Unresolved attention items" in snapshot
        assert "task-1" in snapshot
        assert "reason `FROZEN_STALE`" in snapshot
        assert "pane hash unchanged" in snapshot
        assert "P9/frozen/stuck" in snapshot
        assert "opened 30m ago" in snapshot

    def test_oldest_actionable_uses_priority_then_opened_at_not_waiting_rows(self) -> None:
        now_dt = datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc)
        now = now_dt.timestamp()
        rows: List[Dict[str, Any]] = [
            {
                "task_id": "task-waiting-old",
                "agent": "omp",
                "state": "WAITING_USER",
                "last_output_ts": now - 7200,
            }
        ]
        attention_items: List[Dict[str, Any]] = [
            {
                "id": 1,
                "task_id": "task-low-oldest",
                "reason": "WAITING_USER",
                "priority": 1,
                "opened_at": "2026-06-29T00:30:00+00:00",
            },
            {
                "id": 2,
                "task_id": "task-high-later",
                "reason": "STALE",
                "priority": 5,
                "opened_at": "2026-06-29T01:30:00+00:00",
            },
            {
                "id": 3,
                "task_id": "task-high-earlier",
                "reason": "STALE",
                "priority": 5,
                "opened_at": "2026-06-29T01:00:00+00:00",
            },
        ]

        snapshot = build_snapshot(rows, now=now, attention_items=attention_items)

        assert "Oldest WAITING_USER" not in snapshot
        highlight = snapshot.split("**Oldest actionable item:**", 1)[1]
        assert "task-high-earlier" in highlight
        assert "task-high-later" not in highlight
        assert "task-low-oldest" not in highlight
        assert "task-waiting-old" not in highlight

    def test_empty_rows_with_attention_items_render_safely(self) -> None:
        now_dt = datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc)
        snapshot = build_snapshot(
            [],
            now=now_dt.timestamp(),
            attention_items=[
                {
                    "id": 1,
                    "task_id": "task-attn",
                    "reason": "WAITING_USER",
                    "priority": 1,
                    "opened_at": "2026-06-29T01:59:00+00:00",
                }
            ],
        )

        assert "No active tasks." in snapshot
        assert "task-attn" in snapshot
        assert "reason `WAITING_USER`" in snapshot
        assert "opened 1m ago" in snapshot


class TestBuildRegistrySnapshotAttentionItems:
    def test_registry_snapshot_reads_attention_without_mutating(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now_dt = datetime(2026, 6, 29, 2, 0, tzinfo=timezone.utc)
        now = now_dt.timestamp()
        registry.upsert(
            "task-reg",
            agent="claude",
            state="RUNNING",
            project="hermes-agent",
            last_output_ts=now - 120,
        )
        registry.open_attention_item(
            "task-reg",
            "STALE_FROZEN",
            priority=50,
            detail="pane hash unchanged",
        )
        conn = sqlite3.connect(str(registry._db_path))
        try:
            conn.execute(
                """
                UPDATE session_orchestration_attention_items
                   SET opened_at = ?
                 WHERE task_id = ? AND reason = ?
                """,
                ("2026-06-29T01:00:00+00:00", "task-reg", "STALE_FROZEN"),
            )
            conn.commit()
        finally:
            conn.close()
        before = registry.list_unresolved_attention_items()

        snapshot = build_registry_snapshot(registry, now=now)

        assert registry.list_unresolved_attention_items() == before
        assert "task-reg" in snapshot
        assert "claude/hermes-agent" in snapshot
        assert "reason `STALE_FROZEN`" in snapshot
        assert "P50/frozen/stuck" in snapshot
        assert "opened 1h ago" in snapshot

# ---------------------------------------------------------------------------
# Integration: registry → build_snapshot round-trip
# ---------------------------------------------------------------------------


class TestRegistrySnapshotRoundTrip:
    def test_registry_rows_feed_snapshot_accurately(
        self, registry: SessionOrchestrationRegistry
    ) -> None:
        now = time.time()
        registry.upsert(
            "rt-001",
            agent="claude-code",
            state="STALLED",
            project="/tmp/stalled-proj",
            last_output_ts=now - 400.0,
        )
        rows = registry.list()
        snapshot = build_snapshot(rows, now=now)

        assert "rt-001" in snapshot
        assert "STALLED" in snapshot
        # 400 s = 6m 40s
        assert "6m 40s" in snapshot

    def test_snapshot_fails_gracefully_on_bad_row(self) -> None:
        """A row with missing fields doesn't crash the builder."""
        rows: List[Dict[str, Any]] = [{"task_id": "incomplete", "agent": "x"}]
        snapshot = build_snapshot(rows)
        assert "incomplete" in snapshot

"""Tests for the Progressive Autonomy Tracker."""

import sqlite3
import pytest

from agent.autonomy_tracker import (
    AutonomyLevel,
    AutonomyTracker,
    TaskTypeStats,
    EARNED_ACCURACY_THRESHOLD,
    MIN_ATTEMPTS_FOR_EARNED,
)


def make_tracker(**kwargs) -> AutonomyTracker:
    """Create an in-memory tracker (no DB)."""
    return AutonomyTracker(**kwargs)


def make_db_tracker(**kwargs) -> AutonomyTracker:
    """Create a tracker backed by an in-memory SQLite DB."""
    db = sqlite3.connect(":memory:")
    return AutonomyTracker(db=db, **kwargs)


class TestAutonomyLevels:

    def test_default_level_is_standard(self):
        tracker = make_tracker()
        assert tracker.get_level("file_read") == AutonomyLevel.STANDARD

    def test_earned_after_threshold(self):
        tracker = make_tracker(
            earned_threshold=0.95, min_attempts_earned=20
        )
        for _ in range(20):
            tracker.record_outcome("file_read", success=True)
        assert tracker.get_level("file_read") == AutonomyLevel.EARNED

    def test_not_earned_below_min_attempts(self):
        tracker = make_tracker(earned_threshold=0.95, min_attempts_earned=20)
        for _ in range(19):
            tracker.record_outcome("file_read", success=True)
        assert tracker.get_level("file_read") != AutonomyLevel.EARNED

    def test_restricted_after_many_failures(self):
        tracker = make_tracker(
            restricted_threshold=0.70, min_attempts_restricted=5
        )
        for _ in range(10):
            tracker.record_outcome("api_call", success=False)
        assert tracker.get_level("api_call") == AutonomyLevel.RESTRICTED

    def test_not_restricted_below_min_attempts(self):
        tracker = make_tracker(restricted_threshold=0.70, min_attempts_restricted=5)
        for _ in range(4):
            tracker.record_outcome("api_call", success=False)
        # Not enough attempts to restrict
        assert tracker.get_level("api_call") == AutonomyLevel.STANDARD

    def test_standard_for_mixed_results(self):
        tracker = make_tracker(
            earned_threshold=0.95,
            restricted_threshold=0.70,
            min_attempts_restricted=5,
        )
        for i in range(10):
            tracker.record_outcome("web_search", success=(i % 5 != 0))  # 80% success
        assert tracker.get_level("web_search") == AutonomyLevel.STANDARD


class TestShouldConfirm:

    def test_confirm_when_restricted(self):
        tracker = make_tracker(restricted_threshold=0.70, min_attempts_restricted=5)
        for _ in range(10):
            tracker.record_outcome("external_deploy", success=False)
        assert tracker.should_confirm("external_deploy") is True

    def test_no_confirm_when_earned(self):
        tracker = make_tracker(earned_threshold=0.95, min_attempts_earned=20)
        for _ in range(20):
            tracker.record_outcome("file_read", success=True)
        assert tracker.should_confirm("file_read") is False

    def test_no_confirm_for_unknown_task(self):
        tracker = make_tracker()
        assert tracker.should_confirm("brand_new_task") is False


class TestRecordOutcome:

    def test_returns_level(self):
        tracker = make_tracker()
        level = tracker.record_outcome("file_read", success=True)
        assert isinstance(level, AutonomyLevel)

    def test_accumulates_attempts(self):
        tracker = make_tracker()
        tracker.record_outcome("file_read", success=True)
        tracker.record_outcome("file_read", success=False)
        stats = tracker.get_stats("file_read")
        assert stats is not None
        assert stats.attempts == 2
        assert stats.successes == 1

    def test_accuracy_calculation(self):
        tracker = make_tracker()
        for _ in range(8):
            tracker.record_outcome("web_search", success=True)
        for _ in range(2):
            tracker.record_outcome("web_search", success=False)
        stats = tracker.get_stats("web_search")
        assert abs(stats.accuracy - 0.8) < 0.001


class TestSeedFromHistory:

    def test_seed_bootstrap(self):
        tracker = make_tracker(earned_threshold=0.95, min_attempts_earned=20)
        history = [("file_read", True)] * 20
        tracker.seed_from_history(history)
        assert tracker.get_level("file_read") == AutonomyLevel.EARNED

    def test_seed_mixed(self):
        tracker = make_tracker()
        history = [("api_call", i % 2 == 0) for i in range(10)]
        tracker.seed_from_history(history)
        stats = tracker.get_stats("api_call")
        assert stats.attempts == 10


class TestFormatSummary:

    def test_empty_summary(self):
        tracker = make_tracker()
        summary = tracker.format_summary()
        assert "No autonomy data" in summary

    def test_summary_with_data(self):
        tracker = make_tracker(earned_threshold=0.95, min_attempts_earned=5)
        for _ in range(5):
            tracker.record_outcome("file_read", success=True)
        summary = tracker.format_summary()
        assert "EARNED" in summary
        assert "file_read" in summary


class TestDBPersistence:

    def test_saves_to_db(self):
        tracker = make_db_tracker()
        tracker.record_outcome("file_read", success=True)
        cursor = tracker._db.execute(
            "SELECT attempts, successes FROM agent_autonomy_stats WHERE task_type=?",
            ("file_read",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == 1

    def test_loads_from_db(self):
        db = sqlite3.connect(":memory:")
        tracker1 = AutonomyTracker(db=db)
        for _ in range(10):
            tracker1.record_outcome("file_read", success=True)

        # New tracker using same DB should load existing data
        tracker2 = AutonomyTracker(db=db)
        stats = tracker2.get_stats("file_read")
        assert stats is not None
        assert stats.attempts == 10

    def test_level_persisted(self):
        db = sqlite3.connect(":memory:")
        tracker1 = AutonomyTracker(db=db, earned_threshold=0.95, min_attempts_earned=5)
        for _ in range(5):
            tracker1.record_outcome("file_read", success=True)
        assert tracker1.get_level("file_read") == AutonomyLevel.EARNED

        tracker2 = AutonomyTracker(db=db, earned_threshold=0.95, min_attempts_earned=5)
        assert tracker2.get_level("file_read") == AutonomyLevel.EARNED


class TestTaskTypeStats:

    def test_accuracy_zero_attempts(self):
        stats = TaskTypeStats(task_type="test")
        assert stats.accuracy == 0.0

    def test_failures_count(self):
        stats = TaskTypeStats(task_type="test", attempts=10, successes=7)
        assert stats.failures == 3

    def test_to_from_dict_roundtrip(self):
        stats = TaskTypeStats(
            task_type="file_read",
            attempts=25,
            successes=24,
            level=AutonomyLevel.EARNED,
        )
        d = stats.to_dict()
        restored = TaskTypeStats.from_dict(d)
        assert restored.task_type == stats.task_type
        assert restored.attempts == stats.attempts
        assert restored.successes == stats.successes
        assert restored.level == stats.level

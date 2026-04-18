"""
Progressive Autonomy Tracker for Hermes Agent.

Implements the "earned autonomy" pattern: autonomy is not given upfront but
earned through demonstrated accuracy on each task type.

Inspired by research showing that agents improve more through loop iteration
and feedback than through raw model capability upgrades. When an agent knows
which task types it handles reliably, it can act with confidence in those
areas while being more cautious (and asking for confirmation) in others.

Autonomy Levels:
    RESTRICTED (0): Always ask for confirmation before acting
    STANDARD (1): Default — act, but summarize what was done
    EARNED (2): Act autonomously with minimal reporting for routine tasks

Thresholds (configurable):
    EARNED requires:  accuracy >= EARNED_ACCURACY_THRESHOLD (default: 0.95)
                      AND attempts >= MIN_ATTEMPTS_FOR_EARNED (default: 20)
    RESTRICTED if:    accuracy < RESTRICTED_ACCURACY_THRESHOLD (default: 0.70)

Usage:
    tracker = AutonomyTracker(db)
    tracker.record_outcome("file_read", success=True)
    level = tracker.get_level("file_read")   # AutonomyLevel.EARNED, STANDARD, or RESTRICTED
    if level == AutonomyLevel.RESTRICTED:
        # ask for confirmation before proceeding
    summary = tracker.format_summary()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

EARNED_ACCURACY_THRESHOLD: float = 0.95
RESTRICTED_ACCURACY_THRESHOLD: float = 0.70
MIN_ATTEMPTS_FOR_EARNED: int = 20
MIN_ATTEMPTS_FOR_RESTRICTED: int = 5   # need at least 5 tries before restricting

# ─── Types ────────────────────────────────────────────────────────────────────

class AutonomyLevel(IntEnum):
    RESTRICTED = 0
    STANDARD = 1
    EARNED = 2

    def label(self) -> str:
        return {
            AutonomyLevel.RESTRICTED: "🔴 RESTRICTED",
            AutonomyLevel.STANDARD: "🟡 STANDARD",
            AutonomyLevel.EARNED: "🟢 EARNED",
        }[self]


@dataclass
class TaskTypeStats:
    task_type: str
    attempts: int = 0
    successes: int = 0
    level: AutonomyLevel = AutonomyLevel.STANDARD

    @property
    def accuracy(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def failures(self) -> int:
        return self.attempts - self.successes

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "attempts": self.attempts,
            "successes": self.successes,
            "level": int(self.level),
            "accuracy": round(self.accuracy, 4),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TaskTypeStats":
        stats = cls(task_type=d["task_type"])
        stats.attempts = d.get("attempts", 0)
        stats.successes = d.get("successes", 0)
        stats.level = AutonomyLevel(d.get("level", AutonomyLevel.STANDARD))
        return stats


# ─── Main tracker ─────────────────────────────────────────────────────────────

class AutonomyTracker:
    """Tracks per-task-type accuracy and assigns earned autonomy levels.

    Stores stats in the Hermes state DB (agent_autonomy_stats table) or
    falls back to an in-memory dict if no DB is provided.
    """

    # Well-known task categories (can be extended)
    TASK_TYPES = [
        "file_read",
        "file_write",
        "web_search",
        "web_fetch",
        "shell_exec",
        "memory_store",
        "memory_recall",
        "api_call",
        "code_generation",
        "code_review",
        "external_deploy",
        "send_message",
        "image_analysis",
        "pdf_analysis",
        "data_analysis",
    ]

    def __init__(
        self,
        db: Optional[sqlite3.Connection] = None,
        *,
        earned_threshold: float = EARNED_ACCURACY_THRESHOLD,
        restricted_threshold: float = RESTRICTED_ACCURACY_THRESHOLD,
        min_attempts_earned: int = MIN_ATTEMPTS_FOR_EARNED,
        min_attempts_restricted: int = MIN_ATTEMPTS_FOR_RESTRICTED,
    ):
        self._db = db
        self._earned_threshold = earned_threshold
        self._restricted_threshold = restricted_threshold
        self._min_attempts_earned = min_attempts_earned
        self._min_attempts_restricted = min_attempts_restricted
        self._cache: Dict[str, TaskTypeStats] = {}

        if db is not None:
            self._ensure_table()
            self._load_all()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_outcome(self, task_type: str, *, success: bool) -> AutonomyLevel:
        """Record the outcome of a task and return the updated autonomy level.

        Args:
            task_type: Category of task (e.g. 'file_read', 'api_call')
            success: Whether the task completed successfully

        Returns:
            Current AutonomyLevel for this task type after recording.
        """
        stats = self._get_or_create(task_type)
        stats.attempts += 1
        if success:
            stats.successes += 1

        stats.level = self._compute_level(stats)
        self._save(stats)
        logger.debug(
            "autonomy_tracker: %s → attempts=%d accuracy=%.2f level=%s",
            task_type, stats.attempts, stats.accuracy, stats.level.name,
        )
        return stats.level

    def get_level(self, task_type: str) -> AutonomyLevel:
        """Get the current autonomy level for a task type (no side effects)."""
        stats = self._cache.get(task_type)
        if stats is None:
            return AutonomyLevel.STANDARD
        return stats.level

    def get_stats(self, task_type: str) -> Optional[TaskTypeStats]:
        """Get full stats for a task type."""
        return self._cache.get(task_type)

    def all_stats(self) -> List[TaskTypeStats]:
        """Return stats for all tracked task types, sorted by attempts desc."""
        return sorted(self._cache.values(), key=lambda s: s.attempts, reverse=True)

    def should_confirm(self, task_type: str) -> bool:
        """Return True if the agent should ask for confirmation before this task."""
        return self.get_level(task_type) == AutonomyLevel.RESTRICTED

    def format_summary(self) -> str:
        """Format a human-readable summary of autonomy levels."""
        stats_list = self.all_stats()
        if not stats_list:
            return "No autonomy data recorded yet."

        lines = ["📊 Autonomy Tracker Summary", "─" * 50]
        earned = [s for s in stats_list if s.level == AutonomyLevel.EARNED]
        restricted = [s for s in stats_list if s.level == AutonomyLevel.RESTRICTED]
        standard = [s for s in stats_list if s.level == AutonomyLevel.STANDARD]

        if earned:
            lines.append(f"\n🟢 EARNED ({len(earned)} task types):")
            for s in earned:
                lines.append(
                    f"  {s.task_type:<25} {s.accuracy:.0%} accuracy"
                    f" ({s.successes}/{s.attempts})"
                )

        if restricted:
            lines.append(f"\n🔴 RESTRICTED ({len(restricted)} task types — confirm before acting):")
            for s in restricted:
                lines.append(
                    f"  {s.task_type:<25} {s.accuracy:.0%} accuracy"
                    f" ({s.successes}/{s.attempts})"
                )

        if standard:
            lines.append(f"\n🟡 STANDARD ({len(standard)} task types):")
            for s in standard:
                lines.append(
                    f"  {s.task_type:<25} {s.accuracy:.0%} accuracy"
                    f" ({s.successes}/{s.attempts})"
                )

        total_attempts = sum(s.attempts for s in stats_list)
        total_successes = sum(s.successes for s in stats_list)
        overall = total_successes / total_attempts if total_attempts > 0 else 0.0
        lines.append(f"\nOverall: {overall:.0%} ({total_successes}/{total_attempts})")
        return "\n".join(lines)

    def seed_from_history(self, outcomes: List[Tuple[str, bool]]) -> None:
        """Bootstrap tracker with historical outcomes.

        Useful for initializing from past session logs rather than starting
        from zero accuracy on every task type.

        Args:
            outcomes: List of (task_type, success) tuples
        """
        for task_type, success in outcomes:
            self.record_outcome(task_type, success=success)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _compute_level(self, stats: TaskTypeStats) -> AutonomyLevel:
        """Compute the appropriate autonomy level from stats."""
        if stats.attempts < self._min_attempts_restricted:
            return AutonomyLevel.STANDARD

        if (stats.attempts >= self._min_attempts_earned
                and stats.accuracy >= self._earned_threshold):
            return AutonomyLevel.EARNED

        if stats.accuracy < self._restricted_threshold:
            return AutonomyLevel.RESTRICTED

        return AutonomyLevel.STANDARD

    def _get_or_create(self, task_type: str) -> TaskTypeStats:
        if task_type not in self._cache:
            self._cache[task_type] = TaskTypeStats(task_type=task_type)
        return self._cache[task_type]

    # ── DB persistence ─────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        try:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS agent_autonomy_stats (
                    task_type   TEXT PRIMARY KEY,
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    successes   INTEGER NOT NULL DEFAULT 0,
                    level       INTEGER NOT NULL DEFAULT 1,
                    updated_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
                )
            """)
            self._db.commit()
        except Exception as e:
            logger.warning(f"AutonomyTracker: table creation failed: {e}")

    def _load_all(self) -> None:
        try:
            cursor = self._db.execute(
                "SELECT task_type, attempts, successes, level FROM agent_autonomy_stats"
            )
            for row in cursor.fetchall():
                stats = TaskTypeStats(
                    task_type=row[0],
                    attempts=row[1],
                    successes=row[2],
                    level=AutonomyLevel(row[3]),
                )
                self._cache[stats.task_type] = stats
        except Exception as e:
            logger.warning(f"AutonomyTracker: load failed: {e}")

    def _save(self, stats: TaskTypeStats) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                """
                INSERT INTO agent_autonomy_stats (task_type, attempts, successes, level, updated_at)
                VALUES (?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(task_type) DO UPDATE SET
                    attempts   = excluded.attempts,
                    successes  = excluded.successes,
                    level      = excluded.level,
                    updated_at = excluded.updated_at
                """,
                (stats.task_type, stats.attempts, stats.successes, int(stats.level)),
            )
            self._db.commit()
        except Exception as e:
            logger.warning(f"AutonomyTracker: save failed for {stats.task_type}: {e}")

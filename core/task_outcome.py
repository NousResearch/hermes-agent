#!/usr/bin/env python3
"""
TaskOutcome Recording System

Inspired by ClaudeCodeFramework's SelfOptimization system.
Records task outcomes to SQLite for analysis and optimization.
"""

import enum
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default to ~/.hermes if HOME not set
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_OUTCOMES_DIR = _HERMES_HOME / "outcomes"
_OUTCOMES_DB = _OUTCOMES_DIR / "outcomes.db"


class OutcomeTaxonomy(enum.Enum):
    """Taxonomy classification for task outcomes."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class TaskComplexity(enum.Enum):
    """Complexity level for tasks."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TaskOutcome:
    """Records the outcome of a delegated or executed task."""

    outcome_id: str
    task_description: str
    approach_used: str  # e.g., 'delegate_task', 'direct_execution'
    taxonomy: OutcomeTaxonomy
    files_modified: List[str]
    duration_seconds: float
    task_complexity: TaskComplexity
    task_type: str = "code"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "outcome_id": self.outcome_id,
            "task_description": self.task_description,
            "approach_used": self.approach_used,
            "taxonomy": self.taxonomy.value,
            "files_modified": ",".join(self.files_modified) if self.files_modified else "",
            "duration_seconds": self.duration_seconds,
            "task_complexity": self.task_complexity.value,
            "task_type": self.task_type,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "completed_at": self.completed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskOutcome":
        """Create from dictionary (e.g., database row)."""
        files_str = data.get("files_modified", "")
        files = files_str.split(",") if files_str else []
        return cls(
            outcome_id=data["outcome_id"],
            task_description=data["task_description"],
            approach_used=data["approach_used"],
            taxonomy=OutcomeTaxonomy(data["taxonomy"]),
            files_modified=files,
            duration_seconds=data["duration_seconds"],
            task_complexity=TaskComplexity(data["task_complexity"]),
            task_type=data.get("task_type", "code"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            session_id=data.get("session_id"),
            completed_at=datetime.fromisoformat(data["completed_at"]),
        )


class TaskOutcomeStore:
    """SQLite-backed store for TaskOutcome records."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the store, creating directory and tables if needed."""
        self.db_path = db_path or _OUTCOMES_DB
        self._ensure_directory()
        self._ensure_table()

    def _ensure_directory(self) -> None:
        """Create the outcomes directory if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_table(self) -> None:
        """Create the outcomes table if it doesn't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    approach_used TEXT NOT NULL,
                    taxonomy TEXT NOT NULL,
                    files_modified TEXT,
                    duration_seconds REAL NOT NULL,
                    task_complexity TEXT NOT NULL,
                    task_type TEXT DEFAULT 'code',
                    error_type TEXT,
                    error_message TEXT,
                    session_id TEXT,
                    completed_at TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_completed_at ON outcomes(completed_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_taxonomy ON outcomes(taxonomy)
            """
            )
            conn.commit()

    def record(self, outcome: TaskOutcome) -> str:
        """
        Record a task outcome to the database.

        Args:
            outcome: TaskOutcome instance to record

        Returns:
            The outcome_id of the recorded outcome
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO outcomes (
                    outcome_id, task_description, approach_used, taxonomy,
                    files_modified, duration_seconds, task_complexity, task_type,
                    error_type, error_message, session_id, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.outcome_id,
                    outcome.task_description,
                    outcome.approach_used,
                    outcome.taxonomy.value,
                    ",".join(outcome.files_modified) if outcome.files_modified else "",
                    outcome.duration_seconds,
                    outcome.task_complexity.value,
                    outcome.task_type,
                    outcome.error_type,
                    outcome.error_message,
                    outcome.session_id,
                    outcome.completed_at.isoformat(),
                ),
            )
            conn.commit()
        return outcome.outcome_id

    def get_recent(self, limit: int = 50) -> List[TaskOutcome]:
        """
        Get the most recent outcomes.

        Args:
            limit: Maximum number of outcomes to return (default 50)

        Returns:
            List of TaskOutcome objects, most recent first
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM outcomes
                ORDER BY completed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [TaskOutcome.from_dict(dict(row)) for row in rows]

    def get_by_taxonomy(
        self, taxonomy: OutcomeTaxonomy, limit: int = 50
    ) -> List[TaskOutcome]:
        """
        Get outcomes filtered by taxonomy.

        Args:
            taxonomy: OutcomeTaxonomy value to filter by
            limit: Maximum number of outcomes to return (default 50)

        Returns:
            List of TaskOutcome objects matching the taxonomy
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM outcomes
                WHERE taxonomy = ?
                ORDER BY completed_at DESC
                LIMIT ?
                """,
                (taxonomy.value, limit),
            )
            rows = cursor.fetchall()
            return [TaskOutcome.from_dict(dict(row)) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics about recorded outcomes.

        Returns:
            Dictionary with success_rate, avg_duration, and error_rate
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN taxonomy = 'success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN taxonomy = 'failure' THEN 1 ELSE 0 END) as failures,
                    SUM(CASE WHEN taxonomy = 'partial' THEN 1 ELSE 0 END) as partials,
                    AVG(duration_seconds) as avg_duration,
                    SUM(CASE WHEN error_type IS NOT NULL AND error_type != '' THEN 1 ELSE 0 END) as errors
                FROM outcomes
                """
            )
            row = cursor.fetchone()
            if row is None or row["total"] == 0:
                return {
                    "total": 0,
                    "success_rate": 0.0,
                    "avg_duration": 0.0,
                    "error_rate": 0.0,
                }

            total = row["total"]
            successes = row["successes"] or 0
            failures = row["failures"] or 0
            partials = row["partials"] or 0
            errors = row["errors"] or 0

            return {
                "total": total,
                "successes": successes,
                "failures": failures,
                "partials": partials,
                "success_rate": round(successes / total, 4) if total > 0 else 0.0,
                "avg_duration": round(row["avg_duration"] or 0.0, 2),
                "error_rate": round(errors / total, 4) if total > 0 else 0.0,
            }

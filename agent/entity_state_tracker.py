"""
Entity State Tracker: tracks entity state changes and detects conflicts
after context compression.

Use cases:
- Variable assignments that get truncated/lost
- File paths/config items modified mid-execution
- Key事实陈述被压缩
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any


@dataclasses.dataclass
class StateSnapshot:
    """A snapshot of an entity's state at a point in time."""
    entity_key: str
    value: Any
    timestamp: float


@dataclasses.dataclass
class ConflictRecord:
    """Records a detected conflict after compression."""
    entity_key: str
    pre_compression_value: Any
    post_compression_value: Any
    warning: str


class EntityStateTracker:
    """
    Tracks entity states before/after context compression.

    Example:
        tracker = EntityStateTracker()
        tracker.record("variable_x", 42)
        tracker.record("file_path", "/tmp/output.txt")

        # Before compression - snapshot all tracked entities
        tracker.snapshot_all()

        # ... compression happens, values may change ...

        # After compression - check for conflicts
        conflicts = tracker.check_conflicts()
        if conflicts:
            tracker.inject_warnings_into_summary()
    """

    def __init__(self):
        self._entity_state: dict[str, Any] = {}
        self._entity_history: list[StateSnapshot] = []
        self._pre_compression_snapshot: dict[str, Any] = {}
        self._conflict_log: list[ConflictRecord] = []

    def record(self, entity_key: str, value: Any) -> None:
        """Record the current state of an entity."""
        self._entity_state[entity_key] = copy.deepcopy(value)
        self._entity_history.append(StateSnapshot(
            entity_key=entity_key,
            value=copy.deepcopy(value),
            timestamp=_now(),
        ))

    def snapshot_all(self) -> None:
        """Snapshot all tracked entity states before compression."""
        self._pre_compression_snapshot = copy.deepcopy(self._entity_state)

    def check_conflicts(self) -> list[ConflictRecord]:
        """
        Compare pre/post compression states and detect conflicts.
        Returns list of conflicts found.
        """
        conflicts = []
        for key, pre_value in self._pre_compression_snapshot.items():
            post_value = self._entity_state.get(key)
            if not self._values_equal(pre_value, post_value):
                conflict = ConflictRecord(
                    entity_key=key,
                    pre_compression_value=copy.deepcopy(pre_value),
                    post_compression_value=copy.deepcopy(post_value),
                    warning=self._build_warning(key, pre_value, post_value),
                )
                conflicts.append(conflict)
                self._conflict_log.append(conflict)
        return conflicts

    def has_conflicts(self) -> bool:
        """Quick check if any conflicts were detected."""
        return len(self._conflict_log) > 0

    def get_warnings(self) -> list[str]:
        """Get human-readable warnings for all detected conflicts."""
        return [c.warning for c in self._conflict_log]

    def inject_warnings_into_summary(self) -> str:
        """Generate a summary string with all conflict warnings."""
        if not self._conflict_log:
            return ""
        lines = ["[Entity State Conflict Detected]"]
        for c in self._conflict_log:
            lines.append(f"  - {c.warning}")
        return "\n".join(lines)

    def get_history(self, entity_key: str | None = None) -> list[StateSnapshot]:
        """Get history for all entities or a specific one."""
        if entity_key is None:
            return list(self._entity_history)
        return [s for s in self._entity_history if s.entity_key == entity_key]

    def get_current_state(self, entity_key: str) -> Any:
        """Get current state of a tracked entity."""
        return copy.deepcopy(self._entity_state.get(entity_key))

    def clear(self) -> None:
        """Reset all tracked state."""
        self._entity_state.clear()
        self._entity_history.clear()
        self._pre_compression_snapshot.clear()
        self._conflict_log.clear()

    # --- private helpers ---

    def _values_equal(self, a: Any, b: Any) -> bool:
        try:
            return a == b
        except Exception:
            return False

    def _build_warning(self, key: str, pre: Any, post: Any) -> str:
        pre_repr = _truncate(repr(pre), 50)
        post_repr = _truncate(repr(post), 50)
        return (
            f"Entity '{key}' changed after compression: "
            f"{pre_repr} -> {post_repr}"
        )


def _now() -> float:
    import time
    return time.time()


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."

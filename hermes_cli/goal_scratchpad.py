"""
SOTA structured progress tracker for persistent session goals.

The scratchpad is the agent's working memory for a goal — sub-tasks,
artifacts, decisions, blockers, confidence, and now dependency edges,
error pattern tracking, negative constraints, and turn verdict history.
It persists across turns and is serialized alongside GoalState in SessionDB.

New in v3 (10/10):
- Dependency graph: edges between sub-tasks enable parallel dispatch
- Error pattern tracking: distinguishes transient from systemic failures
- Negative constraints: "do NOT do" list to prevent retrying failing approaches
- Turn verdict history: full record of judge evaluations for trend detection
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────
# SubTask
# ────────────────────────────────────────────────────────────────


@dataclass
class SubTask:
    """A decomposed unit of work within a goal, with dependency edges.

    Each SubTask has a unique id, description, status tracked through
    its lifecycle, optional dependency list (IDs of tasks that must
    complete first), and metadata for timing and blocking.

    Attributes:
        id: Unique identifier within the goal (e.g. 'st01', 'st02').
        description: Human-readable description of the work.
        status: Lifecycle state — pending | in_progress | completed
            | blocked | skipped.
        depends_on: List of task IDs that must complete before this one.
        started_at: Unix timestamp when the task was marked in_progress.
        completed_at: Unix timestamp when the task was marked completed.
        notes: Free-text notes about the task result.
        blocker_reason: Reason the task is blocked, if applicable.
    """

    id: str
    description: str
    status: str = "pending"
    depends_on: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    notes: Optional[str] = None
    blocker_reason: Optional[str] = None

    def mark_started(self) -> None:
        """Record that this task has started execution.

        Sets status to 'in_progress' and records the start timestamp.
        """
        self.status = "in_progress"
        self.started_at = time.time()

    def mark_done(self, notes: str = "") -> None:
        """Record that this task is complete.

        Sets status to 'completed', records completion timestamp, and
        optionally stores notes about the result.

        Args:
            notes: Optional description of what was accomplished.
        """
        self.status = "completed"
        self.completed_at = time.time()
        if notes:
            self.notes = notes

    def mark_blocked(self, reason: str) -> None:
        """Record that this task is blocked and cannot continue.

        Args:
            reason: Description of what is blocking progress.
        """
        self.status = "blocked"
        self.blocker_reason = reason

    @property
    def is_ready(self) -> bool:
        """Check whether this sub-task is ready to start.

        A sub-task is ready when it is pending and has no unresolved
        dependencies (depends_on list is empty).

        Returns:
            True if the task can be started.
        """
        return self.status == "pending" and len(self.depends_on) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this sub-task to a flat dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubTask":
        """Deserialize a sub-task from a dictionary.

        Args:
            data: Dictionary with keys matching SubTask fields.

        Returns:
            A new SubTask instance populated from the dict.
        """
        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            depends_on=data.get("depends_on", []),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            notes=data.get("notes"),
            blocker_reason=data.get("blocker_reason"),
        )


# ────────────────────────────────────────────────────────────────
# Artifact
# ────────────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """A deliverable or file produced during goal execution.

    Tracks files, directories, URLs, services, or other outputs that
    the agent creates. Each artifact can be verified (confirmed to
    exist on disk) for the verification gate.

    Attributes:
        path: Filesystem path, URL, or identifier of the artifact.
        kind: Type — file | directory | url | config | binary
            | service | endpoint.
        description: What this artifact is / what it does.
        created_at: Unix timestamp of creation.
        verified: Whether the artifact has been confirmed to exist.
        verified_at: Unix timestamp of verification.
    """

    path: str
    kind: str
    description: str = ""
    created_at: float = 0.0
    verified: bool = False
    verified_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this artifact to a flat dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Deserialize an artifact from a dictionary.

        Args:
            data: Dictionary with keys matching Artifact fields.

        Returns:
            A new Artifact instance populated from the dict.
        """
        return cls(
            path=data.get("path", ""),
            kind=data.get("kind", "file"),
            description=data.get("description", ""),
            created_at=data.get("created_at", 0.0),
            verified=data.get("verified", False),
            verified_at=data.get("verified_at", 0.0),
        )


# ────────────────────────────────────────────────────────────────
# Decision
# ────────────────────────────────────────────────────────────────


@dataclass
class Decision:
    """A consequential choice made during goal execution.

    Records the context of the decision, what was chosen, the
    rationale, and when (which turn) it was made.

    Attributes:
        context: What was being decided.
        choice: What was chosen.
        why: Rationale for the choice.
        at_turn: Turn number when the decision was made.
    """

    context: str
    choice: str
    why: str = ""
    at_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this decision to a flat dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        """Deserialize a decision from a dictionary.

        Args:
            data: Dictionary with keys matching Decision fields.

        Returns:
            A new Decision instance populated from the dict.
        """
        return cls(
            context=data.get("context", ""),
            choice=data.get("choice", ""),
            why=data.get("why", ""),
            at_turn=data.get("at_turn", 0),
        )


# ────────────────────────────────────────────────────────────────
# GoalScratchpad
# ────────────────────────────────────────────────────────────────


@dataclass
class GoalScratchpad:
    """The agent's durable working memory for a goal.

    Updated by GoalManager during execution. Injected into continuation
    prompts as user message content (preserves prompt caching). Persists
    across turns via serialization to SessionDB's meta store.

    Attributes:
        goal_id: Session ID this scratchpad belongs to.
        decomposition_method: How sub-tasks were created — "auto" | "manual"
            | "none" | "auto_dag".
        sub_tasks: List of SubTask instances with dependency edges.
        artifacts: List of Artifact instances (files/outputs produced).
        decisions: Key decisions made during execution.
        blockers: Active blockers preventing progress.
        notes: Free-form working notes.
        confidence: Self-assessed confidence 0-100.
        total_turns_estimate: Agent's estimate of total turns needed.
        last_updated: Unix timestamp of last modification.
        pivot_count: How many times the strategy has pivoted.
        previous_approaches: Approaches already attempted (for loop avoidance).
        negative_constraints: "Do NOT do" rules accumulated during execution.
        error_patterns: Map of error description → occurrence count.
        history: Turn verdict history for trend/completion tracking.
    """

    goal_id: str = ""
    decomposition_method: str = ""
    sub_tasks: List[SubTask] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    notes: str = ""
    confidence: int = 0
    total_turns_estimate: int = 0
    last_updated: float = 0.0
    pivot_count: int = 0
    previous_approaches: List[str] = field(default_factory=list)
    negative_constraints: List[str] = field(default_factory=list)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ── computed helpers ──────────────────────────────────────

    @property
    def completed_count(self) -> int:
        """Number of sub-tasks with status 'completed'."""
        return sum(1 for st in self.sub_tasks if st.status == "completed")

    @property
    def total_count(self) -> int:
        """Total number of sub-tasks."""
        return len(self.sub_tasks)

    @property
    def blocked_count(self) -> int:
        """Number of sub-tasks with status 'blocked'."""
        return sum(1 for st in self.sub_tasks if st.status == "blocked")

    @property
    def in_progress_count(self) -> int:
        """Number of sub-tasks currently in progress."""
        return sum(1 for st in self.sub_tasks if st.status == "in_progress")

    @property
    def ready_count(self) -> int:
        """Number of pending sub-tasks whose dependencies are all met.

        These tasks can be started in parallel on the next turn.
        """
        completed_ids = {st.id for st in self.sub_tasks if st.status == "completed"}
        return sum(
            1 for st in self.sub_tasks
            if st.status == "pending"
            and all(dep in completed_ids for dep in st.depends_on)
        )

    @property
    def progress_pct(self) -> float:
        """Estimated progress fraction 0.0–1.0.

        When sub-tasks exist, computed from completed (1.0) and in-progress
        (0.5) task weights. When no sub-tasks, falls back to confidence/100.
        """
        if self.total_count == 0:
            return float(self.confidence) / 100.0
        completed_weight = self.completed_count * 1.0
        in_progress_weight = self.in_progress_count * 0.5
        return min(1.0, (completed_weight + in_progress_weight) / self.total_count)

    @property
    def current_task(self) -> Optional[SubTask]:
        """The sub-task currently in progress, or None."""
        for st in self.sub_tasks:
            if st.status == "in_progress":
                return st
        return None

    @property
    def next_pending(self) -> Optional[SubTask]:
        """The next pending sub-task whose dependencies are met.

        Falls back to the first pending task if no tasks have all
        dependencies satisfied.

        Returns:
            A SubTask ready to start, or None if all tasks are done.
        """
        completed_ids = {st.id for st in self.sub_tasks if st.status == "completed"}
        for st in self.sub_tasks:
            if st.status == "pending" and all(dep in completed_ids for dep in st.depends_on):
                return st
        # Fall back to first pending if DAG-aware search yields nothing
        for st in self.sub_tasks:
            if st.status == "pending":
                return st
        return None

    def get_ready_tasks(self) -> List[SubTask]:
        """Return all pending tasks whose dependencies are met and blockers resolved.

        These tasks can be executed in parallel on the next turn.

        Returns:
            List of SubTask instances ready for execution.
        """
        completed_ids = {st.id for st in self.sub_tasks if st.status == "completed"}
        active_blockers = {b.lower() for b in self.blockers}
        return [
            st for st in self.sub_tasks
            if st.status == "pending"
            and all(dep in completed_ids for dep in st.depends_on)
            and not (st.blocker_reason and st.blocker_reason.lower() in active_blockers)
        ]

    def get_unblocked_pending(self) -> List[SubTask]:
        """Return all pending tasks whose individual blockers are resolved.

        Unlike get_ready_tasks, this does not check dependency edges —
        only sub-task-level blockers are considered.

        Returns:
            List of pending SubTask instances without active blockers.
        """
        active_blockers = {b.lower() for b in self.blockers}
        result = []
        for st in self.sub_tasks:
            if st.status != "pending":
                continue
            if st.blocker_reason and st.blocker_reason.lower() in active_blockers:
                continue
            result.append(st)
        return result

    # ── progress bar ──────────────────────────────────────────

    def progress_bar(self, width: int = 20) -> str:
        """Render a visual progress bar string.

        Uses sub-task completion when available, otherwise falls back to
        confidence score. Format: "[███████░░░░] 45%"

        Args:
            width: Character width of the bar (default 20).

        Returns:
            A string like '[██████░░░░░░░░░░░░░] 25%'.
        """
        if self.total_count == 0:
            filled = int(self.confidence / 100 * width)
        else:
            filled = int(self.progress_pct * width)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        pct = int(self.progress_pct * 100)
        return f"[{bar}] {pct}%"

    # ── summary for status display ────────────────────────────

    def summary(self) -> str:
        """Build a human-readable status summary for display.

        Includes progress bar, task counts, current task, blockers,
        negative constraints, recent artifacts, and working notes.

        Returns:
            A multi-line string suitable for CLI or status panel display.
        """
        lines = [self.progress_bar(20)]
        if self.total_count > 0:
            lines.append(
                f"Tasks: {self.completed_count}/{self.total_count} done"
                f"{f', {self.blocked_count} blocked' if self.blocked_count else ''}"
                f"{f', {self.in_progress_count} in progress' if self.in_progress_count else ''}"
                f"{f', {self.ready_count} ready' if self.ready_count else ''}"
            )
        if self.current_task:
            lines.append(f"Current: {self.current_task.description}")
        if self.blockers:
            lines.append(f"Blockers: {', '.join(self.blockers[:3])}")
        if self.negative_constraints:
            lines.append(f"Constraints: {', '.join(self.negative_constraints[:3])}")
        if self.artifacts:
            lines.append(
                f"Artifacts: {', '.join(a.path for a in self.artifacts[-3:])}"
            )
        if self.notes:
            notes_snip = self.notes[:120]
            if len(self.notes) > 120:
                notes_snip += "…"
            lines.append(f"Notes: {notes_snip}")
        return "\n".join(lines)

    # ── context for continuation prompt ───────────────────────

    def context_for_prompt(self) -> str:
        """Build a rich context block for injection into the continuation prompt.

        Includes progress bar and sub-task statuses (with dependency notes),
        artifacts with verification status, key decisions, active blockers,
        negative constraints, approaches tried, recurring errors, and notes.

        Returns:
            A markdown-formatted context string, or empty string if the
            scratchpad has no data.
        """
        parts = []

        if self.sub_tasks:
            statuses = {
                "completed": "✓",
                "in_progress": "→",
                "blocked": "✗",
                "pending": "○",
                "skipped": "—",
            }
            parts.append("## Progress")
            parts.append(self.progress_bar(20))
            parts.append("")
            for st in self.sub_tasks:
                icon = statuses.get(st.status, "?")
                dep_note = ""
                if st.depends_on:
                    dep_note = f" [depends on: {', '.join(st.depends_on)}]"
                parts.append(f"- {icon} {st.description}{dep_note}")
                if st.notes and st.status == "completed":
                    parts.append(f"  ↳ {st.notes}")
                if st.blocker_reason:
                    parts.append(f"  ⚠ Blocked: {st.blocker_reason}")
            parts.append("")

        if self.artifacts:
            parts.append("## Artifacts Created")
            for a in self.artifacts:
                verified_mark = "✓" if a.verified else "?"
                parts.append(f"- [{verified_mark}] {a.path} — {a.description or '(no description)'}")
            parts.append("")

        if self.decisions:
            parts.append("## Key Decisions")
            for d in self.decisions[-5:]:
                parts.append(f"- T{d.at_turn}: {d.context} → {d.choice}")
            parts.append("")

        if self.blockers:
            parts.append("## Active Blockers")
            for b in self.blockers:
                parts.append(f"- ⚠ {b}")
            parts.append("")

        if self.negative_constraints:
            parts.append("## Do NOT Do")
            for nc in self.negative_constraints:
                parts.append(f"- 🚫 {nc}")
            parts.append("")

        if self.previous_approaches:
            parts.append("## Approaches Tried")
            for a in self.previous_approaches[-5:]:
                parts.append(f"- {a}")
            parts.append("")

        if self.error_patterns:
            parts.append("## Recurring Errors")
            for err, count in sorted(self.error_patterns.items(), key=lambda x: -x[1]):
                if count >= 2:
                    parts.append(f"- ⚠ [{count}x] {err[:80]}")
            parts.append("")

        if self.notes:
            parts.append("## Working Notes")
            parts.append(self.notes[:500])
            parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    # ── serialization ─────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the scratchpad to a JSON string for storage in SessionDB.

        Returns:
            JSON string representation of all scratchpad state.
        """
        data = {
            "goal_id": self.goal_id,
            "decomposition_method": self.decomposition_method,
            "sub_tasks": [st.to_dict() for st in self.sub_tasks],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "decisions": [d.to_dict() for d in self.decisions],
            "blockers": self.blockers,
            "notes": self.notes,
            "confidence": self.confidence,
            "total_turns_estimate": self.total_turns_estimate,
            "last_updated": self.last_updated,
            "pivot_count": self.pivot_count,
            "previous_approaches": self.previous_approaches,
            "negative_constraints": self.negative_constraints,
            "error_patterns": self.error_patterns,
            "history": self.history,
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalScratchpad":
        """Deserialize a scratchpad from a JSON string.

        Args:
            raw: JSON string previously produced by to_json().

        Returns:
            A new GoalScratchpad instance populated from the JSON data.
        """
        data = json.loads(raw)
        pad = cls(
            goal_id=data.get("goal_id", ""),
            decomposition_method=data.get("decomposition_method", ""),
            notes=data.get("notes", ""),
            confidence=data.get("confidence", 0),
            total_turns_estimate=data.get("total_turns_estimate", 0),
            last_updated=data.get("last_updated", 0.0),
            pivot_count=data.get("pivot_count", 0),
            previous_approaches=data.get("previous_approaches", []),
            negative_constraints=data.get("negative_constraints", []),
            error_patterns=data.get("error_patterns", {}),
            history=data.get("history", []),
        )
        pad.sub_tasks = [SubTask.from_dict(st) for st in data.get("sub_tasks", [])]
        pad.artifacts = [Artifact.from_dict(a) for a in data.get("artifacts", [])]
        pad.decisions = [Decision.from_dict(d) for d in data.get("decisions", [])]
        pad.blockers = data.get("blockers", [])
        return pad

    @classmethod
    def empty(cls, goal_id: str = "") -> "GoalScratchpad":
        """Create a fresh empty scratchpad for a new goal.

        Args:
            goal_id: Session ID this scratchpad belongs to.

        Returns:
            A new GoalScratchpad with default field values.
        """
        return cls(goal_id=goal_id)

    # ── mutation helpers ──────────────────────────────────────

    def touch(self) -> None:
        """Update the last_updated timestamp to now."""
        self.last_updated = time.time()

    def add_artifact(self, path: str, kind: str = "file", description: str = "", verified: bool = False) -> Artifact:
        """Register a new artifact (file/output) produced during this turn.

        Args:
            path: Filesystem path or identifier.
            kind: Type of artifact (file, directory, url, etc.).
            description: Description of the artifact's purpose.
            verified: Whether the artifact is pre-verified. Default False.

        Returns:
            The newly created Artifact instance.
        """
        now = time.time()
        artifact = Artifact(path=path, kind=kind, description=description, created_at=now, verified=verified, verified_at=now if verified else 0.0)
        self.artifacts.append(artifact)
        self.touch()
        return artifact

    def verify_artifact(self, path: str) -> bool:
        """Mark an artifact as verified if the file exists on disk.

        Calls os.path.isfile() to check the actual filesystem. This
        enables the verification gate to auto-verify artifacts when
        the judge says 'done' — existing files pass, missing files
        trigger a refine_output.

        Args:
            path: The artifact path to verify.

        Returns:
            True if the file exists and the artifact was found and updated.
            False if the file is missing or the artifact is not registered.
        """
        import os
        path_expanded = os.path.abspath(os.path.expanduser(path))
        exists = os.path.isfile(path_expanded)
        for a in self.artifacts:
            if a.path == path:
                a.verified = exists
                a.verified_at = time.time() if exists else 0.0
                self.touch()
                return exists
        return False

    def add_decision(self, context: str, choice: str, why: str = "", at_turn: int = 0) -> Decision:
        """Record a key decision made during execution.

        Args:
            context: What was being decided.
            choice: What was chosen.
            why: Rationale for the choice.
            at_turn: Turn number when the decision was made.

        Returns:
            The newly created Decision instance.
        """
        d = Decision(context=context, choice=choice, why=why, at_turn=at_turn)
        self.decisions.append(d)
        self.touch()
        return d

    def add_blocker(self, reason: str) -> None:
        """Register an active blocker that prevents progress.

        Deduplicates case-insensitively — adding the same reason
        twice is a no-op.

        Args:
            reason: Description of the blocker.
        """
        reason_lower = reason.lower()
        if not any(b.lower() == reason_lower for b in self.blockers):
            self.blockers.append(reason)
        self.touch()

    def resolve_blocker(self, reason: str) -> bool:
        """Remove a previously registered blocker.

        Args:
            reason: The blocker reason text to remove. Case-insensitive.

        Returns:
            True if the blocker was found and removed, False otherwise.
        """
        reason_lower = reason.lower()
        for b in self.blockers:
            if b.lower() == reason_lower:
                self.blockers.remove(b)
                self.touch()
                return True
        return False

    def record_approach(self, description: str) -> None:
        """Record a strategy or approach that was attempted.

        Deduplicates case-insensitively and increments pivot_count.

        Args:
            description: What approach was tried.
        """
        desc_lower = description.lower()
        if not any(a.lower() == desc_lower for a in self.previous_approaches):
            self.previous_approaches.append(description)
        self.pivot_count += 1
        self.touch()

    def add_negative_constraint(self, rule: str) -> None:
        """Add a 'do NOT do' rule to prevent retrying a failing approach.

        Args:
            rule: The constraint text (e.g. 'Do NOT retry the failing test
                without changing the implementation first').
        """
        rule_lower = rule.lower()
        if not any(nc.lower() == rule_lower for nc in self.negative_constraints):
            self.negative_constraints.append(rule)
        self.touch()

    def track_error(self, error_msg: str) -> None:
        """Record and count a recurring error for pattern detection.

        When the same error is tracked enough times, the judge will
        detect it as a systemic error pattern.

        Args:
            error_msg: Description of the error.
        """
        key = error_msg[:120].lower()
        self.error_patterns[key] = self.error_patterns.get(key, 0) + 1
        self.touch()

    def record_verdict(self, verdict: Dict[str, Any]) -> None:
        """Save a turn verdict into the history for trend detection.

        Keeps the last 50 entries maximum to bound memory.

        Args:
            verdict: Dict with keys like action, completion, progress,
                quality, timestamp from the JudgeVerdict.
        """
        self.history.append(verdict)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self.touch()

    def advance_task(self, notes: str = "") -> Optional[SubTask]:
        """Complete the current in-progress task and start the next ready one.

        Marks the current in-progress task as done, then starts the
        next pending task whose dependencies are met.

        Args:
            notes: Optional notes to attach to the completed task.

        Returns:
            The newly started SubTask, or None if no ready tasks remain.
        """
        cur = self.current_task
        if cur:
            cur.mark_done(notes)
        nxt = self.next_pending
        if nxt:
            nxt.mark_started()
        self.touch()
        return nxt

    def set_confidence(self, value: int) -> None:
        """Set the agent's self-assessed confidence (clamped 0-100).

        Args:
            value: Confidence level 0-100.
        """
        self.confidence = max(0, min(100, value))
        self.touch()

    def set_notes(self, text: str) -> None:
        """Update the free-form working notes.

        Args:
            text: New notes content (replaces existing).
        """
        self.notes = text
        self.touch()

    # ── DAG helpers ───────────────────────────────────────────

    def infer_dependencies(self) -> None:
        """Infer dependency edges from sub-task ordering.

        If no explicit dependencies are set, creates a linear chain so
        tasks execute sequentially by default. Safe default — avoids
        accidentally parallelizing tasks that have implicit ordering.

        No-op if any sub-task already has dependencies defined.
        """
        if not self.sub_tasks:
            return

        any_deps = any(st.depends_on for st in self.sub_tasks)
        if any_deps:
            return

        for i in range(1, len(self.sub_tasks)):
            self.sub_tasks[i].depends_on = [self.sub_tasks[i - 1].id]

    def get_parallel_batches(self) -> List[List[SubTask]]:
        """Group sub-tasks into parallel-executable batches.

        Each batch contains tasks whose dependencies are all met.
        Multiple batches are sequential — all tasks in batch N must
        complete before batch N+1 starts.

        Handles deadlock: if no tasks can proceed, the remaining tasks
        are placed in a single batch with a warning.

        Returns:
            List of batches. Each batch is a list of SubTask instances
            that can run concurrently.
        """
        if not self.sub_tasks:
            return []

        completed = {st.id for st in self.sub_tasks if st.status == "completed"}
        batches: List[List[SubTask]] = []
        remaining = [st for st in self.sub_tasks if st.status in ("pending", "in_progress")]

        while remaining:
            batch = [
                st for st in remaining
                if st.status == "pending"
                and all(dep in completed for dep in st.depends_on)
            ]
            if not batch:
                batches.append(remaining)
                break
            batches.append(batch)
            for st in batch:
                completed.add(st.id)
            remaining = [st for st in remaining if st not in batch]

        return batches

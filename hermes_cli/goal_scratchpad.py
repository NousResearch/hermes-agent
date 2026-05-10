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
    """A decomposed unit of work within a goal, with dependency edges."""

    id: str
    description: str
    status: str = "pending"  # pending | in_progress | completed | blocked | skipped
    depends_on: List[str] = field(default_factory=list)  # IDs of tasks that must complete first
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    notes: Optional[str] = None
    blocker_reason: Optional[str] = None

    def mark_started(self) -> None:
        self.status = "in_progress"
        self.started_at = time.time()

    def mark_done(self, notes: str = "") -> None:
        self.status = "completed"
        self.completed_at = time.time()
        if notes:
            self.notes = notes

    def mark_blocked(self, reason: str) -> None:
        self.status = "blocked"
        self.blocker_reason = reason

    @property
    def is_ready(self) -> bool:
        """Sub-task is ready to start if all dependencies are completed."""
        return self.status == "pending" and len(self.depends_on) == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubTask":
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
    """A deliverable or file produced during goal execution."""

    path: str
    kind: str  # file | directory | url | config | binary | service | endpoint
    description: str = ""
    created_at: float = 0.0
    verified: bool = False  # has the agent confirmed it exists/works?
    verified_at: float = 0.0  # when it was verified

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
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
    """A consequential choice made during goal execution."""

    context: str  # what was being decided
    choice: str   # what was chosen
    why: str = ""  # rationale
    at_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
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
    prompts as user message content (preserves prompt caching).
    """

    goal_id: str = ""  # session_id this belongs to
    decomposition_method: str = ""  # "auto" | "manual" | "none" | "auto_dag"
    sub_tasks: List[SubTask] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    notes: str = ""  # free-form agent notes
    confidence: int = 0  # 0-100, agent's self-assessment
    total_turns_estimate: int = 0  # agent's estimate of total turns needed
    last_updated: float = 0.0
    pivot_count: int = 0  # how many times the strategy pivoted
    previous_approaches: List[str] = field(default_factory=list)
    negative_constraints: List[str] = field(default_factory=list)  # "do NOT do" rules
    error_patterns: Dict[str, int] = field(default_factory=dict)  # error → count
    history: List[Dict[str, Any]] = field(default_factory=list)  # turn verdict history

    # ── computed helpers ──────────────────────────────────────

    @property
    def completed_count(self) -> int:
        return sum(1 for st in self.sub_tasks if st.status == "completed")

    @property
    def total_count(self) -> int:
        return len(self.sub_tasks)

    @property
    def blocked_count(self) -> int:
        return sum(1 for st in self.sub_tasks if st.status == "blocked")

    @property
    def in_progress_count(self) -> int:
        return sum(1 for st in self.sub_tasks if st.status == "in_progress")

    @property
    def ready_count(self) -> int:
        """Tasks that can be started now (pending, no blocking deps)."""
        completed_ids = {st.id for st in self.sub_tasks if st.status == "completed"}
        return sum(
            1 for st in self.sub_tasks
            if st.status == "pending"
            and all(dep in completed_ids for dep in st.depends_on)
        )

    @property
    def progress_pct(self) -> float:
        if self.total_count == 0:
            return float(self.confidence) / 100.0
        completed_weight = self.completed_count * 1.0
        in_progress_weight = self.in_progress_count * 0.5
        return min(1.0, (completed_weight + in_progress_weight) / self.total_count)

    @property
    def current_task(self) -> Optional[SubTask]:
        for st in self.sub_tasks:
            if st.status == "in_progress":
                return st
        return None

    @property
    def next_pending(self) -> Optional[SubTask]:
        """Next pending task that has all dependencies met."""
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
        """All tasks that can be executed in parallel right now."""
        completed_ids = {st.id for st in self.sub_tasks if st.status == "completed"}
        active_blockers = {b.lower() for b in self.blockers}
        return [
            st for st in self.sub_tasks
            if st.status == "pending"
            and all(dep in completed_ids for dep in st.depends_on)
            and not (st.blocker_reason and st.blocker_reason.lower() in active_blockers)
        ]

    def get_unblocked_pending(self) -> List[SubTask]:
        """Return pending tasks whose blockers (if any) are resolved."""
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
        """Build a rich context block to inject into the continuation prompt."""
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
            for d in self.decisions[-5:]:  # last 5
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
        return cls(goal_id=goal_id)

    # ── mutation helpers ──────────────────────────────────────

    def touch(self) -> None:
        self.last_updated = time.time()

    def add_artifact(self, path: str, kind: str = "file", description: str = "", verified: bool = False) -> Artifact:
        now = time.time()
        artifact = Artifact(path=path, kind=kind, description=description, created_at=now, verified=verified, verified_at=now if verified else 0.0)
        self.artifacts.append(artifact)
        self.touch()
        return artifact

    def verify_artifact(self, path: str) -> bool:
        """Mark an artifact as verified. Returns True if found."""
        for a in self.artifacts:
            if a.path == path:
                a.verified = True
                a.verified_at = time.time()
                self.touch()
                return True
        return False

    def add_decision(self, context: str, choice: str, why: str = "", at_turn: int = 0) -> Decision:
        d = Decision(context=context, choice=choice, why=why, at_turn=at_turn)
        self.decisions.append(d)
        self.touch()
        return d

    def add_blocker(self, reason: str) -> None:
        """Add a blocker. Deduplicates case-insensitively."""
        reason_lower = reason.lower()
        if not any(b.lower() == reason_lower for b in self.blockers):
            self.blockers.append(reason)
        self.touch()

    def resolve_blocker(self, reason: str) -> bool:
        """Remove a blocker. Returns True if found."""
        reason_lower = reason.lower()
        for b in self.blockers:
            if b.lower() == reason_lower:
                self.blockers.remove(b)
                self.touch()
                return True
        return False

    def record_approach(self, description: str) -> None:
        """Record a strategy attempt. Deduplicates case-insensitively."""
        desc_lower = description.lower()
        if not any(a.lower() == desc_lower for a in self.previous_approaches):
            self.previous_approaches.append(description)
        self.pivot_count += 1
        self.touch()

    def add_negative_constraint(self, rule: str) -> None:
        """Add a 'do NOT do' rule. Deduplicated."""
        rule_lower = rule.lower()
        if not any(nc.lower() == rule_lower for nc in self.negative_constraints):
            self.negative_constraints.append(rule)
        self.touch()

    def track_error(self, error_msg: str) -> None:
        """Track a recurring error for pattern detection."""
        key = error_msg[:120].lower()
        self.error_patterns[key] = self.error_patterns.get(key, 0) + 1
        self.touch()

    def record_verdict(self, verdict: Dict[str, Any]) -> None:
        """Save a verdict in history for trend detection."""
        self.history.append(verdict)
        # Keep last 50 entries max
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self.touch()

    def advance_task(self, notes: str = "") -> Optional[SubTask]:
        """Mark current in-progress task as done and start the next ready task.
        Returns the newly started task, or None if no ready tasks remain."""
        cur = self.current_task
        if cur:
            cur.mark_done(notes)
        nxt = self.next_pending
        if nxt:
            nxt.mark_started()
        self.touch()
        return nxt

    def set_confidence(self, value: int) -> None:
        self.confidence = max(0, min(100, value))
        self.touch()

    def set_notes(self, text: str) -> None:
        self.notes = text
        self.touch()

    # ── DAG helpers ───────────────────────────────────────────

    def infer_dependencies(self) -> None:
        """Infer dependency edges from sub-task ordering if none exist.
        Sequential tasks get linear dependencies; this enables parallel
        dispatch when tasks are truly independent."""
        if not self.sub_tasks:
            return

        any_deps = any(st.depends_on for st in self.sub_tasks)
        if any_deps:
            return  # Already has explicit dependencies

        # Default: linear chain for safety
        for i in range(1, len(self.sub_tasks)):
            self.sub_tasks[i].depends_on = [self.sub_tasks[i - 1].id]

    def get_parallel_batches(self) -> List[List[SubTask]]:
        """Group ready tasks into parallel-executable batches.
        Each batch is a set of tasks with all deps met."""
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
                # Deadlock: put remaining in a single batch
                batches.append(remaining)
                break
            batches.append(batch)
            for st in batch:
                completed.add(st.id)
            remaining = [st for st in remaining if st not in batch]

        return batches

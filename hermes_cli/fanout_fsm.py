"""Finite State Machine for the /fanout command review workflow.

Models the lifecycle of a fanout plan from decomposition through review,
critique, confirmation, and story execution.

States:
    IDLE          - Initial state, no plan exists
    DECOMPOSING   - Subagent is decomposing the task into stories
    PLAN_READY    - Plan has been generated, awaiting user review
    EDITING       - User is editing the plan file (external to FSM)
    CRITIQUING    - User has provided critique; feedback accumulated
    CONFIRMED     - User has accepted the plan
    EXECUTING     - Stories are being executed
    COMPLETE      - All stories finished
    ABORTED       - User aborted or fatal error

Transitions:
    start(task)             IDLE       → DECOMPOSING
    decomposition_done()    DECOMPOSING → PLAN_READY
    decomposition_fail()    DECOMPOSING → IDLE
    critique(text)          PLAN_READY → CRITIQUING
    re_decompose()          CRITIQUING  → DECOMPOSING
    accept()                PLAN_READY → CONFIRMED
    edit()                  PLAN_READY → EDITING
    resume_review()         EDITING    → PLAN_READY
    execute()               CONFIRMED  → EXECUTING
    story_done(sid)         EXECUTING  → EXECUTING (progress)
    all_stories_done()      EXECUTING  → COMPLETE
    abort()                 *          → ABORTED
    retry()                 ABORTED    → IDLE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FanoutState(Enum):
    """States of the fanout review FSM."""
    IDLE = "idle"
    DECOMPOSING = "decomposing"
    PLAN_READY = "plan_ready"
    EDITING = "editing"
    CRITIQUING = "critiquing"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    COMPLETE = "complete"
    ABORTED = "aborted"


class FanoutTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: FanoutState, action: str, allowed: list[str]):
        self.current = current
        self.action = action
        self.allowed = allowed
        super().__init__(
            f"Cannot '{action}' from state '{current.value}'. "
            f"Allowed actions: {', '.join(allowed)}"
        )


@dataclass(frozen=True)
class FanoutStory:
    """A single story in a fanout plan."""
    id: str
    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    acceptance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "acceptance": list(self.acceptance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FanoutStory":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "unnamed")),
            description=str(data.get("description", "")),
            dependencies=list(data.get("dependencies", [])),
            acceptance=list(data.get("acceptance", [])),
        )


@dataclass
class FanoutPlan:
    """Complete fanout plan with stories and tracking."""
    task: str
    stories: list[FanoutStory] = field(default_factory=list)
    completed: set[str] = field(default_factory=set)
    critique_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "stories": [s.to_dict() for s in self.stories],
            "completed": sorted(self.completed),
            "critique_history": list(self.critique_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FanoutPlan":
        stories = [FanoutStory.from_dict(s) for s in data.get("stories", [])]
        completed = set(data.get("completed", []))
        critique_history = list(data.get("critique_history", []))
        return cls(
            task=str(data.get("task", "")),
            stories=stories,
            completed=completed,
            critique_history=critique_history,
        )

    @property
    def is_complete(self) -> bool:
        """All stories have been completed."""
        return bool(self.stories) and self.completed >= {s.id for s in self.stories}

    @property
    def remaining_stories(self) -> list[FanoutStory]:
        """Stories not yet completed, respecting dependency order."""
        return [s for s in self.stories if s.id not in self.completed]

    @property
    def ready_stories(self) -> list[FanoutStory]:
        """Stories whose dependencies are all satisfied and not yet done."""
        ready = []
        for s in self.remaining_stories:
            if all(d in self.completed for d in s.dependencies):
                ready.append(s)
        return ready

    @property
    def blocked_stories(self) -> list[FanoutStory]:
        """Stories blocked by unmet dependencies."""
        blocked = []
        for s in self.remaining_stories:
            unmet = [d for d in s.dependencies if d not in self.completed]
            if unmet:
                blocked.append(s)
        return blocked


_ALLOWED_TRANSITIONS: dict[FanoutState, dict[str, FanoutState]] = {
    FanoutState.IDLE: {
        "start": FanoutState.DECOMPOSING,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.DECOMPOSING: {
        "decomposition_done": FanoutState.PLAN_READY,
        "decomposition_fail": FanoutState.IDLE,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.PLAN_READY: {
        "accept": FanoutState.CONFIRMED,
        "edit": FanoutState.EDITING,
        "critique": FanoutState.CRITIQUING,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.EDITING: {
        "resume_review": FanoutState.PLAN_READY,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.CRITIQUING: {
        "re_decompose": FanoutState.DECOMPOSING,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.CONFIRMED: {
        "execute": FanoutState.EXECUTING,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.EXECUTING: {
        "story_done": FanoutState.EXECUTING,
        "all_stories_done": FanoutState.COMPLETE,
        "abort": FanoutState.ABORTED,
    },
    FanoutState.COMPLETE: {
        "retry": FanoutState.IDLE,
    },
    FanoutState.ABORTED: {
        "retry": FanoutState.IDLE,
    },
}


class FanoutReviewFSM:
    """State machine for the /fanout command review workflow.

    Usage::

        fsm = FanoutReviewFSM()
        fsm.start("Build a web scraper")
        # ... subagent decomposes ...
        fsm.decomposition_done(stories=[...])
        # ... user reviews plan ...
        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        fsm.story_done("002")
        fsm.all_stories_done()
        assert fsm.state == FanoutState.COMPLETE
    """

    def __init__(self) -> None:
        self._state: FanoutState = FanoutState.IDLE
        self._plan: FanoutPlan | None = None
        self._history: list[tuple[str, FanoutState, FanoutState]] = []

    @property
    def state(self) -> FanoutState:
        """Current FSM state."""
        return self._state

    @property
    def plan(self) -> FanoutPlan | None:
        """Current plan, or None if not yet decomposed."""
        return self._plan

    @property
    def history(self) -> list[tuple[str, FanoutState, FanoutState]]:
        """Transition log: (action, from_state, to_state)."""
        return list(self._history)

    # ------------------------------------------------------------------ #
    #  Internal transition helper
    # ------------------------------------------------------------------ #

    def _transition(self, action: str) -> None:
        """Attempt a state transition. Raises FanoutTransitionError if invalid."""
        allowed = _ALLOWED_TRANSITIONS.get(self._state, {})
        if action not in allowed:
            raise FanoutTransitionError(
                self._state, action, sorted(allowed.keys())
            )
        old = self._state
        self._state = allowed[action]
        self._history.append((action, old, self._state))

    # ------------------------------------------------------------------ #
    #  Transitions
    # ------------------------------------------------------------------ #

    def start(self, task: str) -> None:
        """Begin decomposition of a task. IDLE → DECOMPOSING."""
        self._transition("start")
        self._plan = FanoutPlan(task=task)

    def decomposition_done(self, stories: list[FanoutStory]) -> None:
        """Decomposition succeeded; plan is ready for review. DECOMPOSING → PLAN_READY."""
        self._transition("decomposition_done")
        if self._plan is None:
            raise RuntimeError("No plan exists — call start() first")
        self._plan.stories = stories

    def decomposition_fail(self) -> None:
        """Decomposition failed. DECOMPOSING → IDLE."""
        self._transition("decomposition_fail")

    def accept(self) -> None:
        """User accepted the plan. PLAN_READY → CONFIRMED."""
        self._transition("accept")

    def edit(self) -> None:
        """User wants to edit the plan file. PLAN_READY → EDITING."""
        self._transition("edit")

    def resume_review(self) -> None:
        """User finished editing; return to review. EDITING → PLAN_READY."""
        self._transition("resume_review")

    def critique(self, text: str) -> None:
        """User provided critique feedback. PLAN_READY → CRITIQUING."""
        self._transition("critique")
        if self._plan is not None:
            self._plan.critique_history.append(text)

    def re_decompose(self) -> None:
        """Re-decompose with accumulated critique. CRITIQUING → DECOMPOSING."""
        self._transition("re_decompose")

    def execute(self) -> None:
        """Begin story execution. CONFIRMED → EXECUTING."""
        self._transition("execute")

    def story_done(self, story_id: str) -> None:
        """Mark a story as completed. EXECUTING → EXECUTING."""
        self._transition("story_done")
        if self._plan is not None:
            self._plan.completed.add(story_id)

    def all_stories_done(self) -> None:
        """All stories completed. EXECUTING → COMPLETE."""
        self._transition("all_stories_done")

    def abort(self) -> None:
        """Abort the fanout from any state. * → ABORTED."""
        if self._state in (FanoutState.COMPLETE, FanoutState.ABORTED, FanoutState.IDLE):
            # Already terminal or idle — just set aborted
            old = self._state
            self._state = FanoutState.ABORTED
            self._history.append(("abort", old, self._state))
            return
        self._transition("abort")

    def retry(self) -> None:
        """Reset to IDLE for a new attempt. ABORTED/COMPLETE → IDLE."""
        self._transition("retry")
        self._plan = None

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    @property
    def accumulated_critique(self) -> str:
        """Return all accumulated critique as a single string."""
        if self._plan is None:
            return ""
        return "\n\n".join(self._plan.critique_history)

    @property
    def is_terminal(self) -> bool:
        """True when in a terminal state (COMPLETE, ABORTED)."""
        return self._state in (FanoutState.COMPLETE, FanoutState.ABORTED)

    @property
    def pending_story_ids(self) -> list[str]:
        """Story IDs not yet completed."""
        if self._plan is None:
            return []
        return [s.id for s in self._plan.remaining_stories]

    @property
    def ready_story_ids(self) -> list[str]:
        """Story IDs ready to execute (dependencies satisfied)."""
        if self._plan is None:
            return []
        return [s.id for s in self._plan.ready_stories]

    @property
    def blocked_story_ids(self) -> list[str]:
        """Story IDs blocked by unmet dependencies."""
        if self._plan is None:
            return []
        return [s.id for s in self._plan.blocked_stories]

    def story(self, story_id: str) -> FanoutStory | None:
        """Look up a story by ID."""
        if self._plan is None:
            return None
        for s in self._plan.stories:
            if s.id == story_id:
                return s
        return None

    def reset(self) -> None:
        """Hard reset to initial state, discarding all data."""
        self._state = FanoutState.IDLE
        self._plan = None
        self._history.clear()

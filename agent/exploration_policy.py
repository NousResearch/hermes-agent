"""Replayable exploration-policy contracts for staged RSI experiments.

This module deliberately models only bounded orchestration choices.  It does not grant tools,
credentials, approvals, or filesystem/network authority to a policy.  Recorded trees are safe to
serialize because they contain digests and outcome metadata rather than conversation contents.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple


EXPLORATION_TREE_SCHEMA_VERSION = 1
DEFAULT_POLICY_VERSION = "baseline-v1"
EXPLORATION_ACTIONS = frozenset({"continue", "stop", "retry", "backtrack", "delegate", "verify"})
_POLICY_STATUSES = frozenset({"draft", "shadow", "canary", "promoted", "rejected", "rolled_back"})
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _require_non_negative(name: str, value: int) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExplorationCost:
    """Bounded resource accounting attached to one explored branch."""

    token_count: int = 0
    tool_calls: int = 0
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        _require_non_negative("token_count", self.token_count)
        _require_non_negative("tool_calls", self.tool_calls)
        _require_non_negative("elapsed_ms", self.elapsed_ms)

    def to_dict(self) -> Dict[str, int]:
        return {
            "token_count": self.token_count,
            "tool_calls": self.tool_calls,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplorationCost":
        return cls(
            token_count=value.get("token_count", 0),
            tool_calls=value.get("tool_calls", 0),
            elapsed_ms=value.get("elapsed_ms", 0),
        )


@dataclass(frozen=True)
class ExplorationBranch:
    """One action whose downstream result was actually observed in a tree."""

    action: str
    child_ids: Tuple[str, ...] = ()
    score: Optional[float] = None
    terminal: bool = False
    cost: ExplorationCost = field(default_factory=ExplorationCost)

    def __post_init__(self) -> None:
        _require_text("action", self.action)
        if any(not isinstance(child_id, str) or not child_id.strip() for child_id in self.child_ids):
            raise ValueError("child_ids must contain non-empty strings")
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric or None")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "child_ids": list(self.child_ids),
            "score": self.score,
            "terminal": self.terminal,
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplorationBranch":
        return cls(
            action=value["action"],
            child_ids=tuple(value.get("child_ids", ())),
            score=value.get("score"),
            terminal=bool(value.get("terminal", False)),
            cost=ExplorationCost.from_dict(value.get("cost", {})),
        )


@dataclass(frozen=True)
class ExplorationState:
    """The policy-visible, secret-free state at one bounded decision point."""

    decision_id: str
    parent_id: Optional[str]
    state_digest: str
    available_actions: Tuple[str, ...]
    budget_remaining: Optional[int] = None
    depth: int = 0

    def __post_init__(self) -> None:
        _require_text("decision_id", self.decision_id)
        _require_text("state_digest", self.state_digest)
        if not self.available_actions:
            raise ValueError("available_actions must not be empty")
        if any(action not in EXPLORATION_ACTIONS for action in self.available_actions):
            raise ValueError("available_actions contains an unbounded action")
        if self.budget_remaining is not None:
            _require_non_negative("budget_remaining", self.budget_remaining)
        _require_non_negative("depth", self.depth)


@dataclass(frozen=True)
class ExplorationDecision:
    """A policy output; the replay evaluator fences it against the recorded action set."""

    action: str
    branch_count: int = 0

    def __post_init__(self) -> None:
        _require_text("action", self.action)
        _require_non_negative("branch_count", self.branch_count)


class ExplorationPolicy(Protocol):
    """Minimal seam for a deterministic or model-backed orchestration policy."""

    version: str

    def decide(self, state: ExplorationState) -> ExplorationDecision:
        ...


@dataclass(frozen=True)
class ExplorationEvent:
    """One decision point in an exploration tree."""

    decision_id: str
    parent_id: Optional[str]
    policy_version: str
    state_digest: str
    available_actions: Tuple[str, ...]
    chosen_action: str
    branches: Tuple[ExplorationBranch, ...]
    depth: int = 0

    def __post_init__(self) -> None:
        _require_text("decision_id", self.decision_id)
        _require_text("policy_version", self.policy_version)
        _require_text("state_digest", self.state_digest)
        if not self.available_actions:
            raise ValueError("available_actions must not be empty")
        if any(action not in EXPLORATION_ACTIONS for action in self.available_actions):
            raise ValueError("available_actions contains an unbounded action")
        if self.chosen_action not in self.available_actions:
            raise ValueError("chosen_action must be one of available_actions")
        if not self.branches:
            raise ValueError("branches must not be empty")
        if len({branch.action for branch in self.branches}) != len(self.branches):
            raise ValueError("branches must contain at most one entry per action")
        if any(branch.action not in self.available_actions for branch in self.branches):
            raise ValueError("a recorded branch must be available at the decision point")
        if self.chosen_action not in {branch.action for branch in self.branches}:
            raise ValueError("chosen_action must have a recorded branch")
        _require_non_negative("depth", self.depth)

    def state(self) -> ExplorationState:
        return ExplorationState(
            decision_id=self.decision_id,
            parent_id=self.parent_id,
            state_digest=self.state_digest,
            available_actions=self.available_actions,
            depth=self.depth,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "parent_id": self.parent_id,
            "policy_version": self.policy_version,
            "state_digest": self.state_digest,
            "available_actions": list(self.available_actions),
            "chosen_action": self.chosen_action,
            "branches": [branch.to_dict() for branch in self.branches],
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplorationEvent":
        return cls(
            decision_id=value["decision_id"],
            parent_id=value.get("parent_id"),
            policy_version=value["policy_version"],
            state_digest=value["state_digest"],
            available_actions=tuple(value["available_actions"]),
            chosen_action=value["chosen_action"],
            branches=tuple(ExplorationBranch.from_dict(branch) for branch in value["branches"]),
            depth=value.get("depth", 0),
        )


@dataclass(frozen=True)
class ExplorationTree:
    """Versioned, serializable history suitable for offline policy replay."""

    tree_id: str
    policy_version: str
    events: Tuple[ExplorationEvent, ...]
    completed: bool
    schema_version: int = EXPLORATION_TREE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("tree_id", self.tree_id)
        _require_text("policy_version", self.policy_version)
        if self.schema_version != EXPLORATION_TREE_SCHEMA_VERSION:
            raise ValueError(f"unsupported exploration tree schema: {self.schema_version}")
        decision_ids = [event.decision_id for event in self.events]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("exploration event IDs must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tree_id": self.tree_id,
            "policy_version": self.policy_version,
            "completed": self.completed,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplorationTree":
        return cls(
            tree_id=value["tree_id"],
            policy_version=value["policy_version"],
            events=tuple(ExplorationEvent.from_dict(event) for event in value.get("events", ())),
            completed=bool(value.get("completed", False)),
            schema_version=value.get("schema_version", 0),
        )


@dataclass(frozen=True)
class ReplayReport:
    """Offline evaluation metrics; unsupported decisions never receive invented scores."""

    policy_version: str
    tree_id: str
    total_decisions: int
    supported_decisions: int
    scored_decisions: int
    out_of_support_decisions: int
    mean_score: Optional[float]

    @property
    def coverage(self) -> float:
        return self.supported_decisions / self.total_decisions if self.total_decisions else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "tree_id": self.tree_id,
            "total_decisions": self.total_decisions,
            "supported_decisions": self.supported_decisions,
            "scored_decisions": self.scored_decisions,
            "out_of_support_decisions": self.out_of_support_decisions,
            "coverage": self.coverage,
            "mean_score": self.mean_score,
        }


@dataclass(frozen=True)
class PolicyArtifact:
    """Inspectable policy metadata for shadow/canary promotion workflows."""

    policy_version: str
    parent_version: Optional[str]
    policy_name: str
    status: str = "draft"
    provenance: Optional[Mapping[str, Any]] = None
    replay_metrics: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        _require_text("policy_version", self.policy_version)
        _require_text("policy_name", self.policy_name)
        if self.status not in _POLICY_STATUSES:
            raise ValueError(f"unsupported policy status: {self.status}")
        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        if self.replay_metrics is not None and not isinstance(self.replay_metrics, Mapping):
            raise ValueError("replay_metrics must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EXPLORATION_TREE_SCHEMA_VERSION,
            "policy_version": self.policy_version,
            "parent_version": self.parent_version,
            "policy_name": self.policy_name,
            "status": self.status,
            "provenance": dict(self.provenance or {}),
            "replay_metrics": dict(self.replay_metrics or {}),
            "allowed_actions": sorted(EXPLORATION_ACTIONS),
        }


class RecordedExplorationPolicy:
    """Baseline policy that replays the action recorded in each event."""

    def __init__(self, tree: ExplorationTree):
        self.version = tree.policy_version
        self._actions = {event.decision_id: event.chosen_action for event in tree.events}

    def decide(self, state: ExplorationState) -> ExplorationDecision:
        try:
            return ExplorationDecision(action=self._actions[state.decision_id])
        except KeyError as exc:
            raise KeyError(f"no recorded action for decision {state.decision_id}") from exc


def replay_policy(tree: ExplorationTree, policy: ExplorationPolicy) -> ReplayReport:
    """Evaluate a policy over recorded outcomes without invoking a task model.

    A decision is supported only when the policy selects an action with an observed branch.  An
    available-but-unobserved action and an action outside the bounded action set are both counted
    as out of support, with no fabricated reward.
    """
    supported = 0
    scored = 0
    out_of_support = 0
    scores = []
    for event in tree.events:
        decision = policy.decide(event.state())
        if decision.action not in event.available_actions:
            out_of_support += 1
            continue
        branch = next((candidate for candidate in event.branches if candidate.action == decision.action), None)
        if branch is None:
            out_of_support += 1
            continue
        supported += 1
        if branch.score is not None:
            scores.append(float(branch.score))
            scored += 1
    return ReplayReport(
        policy_version=policy.version,
        tree_id=tree.tree_id,
        total_decisions=len(tree.events),
        supported_decisions=supported,
        scored_decisions=scored,
        out_of_support_decisions=out_of_support,
        mean_score=sum(scores) / len(scores) if scores else None,
    )


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tool_call_count(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    return len(_TOOL_CALL_RE.findall(value))


def build_exploration_tree(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    completed: bool,
    policy_version: str = DEFAULT_POLICY_VERSION,
) -> ExplorationTree:
    """Derive a conservative first-generation tree from an existing saved trajectory.

    The derived tree records only continue/stop decisions.  It hashes prior trajectory state,
    never copies prompts, tool arguments, or tool output, and marks unobserved alternatives as
    out of support during replay.  Richer branch instrumentation can add sibling branches later
    without changing the replay contract.
    """
    decisions = [
        (index, message)
        for index, message in enumerate(trajectory)
        if message.get("from") == "gpt"
    ]
    events = []
    for ordinal, (index, message) in enumerate(decisions):
        value = message.get("value", "")
        has_tool_calls = _tool_call_count(value) > 0
        chosen_action = "continue" if has_tool_calls else "stop"
        terminal = ordinal == len(decisions) - 1
        child_ids = (f"decision-{ordinal + 1}",) if not terminal else ()
        score = (1.0 if completed else 0.0) if terminal else None
        state_digest = _stable_digest(list(trajectory[:index]))
        events.append(
            ExplorationEvent(
                decision_id=f"decision-{ordinal}",
                parent_id=f"decision-{ordinal - 1}" if ordinal else None,
                policy_version=policy_version,
                state_digest=state_digest,
                available_actions=("continue", "stop"),
                chosen_action=chosen_action,
                branches=(
                    ExplorationBranch(
                        action=chosen_action,
                        child_ids=child_ids,
                        score=score,
                        terminal=terminal,
                        cost=ExplorationCost(tool_calls=_tool_call_count(value)),
                    ),
                ),
                depth=ordinal,
            )
        )
    return ExplorationTree(
        tree_id=_stable_digest({"policy_version": policy_version, "trajectory": list(trajectory), "completed": completed}),
        policy_version=policy_version,
        events=tuple(events),
        completed=completed,
    )



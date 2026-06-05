"""Inert policy primitives for future Mission Control autonomous lane running.

This module is data-only decision support. Callers must supply observed repo
state and requested tool metadata; the helpers here do not inspect the machine,
load transcripts, persist state, wire runtime behavior, or execute tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LaneState(str, Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    COMPLETE = "complete"


class ApprovalTier(str, Enum):
    READ_ONLY = "read_only"
    CODE_TEST = "code_test"
    ELEVATED = "elevated"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class EvidenceCardModel:
    kind: str
    title: str
    summary: str
    limitations: tuple[str, ...] = ()
    trusted_for_execution: bool = False
    inert_context_only: bool = True
    authorizing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(self, "limitations", _tuple_of_text(self.limitations))
        object.__setattr__(self, "trusted_for_execution", False)
        object.__setattr__(self, "inert_context_only", True)
        object.__setattr__(self, "authorizing", False)


@dataclass(frozen=True)
class TaskControlEnvelopeModel:
    active_lane: str
    mode: str
    lane_state: LaneState
    approval_tier: ApprovalTier
    repo_path: str
    branch: str
    allowed_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    allowed_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()
    allowed_start_gate_dirty_files: tuple[str, ...] = ()
    focused_test_files: tuple[str, ...] = ()
    stop_condition: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_lane", str(self.active_lane))
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "lane_state", LaneState(self.lane_state))
        object.__setattr__(self, "approval_tier", ApprovalTier(self.approval_tier))
        object.__setattr__(self, "repo_path", str(self.repo_path))
        object.__setattr__(self, "branch", str(self.branch))
        object.__setattr__(self, "allowed_actions", _tuple_of_text(self.allowed_actions))
        object.__setattr__(self, "forbidden_actions", _tuple_of_text(self.forbidden_actions))
        object.__setattr__(self, "allowed_files", _tuple_of_text(self.allowed_files))
        object.__setattr__(self, "forbidden_files", _tuple_of_text(self.forbidden_files))
        object.__setattr__(
            self,
            "allowed_start_gate_dirty_files",
            _tuple_of_text(self.allowed_start_gate_dirty_files),
        )
        object.__setattr__(self, "focused_test_files", _tuple_of_text(self.focused_test_files))
        object.__setattr__(self, "stop_condition", str(self.stop_condition))


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    action: str
    target_files: tuple[str, ...] = ()
    writes: bool = False
    executes: bool = False
    uses_network: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", str(self.tool_name))
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "target_files", _tuple_of_text(self.target_files))
        object.__setattr__(self, "writes", bool(self.writes))
        object.__setattr__(self, "executes", bool(self.executes))
        object.__setattr__(self, "uses_network", bool(self.uses_network))


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    approval_tier: ApprovalTier
    reason: str
    violations: tuple[str, ...] = ()
    evidence_cards: tuple[EvidenceCardModel, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "approval_tier", ApprovalTier(self.approval_tier))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "violations", _tuple_of_text(self.violations))
        object.__setattr__(self, "evidence_cards", tuple(self.evidence_cards))


def task_control_envelope_model_from_record(record: dict[str, Any]) -> TaskControlEnvelopeModel:
    """Convert a stored Mission Control envelope record into an inert policy model.

    The record is treated as opaque stored data. This helper does not resolve
    paths, inspect git state, load transcripts, execute tools, or authorize
    future action.
    """
    lane_lock = _dict(record.get("lane_lock"))
    repo_context = _dict(record.get("repo_context"))
    metadata = _dict(record.get("metadata"))
    return TaskControlEnvelopeModel(
        active_lane=_text(lane_lock.get("active_lane"), _text(record.get("title"), "No active lane")),
        mode=_text(record.get("mode"), "unknown"),
        lane_state=_lane_state(record.get("status")),
        approval_tier=_approval_tier(metadata.get("approval_tier")),
        repo_path=_text(repo_context.get("path")),
        branch=_text(repo_context.get("branch")),
        allowed_actions=_string_tuple(record.get("allowed_actions")),
        forbidden_actions=_string_tuple(record.get("forbidden_actions")),
        allowed_files=_string_tuple(metadata.get("allowed_files")),
        forbidden_files=_string_tuple(metadata.get("forbidden_files")),
        allowed_start_gate_dirty_files=_string_tuple(
            metadata.get("allowed_start_gate_dirty_files")
        ),
        focused_test_files=_string_tuple(metadata.get("focused_test_files")),
        stop_condition=_text(metadata.get("stop_condition")),
    )


def task_control_envelope_model_summary(model: TaskControlEnvelopeModel) -> dict[str, Any]:
    return {
        "active_lane": model.active_lane,
        "mode": model.mode,
        "lane_state": model.lane_state.value,
        "approval_tier": model.approval_tier.value,
        "repo_path": model.repo_path,
        "branch": model.branch,
        "allowed_actions": list(model.allowed_actions),
        "forbidden_actions": list(model.forbidden_actions),
        "allowed_files": list(model.allowed_files),
        "forbidden_files": list(model.forbidden_files),
        "allowed_start_gate_dirty_files": list(model.allowed_start_gate_dirty_files),
        "focused_test_files": list(model.focused_test_files),
        "stop_condition": model.stop_condition,
        "trusted_for_execution": False,
        "inert_context_only": True,
    }


def summarize_guard_decision(decision: GuardDecision | dict[str, Any]) -> dict[str, Any]:
    if isinstance(decision, dict):
        allowed = bool(decision.get("allowed"))
        approval_tier = _approval_tier(decision.get("approval_tier"))
        reason = _text(decision.get("reason"), "unknown")
        violations = _string_tuple(decision.get("violations"))
    else:
        allowed = decision.allowed
        approval_tier = decision.approval_tier
        reason = decision.reason
        violations = decision.violations
    return {
        "allowed": allowed,
        "approval_tier": approval_tier.value,
        "reason": reason,
        "violations": list(violations),
        "execution_enabled": False,
        "trusted_for_execution": False,
        "inert_context_only": True,
    }


def next_action_decision_summary(
    envelope: TaskControlEnvelopeModel | None,
    start_gate_decision: GuardDecision | dict[str, Any] | None,
) -> dict[str, Any]:
    decision = (
        summarize_guard_decision(start_gate_decision)
        if start_gate_decision is not None
        else summarize_guard_decision(
            GuardDecision(
                allowed=False,
                approval_tier=ApprovalTier.FORBIDDEN,
                reason="no_start_gate_decision",
            )
        )
    )
    if envelope is None:
        kind = "forbidden"
        label = "Forbidden"
        reason = "no_task_control_envelope"
    elif not decision["allowed"] or envelope.approval_tier is ApprovalTier.FORBIDDEN:
        kind = "forbidden"
        label = "Forbidden"
        reason = decision["reason"]
    elif envelope.approval_tier is ApprovalTier.READ_ONLY:
        kind = "auto"
        label = "Auto read-only"
        reason = "read_only_lane_after_start_gate"
    else:
        kind = "one_click_approval"
        label = "One-click approval required"
        reason = "approval_required_for_non_read_only_lane"
    return {
        "kind": kind,
        "label": label,
        "reason": reason,
        "execution_enabled": False,
        "trusted_for_execution": False,
        "inert_context_only": True,
    }


def validate_start_gate(
    envelope: TaskControlEnvelopeModel,
    *,
    repo_path: str,
    branch: str,
    dirty_files: tuple[str, ...] = (),
) -> GuardDecision:
    violations: list[str] = []
    observed_dirty = _tuple_of_text(dirty_files)
    allowed_dirty = set(envelope.allowed_start_gate_dirty_files)

    if envelope.lane_state is not LaneState.ACTIVE:
        violations.append(f"lane_not_active:{envelope.lane_state.value}")
    if str(repo_path) != envelope.repo_path:
        violations.append("repo_path_mismatch")
    if str(branch) != envelope.branch:
        violations.append("branch_mismatch")
    for dirty_file in observed_dirty:
        if dirty_file not in allowed_dirty:
            violations.append(f"unexpected_dirty_file:{dirty_file}")

    allowed = not violations
    reason = "start_gate_passed" if allowed else "start_gate_blocked"
    return GuardDecision(
        allowed=allowed,
        approval_tier=envelope.approval_tier if allowed else ApprovalTier.FORBIDDEN,
        reason=reason,
        violations=tuple(violations),
        evidence_cards=(
            EvidenceCardModel(
                kind="start_gate",
                title="Autonomous lane runner start gate",
                summary=reason,
                limitations=("caller_supplied_state_only",),
            ),
        ),
    )


def decide_tool_request(
    envelope: TaskControlEnvelopeModel,
    request: ToolRequest,
) -> GuardDecision:
    violations: list[str] = []

    if envelope.lane_state is not LaneState.ACTIVE:
        violations.append(f"lane_not_active:{envelope.lane_state.value}")

    forbidden_actions = set(envelope.forbidden_actions)
    allowed_actions = set(envelope.allowed_actions)
    if request.action in forbidden_actions:
        violations.append(f"forbidden_action:{request.action}")
    elif request.action not in allowed_actions:
        violations.append(f"action_not_allowed:{request.action}")

    if request.uses_network:
        violations.append("network_not_allowed")

    if request.writes:
        allowed_files = set(envelope.allowed_files)
        forbidden_files = set(envelope.forbidden_files)
        for target_file in request.target_files:
            if target_file in forbidden_files:
                violations.append(f"forbidden_file:{target_file}")
            elif target_file not in allowed_files:
                violations.append(f"write_outside_allowed_files:{target_file}")

    if request.action == "run_focused_tests":
        focused_files = set(envelope.focused_test_files)
        for target_file in request.target_files:
            if target_file not in focused_files:
                violations.append(f"test_outside_focused_scope:{target_file}")

    allowed = not violations
    reason = "tool_request_allowed" if allowed else "tool_request_blocked"
    return GuardDecision(
        allowed=allowed,
        approval_tier=envelope.approval_tier if allowed else ApprovalTier.FORBIDDEN,
        reason=reason,
        violations=tuple(violations),
        evidence_cards=(
            EvidenceCardModel(
                kind="tool_policy",
                title=f"Tool request: {request.tool_name}",
                summary=reason,
                limitations=("decision_only_no_execution",),
            ),
        ),
    )


def _tuple_of_text(values: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))


def _lane_state(value: Any) -> LaneState:
    text = _text(value, "stopped")
    if text == "completed":
        return LaneState.COMPLETE
    if text == "archived":
        return LaneState.STOPPED
    try:
        return LaneState(text)
    except ValueError:
        return LaneState.STOPPED


def _approval_tier(value: Any) -> ApprovalTier:
    text = _text(value, "forbidden").lower().replace("-", "_").replace("/", "_")
    text = "_".join(part for part in text.split() if part)
    aliases = {
        "read_only": ApprovalTier.READ_ONLY,
        "read_only_only": ApprovalTier.READ_ONLY,
        "readonly": ApprovalTier.READ_ONLY,
        "code_test": ApprovalTier.CODE_TEST,
        "code_test_only": ApprovalTier.CODE_TEST,
        "code_testing": ApprovalTier.CODE_TEST,
        "code": ApprovalTier.CODE_TEST,
        "elevated": ApprovalTier.ELEVATED,
        "forbidden": ApprovalTier.FORBIDDEN,
    }
    return aliases.get(text, ApprovalTier.FORBIDDEN)

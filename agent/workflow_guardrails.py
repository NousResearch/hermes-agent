"""Forge-style workflow guardrail primitives for Hermes.

This module is intentionally side-effect free. Runtime code tells the
controller which user turn started, which tools succeeded, and what final
response the model wants to return; the controller returns advisory/gate
decisions. Default config is compatibility-first: missing workflow steps are
reported as a footer, not blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class WorkflowStep:
    """A required workflow step satisfied by one of several tool names."""

    key: str
    label: str
    required_any: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class WorkflowSpec:
    """Workflow-level required step specification."""

    key: str
    label: str
    triggers_any: tuple[str, ...]
    steps: tuple[WorkflowStep, ...]


@dataclass(frozen=True)
class WorkflowGuardrailConfig:
    """Config for workflow-level final gates.

    final_gate_mode:
      - off:      disable workflow final evaluation entirely
      - advisory: append/report missing-step advisory but allow final response
      - nudge:    ask the model to continue once before allowing final response
      - block:    return a controlled missing-step response instead of final
    """

    enabled: bool = False
    final_gate_mode: str = "off"
    max_nudges: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "WorkflowGuardrailConfig":
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        mode = str(data.get("final_gate_mode", defaults.final_gate_mode) or "").strip().lower()
        if mode not in {"off", "advisory", "nudge", "block"}:
            mode = defaults.final_gate_mode
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            final_gate_mode=mode,
            max_nudges=_positive_int(data.get("max_nudges"), defaults.max_nudges),
        )


@dataclass(frozen=True)
class WorkflowGuardrailDecision:
    """Decision returned when a final response is evaluated."""

    action: str = "allow"  # allow | advisory | nudge | block
    workflow_key: str = ""
    workflow_label: str = ""
    missing_steps: tuple[WorkflowStep, ...] = ()
    message: str = ""

    @property
    def allows_final_response(self) -> bool:
        return self.action in {"allow", "advisory"}

    def to_metadata(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "workflow_key": self.workflow_key,
            "workflow_label": self.workflow_label,
            "missing_steps": [step.key for step in self.missing_steps],
            "message": self.message,
        }


DEFAULT_WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        key="repo_review",
        label="Repository review",
        triggers_any=("repo", "repository", "github.com", "레포", "레포지토리", "검토", "review"),
        steps=(
            WorkflowStep("inspect_metadata", "Inspect repo metadata", ("terminal",), "branch/status/remote or package metadata"),
            WorkflowStep("inspect_docs", "Inspect docs", ("read_file", "search_files"), "README/docs/config discovery"),
            WorkflowStep("inspect_core", "Inspect core code", ("read_file", "search_files"), "source implementation review"),
            WorkflowStep("verify", "Run tests or smoke checks", ("terminal", "execute_code"), "tests, import smoke, or explicit unavailable reason"),
        ),
    ),
    WorkflowSpec(
        key="devflow",
        label="Code implementation workflow",
        triggers_any=("implement", "fix", "build", "refactor", "구현", "수정", "고쳐", "개발"),
        steps=(
            WorkflowStep("inspect", "Inspect relevant files", ("read_file", "search_files"), "read/search before editing"),
            WorkflowStep("mutate", "Apply scoped changes", ("patch", "write_file"), "code/config/doc changes"),
            WorkflowStep("verify", "Run targeted checks", ("terminal", "execute_code"), "tests, py_compile, lint, smoke, or explicit blocker"),
        ),
    ),
    WorkflowSpec(
        key="weekly_report",
        label="Weekly report workflow",
        triggers_any=("weekly report", "주간보고", "주간 보고"),
        steps=(
            WorkflowStep("collect", "Collect source material", ("session_search", "read_file", "search_files"), "verified source collection"),
            WorkflowStep("preview", "Preview draft before sending", ("send_message",), "Telegram/user preview before delivery"),
        ),
    ),
    WorkflowSpec(
        key="ppt_workflow",
        label="PPT/deck workflow",
        triggers_any=("ppt", "powerpoint", "deck", "슬라이드", "발표자료"),
        steps=(
            WorkflowStep("read_context", "Read source/context", ("read_file", "search_files"), "source/context inspection"),
            WorkflowStep("create_artifact", "Create or edit deck artifact", ("write_file", "patch", "terminal", "execute_code"), "deck/artifact generation"),
            WorkflowStep("visual_or_file_qa", "Run QA", ("terminal", "execute_code", "vision_analyze", "browser_vision"), "layout/file validation"),
        ),
    ),
)


class WorkflowGuardrailController:
    """Per-turn required-step tracker inspired by Forge workflows."""

    def __init__(self, config: WorkflowGuardrailConfig | None = None, workflows: Iterable[WorkflowSpec] | None = None):
        self.config = config or WorkflowGuardrailConfig()
        self.workflows = tuple(workflows or DEFAULT_WORKFLOWS)
        self.reset_for_turn()

    def reset_for_turn(self, user_message: str | None = None) -> None:
        self._user_message = user_message or ""
        self._tool_successes: set[str] = set()
        self._nudges_sent: dict[str, int] = {}
        self._active_workflows = self._classify_workflows(self._user_message)

    @property
    def active_workflows(self) -> tuple[WorkflowSpec, ...]:
        return self._active_workflows

    def record_tool_result(self, tool_name: str, *, failed: bool = False) -> None:
        if not failed and tool_name:
            self._tool_successes.add(tool_name)

    def evaluate_final_response(self, final_response: str | None) -> WorkflowGuardrailDecision:
        if not self.config.enabled or self.config.final_gate_mode == "off":
            return WorkflowGuardrailDecision()
        if not final_response or not self._active_workflows:
            return WorkflowGuardrailDecision()

        workflow = self._active_workflows[0]
        missing = tuple(
            step for step in workflow.steps
            if not any(tool in self._tool_successes for tool in step.required_any)
        )
        if not missing:
            return WorkflowGuardrailDecision(workflow_key=workflow.key, workflow_label=workflow.label)

        message = _format_missing_steps_message(workflow, missing)
        if self.config.final_gate_mode == "advisory":
            return WorkflowGuardrailDecision(
                action="advisory",
                workflow_key=workflow.key,
                workflow_label=workflow.label,
                missing_steps=missing,
                message=message,
            )
        if self.config.final_gate_mode == "nudge":
            sent = self._nudges_sent.get(workflow.key, 0)
            if sent < self.config.max_nudges:
                self._nudges_sent[workflow.key] = sent + 1
                return WorkflowGuardrailDecision(
                    action="nudge",
                    workflow_key=workflow.key,
                    workflow_label=workflow.label,
                    missing_steps=missing,
                    message=message,
                )
            return WorkflowGuardrailDecision(
                action="advisory",
                workflow_key=workflow.key,
                workflow_label=workflow.label,
                missing_steps=missing,
                message=message,
            )
        if self.config.final_gate_mode == "block":
            return WorkflowGuardrailDecision(
                action="block",
                workflow_key=workflow.key,
                workflow_label=workflow.label,
                missing_steps=missing,
                message=message,
            )
        return WorkflowGuardrailDecision()

    def _classify_workflows(self, user_message: str) -> tuple[WorkflowSpec, ...]:
        lowered = (user_message or "").lower()
        if not lowered:
            return ()
        matches = []
        for workflow in self.workflows:
            hits = sum(1 for token in workflow.triggers_any if token.lower() in lowered)
            # Repo review is intentionally stricter: avoid treating every Korean
            # "검토" as a repository review unless repo/GitHub context is present.
            if workflow.key == "repo_review":
                has_repo = any(token in lowered for token in ("repo", "repository", "github.com", "레포", "레포지토리"))
                has_review = any(token in lowered for token in ("review", "검토"))
                if has_repo and has_review:
                    matches.append((10 + hits, workflow))
                continue
            if hits:
                matches.append((hits, workflow))
        matches.sort(key=lambda item: item[0], reverse=True)
        return tuple(workflow for _score, workflow in matches[:1])


def append_workflow_advisory(response: str, decision: WorkflowGuardrailDecision) -> str:
    """Append a concise workflow advisory footer to a string response."""
    if decision.action != "advisory" or not decision.message:
        return response
    return response.rstrip() + "\n\n[Workflow guardrail advisory: " + decision.message + "]"


def workflow_block_response(decision: WorkflowGuardrailDecision) -> str:
    return (
        "I stopped before finalizing because the active workflow is missing "
        f"required step evidence. {decision.message}"
    )


def workflow_nudge_message(decision: WorkflowGuardrailDecision) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "[System workflow guardrail] You are about to give a final response, "
            "but required workflow evidence is missing. Complete the missing steps "
            "with tools if possible, or explicitly state a blocker if a step cannot "
            f"be performed. Missing: {decision.message}"
        ),
    }


def _format_missing_steps_message(workflow: WorkflowSpec, missing: tuple[WorkflowStep, ...]) -> str:
    parts = []
    for step in missing:
        tools = "/".join(step.required_any) if step.required_any else "manual evidence"
        parts.append(f"{step.label} ({tools})")
    return f"{workflow.label} missing required step evidence: " + "; ".join(parts)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default

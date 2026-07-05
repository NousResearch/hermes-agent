"""Orchestrator Interface (Phase 2 minimal stub).

Wraps the (existing) kanban swarm v1 for use by Conversation API.
Phase 2: returns OrchestratorDecision only. NO actual kanban creation.
Phase 3 will populate task_ids via existing kanban CLI subprocess.

Cardinal rules:
- NO real kanban DB mutations in Phase 2
- NO workers (Phase 3)
- NO delegate_task (Phase 3)
- NO gbrain CLI invocation (Phase 4)
- NO LLM calls (Phase 2)
- NO hermes artifact modification
- NO R7 file modification
- NO config.yaml modification
- NO gateway restart
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.intent_router import IntentClassification


@dataclass
class TaskSpec:
    task_id: str  # empty in Phase 2; populated by execute() in Phase 3
    description: str
    assigned_profile: str
    inputs: dict = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    timeout_s: int = 60
    requires_user_input: bool = False
    approval_id: str | None = None


@dataclass
class TaskAssignment:
    task_id: str
    profile: str
    workspace: str
    assigned_at_utc: str
    status: str = "assigned"  # assigned|running|completed|failed|blocked


@dataclass
class OrchestratorDecision:
    plan: list[TaskSpec] = field(default_factory=list)
    task_assignments: list[TaskAssignment] = field(default_factory=list)
    requires_approval: bool = False
    approval_prompt: str | None = None
    rationale: str = ""
    estimated_complexity: str = "trivial"
    orchestrator_model: str = "phase2_stub_no_llm"
    decided_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "assigned_profile": t.assigned_profile,
                    "inputs": dict(t.inputs),
                    "expected_outputs": list(t.expected_outputs),
                    "dependencies": list(t.dependencies),
                    "timeout_s": t.timeout_s,
                    "requires_user_input": t.requires_user_input,
                    "approval_id": t.approval_id,
                }
                for t in self.plan
            ],
            "task_assignments": [
                {
                    "task_id": a.task_id,
                    "profile": a.profile,
                    "workspace": a.workspace,
                    "assigned_at_utc": a.assigned_at_utc,
                    "status": a.status,
                }
                for a in self.task_assignments
            ],
            "requires_approval": self.requires_approval,
            "approval_prompt": self.approval_prompt,
            "rationale": self.rationale,
            "estimated_complexity": self.estimated_complexity,
            "orchestrator_model": self.orchestrator_model,
            "decided_at_utc": self.decided_at_utc,
        }


def _now_iso_utc() -> str:
    """Return current UTC as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_phase2_plan(intent: IntentClassification) -> tuple[list[TaskSpec], str]:
    """Build a Phase 2 plan from an IntentClassification.

    Phase 2 stub: returns an empty plan and trivial complexity.
    Phase 3 will populate this with real TaskSpecs via kanban CLI.

    Returns: (plan, estimated_complexity)
    """
    if intent.intent_type in {"delegate", "research", "code", "kanban"}:
        # Phase 2: empty plan (no actual kanban creation)
        # Phase 3: would build a TaskSpec here and create via kanban CLI
        return [], "simple"

    if intent.intent_type == "composite":
        return [], "moderate"

    return [], "trivial"


class OrchestratorInterface:
    """Phase 2 stub interface to the kanban swarm.

    Returns OrchestratorDecision with:
    - plan: empty list in Phase 2 (no kanban creation)
    - task_assignments: empty list in Phase 2 (Phase 3)
    - requires_approval: True if routing strategy/mode/approval intent/critical safety flag requires approval
    - rationale: text describing the decision
    - estimated_complexity: trivial|simple|moderate|complex

    Does NOT:
    - Spawn workers (Phase 3)
    - Call delegate_task (Phase 3)
    - Touch Kanban DB (Phase 3)
    - Invoke GBrain (Phase 4)
    - Call LLM (Phase 2)
    - Modify hermes artifacts
    """

    def __init__(self, conversation_persistence=None, kanban_cli_path: str = "hermes"):
        """Initialize orchestrator interface.

        Args:
            conversation_persistence: accepted for backward compatibility, unused in Phase 2
            kanban_cli_path: path to existing kanban CLI (default "hermes", unused in Phase 2)
        """
        self._kanban_cli_path = kanban_cli_path

    def orchestrate(
        self,
        intent: IntentClassification,
        conversation_context: dict | None = None,
        mode: str = "auto",
    ) -> OrchestratorDecision:
        """Decide how to handle the intent.

        Returns OrchestratorDecision with:
        - plan: list[TaskSpec] (empty in Phase 2; Phase 3 will populate)
        - task_assignments: list[TaskAssignment] (empty in Phase 2)
        - requires_approval: bool
        - approval_prompt: str | None
        - rationale: str
        - estimated_complexity: trivial|simple|moderate|complex
        - orchestrator_model: phase2_stub_no_llm
        - decided_at_utc: str

        Phase 2 may return plan with task_ids empty (no actual kanban creation).
        Phase 3 will populate task_ids via existing kanban CLI subprocess.
        """
        # Determine if approval is required
        requires_approval = (
            intent.routing_strategy == "approval_required"
            or mode == "approval_required"
            or intent.intent_type == "approval"
            or any(flag.severity == "critical" for flag in intent.safety_flags)
        )

        # Build a Phase 2 plan (empty list; Phase 3 will populate)
        plan, complexity = _build_phase2_plan(intent)

        approval_prompt = None
        if requires_approval:
            approval_prompt = (
                f"Intent '{intent.intent_type}' requires approval "
                f"(safety_flags={[f.flag_type for f in intent.safety_flags]})."
            )

        rationale = (
            f"Phase 2 stub: classified as '{intent.intent_type}' "
            f"(confidence={intent.confidence:.2f}, "
            f"strategy={intent.routing_strategy}, mode={mode}). "
            f"Phase 3 will populate plan via kanban CLI."
        )

        decision = OrchestratorDecision(
            plan=plan,
            task_assignments=[],
            requires_approval=requires_approval,
            approval_prompt=approval_prompt,
            rationale=rationale,
            estimated_complexity=complexity,
            orchestrator_model="phase2_stub_no_llm",
            decided_at_utc=_now_iso_utc(),
        )

        return decision

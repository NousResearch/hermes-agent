"""Conversational Operating Layer contracts and orchestration helpers.

The COL turns a natural-language user request into a Hermes OS workflow plan.
It is deliberately deterministic here: Hermes OS owns state and governance,
while model-assisted interpretation can be added behind these contracts later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COL_VERSION = "1.0"

INTENT_NEW_PROJECT = "new_project"
INTENT_EXISTING_PROJECT = "existing_project"
INTENT_RESEARCH = "research"
INTENT_ARCHITECTURE = "architecture"
INTENT_TASK_WORK = "task_work"
INTENT_REVIEW = "review"
INTENT_UNKNOWN = "unknown"

WORKFLOW_STATUS_PLANNED = "planned"
WORKFLOW_STATUS_WAITING = "waiting_for_approval"


@dataclass(frozen=True)
class ChatEnvelope:
    message: str
    user_id: str = "operator"
    project_id: str = "workspace"
    session_id: str = "local"
    active_goal: str = ""
    active_initiative: str = ""
    slash_command: str = ""
    dry_run: bool = True


@dataclass(frozen=True)
class IntentRoute:
    intent: str
    confidence: float
    workflow: str
    alternatives: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    clarification: str = ""


@dataclass(frozen=True)
class MemoryContext:
    project_id: str
    project_path: str
    loaded_sources: List[str]
    active_goal: str = ""
    active_initiative: str = ""
    recent_decisions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    name: str
    owner: str
    action: str
    expected_artifacts: List[str] = field(default_factory=list)
    requires_approval: bool = False
    status: str = WORKFLOW_STATUS_PLANNED


@dataclass(frozen=True)
class DelegationRecord:
    delegation_id: str
    agent_role: str
    layer: str
    input_artifacts: List[str]
    output_artifacts: List[str]
    status: str = "planned"
    risk: str = "normal"


@dataclass(frozen=True)
class Recommendation:
    title: str
    reason: str
    priority: str = "normal"


@dataclass(frozen=True)
class ChiefOfStaffPlan:
    request: ChatEnvelope
    route: IntentRoute
    memory: MemoryContext
    status: str
    steps: List[WorkflowStep]
    delegations: List[DelegationRecord]
    recommendations: List[Recommendation]
    audit_events: List[Dict[str, Any]]
    commands: List[str]
    guardrails: List[str]


MANAGEMENT_ROLES = {
    "planner": "Plans multi-step work and task decomposition.",
    "architect": "Reviews architecture, domain boundaries, and work graphs.",
    "research_lead": "Coordinates research and evidence collection.",
    "engineering_lead": "Coordinates implementation and validation work.",
    "product_lead": "Keeps workflows tied to user outcomes.",
}

WORKER_ROLES = {
    "engineer": "Produces implementation artifacts.",
    "reviewer": "Reviews outputs before promotion.",
    "researcher": "Collects source-backed evidence.",
    "analyst": "Synthesizes evidence and tradeoffs.",
    "qa": "Validates behavior and regression coverage.",
    "writer": "Produces user-facing documentation.",
}

SKILL_COMMANDS = {
    "/grill-me": "grill-me",
    "/prd": "to-prd",
    "/architecture": "architect review",
    "/plan": "plan",
    "/tasks": "to-issues",
    "/review": "review",
    "/research": "research",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_col_config(enabled: bool = True) -> Dict[str, Any]:
    return {
        "version": COL_VERSION,
        "enabled": enabled,
        "feature_flags": {
            "hermes_chat_col_preview": True,
            "chief_of_staff": True,
            "intent_router": True,
            "workflow_engine": True,
            "dynamic_commands": True,
        },
        "source_of_truth": "hermes_os",
    }


def normalize_slash_command(message: str) -> str:
    stripped = message.strip()
    if not stripped.startswith("/"):
        return ""
    first = stripped.split(maxsplit=1)[0].lower()
    return first if first in SKILL_COMMANDS else ""


def route_intent(message: str, slash_command: str = "") -> IntentRoute:
    text = message.strip().lower()
    if slash_command in {"/grill-me", "/prd", "/architecture", "/plan", "/tasks"}:
        return IntentRoute(INTENT_NEW_PROJECT, 0.92, "new_project_launch", [INTENT_ARCHITECTURE])
    if slash_command == "/research":
        return IntentRoute(INTENT_RESEARCH, 0.94, "research")
    if slash_command == "/review":
        return IntentRoute(INTENT_REVIEW, 0.9, "review")

    if any(word in text for word in ("build ", "create ", "start ", "launch ")):
        return IntentRoute(INTENT_NEW_PROJECT, 0.88, "new_project_launch", [INTENT_ARCHITECTURE])
    if any(word in text for word in ("continue", "update", "fix", "improve", "switch")):
        return IntentRoute(INTENT_EXISTING_PROJECT, 0.84, "existing_project_work", [INTENT_TASK_WORK])
    if any(word in text for word in ("research", "analyze", "compare", "investigate")):
        return IntentRoute(INTENT_RESEARCH, 0.86, "research", [INTENT_ARCHITECTURE])
    if any(word in text for word in ("architecture", "architect", "refactor", "design workflow")):
        return IntentRoute(INTENT_ARCHITECTURE, 0.86, "architecture_review", [INTENT_TASK_WORK])
    if any(word in text for word in ("implement task", "complete task", "task-")):
        return IntentRoute(INTENT_TASK_WORK, 0.86, "task_execution", [INTENT_REVIEW])
    if any(word in text for word in ("review task", "approve", "reject")):
        return IntentRoute(INTENT_REVIEW, 0.86, "review")
    return IntentRoute(
        INTENT_UNKNOWN,
        0.2,
        "clarify",
        requires_clarification=True,
        clarification="What outcome should Hermes OS plan, research, review, or implement?",
    )


def load_memory_context(
    project: str = ".",
    active_goal: str = "",
    active_initiative: str = "",
    recent_decisions: Optional[Iterable[str]] = None,
) -> MemoryContext:
    project_path = Path(project).resolve()
    sources = []
    warnings = []
    for name in (
        "project.md",
        "architecture.md",
        "decision-log.md",
        "tracker.md",
        "agents.md",
        "TASKS.md",
    ):
        candidate = project_path / name
        if candidate.exists():
            sources.append(str(candidate))
    if not sources:
        warnings.append("No core Hermes project memory files were found for this project.")
    return MemoryContext(
        project_id=project_path.name or "workspace",
        project_path=str(project_path),
        loaded_sources=sources,
        active_goal=active_goal,
        active_initiative=active_initiative,
        recent_decisions=list(recent_decisions or []),
        warnings=warnings,
    )


def workflow_steps_for(route: IntentRoute) -> List[WorkflowStep]:
    if route.workflow == "new_project_launch":
        return [
            WorkflowStep("grill-me", "Clarify product idea", "Chief of Staff", "Run grill-me", ["grill-me.md"]),
            WorkflowStep("prd", "Create PRD", "Product Lead", "Generate product requirements", ["prd.md"]),
            WorkflowStep("architecture", "Review architecture", "Architect", "Generate architecture review", ["architecture.md"], True),
            WorkflowStep("plan", "Create implementation plan", "Planner", "Compile plan", ["plan.md"]),
            WorkflowStep("tasks", "Generate tasks", "Planner", "Create backlog tasks", ["TASKS.md"]),
            WorkflowStep("agents", "Create agent roster", "Chief of Staff", "Write agents.md", ["agents.md"]),
            WorkflowStep("tracker", "Create tracker", "Chief of Staff", "Write tracker.md", ["tracker.md"]),
            WorkflowStep("dashboard", "Create dashboard contract", "Engineer", "Prepare dashboard", ["dashboard.md"]),
            WorkflowStep("workspace", "Open project workspace", "Chief of Staff", "Prepare workspace automation", ["workspace-snapshot.json"], True),
        ]
    if route.workflow == "existing_project_work":
        return [
            WorkflowStep("load-context", "Load project context", "Chief of Staff", "Load memory", ["context-pack.json"]),
            WorkflowStep("route", "Route requested change", "Chief of Staff", "Select workflow", ["route.json"]),
            WorkflowStep("plan-work", "Plan work", "Planner", "Create task plan", ["work-plan.md"]),
            WorkflowStep("delegate", "Delegate work", "Engineering Lead", "Assign worker agents", ["delegation.json"], True),
            WorkflowStep("review", "Review output", "Reviewer", "Validate artifacts", ["review.md"], True),
        ]
    if route.workflow == "research":
        return [
            WorkflowStep("research-plan", "Plan research", "Research Lead", "Define questions", ["research-plan.md"]),
            WorkflowStep("evidence", "Collect evidence", "Researcher", "Gather evidence", ["evidence.md"]),
            WorkflowStep("analysis", "Analyze evidence", "Analyst", "Synthesize findings", ["analysis.md"]),
            WorkflowStep("report", "Write research report", "Writer", "Draft report", ["research-report.md"], True),
        ]
    if route.workflow == "architecture_review":
        return [
            WorkflowStep("scan", "Scan architecture", "Architect", "Load project architecture", ["architecture-scan.json"]),
            WorkflowStep("review", "Review architecture", "Architect", "Find gaps", ["architecture-review.md"]),
            WorkflowStep("work-graph", "Compile work graph", "Planner", "Generate work graph", ["work-graph.json"]),
        ]
    if route.workflow == "task_execution":
        return [
            WorkflowStep("select-task", "Select task", "Chief of Staff", "Resolve task id", ["task-context.json"]),
            WorkflowStep("execute", "Execute task", "Engineer", "Delegate implementation", ["artifact-manifest.json"], True),
            WorkflowStep("qa", "Validate task", "QA", "Run checks", ["validation.md"]),
            WorkflowStep("review", "Review task", "Reviewer", "Approve handoff", ["review.md"], True),
        ]
    if route.workflow == "review":
        return [
            WorkflowStep("load-review", "Load review item", "Reviewer", "Load artifact", ["review-context.json"]),
            WorkflowStep("decide", "Record decision", "Reviewer", "Approve, reject, or request context", ["review-decision.json"], True),
        ]
    return [WorkflowStep("clarify", "Clarify request", "Chief of Staff", "Ask a follow-up question")]


def delegation_records_for(steps: Iterable[WorkflowStep]) -> List[DelegationRecord]:
    records = []
    for index, step in enumerate(steps, start=1):
        owner_key = step.owner.lower().replace(" ", "_")
        if step.owner == "Chief of Staff":
            layer = "coordination"
        elif owner_key in MANAGEMENT_ROLES:
            layer = "management"
        else:
            layer = "worker"
        records.append(
            DelegationRecord(
                delegation_id=f"delegation-{index:02d}",
                agent_role=owner_key,
                layer=layer,
                input_artifacts=[] if index == 1 else [f"step-{index - 1}-output"],
                output_artifacts=step.expected_artifacts,
                risk="approval_required" if step.requires_approval else "normal",
            )
        )
    return records


def dynamic_commands_for(route: IntentRoute) -> List[str]:
    if route.workflow == "new_project_launch":
        return ["/grill-me", "/prd", "/architecture", "/plan", "/tasks"]
    if route.workflow == "research":
        return ["/research"]
    if route.workflow in {"architecture_review", "task_execution", "review"}:
        return ["/architecture", "/tasks", "/review"]
    return sorted(SKILL_COMMANDS)


def recommendations_for(route: IntentRoute, memory: MemoryContext) -> List[Recommendation]:
    recommendations = []
    if route.requires_clarification:
        recommendations.append(Recommendation("Clarify request", route.clarification, "high"))
    if memory.warnings:
        recommendations.append(Recommendation("Create project memory", memory.warnings[0], "high"))
    if route.workflow == "new_project_launch":
        recommendations.append(Recommendation("Preview launch workflow", "Review generated steps before live project creation.", "high"))
    elif route.workflow == "existing_project_work":
        recommendations.append(Recommendation("Load active project", "Confirm the selected project before delegation.", "high"))
    else:
        recommendations.append(Recommendation("Run dry-run first", "Preview agents, artifacts, and approvals before writes.", "normal"))
    return recommendations


def guardrails() -> List[str]:
    return [
        "Hermes OS owns source-of-truth state.",
        "Chief of Staff coordinates but does not perform worker implementation directly.",
        "Agents and skills produce artifacts for review.",
        "Write-capable, costly, or destructive actions require approval.",
        "Runtime memory cannot overwrite project memory records.",
    ]


def chief_of_staff_plan(envelope: ChatEnvelope, project_path: str = ".") -> ChiefOfStaffPlan:
    slash = envelope.slash_command or normalize_slash_command(envelope.message)
    route = route_intent(envelope.message, slash)
    memory = load_memory_context(
        project_path,
        active_goal=envelope.active_goal,
        active_initiative=envelope.active_initiative,
    )
    steps = workflow_steps_for(route)
    delegations = delegation_records_for(steps)
    status = WORKFLOW_STATUS_WAITING if any(step.requires_approval for step in steps) else WORKFLOW_STATUS_PLANNED
    events = [
        {"type": "conversation.received", "at": _now(), "session_id": envelope.session_id},
        {"type": "intent.routed", "at": _now(), "intent": route.intent, "workflow": route.workflow},
        {"type": "workflow.planned", "at": _now(), "step_count": len(steps)},
        {"type": "delegation.planned", "at": _now(), "delegation_count": len(delegations)},
    ]
    return ChiefOfStaffPlan(
        request=ChatEnvelope(
            message=envelope.message,
            user_id=envelope.user_id,
            project_id=envelope.project_id,
            session_id=envelope.session_id,
            active_goal=envelope.active_goal,
            active_initiative=envelope.active_initiative,
            slash_command=slash,
            dry_run=envelope.dry_run,
        ),
        route=route,
        memory=memory,
        status=status,
        steps=steps,
        delegations=delegations,
        recommendations=recommendations_for(route, memory),
        audit_events=events,
        commands=dynamic_commands_for(route),
        guardrails=guardrails(),
    )


def ask_col(message: str, project_path: str = ".", **kwargs: Any) -> Dict[str, Any]:
    envelope = ChatEnvelope(message=message, **kwargs)
    return plan_to_dict(chief_of_staff_plan(envelope, project_path=project_path))


def plan_to_dict(plan: ChiefOfStaffPlan) -> Dict[str, Any]:
    return asdict(plan)


def col_dashboard_panels(project: str = ".") -> List[Dict[str, Any]]:
    sample = chief_of_staff_plan(
        ChatEnvelope(
            message="Build a CRM for wholesalers.",
            project_id=Path(project).resolve().name or "workspace",
            active_goal="Conversational Hermes OS",
            active_initiative="Chief of Staff + Chat UX",
        ),
        project_path=project,
    )
    return [
        {
            "panel_id": "col-active-context",
            "title": "Conversational Context",
            "data": {
                "enabled": True,
                "version": COL_VERSION,
                "active_project": sample.memory.project_id,
                "active_goal": sample.memory.active_goal,
                "active_initiative": sample.memory.active_initiative,
                "loaded_source_count": len(sample.memory.loaded_sources),
                "warnings": sample.memory.warnings,
            },
        },
        {
            "panel_id": "col-chief-of-staff",
            "title": "Chief of Staff",
            "data": {
                "status": sample.status,
                "intent": sample.route.intent,
                "workflow": sample.route.workflow,
                "step_count": len(sample.steps),
                "delegation_count": len(sample.delegations),
                "recommendations": [asdict(item) for item in sample.recommendations],
            },
        },
        {
            "panel_id": "col-workflow-preview",
            "title": "Conversational Workflow Preview",
            "data": {
                "steps": [asdict(step) for step in sample.steps],
                "commands": sample.commands,
                "guardrails": sample.guardrails,
            },
        },
        {
            "panel_id": "col-agent-hierarchy",
            "title": "Agent Hierarchy",
            "data": {
                "management_roles": MANAGEMENT_ROLES,
                "worker_roles": WORKER_ROLES,
                "delegations": [asdict(item) for item in sample.delegations],
            },
        },
    ]

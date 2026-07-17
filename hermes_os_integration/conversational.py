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
WORKFLOW_STATES = ["planned", "running", "waiting", "delegated", "validating", "completed", "failed", "canceled"]


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
class ConversationalSession:
    session_id: str
    user_id: str
    project_id: str
    goal: str = ""
    initiative: str = ""
    transcript_ref: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ChatResponseEnvelope:
    session_id: str
    status: str
    message: str
    route: Dict[str, Any] = field(default_factory=dict)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationTranscript:
    session_id: str
    project_id: str
    turns: List[Dict[str, Any]]
    source_refs: List[str] = field(default_factory=list)
    stored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ConversationalWorkflow:
    workflow_id: str
    route: str
    status: str
    steps: List[WorkflowStep]
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class DelegationProtocol:
    assignment_id: str
    agent_role: str
    input_artifacts: List[str]
    output_artifacts: List[str]
    completion_evidence: List[str] = field(default_factory=list)
    status: str = "planned"


@dataclass(frozen=True)
class AgentAssignment:
    assignment_id: str
    agent_role: str
    step_id: str
    confidence: float
    risk: str = "normal"
    fallback: bool = False
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

CHAT_CAPABLE_COMMANDS = {
    "architect": "architecture review",
    "plan": "work graph planning",
    "workspace": "workspace summary",
    "projects": "project registry",
    "switch": "project switch",
    "start": "runtime startup",
    "snapshot": "workspace snapshots",
    "ask": "one-shot conversational planning",
}

SKILL_MANIFEST_INDEX = {
    "grill-me": {"aliases": ["/grill-me", "/clarify"], "goals": ["clarify idea", "interview project"]},
    "plan": {"aliases": ["/plan", "/tasks"], "goals": ["create plan", "generate tasks"]},
    "architect": {"aliases": ["/architecture", "/review-architecture"], "goals": ["review architecture", "find system gaps"]},
    "research": {"aliases": ["/research", "/analyze"], "goals": ["research topic", "compare options"]},
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


def col_contracts() -> Dict[str, Any]:
    return {
        "ownership": {
            "hermes_os": ["state", "governance", "approvals", "artifacts"],
            "chief_of_staff": ["route", "plan", "delegate", "report"],
            "worker_agents": ["produce artifacts for review"],
        },
        "boundaries": guardrails(),
    }


def create_conversational_session(session_id: str, user_id: str, project_id: str, *, goal: str = "", initiative: str = "") -> ConversationalSession:
    return ConversationalSession(session_id=session_id, user_id=user_id, project_id=project_id, goal=goal, initiative=initiative)


def chat_response_from_plan(plan: ChiefOfStaffPlan) -> ChatResponseEnvelope:
    return ChatResponseEnvelope(
        session_id=plan.request.session_id,
        status=plan.status,
        message="Hermes OS prepared a dry-run workflow plan.",
        route=asdict(plan.route),
        approvals=[asdict(step) for step in plan.steps if step.requires_approval],
        artifacts=[artifact for step in plan.steps for artifact in step.expected_artifacts],
        audit_events=plan.audit_events,
    )


def transcript_from_turns(session: ConversationalSession, turns: List[Dict[str, Any]], source_refs: Optional[List[str]] = None) -> ConversationTranscript:
    return ConversationTranscript(session_id=session.session_id, project_id=session.project_id, turns=list(turns), source_refs=list(source_refs or []))


def col_audit_event(event_type: str, session_id: str, *, route: str = "", delegation_id: str = "", approval_id: str = "", artifact_ref: str = "") -> Dict[str, Any]:
    return {
        "type": event_type,
        "session_id": session_id,
        "route": route,
        "delegation_id": delegation_id,
        "approval_id": approval_id,
        "artifact_ref": artifact_ref,
        "timestamp": _now(),
    }


def discover_chat_commands() -> Dict[str, Any]:
    return {"commands": [{"name": name, "capability": capability} for name, capability in sorted(CHAT_CAPABLE_COMMANDS.items())]}


def chat_cli_contract(*, interactive: bool = False, feature_enabled: bool = True) -> Dict[str, Any]:
    return {
        "command": "hermes chat",
        "interactive": interactive,
        "feature_enabled": feature_enabled,
        "executes_work": False,
        "mode": "prototype" if feature_enabled else "disabled",
    }


def ask_command_contract(message: str, *, project: str = ".") -> Dict[str, Any]:
    return {"command": "hermes ask", "message": message, "project": project, "dry_run": True}


def interactive_shell_contract(*, feature_enabled: bool = False) -> Dict[str, Any]:
    return {"command": "hermes", "mode": "chat-shell" if feature_enabled else "classic-cli", "feature_flag": "hermes_chat_shell"}


def parse_conversational_shortcut(message: str) -> Dict[str, Any]:
    command = normalize_slash_command(message)
    return {"slash_command": command, "route": SKILL_COMMANDS.get(command, ""), "valid": bool(command)}


def streaming_status_updates(plan: ChiefOfStaffPlan) -> List[Dict[str, Any]]:
    return [{"event": "workflow.step", "step_id": step.step_id, "status": step.status} for step in plan.steps]


def save_transcript_contract(transcript: ConversationTranscript) -> Dict[str, Any]:
    return {"path": f".hermes/conversations/{transcript.session_id}.json", "schema": "hermes-col-transcript-v1", "turn_count": len(transcript.turns)}


def load_transcript_contract(session_id: str) -> Dict[str, Any]:
    return {"path": f".hermes/conversations/{session_id}.json", "required": False}


def project_switch_from_chat(message: str, available_projects: Iterable[str]) -> Dict[str, Any]:
    text = message.lower()
    for project in available_projects:
        if project.lower() in text:
            return {"project_id": project, "command": f"hermes switch {project}", "dry_run": True}
    return {"project_id": "", "command": "", "dry_run": True}


def chat_error_envelope(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}, "data": {}}


def chat_help_examples() -> List[str]:
    return [
        "hermes ask \"Build a CRM for wholesalers\"",
        "hermes chat --project khashi-vc",
        "/architecture Review this project",
        "/research Compare provider options",
    ]


def cli_smoke_contracts() -> List[Dict[str, Any]]:
    return [
        {"command": "hermes chat --help", "expected": "usage"},
        {"command": "hermes ask \"Build a CRM\"", "expected": "dry-run workflow"},
        {"command": "hermes", "expected": "classic cli or chat shell"},
    ]


def chief_of_staff_role() -> Dict[str, Any]:
    return {
        "role": "Chief of Staff",
        "responsibilities": ["understand", "route", "plan", "delegate", "track", "report"],
        "prohibitions": ["direct worker implementation", "source-of-truth overwrite", "unapproved destructive action"],
        "outputs": ["workflow plan", "delegation records", "approval prompts", "status report"],
    }


def chief_of_staff_policy(action: str) -> Dict[str, Any]:
    blocked = action in {"write_code", "deploy", "delete"}
    return {"allowed": not blocked, "reason": "Chief of Staff coordinates; workers execute." if blocked else "coordination action"}


def chief_of_staff_decision_loop(envelope: ChatEnvelope, project_path: str = ".") -> Dict[str, Any]:
    plan = chief_of_staff_plan(envelope, project_path=project_path)
    return {
        "loop": ["understand", "route", "plan", "delegate", "track", "report"],
        "route": asdict(plan.route),
        "step_count": len(plan.steps),
        "delegation_count": len(plan.delegations),
        "status": plan.status,
    }


def action_plan_schema(plan: ChiefOfStaffPlan) -> Dict[str, Any]:
    return {
        "selected_workflow": plan.route.workflow,
        "agents": [item.agent_role for item in plan.delegations],
        "approvals": [asdict(step) for step in plan.steps if step.requires_approval],
        "expected_artifacts": [artifact for step in plan.steps for artifact in step.expected_artifacts],
    }


def conversational_status_report(workflow: ConversationalWorkflow) -> Dict[str, Any]:
    blocked = [step.step_id for step in workflow.steps if step.requires_approval]
    return {"workflow_id": workflow.workflow_id, "status": workflow.status, "blocked_steps": blocked, "step_count": len(workflow.steps)}


def clarification_prompt(route: IntentRoute) -> Dict[str, Any]:
    return {"required": route.requires_clarification, "question": route.clarification or "Confirm the desired outcome before proceeding."}


def approval_prompt_contract(plan: ChiefOfStaffPlan) -> Dict[str, Any]:
    approvals = [step for step in plan.steps if step.requires_approval]
    return {"required": bool(approvals), "approvals": [asdict(item) for item in approvals]}


def chief_of_staff_memory_handoff(plan: ChiefOfStaffPlan) -> Dict[str, Any]:
    return {"project_id": plan.request.project_id, "session_id": plan.request.session_id, "memory_refs": plan.memory.loaded_sources, "summary": plan.route.workflow}


def model_assisted_intent_fallback(message: str) -> Dict[str, Any]:
    return {"enabled": False, "reason": "deterministic router used first", "message": message}


def intent_evaluation_set() -> List[Dict[str, str]]:
    return [
        {"input": "Build a CRM", "expected": INTENT_NEW_PROJECT},
        {"input": "Continue Khashi dashboard", "expected": INTENT_EXISTING_PROJECT},
        {"input": "Research providers", "expected": INTENT_RESEARCH},
        {"input": "Review architecture", "expected": INTENT_ARCHITECTURE},
        {"input": "Implement task-123", "expected": INTENT_TASK_WORK},
    ]


def workflow_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "new_project_launch": {"route": INTENT_NEW_PROJECT, "steps": [step.step_id for step in workflow_steps_for(IntentRoute(INTENT_NEW_PROJECT, 1.0, "new_project_launch"))]},
        "existing_project_work": {"route": INTENT_EXISTING_PROJECT, "steps": [step.step_id for step in workflow_steps_for(IntentRoute(INTENT_EXISTING_PROJECT, 1.0, "existing_project_work"))]},
        "research": {"route": INTENT_RESEARCH, "steps": [step.step_id for step in workflow_steps_for(IntentRoute(INTENT_RESEARCH, 1.0, "research"))]},
        "architecture_review": {"route": INTENT_ARCHITECTURE, "steps": [step.step_id for step in workflow_steps_for(IntentRoute(INTENT_ARCHITECTURE, 1.0, "architecture_review"))]},
        "task_execution": {"route": INTENT_TASK_WORK, "steps": [step.step_id for step in workflow_steps_for(IntentRoute(INTENT_TASK_WORK, 1.0, "task_execution"))]},
    }


def create_conversational_workflow(route: IntentRoute) -> ConversationalWorkflow:
    steps = workflow_steps_for(route)
    return ConversationalWorkflow(workflow_id=f"workflow:{route.workflow}", route=route.workflow, status="planned", steps=steps)


def transition_conversational_workflow(workflow: ConversationalWorkflow, new_status: str) -> ConversationalWorkflow:
    if new_status not in WORKFLOW_STATES:
        raise ValueError("unknown conversational workflow state: " + new_status)
    return ConversationalWorkflow(workflow_id=workflow.workflow_id, route=workflow.route, status=new_status, steps=workflow.steps, checkpoints=workflow.checkpoints)


def checkpoint_workflow(workflow: ConversationalWorkflow, step_id: str, artifact_ref: str) -> ConversationalWorkflow:
    checkpoints = list(workflow.checkpoints) + [{"step_id": step_id, "artifact_ref": artifact_ref, "timestamp": _now()}]
    return ConversationalWorkflow(workflow.workflow_id, workflow.route, workflow.status, workflow.steps, checkpoints)


def workflow_preview(workflow: ConversationalWorkflow) -> Dict[str, Any]:
    return {"workflow_id": workflow.workflow_id, "steps": [asdict(item) for item in workflow.steps], "approvals": [asdict(item) for item in workflow.steps if item.requires_approval], "artifacts": [artifact for step in workflow.steps for artifact in step.expected_artifacts]}


def workflow_recovery_contract(workflow: ConversationalWorkflow, action: str) -> Dict[str, Any]:
    return {"workflow_id": workflow.workflow_id, "action": action, "valid": action in {"resume", "cancel", "recover"}, "dry_run": True}


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


def project_identity_memory(project_id: str, workspace_path: str) -> Dict[str, Any]:
    return {"project_id": project_id, "workspace_path": str(Path(workspace_path).resolve()), "source": "session"}


def core_project_memory_loader(project_path: str) -> Dict[str, Any]:
    root = Path(project_path)
    names = ["project.md", "architecture.md", "decisions.md", "progress.md", "tracker.md", "backlog.md", "agents.md", "TASKS.md"]
    loaded = {}
    missing = []
    for name in names:
        path = root / name
        if path.exists():
            loaded[name] = path.read_text(encoding="utf-8")
        else:
            missing.append(name)
    return {"project_path": str(root.resolve()), "loaded": loaded, "missing": missing}


def working_context_builder(memory: Dict[str, Any], *, max_chars: int = 2000) -> Dict[str, Any]:
    text = "\n".join(str(value) for value in memory.get("loaded", {}).values())
    return {"context": text[:max_chars], "source_count": len(memory.get("loaded", {})), "truncated": len(text) > max_chars}


def retrieve_decision_progress(records: Iterable[Dict[str, Any]], *, topic: str = "", min_confidence: float = 0.0) -> List[Dict[str, Any]]:
    return [record for record in records if (not topic or record.get("topic") == topic) and float(record.get("confidence", 1.0)) >= min_confidence]


def rolling_conversation_memory(turns: List[Dict[str, Any]], *, limit: int = 5) -> Dict[str, Any]:
    recent = turns[-limit:]
    return {"recent_turns": recent, "unresolved_questions": [turn for turn in recent if str(turn.get("content", "")).endswith("?")]}


def active_session_store(session: ConversationalSession) -> Dict[str, Any]:
    return {"active_project": session.project_id, "active_goal": session.goal, "active_initiative": session.initiative, "session_id": session.session_id}


def memory_freshness_warnings(memory: Dict[str, Any], *, max_missing: int = 3) -> List[str]:
    missing = list(memory.get("missing", []))
    warnings = []
    if len(missing) > max_missing:
        warnings.append("Project memory is sparse; core context documents are missing.")
    if "decisions.md" in missing:
        warnings.append("Decision memory is missing.")
    return warnings


def redact_memory_context(context: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in context.items():
        if any(token in key.lower() for token in ("secret", "token", "password", "api_key")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def agent_context_package(project_memory: Dict[str, Any], task_context: Dict[str, Any], workflow: ConversationalWorkflow) -> Dict[str, Any]:
    return {"project_memory": redact_memory_context(project_memory), "task_context": task_context, "workflow_id": workflow.workflow_id, "checkpoint_count": len(workflow.checkpoints)}


def conversational_agent_roles() -> Dict[str, Dict[str, str]]:
    return {"management": MANAGEMENT_ROLES, "worker": WORKER_ROLES}


def delegation_protocol_for(step: WorkflowStep, agent_role: str) -> DelegationProtocol:
    return DelegationProtocol(assignment_id=f"{step.step_id}:{agent_role}", agent_role=agent_role, input_artifacts=[], output_artifacts=step.expected_artifacts)


def plan_agent_assignment(step: WorkflowStep, candidates: Iterable[Dict[str, Any]]) -> AgentAssignment:
    candidates_list = list(candidates)
    selected = candidates_list[0] if candidates_list else {"role": step.owner.lower().replace(" ", "_"), "confidence": 0.5}
    return AgentAssignment(
        assignment_id=f"{step.step_id}:{selected.get('role')}",
        agent_role=str(selected.get("role")),
        step_id=step.step_id,
        confidence=float(selected.get("confidence", 0.5)),
        risk="approval_required" if step.requires_approval else "normal",
        fallback=not bool(candidates_list),
    )


def validate_artifact_handoff(protocol: DelegationProtocol) -> Dict[str, Any]:
    return {"valid": bool(protocol.output_artifacts), "assignment_id": protocol.assignment_id, "missing": [] if protocol.output_artifacts else ["output_artifacts"]}


def multi_agent_trace(delegations: Iterable[DelegationRecord]) -> List[Dict[str, Any]]:
    return [{"agent_role": item.agent_role, "layer": item.layer, "status": item.status, "outputs": item.output_artifacts} for item in delegations]


def reviewer_approval_flow(artifact_ref: str, *, reviewer: str = "reviewer") -> Dict[str, Any]:
    return {"artifact_ref": artifact_ref, "reviewer": reviewer, "actions": ["approve", "reject", "request-more-context"], "requires_reason": True}


def agent_failure_fallback(agent_role: str, *, reason: str) -> Dict[str, Any]:
    return {"agent_role": agent_role, "reason": reason, "fallback": "reviewer" if agent_role != "reviewer" else "Chief of Staff", "escalate": True}


def chief_of_staff_api_contract() -> Dict[str, Any]:
    return {"endpoints": ["/api/col/chat", "/api/col/workflow", "/api/col/recommendations", "/api/col/context"], "mode": "contract"}


def open_webui_integration_contract(session: ConversationalSession) -> Dict[str, Any]:
    return {"adapter": "open-webui", "session_id": session.session_id, "project_id": session.project_id, "streaming": True}


def hermes_chat_page_contract(session: ConversationalSession, workflow: ConversationalWorkflow) -> Dict[str, Any]:
    return {"page": "hermes-chat", "session": asdict(session), "workflow_status": workflow.status, "persistent_conversation": True}


def project_switcher_contract(projects: Iterable[str], active_project: str = "") -> Dict[str, Any]:
    project_list = list(projects)
    return {"projects": project_list, "active_project": active_project or (project_list[0] if project_list else "")}


def task_backlog_view(tasks: Iterable[Dict[str, Any]], *, project_id: str, workflow_id: str = "") -> Dict[str, Any]:
    selected = [task for task in tasks if task.get("project_id") == project_id and (not workflow_id or task.get("workflow_id") == workflow_id)]
    return {"project_id": project_id, "workflow_id": workflow_id, "tasks": selected, "count": len(selected)}


def review_queue_view(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    pending = [item for item in items if item.get("status") == "pending"]
    return {"count": len(pending), "items": pending, "actions": ["approve", "reject", "request-more-context"]}


def recommendation_panel(plan: ChiefOfStaffPlan) -> Dict[str, Any]:
    return {"panel_id": "col-recommendations", "recommendations": [asdict(item) for item in plan.recommendations]}


def active_agent_activity(assignments: Iterable[AgentAssignment]) -> Dict[str, Any]:
    rows = [asdict(item) for item in assignments]
    return {"panel_id": "active-agent-activity", "count": len(rows), "assignments": rows}


def dashboard_status_widgets(session: ConversationalSession, *, open_tasks: int = 0, pending_reviews: int = 0) -> Dict[str, Any]:
    return {"active_project": session.project_id, "active_initiative": session.initiative, "open_tasks": open_tasks, "pending_reviews": pending_reviews, "dashboard": "ready"}


def ui_test_contracts() -> List[Dict[str, str]]:
    return [{"scenario": name, "status": "contract"} for name in ["chat", "project switching", "task viewing", "review actions", "status refresh"]]


def skill_manifest_index() -> Dict[str, Any]:
    return {"skills": SKILL_MANIFEST_INDEX}


def slash_command_aliases() -> Dict[str, str]:
    aliases = {}
    for skill, payload in SKILL_MANIFEST_INDEX.items():
        for alias in payload["aliases"]:
            aliases[alias] = skill
    return aliases


def natural_language_skill_invocation(message: str) -> Dict[str, Any]:
    text = message.lower()
    for skill, payload in SKILL_MANIFEST_INDEX.items():
        if any(goal in text for goal in payload["goals"]):
            return {"skill": skill, "confidence": 0.8, "dry_run": True}
    return {"skill": "", "confidence": 0.0, "dry_run": True}


def contextual_command_recommendation(active_project: str, workflow: str, recent_messages: Iterable[str]) -> Dict[str, Any]:
    messages = " ".join(recent_messages).lower()
    if "research" in messages or workflow == "research":
        command = "/research"
    elif workflow == "new_project_launch":
        command = "/grill-me"
    else:
        command = "/plan"
    return {"active_project": active_project, "workflow": workflow, "command": command}


def generated_command_permission_check(command: str, *, approved: bool = False) -> Dict[str, Any]:
    high_risk = any(token in command for token in ("deploy", "delete", "purchase"))
    return {"allowed": not high_risk or approved, "requires_approval": high_risk and not approved, "command": command}


def chief_of_staff_dashboard(plan: ChiefOfStaffPlan) -> Dict[str, Any]:
    return {
        "active_project": plan.request.project_id,
        "initiative": plan.request.active_initiative,
        "tasks": [step.step_id for step in plan.steps],
        "reviews": [step.step_id for step in plan.steps if step.requires_approval],
        "agents": [item.agent_role for item in plan.delegations],
        "recommendations": [item.title for item in plan.recommendations],
    }


def crm_launch_success_scenario(project_name: str = "wholesaler-crm") -> Dict[str, Any]:
    envelope = ChatEnvelope(message="Build a CRM for wholesalers", project_id=project_name)
    plan = chief_of_staff_plan(envelope)
    return {"project_id": project_name, "workflow": plan.route.workflow, "success_path": [step.step_id for step in plan.steps], "dry_run": True}


def conversational_user_guide() -> Dict[str, Any]:
    return {"title": "Conversational Hermes OS Workflows", "sections": ["ask", "chat", "approvals", "artifacts", "operator expectations"]}


def command_first_to_chat_first_rollout() -> Dict[str, Any]:
    return {"phases": ["contract", "dry-run", "operator-gated live", "default chat"], "migration": "command aliases remain available"}


def dynamic_command_regression_tests() -> List[Dict[str, str]]:
    return [{"case": "dynamic command recommendation"}, {"case": "launch workflow"}, {"case": "dashboard panels"}, {"case": "documentation links"}]

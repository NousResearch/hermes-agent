"""Architecture-first contracts and gates for Hermes OS."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .errors import PERMISSION_DENIED, VALIDATION_ERROR, AdapterError, adapter_error


CONSTITUTION_RULES = [
    "Business logic before implementation.",
    "Dashboards before automation.",
    "Workflows before agents.",
    "Agents are workers, not owners.",
    "Persistent state belongs to Hermes OS.",
    "Agent memory is not source of truth.",
    "Every project requires a domain model.",
    "Every project requires measurable outcomes.",
    "High-risk actions require human approval.",
    "Coding begins only after architecture review.",
    "Specifications generate tasks.",
    "Tasks generate execution.",
    "Execution generates artifacts.",
    "Artifacts generate dashboards.",
    "Dashboards generate feedback loops.",
]

ARCHITECTURE_ORDER = [
    "business_system",
    "control_plane",
    "domain_models",
    "workflows",
    "dashboards",
    "metrics",
    "approval_gates",
    "agents",
    "implementation",
    "optimization",
]

REQUIRED_PROJECT_DOCS = [
    "PROJECT.md",
    "DOMAIN.md",
    "WORKFLOWS.md",
    "DASHBOARD.md",
    "METRICS.md",
    "APPROVALS.md",
    "AGENTS.md",
    "TASKS.md",
    "DECISIONS.md",
    "ROADMAP.md",
    "ARCHITECTURE.md",
]

REVIEW_CATEGORIES = [
    "business_model",
    "control_plane",
    "domain_model",
    "workflows",
    "dashboards",
    "metrics",
    "approvals",
    "agents",
    "data_quality",
    "automation_opportunities",
    "scalability",
    "technical_debt",
]

GRILL_ME_CATEGORIES = [
    "business",
    "domain",
    "workflow",
    "metrics",
    "dashboard",
    "approvals",
    "automation",
    "agents",
    "scalability",
    "monetization",
    "risk",
    "data",
]

GRILL_ME_QUESTIONS = {
    "business": [
        "Who is the user?",
        "What problem is solved?",
        "How is value measured?",
    ],
    "domain": [
        "What entities exist?",
        "How are entities related?",
        "What data must persist?",
    ],
    "workflow": [
        "What starts the workflow?",
        "What completes it?",
        "What exceptions exist?",
    ],
    "metrics": [
        "What outcome proves progress?",
        "What indicates failure?",
        "What metric can be derived from state?",
    ],
    "dashboard": [
        "What must be visible daily?",
        "What reports are required?",
        "What indicates opportunity?",
    ],
    "approvals": [
        "Which actions are high risk?",
        "Who approves irreversible changes?",
        "What must block automation?",
    ],
    "automation": [
        "What can be automated only after a dashboard exists?",
        "What needs human review?",
        "What should remain manual?",
    ],
    "agents": [
        "What artifact should an agent produce?",
        "What state must the agent not own?",
        "Which tools may the agent use?",
    ],
    "scalability": [
        "What breaks when usage doubles?",
        "What must be cached?",
        "What must be queued?",
    ],
    "monetization": [
        "How does this create or protect value?",
        "What is the business model?",
        "What changes willingness to pay?",
    ],
    "risk": [
        "What can go wrong?",
        "What is irreversible?",
        "What needs an escalation path?",
    ],
    "data": [
        "What data is authoritative?",
        "What data is derived?",
        "What data quality checks are required?",
    ],
}

ZOD_FIRST_SCHEMA_CATALOG = [
    "Project",
    "Task",
    "Workflow",
    "Experiment",
    "ResearchReport",
    "AgentResult",
    "Approval",
    "DashboardMetric",
    "Decision",
    "Portfolio",
    "Market",
    "Bucket",
    "Observation",
    "Hypothesis",
    "Validation",
    "PromotionDecision",
]

AGENT_ALLOWED_RESPONSIBILITIES = {
    "research",
    "analyze",
    "validate",
    "generate",
    "review",
    "document",
    "test",
}

AGENT_PROHIBITED_OWNERSHIP = {
    "projects",
    "workflows",
    "dashboards",
    "approvals",
    "business_logic",
    "source_of_truth_state",
}


@dataclass(frozen=True)
class ArchitectureViolation:
    requested_stage: str
    missing_prerequisites: List[str]
    severity: str = "warning"
    message: str = ""


@dataclass(frozen=True)
class ArchitectureReviewRequest:
    project_id: str
    project_path: str
    present_documents: List[str] = field(default_factory=list)
    completed_stages: List[str] = field(default_factory=list)
    scope: List[str] = field(default_factory=lambda: list(REVIEW_CATEGORIES))


@dataclass(frozen=True)
class ArchitectureReviewReport:
    project_id: str
    architecture_score: int
    critical_gaps: List[str]
    missing_documents: List[str]
    missing_schemas: List[str]
    missing_dashboards: List[str]
    missing_approvals: List[str]
    automation_opportunities: List[str]
    recommendations: List[str]
    priority_roadmap: List[str]
    blocked: bool = False


@dataclass(frozen=True)
class GrillMeSession:
    project_id: str
    questions: Dict[str, List[str]]
    answers: Dict[str, str] = field(default_factory=dict)
    unresolved_assumptions: List[str] = field(default_factory=list)
    status: str = "open"


@dataclass(frozen=True)
class WorkflowDefinition:
    trigger: str
    inputs: List[str]
    steps: List[str]
    outputs: List[str]
    approvals: List[str]
    metrics: List[str]
    failure_states: List[str]
    escalation_rules: List[str]


@dataclass(frozen=True)
class DashboardRequirements:
    daily_visibility: List[str]
    success_metrics: List[str]
    failure_indicators: List[str]
    opportunity_indicators: List[str]
    required_reports: List[str]


@dataclass(frozen=True)
class ArtifactIngestionResult:
    accepted: bool
    artifact_ref: Optional[str] = None
    error: Optional[AdapterError] = None
    provenance: Dict[str, str] = field(default_factory=dict)


def load_constitution(project_rules: Optional[Sequence[str]] = None):
    rules = list(CONSTITUTION_RULES)
    if project_rules:
        rules.extend(str(rule) for rule in project_rules)
    return {
        "source": ".hermes/constitution.md",
        "rules": rules,
        "precedence": ["project", "workspace", "global", "built-in"],
    }


def check_architecture_order(completed_stages: Iterable[str], requested_stage: str):
    completed = set(completed_stages)
    if requested_stage not in ARCHITECTURE_ORDER:
        return ArchitectureViolation(
            requested_stage=requested_stage,
            missing_prerequisites=[],
            severity="block",
            message="Unknown architecture stage: " + requested_stage,
        )
    requested_index = ARCHITECTURE_ORDER.index(requested_stage)
    missing = [stage for stage in ARCHITECTURE_ORDER[:requested_index] if stage not in completed]
    severity = "block" if requested_stage in {"implementation", "agents"} and missing else "warning"
    message = "Missing architecture prerequisites: " + ", ".join(missing) if missing else "Architecture order satisfied."
    return ArchitectureViolation(
        requested_stage=requested_stage,
        missing_prerequisites=missing,
        severity=severity,
        message=message,
    )


def validate_architecture_review_request(payload):
    if not isinstance(payload, dict):
        return None, adapter_error(VALIDATION_ERROR, "Architecture review request must be an object")
    missing = [key for key in ["project_id", "project_path"] if not payload.get(key)]
    if missing:
        return None, adapter_error(VALIDATION_ERROR, "Missing required fields: " + ", ".join(missing))
    return ArchitectureReviewRequest(
        project_id=str(payload["project_id"]),
        project_path=str(payload["project_path"]),
        present_documents=[str(item) for item in payload.get("present_documents", [])],
        completed_stages=[str(item) for item in payload.get("completed_stages", [])],
        scope=[str(item) for item in payload.get("scope", REVIEW_CATEGORIES)],
    ), None


def review_architecture(request: ArchitectureReviewRequest):
    missing_documents = [doc for doc in REQUIRED_PROJECT_DOCS if doc not in request.present_documents]
    violation = check_architecture_order(request.completed_stages, "implementation")
    missing_schemas = [] if "DOMAIN.md" in request.present_documents else list(ZOD_FIRST_SCHEMA_CATALOG[:8])
    missing_dashboards = [] if "DASHBOARD.md" in request.present_documents else ["daily_visibility", "failure_indicators", "opportunity_indicators"]
    missing_approvals = [] if "APPROVALS.md" in request.present_documents else ["high_risk_actions", "human_approval_gate"]
    gap_count = len(missing_documents) + len(violation.missing_prerequisites) + len(missing_schemas)
    score = max(0, 100 - gap_count * 5)
    critical_gaps = []
    if missing_documents:
        critical_gaps.append("Missing required project documents")
    if violation.missing_prerequisites:
        critical_gaps.append("Architecture order incomplete")
    if missing_schemas:
        critical_gaps.append("Domain schemas missing")
    recommendations = [
        "Complete missing source-of-truth project documents",
        "Run grill-me before task generation",
        "Generate dashboards before automation",
        "Keep agent outputs as artifacts until Hermes OS validates them",
    ]
    return ArchitectureReviewReport(
        project_id=request.project_id,
        architecture_score=score,
        critical_gaps=critical_gaps,
        missing_documents=missing_documents,
        missing_schemas=missing_schemas,
        missing_dashboards=missing_dashboards,
        missing_approvals=missing_approvals,
        automation_opportunities=["Generate tasks from approved specifications"],
        recommendations=recommendations,
        priority_roadmap=missing_documents[:5] + violation.missing_prerequisites[:5],
        blocked=bool(violation.missing_prerequisites),
    )


def architect_cli_spec():
    return {
        "command": "hermes architect review <project>",
        "options": ["--json", "--scope", "--block-on-critical", "--write-report"],
        "exit_codes": {"0": "pass", "2": "warning", "3": "blocked", "64": "invalid_request"},
        "outputs": ["architecture_score", "critical_gaps", "recommendations", "priority_roadmap"],
    }


def render_review_report(report: ArchitectureReviewReport):
    lines = [
        "# Architecture Review",
        "",
        "Project: " + report.project_id,
        "Score: " + str(report.architecture_score),
        "Blocked: " + str(report.blocked).lower(),
        "",
        "## Critical Gaps",
    ]
    lines.extend("- " + item for item in (report.critical_gaps or ["None"]))
    lines.append("")
    lines.append("## Priority Roadmap")
    lines.extend("- " + item for item in (report.priority_roadmap or ["No immediate roadmap items"]))
    return "\n".join(lines) + "\n"


def create_grill_me_session(project_id: str):
    return GrillMeSession(project_id=project_id, questions={key: list(value) for key, value in GRILL_ME_QUESTIONS.items()})


def assumption_challenge_output(session: GrillMeSession):
    unresolved = list(session.unresolved_assumptions)
    unanswered = [category for category in GRILL_ME_CATEGORIES if not session.answers.get(category)]
    return {
        "project_id": session.project_id,
        "challenged_assumptions": unresolved,
        "unanswered_categories": unanswered,
        "blocks_task_generation": bool(unresolved or unanswered),
        "required_follow_up": unresolved + ["Answer " + category + " questions" for category in unanswered],
    }


def project_document_templates():
    return {
        doc: "# " + doc.replace(".md", "").title() + "\n\n## Required Fields\n\n- Purpose\n- Source of truth\n- Open questions\n"
        for doc in REQUIRED_PROJECT_DOCS
    }


def specification_to_tasks(specification: Dict[str, Any], architecture_approved: bool):
    if not architecture_approved:
        return None, adapter_error(PERMISSION_DENIED, "Cannot generate tasks before architecture approval")
    required = ["spec_id", "summary", "acceptance_criteria"]
    missing = [key for key in required if not specification.get(key)]
    if missing:
        return None, adapter_error(VALIDATION_ERROR, "Missing specification fields: " + ", ".join(missing))
    return [{
        "title": "Implement " + str(specification["summary"]),
        "traceability": str(specification["spec_id"]),
        "acceptanceCriteria": [str(item) for item in specification["acceptance_criteria"]],
        "risk": str(specification.get("risk", "medium")),
    }], None


def validate_workflow_definition(workflow: WorkflowDefinition):
    missing = []
    for field_name in ["trigger", "inputs", "steps", "outputs", "metrics", "failure_states", "escalation_rules"]:
        value = getattr(workflow, field_name)
        if not value:
            missing.append(field_name)
    if missing:
        return None, adapter_error(VALIDATION_ERROR, "Missing workflow fields: " + ", ".join(missing))
    return workflow, None


def dashboard_readiness(requirements: DashboardRequirements):
    missing = []
    for field_name in ["daily_visibility", "success_metrics", "failure_indicators", "opportunity_indicators", "required_reports"]:
        if not getattr(requirements, field_name):
            missing.append(field_name)
    return {
        "ready": not missing,
        "missing": missing,
        "severity": "block" if missing else "pass",
    }


def dashboard_feedback(metric_name: str, observation: str):
    return {
        "metric": metric_name,
        "observation": observation,
        "generated_task": "Investigate " + metric_name,
        "traceability": "dashboard://" + metric_name,
    }


def check_agent_ownership(responsibilities: Iterable[str]):
    prohibited = sorted(set(responsibilities).intersection(AGENT_PROHIBITED_OWNERSHIP))
    if prohibited:
        return None, adapter_error(PERMISSION_DENIED, "Agent cannot own: " + ", ".join(prohibited))
    return {"allowed": sorted(set(responsibilities).intersection(AGENT_ALLOWED_RESPONSIBILITIES))}, None


def validate_artifact_ingestion(payload: Dict[str, Any]):
    required = ["artifact_id", "agent_id", "schema", "content_ref"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        return ArtifactIngestionResult(
            accepted=False,
            error=adapter_error(VALIDATION_ERROR, "Missing artifact fields: " + ", ".join(missing)),
        )
    return ArtifactIngestionResult(
        accepted=True,
        artifact_ref="hermes-os://artifact/" + str(payload["artifact_id"]),
        provenance={"agent_id": str(payload["agent_id"]), "schema": str(payload["schema"])},
    )


def runtime_delegation_readiness(completed_stages: Iterable[str], dry_run: bool = False):
    violation = check_architecture_order(completed_stages, "implementation")
    if dry_run:
        return {"ready": True, "mode": "dry_run", "missing": violation.missing_prerequisites}
    return {"ready": not violation.missing_prerequisites, "mode": "live", "missing": violation.missing_prerequisites}


def existing_project_review_targets():
    return {
        "kalshi-vc": ["experiment success rate", "promotion rate", "bucket performance", "evidence coverage"],
        "investing-system": ["watchlist size", "thesis status", "portfolio exposure", "risk metrics"],
        "media-engine": ["stories published", "coverage mix", "brand growth", "approval time"],
        "rinseables": ["business model", "workflows", "dashboards", "approvals"],
    }

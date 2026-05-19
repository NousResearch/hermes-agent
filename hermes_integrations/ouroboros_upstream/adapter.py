"""Hermes wrapper around the vendored Ouroboros Interview/Seed subset."""

from __future__ import annotations

from typing import Any

from hermes_integrations.ouroboros_upstream.bigbang.interview import (
    InterviewRound,
    InterviewState,
    InterviewStatus,
)
from hermes_integrations.ouroboros_upstream.bigbang.seed_generator import SeedGenerator
from hermes_integrations.ouroboros_upstream.auto.seed_repairer import SeedRepairer
from hermes_integrations.ouroboros_upstream.auto.seed_reviewer import SeedReviewer
from hermes_integrations.ouroboros_upstream.core.seed import (
    BrownfieldContext,
    ContextReference,
    EvaluationPrinciple,
    ExitCondition,
    OntologyField,
    OntologySchema,
    Seed,
    SeedMetadata,
)


def as_list(value: Any, *, default: list[str] | None = None) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default or [])
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    parts = [p.strip() for p in __import__('re').split(r"[,;|]", text) if p.strip()]
    return parts or [text]


def start_interview_state(session_id: str, values: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    context = str(values.get("context") or "").strip()
    state = InterviewState(
        interview_id=session_id,
        initial_context=str(values.get("goal") or "").strip(),
        is_brownfield=bool(__import__('re').search(r"brownfield|gateway|repo|runtime|existing", " ".join([str(values.get('goal') or ''), context]), __import__('re').IGNORECASE)),
        codebase_context=context,
        rounds=[InterviewRound(round_number=1, question=str(question.get("text") or ""))],
    )
    return state.model_dump(mode="json")


def record_answer(state_payload: dict[str, Any] | None, answer: str, next_question: dict[str, Any] | None, *, ambiguity_score: float | None = None, ambiguity_breakdown: dict[str, Any] | None = None, completed: bool = False) -> dict[str, Any]:
    if not isinstance(state_payload, dict):
        state = InterviewState(interview_id="unknown")
    else:
        state = InterviewState.model_validate(state_payload)
    if state.rounds:
        state.rounds[-1].user_response = answer
    if ambiguity_score is not None:
        state.store_ambiguity(score=float(ambiguity_score), breakdown=ambiguity_breakdown or {})
    else:
        state.mark_updated()
    if completed:
        state.status = InterviewStatus.COMPLETED
    elif next_question is not None:
        state.rounds.append(InterviewRound(round_number=state.current_round_number, question=str(next_question.get("text") or "")))
    return state.model_dump(mode="json")


def _ontology(values: dict[str, Any]) -> OntologySchema:
    project = str(values.get("project") or "bo").strip().lower()
    return OntologySchema(
        name=f"{project.upper()}Admission",
        description="Ouroboros Seed ontology projected through Hermes Kanban admission wrapper.",
        fields=(
            OntologyField(name="goal", field_type="string", description="Primary user-visible objective"),
            OntologyField(name="scope", field_type="string", description="Included boundaries and non-goals"),
            OntologyField(name="verification", field_type="array", description="Observable proof required before closeout"),
        ),
    )


def _evaluation_principles(values: dict[str, Any]) -> tuple[EvaluationPrinciple, ...]:
    raw = as_list(values.get("evaluation_principles"))
    if not raw:
        return (
            EvaluationPrinciple(name="completeness", description="All acceptance criteria are satisfied without violating constraints", weight=1.0),
            EvaluationPrinciple(name="authority_boundary", description="Kanban/Hermes side-effect boundaries remain intact", weight=1.0),
        )
    principles = []
    for item in raw:
        name, _, rest = item.partition(":")
        desc, _, weight_text = rest.partition(":")
        try:
            weight = float(weight_text) if weight_text else 1.0
        except ValueError:
            weight = 1.0
        principles.append(EvaluationPrinciple(name=name.strip() or "principle", description=desc.strip() or item, weight=max(0.0, min(1.0, weight))))
    return tuple(principles)


def _exit_conditions(values: dict[str, Any]) -> tuple[ExitCondition, ...]:
    raw = as_list(values.get("exit_conditions"))
    if not raw:
        return (
            ExitCondition(name="all_criteria_met", description="All acceptance criteria are satisfied", evaluation_criteria="Observable verification evidence exists for every criterion"),
            ExitCondition(name="authority_preserved", description="Admission never grants execution authority", evaluation_criteria="executor_dispatch remains forbidden until Chris/Kanban approval"),
        )
    conditions = []
    for item in raw:
        name, _, rest = item.partition(":")
        desc, _, criteria = rest.partition(":")
        conditions.append(ExitCondition(name=name.strip() or "condition", description=desc.strip() or item, evaluation_criteria=criteria.strip() or desc.strip() or item))
    return tuple(conditions)


def _structured_seed_extraction(values: dict[str, Any]) -> str:
    """Build the upstream SeedGenerator structured extraction format."""

    goal = str(values.get("goal") or "").strip()
    context = str(values.get("context") or "").strip()
    is_brownfield = bool(__import__('re').search(r"brownfield|gateway|repo|runtime|existing", " ".join([goal, context]), __import__('re').IGNORECASE))
    project = str(values.get("project") or "bo").strip().lower()
    constraints = " | ".join(as_list(values.get("constraints"), default=["Follow existing project patterns unless acceptance criteria require otherwise"]))
    acceptance = " | ".join(as_list(values.get("acceptance_criteria"), default=["A command/API check returns stable observable output or artifacts proving the goal"]))
    existing_patterns = " | ".join(as_list(values.get("existing_patterns"), default=[context] if context else []))
    existing_dependencies = " | ".join(as_list(values.get("existing_dependencies")))
    context_refs = ""
    if context and is_brownfield:
        context_refs = f"hermes://context:reference:{context}"
    evaluation = " | ".join(
        f"{item.name}:{item.description}:{item.weight}"
        for item in _evaluation_principles(values)
    )
    exits = " | ".join(
        f"{item.name}:{item.description}:{item.evaluation_criteria}"
        for item in _exit_conditions(values)
    )
    return "\n".join([
        f"GOAL: {goal}",
        f"CONSTRAINTS: {constraints}",
        f"ACCEPTANCE_CRITERIA: {acceptance}",
        f"ONTOLOGY_NAME: {project.upper()}Admission",
        "ONTOLOGY_DESCRIPTION: Ouroboros Seed ontology projected through Hermes Kanban admission wrapper.",
        "ONTOLOGY_FIELDS: goal:string:Primary user-visible objective | scope:string:Included boundaries and non-goals | verification:array:Observable proof required before closeout",
        f"EVALUATION_PRINCIPLES: {evaluation}",
        f"EXIT_CONDITIONS: {exits}",
        f"PROJECT_TYPE: {'brownfield' if is_brownfield else 'greenfield'}",
        f"CONTEXT_REFERENCES: {context_refs}",
        f"EXISTING_PATTERNS: {existing_patterns}",
        f"EXISTING_DEPENDENCIES: {existing_dependencies}",
    ])


def build_seed(values: dict[str, Any], review: dict[str, Any], *, session_id: str | None) -> Seed:
    state = InterviewState(interview_id=session_id or "unknown", initial_context=str(values.get("goal") or ""))
    generator = SeedGenerator()
    return generator.build_from_structured_response(
        _structured_seed_extraction(values),
        state=state,
        ambiguity_score=float(review.get("ambiguity_score", 0.15)),
    )

def build_seed_dict(values: dict[str, Any], review: dict[str, Any], *, session_id: str | None) -> dict[str, Any]:
    seed = build_seed(values, review, session_id=session_id)
    payload = seed.to_dict()
    # Prove we did not create a lookalike dict: validate through upstream model.
    return Seed.from_dict(payload).to_dict()


def review_and_repair_seed_dict(seed_payload: dict[str, Any], *, max_iterations: int = 2) -> dict[str, Any]:
    """Review and bounded-repair a Seed through vendored upstream auto primitives.

    This is intentionally advisory for Hermes admission: upstream `may_run` is
    recorded for Seed quality, but Hermes still forbids executor dispatch until
    Chris/Kanban separately approves execution.
    """

    seed = Seed.from_dict(seed_payload)
    reviewer = SeedReviewer()
    repairer = SeedRepairer(reviewer=reviewer, max_iterations=max_iterations)
    repaired_seed, review, history = repairer.converge(seed)
    final_payload = Seed.from_dict(repaired_seed.to_dict()).to_dict()
    return {
        "seed": final_payload,
        "review": {
            "grade": review.grade_result.grade.value,
            "scores": review.grade_result.scores,
            "may_run": review.may_run,
            "can_repair": review.grade_result.can_repair,
            "findings": [finding.__dict__ for finding in review.findings],
            "blockers": [blocker.to_dict() for blocker in review.grade_result.blockers],
        },
        "repair_history": [
            {
                "changed": item.changed,
                "applied_repairs": list(item.applied_repairs),
                "unresolved_findings": [finding.__dict__ for finding in item.unresolved_findings],
                "blocker": item.blocker,
            }
            for item in history
        ],
    }

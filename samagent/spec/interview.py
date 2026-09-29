"""Adaptive interview engine (≤ 5 questions, recommended defaults, skip-safe -> assumptions ledger)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from samagent.spec.models import (
    AssumptionEntry,
    AutonomyLevel,
    BudgetSpec,
    ModuleSpec,
    SpecDocument,
    UserStory,
)


@dataclass(frozen=True)
class InterviewOption:
    id: str
    label: str
    description: str
    assumption_text: str


@dataclass(frozen=True)
class InterviewQuestion:
    id: str
    question: str
    recommended_id: str
    options: List[InterviewOption]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_interview_questions(brief_text: str) -> List[InterviewQuestion]:
    """Generate at most 5 adaptive interview questions with explicit recommended defaults.

    Every question has a deterministic fallback assumption so skipping 1 or all questions
    never blocks execution (05-final-plan.md §4 step 2).
    """
    lower = (brief_text or "").lower()
    needs_auth = any(w in lower for w in ("book", "user", "member", "admin", "account", "login", "dashboard", "crud", "saas", "studio", "store"))
    questions: List[InterviewQuestion] = [
        InterviewQuestion(
            id="Q1_AUTH_ROLES",
            question="Who uses this app and do they need separate accounts/roles?",
            recommended_id="roles_3" if needs_auth else "public_only",
            options=[
                InterviewOption(
                    id="roles_3",
                    label="Visitor + Member + Admin (Recommended)" if needs_auth else "Visitor + Member + Admin",
                    description="Public pages for visitors, authenticated actions for members, management for admins.",
                    assumption_text="Three roles (visitor, member, admin) with session-based auth and row-level ownership.",
                ),
                InterviewOption(
                    id="public_only",
                    label="Single-user / Public (No login)" + (" (Recommended)" if not needs_auth else ""),
                    description="Anyone with the link can use the app; no login screen.",
                    assumption_text="Single-role public app without authentication walls.",
                ),
            ],
        ),
        InterviewQuestion(
            id="Q2_DATA_STORAGE",
            question="How should application data be stored?",
            recommended_id="sqlite_local",
            options=[
                InterviewOption(
                    id="sqlite_local",
                    label="Local SQLite database (Recommended)",
                    description="Zero external setup, committed schema in .samagent/contract/db/schema.sql.",
                    assumption_text="Data persists in a local SQLite database with strict foreign-key and schema constraints.",
                ),
                InterviewOption(
                    id="in_memory",
                    label="In-memory / JSON seed only",
                    description="Fastest prototype for pure UI demos; resets on restart.",
                    assumption_text="Prototype uses seeded in-memory/JSON state without durable DB migrations.",
                ),
            ],
        ),
        InterviewQuestion(
            id="Q3_PAYMENTS_INTEGRATIONS",
            question="How should payments or external third-party APIs be handled in v1?",
            recommended_id="mock_out_of_scope",
            options=[
                InterviewOption(
                    id="mock_out_of_scope",
                    label="Mock / Out of scope for v1 (Recommended)",
                    description="Keep v1 self-contained with deterministic local mocks; no live API keys required.",
                    assumption_text="Live third-party payment/email APIs are out of scope for v1; deterministic local mocks are used.",
                ),
                InterviewOption(
                    id="webhook_stub",
                    label="Include webhook & adapter interfaces",
                    description="Generate typed adapter interfaces ready for real keys later.",
                    assumption_text="External integrations use typed adapter stubs with local test doubles.",
                ),
            ],
        ),
        InterviewQuestion(
            id="Q4_PRIVACY_ROUTING",
            question="Where can models run when building your project?",
            recommended_id="hybrid_local_first",
            options=[
                InterviewOption(
                    id="hybrid_local_first",
                    label="Local-first + Strongest model for Spec/Contract/Judge (Recommended)",
                    description="Uses strong cloud (if configured) for spec/contract/judge and local models for workers.",
                    assumption_text="Router policy 'default': strongest available model freezes contract & judges; local-first for workers.",
                ),
                InterviewOption(
                    id="local_strict",
                    label="Strict Local-Only (Never send code or prompts to cloud)",
                    description="100% local execution (`router.policy: local_strict`).",
                    assumption_text="Router policy 'local_strict': no prompts or code leave the local machine.",
                ),
            ],
        ),
        InterviewQuestion(
            id="Q5_AUTONOMY",
            question="How much control do you want during the build?",
            recommended_id="milestones",
            options=[
                InterviewOption(
                    id="milestones",
                    label="Check in at milestones (Recommended)",
                    description="Approve the Plan Card first, then run to completion unless budget or security gates trigger.",
                    assumption_text="Autonomy mode 'milestones': pause at Plan Card and before any external/deploy action.",
                ),
                InterviewOption(
                    id="hands_off",
                    label="Hands-off (One shot start-to-finish)",
                    description="Execute scaffold, workers, verification, and repair loops automatically within budget.",
                    assumption_text="Autonomy mode 'hands_off': execute end-to-end automatically within the approved budget cap.",
                ),
                InterviewOption(
                    id="plan_only",
                    label="Plan & Contract only",
                    description="Stop after generating spec.yaml, contract/, and red-first acceptance tests.",
                    assumption_text="Autonomy mode 'plan_only': freeze spec and contract without running implementation workers.",
                ),
            ],
        ),
    ]
    return questions[:5]


def synthesize_spec_from_brief(
    brief_text: str,
    answers: Optional[Dict[str, str]] = None,
    *,
    max_usd: float = 6.0,
    max_minutes: int = 45,
) -> SpecDocument:
    """Turn a user brief + optional interview answers into a complete SpecDocument.

    Any question omitted from ``answers`` (or when the user skips the interview) is
    resolved using its ``recommended_id`` and recorded in ``spec.assumptions`` with
    ``source='default'``. Answered questions are recorded with ``source='user'``.
    """
    clean_brief = (brief_text or "Web application").strip()
    questions = generate_interview_questions(clean_brief)
    ans_map = dict(answers or {})

    assumptions: List[AssumptionEntry] = []
    chosen: Dict[str, str] = {}
    for idx, q in enumerate(questions, start=1):
        raw_choice = ans_map.get(q.id)
        opt_map = {o.id: o for o in q.options}
        if raw_choice and raw_choice in opt_map:
            opt = opt_map[raw_choice]
            src = "user"
        else:
            opt = opt_map[q.recommended_id]
            src = "default"
        chosen[q.id] = opt.id
        assumptions.append(
            AssumptionEntry(
                id=f"X{idx}",
                text=opt.assumption_text,
                source=src,
                question_id=q.id,
            )
        )

    has_roles = chosen.get("Q1_AUTH_ROLES") == "roles_3"
    roles = ["visitor", "member", "admin"] if has_roles else ["visitor"]
    stack = "web-auth-crud" if has_roles else "web-basic"
    router_policy = "local_strict" if chosen.get("Q4_PRIVACY_ROUTING") == "local_strict" else "default"
    autonomy = chosen.get("Q5_AUTONOMY") or AutonomyLevel.MILESTONES.value

    stories: List[UserStory] = [
        UserStory(
            id="S1",
            as_role="visitor",
            can=f"view the public landing and core items for: {clean_brief}",
            accept="GET /api/items returns HTTP 200 with a JSON list containing at least 1 seeded item; no login required",
            route="/api/items",
            method="GET",
            auth_required=False,
        )
    ]
    if has_roles:
        stories.extend(
            [
                UserStory(
                    id="S2",
                    as_role="member",
                    can="create or book an item tied to my authenticated account",
                    accept=(
                        "POST /api/bookings with a valid member token creates a record (HTTP 201); "
                        "anonymous POST is rejected (HTTP 401); duplicate booking of the same slot is rejected (HTTP 409)"
                    ),
                    route="/api/bookings",
                    method="POST",
                    auth_required=True,
                ),
                UserStory(
                    id="S3",
                    as_role="member",
                    can="view only my own bookings and never another member's private records",
                    accept=(
                        "GET /api/bookings/{id} returns HTTP 200 for the owner and HTTP 403/404 for a different member (IDOR check)"
                    ),
                    route="/api/bookings/{id}",
                    method="GET",
                    auth_required=True,
                ),
                UserStory(
                    id="S4",
                    as_role="admin",
                    can="create new catalog items and view system summary",
                    accept=(
                        "POST /api/items with admin token returns HTTP 201; with member or anonymous token returns HTTP 403/401"
                    ),
                    route="/api/items",
                    method="POST",
                    auth_required=True,
                ),
            ]
        )
    else:
        stories.append(
            UserStory(
                id="S2",
                as_role="visitor",
                can="submit a new entry through the public API with input validation",
                accept="POST /api/items with valid payload returns HTTP 201; empty title returns HTTP 400/422",
                route="/api/items",
                method="POST",
                auth_required=False,
            )
        )

    modules = [
        ModuleSpec(
            name="backend_api",
            description="Database models, RBAC auth middleware, and REST API endpoints matching contract/openapi.yaml",
            owned_globs=["backend/**", "app/api/**", "app/db/**", "app/main.py"],
            depends_on=[],
            estimated_minutes=10.0 if has_roles else 6.0,
        ),
        ModuleSpec(
            name="frontend_ui",
            description="Interactive web UI and client views consuming /api/* endpoints",
            owned_globs=["frontend/**", "app/static/**", "app/templates/**"],
            depends_on=[],
            estimated_minutes=8.0 if has_roles else 5.0,
        ),
    ]

    non_goals = [
        "Live third-party payment processing or external SMTP delivery in v1"
        if chosen.get("Q3_PAYMENTS_INTEGRATIONS") == "mock_out_of_scope"
        else "Production cloud deployment credentials inside agent workspace"
    ]

    return SpecDocument(
        goal=clean_brief,
        roles=roles,
        stories=stories,
        non_goals=non_goals,
        stack=stack,
        assumptions=assumptions,
        modules=modules,
        budget=BudgetSpec(max_usd=max_usd, max_minutes=max_minutes),
        autonomy=autonomy,
        router_policy=router_policy,
    )

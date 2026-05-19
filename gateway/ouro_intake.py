"""Ouroboros-faithful /ouro-intake admission controller.

The command intentionally models only the front half of Ouroboros:
Interview -> Seed draft -> Seed QA/repair -> Kanban admission.

It never starts workers, mutates repos, opens PRs, restarts the gateway, or grants
execution authority.  A Seed admitted to Kanban is source material only; the
Kanban card remains the authority after admission.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_native_admission import next_public_id, normalize_namespace

_ALLOWED_PROJECTS = {
    "bo": "BO",
    "brain": "BO",
    "brain-os": "BO",
    "dc": "DC",
    "dailychingu": "DC",
    "ws": "WS",
    "whystarve": "WS",
    "rs": "RS",
    "risu": "RS",
}

_KEY_ALIASES = {
    "goal": "goal",
    "project": "project",
    "namespace": "project",
    "tenant": "tenant",
    "context": "context",
    "non-goals": "non_goals",
    "non_goals": "non_goals",
    "constraints": "constraints",
    "constraint": "constraints",
    "acceptance": "acceptance_criteria",
    "acceptance-criteria": "acceptance_criteria",
    "acceptance_criteria": "acceptance_criteria",
    "verify": "verification_requirements",
    "verification": "verification_requirements",
    "risks": "risks",
    "questions": "open_questions",
    "routing": "initial_routing",
    "scope": "scope",
    "answer": "answer",
    "session": "session_id",
    "session_id": "session_id",
    "side-effects": "side_effect_boundary_note",
    "side_effects": "side_effect_boundary_note",
}

_VALID_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:")
_SENSITIVE_RE = re.compile(
    r"\b(prod|production|billing|paddle|payment|secret|env|customer|email|refund|deploy|restart|release|migration|cohort)\b",
    re.IGNORECASE,
)
_OBSERVABLE_RE = re.compile(
    r"\b(test|pytest|command|exit code|returns|prints|file|artifact|api|status|readback|proof|verif|확인|검증|테스트)\b",
    re.IGNORECASE,
)
_VAGUE_RE = re.compile(r"\b(improve|better|robust|easy|clean|nice|적당|잘|개선|좋게)\b", re.IGNORECASE)

AMBIGUITY_READY_THRESHOLD = 0.20
SENSITIVE_READY_THRESHOLD = 0.15


@dataclass(frozen=True)
class OuroIntakeResult:
    action: str
    message: str
    mutated: bool = False
    dispatched: bool = False
    task_id: str | None = None
    public_id: str | None = None
    session_id: str | None = None
    error: str | None = None


def _help_message() -> str:
    return (
        "/ouro-intake runs an Ouroboros-faithful Interview -> Seed -> QA/repair -> Kanban admission flow.\n\n"
        "Usage:\n"
        "  /ouro-intake goal:<text> project:<bo|dc|ws|rs> [tenant:<name>] [context:<text>]\n"
        "  /ouro-intake answer session:<id> answer:<text>\n"
        "  /ouro-intake seed session:<id>\n"
        "  /ouro-intake admit session:<id>\n\n"
        "Boundary: admission only. executor_dispatch=forbidden_during_admission; "
        "no worker, repo mutation, PR, gateway restart, secret/env change, or live rollout is approved."
    )


def _split_tokens(raw_args: str) -> list[str]:
    try:
        return shlex.split(raw_args)
    except ValueError:
        # Fall back to whitespace splitting so malformed quotes fail closed as
        # ordinary text rather than crashing the gateway handler.
        return raw_args.split()


def _parse_args(raw_args: str) -> dict[str, Any]:
    values: dict[str, Any] = {"project": "bo", "tenant": "intake"}
    free_goal: list[str] = []
    for token in _split_tokens(raw_args):
        if _VALID_KEY_RE.match(token):
            key, value = token.split(":", 1)
            canonical = _KEY_ALIASES.get(key.strip().lower())
            if canonical:
                values[canonical] = value.strip()
                continue
        free_goal.append(token)
    if not str(values.get("goal") or "").strip() and free_goal:
        values["goal"] = " ".join(free_goal).strip()
    return values


def _namespace_for(project: str) -> str:
    key = (project or "bo").strip().lower()
    ns = _ALLOWED_PROJECTS.get(key, project.strip().upper())
    return normalize_namespace(ns)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_list(value: Any, *, default: list[str] | None = None) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default or [])
    text = str(value).strip()
    parts = [p.strip() for p in re.split(r"[,;|]", text) if p.strip()]
    return parts or [text]


def _session_store_path() -> Any:
    return get_hermes_home() / "ouro_intake_sessions.json"


def _load_sessions() -> dict[str, Any]:
    path = _session_store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_sessions(sessions: dict[str, Any]) -> None:
    path = _session_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _session_id_for(values: dict[str, Any]) -> str:
    raw = f"{time.time_ns()}:{values.get('goal','')}:{values.get('project','bo')}"
    return "oi_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _score_dimension(*, clarity: float, weight: float, name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "clarity_score": round(clarity, 2), "weight": weight, "reason": reason}


def _ambiguity_analysis(values: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic ambiguity score, level, ledger, and next questions.

    This mirrors the parts of upstream Ouroboros that are safe to embed in the
    gateway frontdoor: explicit scored dimensions, per-dimension blockers, and a
    visible ledger.  It does not use Ouroboros's Execute/Ralph authority model.
    """

    goal = str(values.get("goal") or "").strip()
    context = str(values.get("context") or "").strip()
    constraints = _as_list(values.get("constraints"))
    non_goals = _as_list(values.get("non_goals"))
    acceptance = _as_list(values.get("acceptance_criteria"))
    verification = _as_list(values.get("verification_requirements"))
    side_effect_note = str(values.get("side_effect_boundary_note") or "").strip()
    sensitive = bool(_SENSITIVE_RE.search(" ".join([goal, context, side_effect_note])))

    goal_words = goal.split()
    goal_clarity = 0.90
    goal_reason = "goal is specific enough to evaluate"
    if not goal:
        goal_clarity, goal_reason = 0.0, "missing goal"
    elif len(goal_words) < 4:
        goal_clarity, goal_reason = 0.35, "goal is too short to execute safely"
    elif _VAGUE_RE.search(goal):
        goal_clarity, goal_reason = 0.60, "goal contains vague improvement language"

    # A side-effect boundary is more valuable than a generic constraint in this
    # intake flow because it directly preserves the admission/execution split.
    # Count it strongly when present so an answer like "no repo mutation or
    # gateway restart without approval" can close the constraint dimension.
    constraint_signals = len(constraints) + len(non_goals) + (3 if side_effect_note else 0)
    constraint_clarity = min(0.90, 0.35 + 0.18 * constraint_signals)
    constraint_reason = "constraints/non-goals/side-effect boundary are explicit"
    if sensitive and not side_effect_note:
        constraint_clarity = min(constraint_clarity, 0.45)
        constraint_reason = "sensitive domain lacks explicit side-effect boundary"
    elif constraint_signals == 0:
        constraint_reason = "constraints and non-goals are not explicit"

    if not acceptance:
        criteria_clarity, criteria_reason = 0.20, "missing acceptance criteria"
    else:
        observable_count = sum(1 for item in acceptance if _OBSERVABLE_RE.search(item))
        if observable_count:
            # One concrete proof statement is enough to make the acceptance
            # dimension seedable; additional boundary clauses split out of the
            # same answer should not drag it back below the close threshold.
            criteria_clarity = max(0.82, 0.55 + 0.35 * (observable_count / max(len(acceptance), 1)))
            criteria_reason = "acceptance criteria include observable proof"
        else:
            criteria_clarity = 0.55
            criteria_reason = "acceptance criteria are present but not observably testable"

    if context:
        context_clarity = 0.85 if len(context.split()) >= 3 else 0.55
        context_reason = "context supplied"
    else:
        context_clarity, context_reason = 0.25, "missing current repo/runtime/product context"

    dims = [
        _score_dimension(clarity=goal_clarity, weight=0.35, name="goal", reason=goal_reason),
        _score_dimension(clarity=constraint_clarity, weight=0.25, name="constraints", reason=constraint_reason),
        _score_dimension(clarity=criteria_clarity, weight=0.25, name="acceptance_criteria", reason=criteria_reason),
        _score_dimension(clarity=context_clarity, weight=0.15, name="context", reason=context_reason),
    ]
    weighted_clarity = sum(item["clarity_score"] * item["weight"] for item in dims)
    score = round(max(0.0, min(1.0, 1.0 - weighted_clarity)), 4)
    if score <= 0.20:
        level = "clear"
    elif score <= 0.40:
        level = "moderate"
    elif score <= 0.70:
        level = "high"
    else:
        level = "critical"

    flags: list[str] = []
    questions: list[str] = []
    for dim in dims:
        if dim["clarity_score"] < 0.75:
            flags.append(f"{dim['name']}_unclear")
    if not goal:
        questions.append("What concrete outcome should this intake produce?")
    elif goal_clarity < 0.75:
        questions.append("What is the specific deliverable/output, in one sentence, without broad improvement words?")
    if context_clarity < 0.75:
        questions.append("What existing repo/runtime/product context should be considered before implementation?")
    if criteria_clarity < 0.75:
        questions.append("What observable proof should make this accepted as Done?")
    if constraint_clarity < 0.75:
        questions.append("What is explicitly out of scope, and which side effects remain approval-gated?")
    if sensitive:
        flags.append("sensitive_side_effect_domain")
        if not side_effect_note:
            questions.append("This touches a sensitive domain. Which side effects are explicitly allowed, and which are forbidden until a later approval?")

    threshold = SENSITIVE_READY_THRESHOLD if sensitive else AMBIGUITY_READY_THRESHOLD
    return {
        "ambiguity_score": score,
        "ambiguity_level": level,
        "threshold": threshold,
        "sensitive_domain": sensitive,
        "seed_ready": score <= threshold and not questions,
        "ambiguity_ledger": dims,
        "ambiguity_flags": list(dict.fromkeys(flags)),
        "blocking_questions": list(dict.fromkeys(questions))[:4],
    }


def _seed_review(values: dict[str, Any]) -> dict[str, Any]:
    analysis = _ambiguity_analysis(values)
    mode = "seed_ready_for_admission" if analysis["seed_ready"] else "decision_gate_only"
    return {
        "mode": mode,
        "ambiguity_score": analysis["ambiguity_score"],
        "ambiguity_level": analysis["ambiguity_level"],
        "ambiguity_threshold": analysis["threshold"],
        "ambiguity_flags": analysis["ambiguity_flags"],
        "ambiguity_ledger": analysis["ambiguity_ledger"],
        "blocking_questions": analysis["blocking_questions"],
        "dispatch_allowed": False,
    }


def _ontology_for(values: dict[str, Any]) -> dict[str, Any]:
    project = str(values.get("project") or "bo").strip().lower()
    return {
        "name": "OuroIntakeSeed",
        "description": "Admission source material produced by an Ouroboros-style interview before Kanban authority takes over.",
        "fields": [
            {"name": "goal", "type": "string", "description": "Primary requested outcome"},
            {"name": "project", "type": "string", "description": f"Target project namespace ({project})"},
            {"name": "ambiguity_score", "type": "number", "description": "Deterministic interview ambiguity score"},
            {"name": "approval_required_for", "type": "array", "description": "Actions still requiring Chris approval"},
        ],
    }


def _qa_and_repair_seed(seed: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic Seed QA and small structural repairs before admission."""

    findings: list[dict[str, str]] = []
    repairs: list[str] = []
    if not seed.get("goal"):
        findings.append({"code": "missing_goal", "severity": "high", "message": "Seed goal is missing"})
    if not seed.get("acceptance_criteria"):
        findings.append({"code": "missing_acceptance", "severity": "high", "message": "Seed needs observable acceptance criteria"})
    if not seed.get("ontology"):
        seed["ontology"] = _ontology_for(seed)
        repairs.append("added ontology")
    if not seed.get("verification_requirements"):
        seed["verification_requirements"] = ["Kanban readback confirms admission metadata and no task runs exist"]
        repairs.append("added verification requirement")
    vague_acceptance = [item for item in seed.get("acceptance_criteria", []) if not _OBSERVABLE_RE.search(str(item))]
    if vague_acceptance:
        findings.append({"code": "weak_acceptance_observability", "severity": "medium", "message": "Some acceptance criteria lack observable proof language"})
        seed["acceptance_criteria"] = [
            item if _OBSERVABLE_RE.search(str(item)) else f"Observable proof exists for: {item}"
            for item in seed.get("acceptance_criteria", [])
        ]
        repairs.append("made weak acceptance criteria observable")
    passed = not any(f["severity"] == "high" for f in findings)
    return {"passed": passed, "findings": findings, "repairs": repairs, "max_iterations": 1}


def _build_seed_contract(values: dict[str, Any], *, public_id: str | None, actor: str | None, session_id: str | None = None) -> dict[str, Any]:
    goal = str(values.get("goal") or "").strip()
    project = str(values.get("project") or "bo").strip().lower()
    namespace = _namespace_for(project)
    tenant = str(values.get("tenant") or "intake").strip() or "intake"
    context = str(values.get("context") or "").strip()
    review = _seed_review(values)
    non_goals = _as_list(values.get("non_goals"), default=[
        "executor dispatch",
        "repo mutation",
        "PR creation/merge",
        "gateway restart/reload",
        "secret/env mutation",
        "production or customer-visible change",
    ])
    acceptance = _as_list(values.get("acceptance_criteria"), default=[
        "Kanban readback confirms authority/admission metadata and task_runs equals 0",
        "open questions are visible before implementation begins",
        "Chris explicitly approves any execution/routing transition after admission",
    ])
    verification = _as_list(values.get("verification_requirements"), default=[
        "Kanban readback confirms authority/admission metadata",
        "task remains unassigned and non-dispatchable",
        "task_runs is empty for the admission card",
    ])
    risks = _as_list(values.get("risks"), default=[
        "ambiguous goal may require additional Socratic interview before execution",
        "auto-decompose/dispatcher must not treat intake as runnable work",
    ])
    open_questions = _as_list(values.get("open_questions"), default=[
        "What exact scope should be included/excluded?",
        "What proof would make this Done?",
        "Which repo/runtime, if any, becomes relevant after approval?",
    ])
    if review["blocking_questions"]:
        open_questions = list(dict.fromkeys([*review["blocking_questions"], *open_questions]))
    suggested_breakdown = [
        {
            "title": "Clarify Seed Contract until ambiguity flags are resolved",
            "link_type": "hierarchy",
            "dispatch_allowed": False,
        }
    ] if review["mode"] == "decision_gate_only" else []
    seed = {
        "source": "ouro_intake",
        "session_id": session_id,
        "public_id": public_id,
        "created_by": actor or "unknown",
        "created_at": int(time.time()),
        "goal": goal,
        "context": context,
        "non_goals": non_goals,
        "constraints": _as_list(values.get("constraints"), default=["Seed is Kanban admission source material only"]),
        "ontology": _ontology_for(values),
        "ambiguity_score": review["ambiguity_score"],
        "ambiguity_level": review["ambiguity_level"],
        "ambiguity_ledger": review["ambiguity_ledger"],
        "seed_review": review,
        "authority": {
            "after_admission": f"Kanban {public_id}" if public_id else "pending Kanban admission",
            "seed_contract_is_source_material_only": True,
        },
        "scope": str(values.get("scope") or "decision-gate/admission").strip(),
        "side_effect_boundary": {
            "executor_dispatch": "forbidden_during_admission",
            "repo_mutation": False,
            "pull_request": False,
            "gateway_restart_or_reload": False,
            "secret_or_env_mutation": False,
            "prod_or_customer_visible_change": False,
            "note": str(values.get("side_effect_boundary_note") or "").strip(),
        },
        "acceptance_criteria": acceptance,
        "verification_requirements": verification,
        "risks": risks,
        "open_questions": open_questions,
        "suggested_breakdown": suggested_breakdown,
        "initial_routing": {
            "status": "proposed_only",
            "verdict": str(values.get("initial_routing") or "blocked").strip() or "blocked",
            "reason": "Ouroboros-style intake records routing intent only; execution requires Chris approval plus Kanban promotion.",
        },
        "approval_required_for": [
            "worker dispatch",
            "repo/worktree mutation",
            "PR creation/update/merge",
            "gateway restart/reload",
            "secret/env/config mutation",
            "live rollout or customer-visible change",
        ],
        "kanban": {
            "namespace": namespace,
            "tenant": tenant,
            "idempotency_key": f"ouro-intake:{hashlib.sha256(goal.encode('utf-8')).hexdigest()[:16]}",
        },
    }
    seed["seed_qa"] = _qa_and_repair_seed(seed)
    seed["seed_contract_sha256"] = hashlib.sha256(_canonical_json(seed).encode("utf-8")).hexdigest()
    return seed


def _body(seed: dict[str, Any]) -> str:
    return (
        f"Ouroboros-style Seed Contract admission `{seed['public_id']}`.\n\n"
        "This card is the Kanban authority after admission. The Seed Contract below is source material only; "
        "it is not execution approval.\n\n"
        "Admission boundary: `executor_dispatch=forbidden_during_admission`. No worker, repo mutation, PR, "
        "gateway restart/reload, secret/env change, prod/customer-visible change, or live rollout is approved by this intake.\n\n"
        "```json seed_contract\n"
        + json.dumps(seed, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n```\n"
    )


def _readback(conn: Any, task_id: str) -> dict[str, Any]:
    task = kb.get_task(conn, task_id)
    if task is None:
        raise RuntimeError(f"created task did not read back: {task_id}")
    runs = conn.execute("SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()["n"]
    return {
        "task_id": task.id,
        "public_id": task.public_id,
        "status": task.status,
        "assignee": task.assignee,
        "claim_lock": task.claim_lock,
        "worker_pid": task.worker_pid,
        "task_runs": int(runs or 0),
        "admission_executor_dispatch": (task.admission_snapshot or {}).get("executor_dispatch"),
        "routing_status": (task.routing_verdict or {}).get("status"),
    }


def _start_interview(raw_args: str, *, actor: str | None) -> OuroIntakeResult:
    values = _parse_args(raw_args)
    if not str(values.get("goal") or "").strip():
        return OuroIntakeResult(
            action="error",
            error="missing_goal",
            message="Missing goal. Use `/ouro-intake goal:<text> project:<bo|dc|ws|rs>`; no Kanban action was taken.",
        )
    try:
        _namespace_for(str(values.get("project") or "bo"))
    except ValueError as exc:
        return OuroIntakeResult(
            action="error",
            error="invalid_project",
            message=f"Invalid project/namespace: {exc}. No action was taken.",
        )
    session_id = _session_id_for(values)
    review = _seed_review(values)
    seed = _build_seed_contract(values, public_id=None, actor=actor, session_id=session_id)
    sessions = _load_sessions()
    sessions[session_id] = {
        "session_id": session_id,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "actor": actor or "unknown",
        "values": values,
        "rounds": [],
        "seed": seed if review["mode"] == "seed_ready_for_admission" else None,
        "status": "seed_ready" if review["mode"] == "seed_ready_for_admission" else "interviewing",
    }
    _save_sessions(sessions)
    if review["mode"] == "seed_ready_for_admission":
        message = (
            f"Started /ouro-intake interview session {session_id}.\n"
            f"Ambiguity: {review['ambiguity_score']:.2f} ({review['ambiguity_level']}) — Seed draft + QA is ready, but not admitted.\n"
            f"Seed QA passed: {seed['seed_qa']['passed']}.\n"
            f"Next: `/ouro-intake admit session:{session_id}` to create the blocked Kanban admission card."
        )
    else:
        questions = "\n".join(f"- {q}" for q in review["blocking_questions"])
        message = (
            f"Started /ouro-intake interview session {session_id}.\n"
            f"Ambiguity: {review['ambiguity_score']:.2f} ({review['ambiguity_level']}); Seed is decision-gated.\n"
            "Socratic blockers tied to the ambiguity ledger:\n"
            f"{questions}\n"
            f"Reply with `/ouro-intake answer session:{session_id} answer:<answer>`; no Kanban card or worker was created."
        )
    return OuroIntakeResult(action="interview_started", mutated=True, dispatched=False, session_id=session_id, message=message)


def _merge_answer(values: dict[str, Any], parsed: dict[str, Any], free_answer: str) -> dict[str, Any]:
    merged = dict(values)
    for key in ("context", "acceptance_criteria", "verification_requirements", "constraints", "non_goals", "side_effect_boundary_note", "scope"):
        if parsed.get(key):
            existing = str(merged.get(key) or "").strip()
            merged[key] = f"{existing}; {parsed[key]}" if existing else parsed[key]
    answer = str(parsed.get("answer") or free_answer or "").strip()
    if answer:
        existing_context = str(merged.get("context") or "").strip()
        merged["context"] = f"{existing_context}; interview answer: {answer}" if existing_context else f"interview answer: {answer}"
        if _OBSERVABLE_RE.search(answer) and not str(merged.get("acceptance_criteria") or "").strip():
            merged["acceptance_criteria"] = answer
        if re.search(r"\b(no|forbid|forbidden|approval|gate|금지|승인)\b", answer, re.IGNORECASE):
            existing = str(merged.get("side_effect_boundary_note") or "").strip()
            merged["side_effect_boundary_note"] = f"{existing}; {answer}" if existing else answer
    return merged


def _answer_interview(raw_args: str, *, actor: str | None) -> OuroIntakeResult:
    parsed = _parse_args(raw_args)
    session_id = str(parsed.get("session_id") or "").strip()
    if not session_id:
        return OuroIntakeResult(action="error", error="missing_session", message="Missing session. Use `/ouro-intake answer session:<id> answer:<text>`. No action was taken.")
    sessions = _load_sessions()
    session = sessions.get(session_id)
    if not isinstance(session, dict):
        return OuroIntakeResult(action="error", error="unknown_session", message=f"Unknown ouro-intake session {session_id}. No action was taken.")
    values = _merge_answer(dict(session.get("values") or {}), parsed, str(parsed.get("goal") or ""))
    review = _seed_review(values)
    seed = _build_seed_contract(values, public_id=None, actor=actor or session.get("actor"), session_id=session_id)
    session["values"] = values
    session["updated_at"] = int(time.time())
    session.setdefault("rounds", []).append({"at": int(time.time()), "answer": parsed.get("answer") or parsed.get("goal") or "", "review": review})
    session["seed"] = seed if review["mode"] == "seed_ready_for_admission" else None
    session["status"] = "seed_ready" if review["mode"] == "seed_ready_for_admission" else "interviewing"
    sessions[session_id] = session
    _save_sessions(sessions)
    if review["mode"] == "seed_ready_for_admission":
        message = (
            f"Updated /ouro-intake session {session_id}.\n"
            f"Ambiguity: {review['ambiguity_score']:.2f} ({review['ambiguity_level']}) — Seed draft + QA is ready, not admitted.\n"
            f"Seed QA passed: {seed['seed_qa']['passed']}.\n"
            f"Next: `/ouro-intake admit session:{session_id}`."
        )
    else:
        questions = "\n".join(f"- {q}" for q in review["blocking_questions"])
        message = (
            f"Updated /ouro-intake session {session_id}.\n"
            f"Ambiguity: {review['ambiguity_score']:.2f} ({review['ambiguity_level']}); Seed remains decision-gated.\n"
            f"Next Socratic blockers:\n{questions}"
        )
    return OuroIntakeResult(action="interview_updated", mutated=True, dispatched=False, session_id=session_id, message=message)


def _show_or_seed(raw_args: str, *, actor: str | None) -> OuroIntakeResult:
    parsed = _parse_args(raw_args)
    session_id = str(parsed.get("session_id") or "").strip()
    sessions = _load_sessions()
    session = sessions.get(session_id)
    if not session_id or not isinstance(session, dict):
        return OuroIntakeResult(action="error", error="unknown_session", message="Missing or unknown session. No action was taken.")
    values = dict(session.get("values") or {})
    seed = _build_seed_contract(values, public_id=None, actor=actor or session.get("actor"), session_id=session_id)
    session["seed"] = seed
    session["status"] = "seed_ready" if seed["seed_review"]["mode"] == "seed_ready_for_admission" else "decision_gate_only"
    session["updated_at"] = int(time.time())
    sessions[session_id] = session
    _save_sessions(sessions)
    return OuroIntakeResult(
        action="seed_rendered",
        mutated=True,
        dispatched=False,
        session_id=session_id,
        message=(
            f"Seed draft for session {session_id} ({seed['seed_review']['mode']}).\n"
            f"Ambiguity: {seed['ambiguity_score']:.2f} ({seed['ambiguity_level']}); QA passed: {seed['seed_qa']['passed']}.\n"
            "```json seed_contract\n" + json.dumps(seed, indent=2, sort_keys=True, ensure_ascii=False) + "\n```"
        ),
    )


def _admit_seed(raw_args: str) -> OuroIntakeResult:
    parsed = _parse_args(raw_args)
    session_id = str(parsed.get("session_id") or "").strip()
    sessions = _load_sessions()
    session = sessions.get(session_id)
    if not session_id or not isinstance(session, dict):
        return OuroIntakeResult(action="error", error="unknown_session", message="Missing or unknown session. No Kanban action was taken.")
    values = dict(session.get("values") or {})
    with kb.connect() as conn:
        namespace = _namespace_for(str(values.get("project") or "bo"))
        public_id = next_public_id(conn, namespace)
        seed = _build_seed_contract(values, public_id=public_id, actor=session.get("actor"), session_id=session_id)
        title = f"{public_id} — Ouro intake: {seed['goal'][:80]}"
        task_id = kb.create_task(
            conn,
            title=title,
            body=_body(seed),
            assignee=None,
            created_by="ouro-intake",
            workspace_kind="scratch",
            tenant=seed["kanban"]["tenant"],
            priority=0,
            parents=(),
            triage=False,
            idempotency_key=seed["kanban"]["idempotency_key"],
            public_id=public_id,
            skills=None,
            review_phase=None,
            routing_verdict={
                "verdict": "blocked",
                "status": "proposed_only",
                "reason": "Ouro intake admission records a Seed Contract only; execution requires Chris approval and Kanban promotion.",
            },
            admission_snapshot={
                "source": "ouro_intake",
                "session_id": session_id,
                "public_id": public_id,
                "tenant": seed["kanban"]["tenant"],
                "executor_dispatch": "forbidden_during_admission",
                "approval_boundary": "human_approval_required",
                "seed_review_mode": seed["seed_review"]["mode"],
                "ambiguity_score": seed["ambiguity_score"],
                "ambiguity_level": seed["ambiguity_level"],
                "ambiguity_flags": seed["seed_review"]["ambiguity_flags"],
                "seed_qa_passed": seed["seed_qa"]["passed"],
                "seed_contract_sha256": seed["seed_contract_sha256"],
            },
            closeout_evidence={
                "policy": "admission_only_no_execution",
                "evidence_status": "not_started",
            },
        )
        kb.block_task(conn, task_id, reason=None)
        kb.add_comment(
            conn,
            task_id,
            "ouro-intake",
            "Admission-only block: Seed Contract is source material, not executable authority. "
            "executor_dispatch=forbidden_during_admission; Chris approval plus Kanban promotion is required before implementation.",
        )
        rb = _readback(conn, task_id)
    session["admitted_task_id"] = task_id
    session["admitted_public_id"] = public_id
    session["updated_at"] = int(time.time())
    sessions[session_id] = session
    _save_sessions(sessions)
    message = (
        f"Created Ouro intake admission card {public_id} ({task_id}) from session {session_id}.\n"
        f"Status: {rb['status']} | assignee: {rb['assignee'] or 'none'} | task_runs: {rb['task_runs']}\n"
        f"Seed mode: {seed['seed_review']['mode']} | ambiguity: {seed['ambiguity_score']:.2f} ({seed['ambiguity_level']}) | QA passed: {seed['seed_qa']['passed']}\n"
        "Boundary: no worker dispatched; execution remains blocked until Chris explicitly approves a Kanban transition."
    )
    return OuroIntakeResult(
        action="created",
        mutated=True,
        dispatched=False,
        task_id=task_id,
        public_id=public_id,
        session_id=session_id,
        message=message,
    )


def handle_ouro_intake_command(raw_args: str = "", *, actor: str | None = None) -> OuroIntakeResult:
    """Run explicit Ouroboros-style intake without granting execution authority."""

    raw_args = (raw_args or "").strip()
    if not raw_args or raw_args in {"help", "--help", "-h"}:
        return OuroIntakeResult(action="help", message=_help_message())

    tokens = _split_tokens(raw_args)
    subcommand = tokens[0].lower() if tokens else "start"
    # Preserve original quoting in the subcommand remainder. Re-joining shlex
    # tokens would turn `answer:"multi word"` into `answer:multi word`, causing
    # only the first word to bind to the key and losing the actual interview
    # answer that reduces ambiguity.
    rest = raw_args[len(raw_args.split(maxsplit=1)[0]) :].strip() if subcommand in {"start", "answer", "continue", "seed", "show", "admit"} else raw_args

    if subcommand in {"answer", "continue"}:
        return _answer_interview(rest, actor=actor)
    if subcommand in {"seed", "show"}:
        return _show_or_seed(rest, actor=actor)
    if subcommand == "admit":
        return _admit_seed(rest)
    if subcommand == "start":
        return _start_interview(rest, actor=actor)
    return _start_interview(raw_args, actor=actor)

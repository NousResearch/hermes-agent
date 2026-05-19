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
from hermes_integrations.ouroboros_upstream.adapter import (
    build_seed_dict as _vendored_ouroboros_seed_dict,
    record_answer as _vendored_ouroboros_record_answer,
    review_and_repair_seed_dict as _vendored_ouroboros_review_and_repair_seed_dict,
    start_interview_state as _vendored_ouroboros_start_state,
)

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
    "hermes-extraction": "hermes_extraction",
    "hermes_extraction": "hermes_extraction",
    "seed-extraction": "seed_extraction",
    "seed_extraction": "seed_extraction",
    "structured-extraction": "structured_extraction",
    "structured_extraction": "structured_extraction",
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
ACTIVE_SESSION_TTL_SECONDS = 24 * 60 * 60
_ACTIVE_STATUSES = {"interviewing", "restate_pending", "refine_pending"}
_ESCAPE_REPLIES = {"취소", "그만", "중단", "탈출", "나가기", "cancel", "stop", "exit", "quit"}
_AUTOPILOT_AXIS_OPTIONS = {
    "a": "intake/cardization",
    "b": "kanban_execution_prep",
    "c": "git_pr_ci_operations",
    "d": "gateway_cron_runtime_monitoring",
    "e": "other",
}
_AUTOPILOT_AXIS_LABELS = {
    "intake/cardization": "A) intake/cardization",
    "kanban_execution_prep": "B) Kanban execution prep",
    "git_pr_ci_operations": "C) git/PR/CI operations",
    "gateway_cron_runtime_monitoring": "D) gateway/cron/runtime monitoring",
    "other": "E) other",
}


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
        "  /ouro-intake admit session:<id>\n"
        "  /ouro-intake cancel [session:<id>]\n"
        "After start, same-thread replies are captured as answers; reply `그만`/`탈출` to cancel.\n\n"
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


def origin_from_source(source: Any) -> dict[str, Any] | None:
    """Return stable origin fields for binding an intake session to a chat thread.

    Kept in this module so gateway routing and tests use the same matching
    semantics as the controller.  Message ids are intentionally excluded: the
    binding is to the conversation surface and author, not a single message.
    """

    if source is None:
        return None
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    return {
        "platform": str(platform or ""),
        "chat_id": str(getattr(source, "chat_id", "") or ""),
        "thread_id": str(getattr(source, "thread_id", "") or ""),
        "user_id": str(getattr(source, "user_id", "") or ""),
        "user_name": str(getattr(source, "user_name", "") or ""),
        "chat_type": str(getattr(source, "chat_type", "") or ""),
    }


def _origin_key(origin: dict[str, Any] | None) -> str | None:
    if not isinstance(origin, dict):
        return None
    platform = str(origin.get("platform") or "").strip()
    chat_id = str(origin.get("chat_id") or "").strip()
    thread_id = str(origin.get("thread_id") or "").strip()
    user = str(origin.get("user_id") or origin.get("user_name") or "").strip()
    if not platform or not chat_id or not user:
        return None
    return "|".join([platform, chat_id, thread_id, user])


def _active_session_for_origin(origin: dict[str, Any] | None, *, now: int | None = None) -> tuple[str, dict[str, Any]] | None:
    key = _origin_key(origin)
    if key is None:
        return None
    now = int(now or time.time())
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for session_id, session in _load_sessions().items():
        if not isinstance(session, dict):
            continue
        if session.get("status") not in _ACTIVE_STATUSES:
            continue
        binding = session.get("origin_binding") if isinstance(session.get("origin_binding"), dict) else None
        if not binding or binding.get("key") != key:
            continue
        expires_at = int(binding.get("expires_at") or 0)
        if expires_at and expires_at < now:
            continue
        candidates.append((int(session.get("updated_at") or session.get("created_at") or 0), str(session_id), session))
    if not candidates:
        return None
    _, session_id, session = sorted(candidates, reverse=True)[0]
    return session_id, session



def _active_sessions_for_origin(origin: dict[str, Any] | None, *, now: int | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Return every active session bound to this origin, newest first."""

    key = _origin_key(origin)
    if key is None:
        return []
    now = int(now or time.time())
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for session_id, session in _load_sessions().items():
        if not isinstance(session, dict):
            continue
        if session.get("status") not in _ACTIVE_STATUSES:
            continue
        binding = session.get("origin_binding") if isinstance(session.get("origin_binding"), dict) else None
        if not binding or binding.get("key") != key:
            continue
        expires_at = int(binding.get("expires_at") or 0)
        if expires_at and expires_at < now:
            continue
        candidates.append((int(session.get("updated_at") or session.get("created_at") or 0), str(session_id), session))
    return [(sid, sess) for _updated, sid, sess in sorted(candidates, reverse=True)]


def _is_escape_reply(text: str) -> bool:
    normalized = re.sub(r"[.!?。！？\s]+", "", (text or "").strip().lower())
    return normalized in _ESCAPE_REPLIES


def _cancel_sessions(session_ids: list[str], *, actor: str | None, reason: str = "user_cancelled") -> OuroIntakeResult:
    sessions = _load_sessions()
    now = int(time.time())
    cancelled: list[str] = []
    for session_id in session_ids:
        session = sessions.get(session_id)
        if not isinstance(session, dict):
            continue
        if session.get("status") == "cancelled":
            continue
        session["status"] = "cancelled"
        session["phase"] = "cancelled"
        session["cancelled_at"] = now
        session["cancelled_by"] = actor or "unknown"
        session["cancel_reason"] = reason
        session["updated_at"] = now
        binding = session.get("origin_binding") if isinstance(session.get("origin_binding"), dict) else None
        if binding:
            binding["expires_at"] = now - 1
            binding["cancelled_at"] = now
        session.setdefault("turns", []).append({"at": now, "phase": "cancelled", "answer": reason})
        sessions[session_id] = session
        cancelled.append(session_id)
    if cancelled:
        _save_sessions(sessions)
        joined = ", ".join(cancelled)
        return OuroIntakeResult(
            action="cancelled",
            mutated=True,
            dispatched=False,
            session_id=cancelled[0],
            message=f"/ouro-intake session {joined} 취소했습니다. 이후 일반 메시지는 일반 Hermes 대화로 처리됩니다.",
        )
    return OuroIntakeResult(
        action="cancel_noop",
        mutated=False,
        dispatched=False,
        message="취소할 active /ouro-intake 세션이 없습니다. 일반 대화는 이미 정상 라우팅됩니다.",
    )


def _cancel_interview(raw_args: str = "", *, actor: str | None = None, origin: dict[str, Any] | None = None) -> OuroIntakeResult:
    parsed = _parse_args(raw_args)
    session_id = str(parsed.get("session_id") or parsed.get("goal") or "").strip()
    if session_id and session_id.lower() not in {"cancel", "stop", "exit", "quit"}:
        return _cancel_sessions([session_id], actor=actor, reason="explicit_cancel")
    active = _active_sessions_for_origin(origin)
    if active:
        return _cancel_sessions([session_id for session_id, _session in active], actor=actor, reason="origin_cancel")
    return OuroIntakeResult(
        action="cancel_noop",
        mutated=False,
        dispatched=False,
        message="취소할 active /ouro-intake 세션이 없습니다. 일반 대화는 이미 정상 라우팅됩니다.",
    )


def _detect_autopilot_axes(answer: str) -> list[str]:
    text = (answer or "").strip().lower()
    axes: list[str] = []
    for letter, axis in _AUTOPILOT_AXIS_OPTIONS.items():
        if re.search(rf"(?<![a-z]){letter}(?![a-z])", text):
            axes.append(axis)
    if "둘 다" in text or "both" in text:
        axes.extend(["intake/cardization", "kanban_execution_prep"])
    if "intake" in text or "카드" in text or "card" in text:
        axes.append("intake/cardization")
    if "kanban" in text or "실행 준비" in text or "execution prep" in text:
        axes.append("kanban_execution_prep")
    if "git" in text or "pr" in text or "ci" in text:
        axes.append("git_pr_ci_operations")
    if "gateway" in text or "cron" in text or "런타임" in text:
        axes.append("gateway_cron_runtime_monitoring")
    return list(dict.fromkeys(axes))


def _refine_answer(values: dict[str, Any], answer: str, previous_question: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve and structure free-text answers before state updates.

    This mirrors upstream Ouroboros' main-session Refine gate contract in a
    deterministic, gateway-safe form: never collapse meaningful free text into a
    single option label; keep raw text plus structured decision/context fields.
    """

    raw = (answer or "").strip()
    refined: dict[str, Any] = {
        "source": "from-user",
        "refined": bool(raw),
        "raw_answer": raw,
        "decision": raw,
        "reasoning": [],
        "constraints": [],
        "out_of_scope": [],
        "context": [],
        "source_prefix": "[from-user][refined]",
        "needs_confirmation": False,
    }
    if not raw:
        return refined
    qid = (previous_question or {}).get("id")
    if qid == "autopilot_axis" or re.search(r"오토파일럿|autopilot", " ".join(str(values.get(k) or "") for k in ("goal", "scope")), re.IGNORECASE):
        axes = _detect_autopilot_axes(raw)
        if axes:
            refined["decision"] = " + ".join(_AUTOPILOT_AXIS_LABELS.get(axis, axis) for axis in axes)
            refined["scope_axes"] = axes
            refined["scope_confidence"] = "compound" if len(axes) > 1 else "single"
            refined["needs_confirmation"] = True
            if {"intake/cardization", "kanban_execution_prep"}.issubset(set(axes)):
                refined["reasoning"].append("User wants both intake/cardization and Kanban execution preparation, not a single-axis choice.")
                refined["constraints"].append("This scope does not by itself approve worker dispatch or repo mutation.")
                refined["out_of_scope"].extend([
                    "git/PR/CI operations unless Chris later includes them",
                    "gateway/cron/runtime monitoring unless Chris later includes it",
                ])
    if _OBSERVABLE_RE.search(raw):
        refined["acceptance_criteria"] = raw
        refined["needs_confirmation"] = True
    if re.search(r"\b(no|forbid|forbidden|approval|gate|금지|승인|하지마|말고)\b", raw, re.IGNORECASE):
        refined["constraints"].append(raw)
        refined["side_effect_boundary_note"] = raw
        refined["needs_confirmation"] = True
    return refined


def _is_refine_approval(answer: str) -> bool:
    text = (answer or "").strip().lower()
    return bool(re.search(r"\b(send as-is|send|approve|approved|confirm|confirmed|yes|y|ok|okay)\b|승인|그대로|보내|맞아|확인|좋아", text))


def _format_refined_payload(refinement: dict[str, Any]) -> str:
    def _section(title: str, value: Any) -> str:
        if isinstance(value, list):
            body = "\n".join(f"- {item}" for item in value) if value else "- None"
        else:
            body = str(value or "None")
        return f"{title}:\n{body}"

    return "\n\n".join([
        str(refinement.get("source_prefix") or "[from-user][refined]"),
        _section("Decision", refinement.get("decision")),
        _section("Reasoning", refinement.get("reasoning") or []),
        _section("Constraints", refinement.get("constraints") or []),
        _section("Out of scope", refinement.get("out_of_scope") or []),
        _section("Codebase context", refinement.get("context") or []),
    ])


def _format_refine_gate(session_id: str, refinement: dict[str, Any]) -> str:
    return (
        f"/ouro-intake session {session_id} 답변을 upstream Refine gate 형식으로 구조화했습니다.\n\n"
        f"{_format_refined_payload(refinement)}\n\n"
        "누락되거나 왜곡된 게 없으면 `승인`이라고 답해주세요. "
        "수정할 내용이 있으면 그대로 적어주세요. 승인 전에는 다음 질문으로 넘기지 않습니다."
    )


def _needs_refine_gate(refinement: dict[str, Any], *, previous_question: dict[str, Any] | None = None, restate_correction: bool = False) -> bool:
    if restate_correction:
        return True
    return bool(refinement.get("needs_confirmation"))


def _hermes_gateway_seed_closer_fallback(values: dict[str, Any], review: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hermes gateway fallback for upstream Seed Closer material-decision audit.

    BO-062 keeps this clearly named as a fallback, not a claimed full upstream
    port. The canonical upstream Seed quality gate now comes from the vendored
    SeedReviewer/SeedRepairer path attached to the Seed contract.
    """

    goal = str(values.get("goal") or "")
    context = str(values.get("context") or "")
    scope = str(values.get("scope") or "")
    combined = " ".join([goal, context, scope]).lower()
    blockers: list[str] = []
    questions: list[str] = []
    is_system = any(token in combined for token in ("brownfield", "system-level", "migration", "protocol redesign"))
    if is_system:
        facts = " ".join(str(values.get(key) or "") for key in ("context", "constraints", "side_effect_boundary_note", "acceptance_criteria", "first_slice"))
        checks = [
            ("ownership/SSOT", r"ownership|owner|ssot|authority|source of truth|소유|권한"),
            ("API/protocol contract", r"api|protocol|contract|command|slash|route|handler|dispatch"),
            ("lifecycle/recovery", r"lifecycle|recovery|rollback|cancel|resume|restart|복구|취소"),
            ("verification", r"pytest|test|검증|readback|proof|exit code|확인"),
        ]
        for label, pattern in checks:
            if not re.search(pattern, facts, re.IGNORECASE):
                blockers.append(f"{label} is not explicit enough for Seed Closer")
        if blockers:
            questions.append("Seed Closer 기준으로 ownership/SSOT, API/protocol, lifecycle/recovery, verification 중 빠진 실행 변경 결정을 먼저 확인해야 합니다. 어떤 경계가 맞나요?")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "questions": questions,
    }


def handle_ouro_intake_plain_reply(text: str, *, origin: dict[str, Any] | None = None, actor: str | None = None) -> OuroIntakeResult | None:
    """Route a normal same-thread/user message into an active intake interview.

    Returns None when no active bound /ouro-intake session exists, allowing the
    gateway to continue normal agent routing. Slash commands are never captured
    here; explicit /ouro-intake seed/admit/cancel style controls stay commands.
    """

    answer = (text or "").strip()
    if not answer or answer.startswith("/"):
        return None
    active = _active_session_for_origin(origin)
    if active is None:
        return None
    if _is_escape_reply(answer):
        return _cancel_interview(actor=actor, origin=origin)
    session_id, _session = active
    return _answer_interview(f"session:{session_id} answer:{shlex.quote(answer)}", actor=actor)


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
    scope_axes = values.get("scope_axes") if isinstance(values.get("scope_axes"), list) else []
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
    elif scope_axes:
        context_clarity, context_reason = 0.70, "structured scope decision supplied; concrete runtime context still useful"
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
    closer = _hermes_gateway_seed_closer_fallback(values, analysis)
    seed_ready = analysis["seed_ready"] and closer["ready"]
    mode = "seed_ready_for_admission" if seed_ready else "decision_gate_only"
    return {
        "mode": mode,
        "ambiguity_score": analysis["ambiguity_score"],
        "ambiguity_level": analysis["ambiguity_level"],
        "ambiguity_threshold": analysis["threshold"],
        "ambiguity_flags": analysis["ambiguity_flags"],
        "ambiguity_ledger": analysis["ambiguity_ledger"],
        "blocking_questions": list(dict.fromkeys([*analysis["blocking_questions"], *closer["questions"]])),
        "seed_closer": closer,
        "dispatch_allowed": False,
    }



def _detect_language(values: dict[str, Any]) -> str:
    text = " ".join(str(values.get(key) or "") for key in ("goal", "context", "scope", "answer"))
    return "ko" if re.search(r"[가-힣]", text) else "en"


def _infer_track(values: dict[str, Any]) -> str:
    text = " ".join(str(values.get(key) or "") for key in ("goal", "context", "scope"))
    lowered = text.lower()
    if re.search(r"오토파일럿|autopilot", text, re.IGNORECASE) and not values.get("scope_axes"):
        return "autopilot"
    if "gateway" in lowered or "discord" in lowered or "command" in lowered:
        return "trigger_surface"
    if any(word in lowered for word in ("billing", "paddle", "payment", "prod", "production")):
        return "side_effect_boundary"
    return "scope"


def _restate_goal(values: dict[str, Any]) -> str:
    goal = str(values.get("goal") or "").strip()
    scope = str(values.get("scope") or "").strip()
    axes = values.get("scope_axes") if isinstance(values.get("scope_axes"), list) else []
    context = str(values.get("context") or "").strip()
    if axes:
        base = f"{goal} — scope: {' + '.join(_AUTOPILOT_AXIS_LABELS.get(axis, axis) for axis in axes)}"
    elif scope:
        base = f"{goal} — scope: {scope}"
    elif context:
        base = f"{goal} — context: {context[:160]}"
    else:
        base = goal
    return base[:260]


def _is_restate_approval(answer: str) -> bool:
    text = (answer or "").strip().lower()
    return bool(re.search(r"\b(yes|y|approve|approved|confirm|confirmed|correct|ok|okay)\b|맞아|승인|확인|그대로|좋아|진행", text))


def _hermes_gateway_fallback_question(values: dict[str, Any], review: dict[str, Any], *, previous_track: str | None = None) -> dict[str, Any]:
    language = _detect_language(values)
    track = _infer_track(values)
    flags = set(review.get("ambiguity_flags") or [])
    blockers = review.get("blocking_questions") or []
    if previous_track == track and blockers:
        # Keep turns moving across independent ambiguity tracks instead of asking
        # the same broad question twice.
        order = ["authority", "verification", "non_goals", "brownfield_context", "scope"]
        track = next((candidate for candidate in order if candidate != previous_track), track)

    if track == "autopilot":
        if language == "ko":
            text = (
                "오토파일럿이라고 할 때, 먼저 자동화하려는 축이 어느 쪽이에요? "
                "복수 선택도 괜찮습니다: A) intake/카드화 자동화, B) Kanban 실행 준비 자동화, "
                "C) git·PR·CI 운영 자동화, D) gateway·cron·런타임 감시 자동화, E) 다른 범위."
            )
        else:
            text = (
                "When you say autopilot, which axis should it automate first? Multiple axes are fine: "
                "A) intake/cardization, B) Kanban execution prep, C) git/PR/CI operations, "
                "D) gateway/cron/runtime monitoring, or E) another scope."
            )
        options = ["intake/cardization", "Kanban execution prep", "git/PR/CI operations", "gateway/cron/runtime monitoring", "other"]
        return {"id": "autopilot_axis", "track": "scope", "text": text, "options": options}

    if values.get("scope_axes") and not str(values.get("first_slice") or "").strip():
        text = (
            "좋아요. A/B를 함께 본다면 첫 버전은 어디까지를 완료로 볼까요? 예: ‘요청 → refined Seed 초안 → Kanban admission/실행준비 카드까지, 실제 worker dispatch는 제외’."
            if language == "ko"
            else "Good. If this is a combined scope, where should v1 stop? For example: request -> refined Seed draft -> Kanban admission/execution-prep card, excluding worker dispatch."
        )
        return {"id": "first_slice", "track": "scope", "text": text, "options": []}

    closer = review.get("seed_closer") if isinstance(review.get("seed_closer"), dict) else {"ready": True, "questions": []}
    if not closer.get("ready"):
        question_text = (closer.get("questions") or ["What material Seed Closer decision is still unresolved?"])[0]
        return {"id": "seed_closer_material_gap", "track": "closure", "text": question_text, "options": []}

    if "goal_unclear" in flags:
        text = "결과물을 한 문장으로 좁히면 무엇인가요?" if language == "ko" else "What concrete output should this produce, in one sentence?"
        return {"id": "goal_output", "track": "scope", "text": text, "options": []}
    if "context_unclear" in flags:
        text = "어느 기존 시스템·repo·런타임 맥락을 먼저 기준으로 삼아야 해요?" if language == "ko" else "Which existing repo/runtime/product context should be treated as the baseline?"
        return {"id": "brownfield_context", "track": "brownfield_context", "text": text, "options": []}
    if "acceptance_criteria_unclear" in flags:
        text = "완료 판정은 어떤 관측 가능한 증거 하나로 할까요?" if language == "ko" else "What single observable proof should make this Done?"
        return {"id": "observable_proof", "track": "verification", "text": text, "options": []}
    if "constraints_unclear" in flags or "sensitive_side_effect_domain" in flags:
        text = "이번 Seed에서 금지할 부작용은 무엇인가요? 예: worker dispatch, PR, gateway restart, secret/env 변경." if language == "ko" else "Which side effects are forbidden for this Seed? For example: worker dispatch, PR, gateway restart, secret/env mutation."
        return {"id": "side_effect_boundary", "track": "authority", "text": text, "options": []}
    text = "제가 이해한 목표를 한 문장으로 재확인할게요. 이 문장으로 Seed를 만들어도 되나요?" if language == "ko" else "I will restate the goal in one sentence before Seed. Is this correct?"
    return {"id": "restate_confirmation", "track": "restate", "text": text, "options": []}


def _format_interview_question(session_id: str, review: dict[str, Any], question: dict[str, Any]) -> str:
    return (
        f"/ouro-intake 인터뷰를 시작했습니다 (`{session_id}`).\n"
        "답은 이 thread에 그냥 평문으로 보내면 됩니다. 중단하려면 `/ouro-intake cancel` 또는 `그만`이라고 보내세요.\n\n"
        f"질문: {question['text']}\n\n"
        "아직 Kanban 카드나 worker는 만들지 않았습니다."
    )


def _format_next_question(session_id: str, review: dict[str, Any], question: dict[str, Any]) -> str:
    return (
        f"Updated /ouro-intake session {session_id}.\n"
        "좋아요. 답변은 구조화해서 반영했습니다.\n\n"
        f"다음 질문: {question['text']}"
    )


def _format_restate(session_id: str, values: dict[str, Any], review: dict[str, Any]) -> str:
    restatement = _restate_goal(values)
    return (
        f"Updated /ouro-intake session {session_id}.\n"
        "제가 이해한 목표를 한 문장으로 확인할게요.\n\n"
        f"Restate: {restatement}\n\n"
        "맞으면 `승인`이라고 답해주세요. 아니면 수정할 내용을 그대로 보내면 됩니다. Seed는 승인 전까지 막혀 있습니다."
    )

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
    """Run deterministic Seed QA and explicit repair/adoption checks."""

    findings: list[dict[str, str]] = []
    safe_repairs: list[str] = []
    proposed_repairs: list[dict[str, str]] = []
    adoption_required: list[str] = []
    if not seed.get("goal"):
        findings.append({"code": "missing_goal", "severity": "high", "message": "Seed goal is missing"})
    if not seed.get("acceptance_criteria"):
        findings.append({"code": "missing_acceptance", "severity": "high", "message": "Seed needs observable acceptance criteria"})
    if not seed.get("ontology"):
        seed["ontology"] = _ontology_for(seed)
        safe_repairs.append("added ontology")
    if not seed.get("verification_requirements"):
        seed["verification_requirements"] = ["Kanban readback confirms admission metadata and no task runs exist"]
        safe_repairs.append("added verification requirement")
    vague_acceptance = [item for item in seed.get("acceptance_criteria", []) if not _OBSERVABLE_RE.search(str(item))]
    if vague_acceptance:
        findings.append({"code": "weak_acceptance_observability", "severity": "medium", "message": "Some acceptance criteria lack observable proof language"})
        for item in vague_acceptance:
            proposed_repairs.append({
                "code": "make_acceptance_observable",
                "from": str(item),
                "proposal": f"Define observable proof for: {item}",
                "adoption": "requires_user_or_followup_gate",
            })
        adoption_required.append("weak_acceptance_observability")
    passed = not any(f["severity"] == "high" for f in findings)
    return {
        "passed": passed,
        "findings": findings,
        "safe_repairs": safe_repairs,
        "proposed_repairs": proposed_repairs,
        "adoption_required": adoption_required,
        "repairs": safe_repairs,
        "max_iterations": 1,
    }


def _upstream_seed_projection(values: dict[str, Any], review: dict[str, Any], *, session_id: str | None) -> dict[str, Any]:
    """Build the upstream Seed payload through the vendored Ouroboros model.

    BO-062 intentionally stops hand-rolling a lookalike dict here.  The return
    value is produced by the vendored upstream `Seed` model and round-tripped
    through `Seed.from_dict()`, then wrapped by Hermes/Kanban authority fields
    outside this object.
    """

    return _vendored_ouroboros_seed_dict(values, review, session_id=session_id)


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
    upstream_seed = _upstream_seed_projection(values, review, session_id=session_id)
    upstream_auto = _vendored_ouroboros_review_and_repair_seed_dict(upstream_seed)
    upstream_seed = upstream_auto["seed"]
    extraction_source = "hermes_agent_structured_extraction" if any(
        isinstance(values.get(key), str) and values.get(key, "").strip()
        for key in ("hermes_extraction", "seed_extraction", "structured_extraction")
    ) else "hermes_gateway_confirmed_values"
    seed = {
        "source": "ouro_intake",
        "session_id": session_id,
        "public_id": public_id,
        "created_by": actor or "unknown",
        "created_at": int(time.time()),
        "goal": goal,
        "context": context,
        "interview_refinements": values.get("refined_answers", []),
        "seed_extraction": {
            "source": extraction_source,
            "parser": "vendored_q00_ouroboros_seed_generator",
            "agent_prompt": "hermes_integrations/ouroboros_upstream/agents/hermes-seed-extractor.md",
            "provider_call": False,
        },
        "upstream_seed": upstream_seed,
        "upstream_auto_review": upstream_auto["review"],
        "upstream_auto_repair_history": upstream_auto["repair_history"],
        "non_goals": non_goals,
        "constraints": _as_list(values.get("constraints"), default=["Seed is Kanban admission source material only"]),
        "ontology": _ontology_for(values),
        "ambiguity_score": review["ambiguity_score"],
        "ambiguity_level": review["ambiguity_level"],
        "ambiguity_ledger": review["ambiguity_ledger"],
        "seed_review": review,
        "seed_closer": review.get("seed_closer"),
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


def _start_interview(raw_args: str, *, actor: str | None, origin: dict[str, Any] | None = None) -> OuroIntakeResult:
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
    sessions = _load_sessions()
    status = "restate_pending" if review["mode"] == "seed_ready_for_admission" else "interviewing"
    question = _hermes_gateway_fallback_question(values, review)
    sessions[session_id] = {
        "session_id": session_id,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "actor": actor or "unknown",
        "values": values,
        "turns": [],
        "rounds": [],
        "last_question": question if status == "interviewing" else {"id": "restate_confirmation", "track": "restate", "text": _restate_goal(values), "options": []},
        "upstream_interview_state": _vendored_ouroboros_start_state(
            session_id,
            values,
            question if status == "interviewing" else {"text": _restate_goal(values)},
        ),
        "upstream_interview_provider": "vendored_q00_ouroboros_subset",
        "track": question["track"] if status == "interviewing" else "restate",
        "language": _detect_language(values),
        "phase": status,
        "seed": None,
        "status": status,
        "restate": _restate_goal(values) if status == "restate_pending" else None,
        "ambiguity_ledger": review["ambiguity_ledger"],
        "seed_closer": review.get("seed_closer"),
        "dialectic": {"non_user_streak": 0},
        "origin_binding": {
            "key": _origin_key(origin),
            "origin": origin or {},
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + ACTIVE_SESSION_TTL_SECONDS,
        } if _origin_key(origin) else None,
    }
    _save_sessions(sessions)
    if status == "restate_pending":
        message = _format_restate(session_id, values, review)
    else:
        message = _format_interview_question(session_id, review, question)
    return OuroIntakeResult(action="interview_started", mutated=True, dispatched=False, session_id=session_id, message=message)


def _merge_answer(values: dict[str, Any], parsed: dict[str, Any], free_answer: str, refinement: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(values)
    for key in ("context", "acceptance_criteria", "verification_requirements", "constraints", "non_goals", "side_effect_boundary_note", "scope"):
        if parsed.get(key):
            existing = str(merged.get(key) or "").strip()
            merged[key] = f"{existing}; {parsed[key]}" if existing else parsed[key]
    answer = str(parsed.get("answer") or free_answer or "").strip()
    if answer:
        existing_context = str(merged.get("context") or "").strip()
        merged["context"] = f"{existing_context}; interview answer: {answer}" if existing_context else f"interview answer: {answer}"
    if refinement and refinement.get("raw_answer"):
        merged.setdefault("refined_answers", [])
        merged["refined_answers"] = [*list(merged.get("refined_answers") or []), refinement]
        axes = refinement.get("scope_axes") if isinstance(refinement.get("scope_axes"), list) else []
        if axes:
            existing_axes = list(merged.get("scope_axes") or []) if isinstance(merged.get("scope_axes"), list) else []
            merged["scope_axes"] = list(dict.fromkeys([*existing_axes, *axes]))
            merged["scope"] = " + ".join(_AUTOPILOT_AXIS_LABELS.get(axis, axis) for axis in merged["scope_axes"])
        if refinement.get("acceptance_criteria") and not str(merged.get("acceptance_criteria") or "").strip():
            merged["acceptance_criteria"] = refinement["acceptance_criteria"]
        for key in ("constraints", "non_goals"):
            items = refinement.get("constraints" if key == "constraints" else "out_of_scope")
            if items:
                existing = str(merged.get(key) or "").strip()
                addition = "; ".join(str(item) for item in items if str(item).strip())
                merged[key] = f"{existing}; {addition}" if existing else addition
        if refinement.get("side_effect_boundary_note"):
            existing = str(merged.get("side_effect_boundary_note") or "").strip()
            merged["side_effect_boundary_note"] = f"{existing}; {refinement['side_effect_boundary_note']}" if existing else refinement["side_effect_boundary_note"]
    if answer:
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
    answer_text = str(parsed.get("answer") or parsed.get("goal") or "").strip()
    previous_question = session.get("last_question") if isinstance(session.get("last_question"), dict) else None
    if session.get("status") == "refine_pending":
        if _is_refine_approval(answer_text):
            pending_refinement = dict(session.get("pending_refinement") or {})
            pending_answer = str(session.get("pending_answer_text") or pending_refinement.get("raw_answer") or "")
            pending_parsed = dict(session.get("pending_parsed") or {})
            pending_question = session.get("pending_previous_question") if isinstance(session.get("pending_previous_question"), dict) else previous_question
            base_values = dict(session.get("values") or {})
            values = _merge_answer(base_values, pending_parsed, pending_answer, refinement=pending_refinement)
            if (pending_question or {}).get("id") == "first_slice" and pending_answer:
                values["first_slice"] = pending_answer
            review = _seed_review(values)
            session["values"] = values
            session["updated_at"] = int(time.time())
            session.setdefault("turns", []).append({"at": int(time.time()), "question": pending_question, "answer": pending_answer, "refined_answer": pending_refinement, "review": review})
            session.setdefault("rounds", []).append({"at": int(time.time()), "answer": pending_answer, "review": review})
            session["language"] = _detect_language(values)
            session["ambiguity_ledger"] = review["ambiguity_ledger"]
            session["seed_closer"] = review.get("seed_closer")
            session["seed"] = None
            session.pop("pending_refinement", None)
            session.pop("pending_answer_text", None)
            session.pop("pending_parsed", None)
            session.pop("pending_previous_question", None)
            if review["mode"] == "seed_ready_for_admission":
                session["status"] = "restate_pending"
                session["phase"] = "restate_pending"
                session["restate"] = _restate_goal(values)
                session["last_question"] = {"id": "restate_confirmation", "track": "restate", "text": session["restate"], "options": []}
                session["upstream_interview_state"] = _vendored_ouroboros_record_answer(
                    session.get("upstream_interview_state"),
                    pending_answer,
                    session["last_question"],
                    ambiguity_score=review["ambiguity_score"],
                    ambiguity_breakdown=review,
                )
                message = _format_restate(session_id, values, review)
            else:
                question = _hermes_gateway_fallback_question(values, review, previous_track=(pending_question or {}).get("track"))
                session["status"] = "interviewing"
                session["phase"] = "interviewing"
                session["track"] = question["track"]
                session["last_question"] = question
                session["upstream_interview_state"] = _vendored_ouroboros_record_answer(
                    session.get("upstream_interview_state"),
                    pending_answer,
                    question,
                    ambiguity_score=review["ambiguity_score"],
                    ambiguity_breakdown=review,
                )
                message = _format_next_question(session_id, review, question)
            sessions[session_id] = session
            _save_sessions(sessions)
            return OuroIntakeResult(action="interview_updated", mutated=True, dispatched=False, session_id=session_id, message=message)
        refinement = _refine_answer(dict(session.get("values") or {}), answer_text, previous_question)
        refinement["needs_confirmation"] = True
        session["pending_refinement"] = refinement
        session["pending_answer_text"] = answer_text
        session["pending_parsed"] = parsed
        session["updated_at"] = int(time.time())
        sessions[session_id] = session
        _save_sessions(sessions)
        return OuroIntakeResult(action="refine_pending", mutated=True, dispatched=False, session_id=session_id, message=_format_refine_gate(session_id, refinement))
    if session.get("status") == "restate_pending" and _is_restate_approval(answer_text):
        values = dict(session.get("values") or {})
        review = _seed_review(values)
        seed = _build_seed_contract(values, public_id=None, actor=actor or session.get("actor"), session_id=session_id)
        session.setdefault("turns", []).append({"at": int(time.time()), "question": previous_question, "answer": answer_text, "phase": "restate_approved"})
        session.setdefault("rounds", []).append({"at": int(time.time()), "answer": answer_text, "review": review})
        session["seed"] = seed
        session["status"] = "seed_ready"
        session["phase"] = "seed_ready"
        session["upstream_interview_state"] = _vendored_ouroboros_record_answer(
            session.get("upstream_interview_state"),
            answer_text,
            None,
            ambiguity_score=review["ambiguity_score"],
            ambiguity_breakdown=review,
            completed=True,
        )
        session["updated_at"] = int(time.time())
        sessions[session_id] = session
        _save_sessions(sessions)
        message = (
            f"Updated /ouro-intake session {session_id}.\n"
            f"Restate approved. Seed draft + QA is ready, not admitted.\n"
            f"Seed QA passed: {seed['seed_qa']['passed']}.\n"
            f"Next: `/ouro-intake seed session:{session_id}` to inspect, then `/ouro-intake admit session:{session_id}` to create the blocked Kanban admission card."
        )
        return OuroIntakeResult(action="interview_updated", mutated=True, dispatched=False, session_id=session_id, message=message)

    if session.get("status") == "restate_pending" and answer_text and not _is_restate_approval(answer_text):
        base_values = dict(session.get("values") or {})
        refinement = _refine_answer(base_values, answer_text, previous_question)
        refinement["restate_correction"] = True
        refinement["needs_confirmation"] = True
        session["status"] = "refine_pending"
        session["phase"] = "refine_pending"
        session["pending_refinement"] = refinement
        session["pending_answer_text"] = answer_text
        session["pending_parsed"] = parsed
        session["pending_previous_question"] = previous_question
        session["updated_at"] = int(time.time())
        sessions[session_id] = session
        _save_sessions(sessions)
        return OuroIntakeResult(action="refine_pending", mutated=True, dispatched=False, session_id=session_id, message=_format_refine_gate(session_id, refinement))

    base_values = dict(session.get("values") or {})
    refinement = _refine_answer(base_values, answer_text, previous_question)
    if _needs_refine_gate(refinement, previous_question=previous_question):
        session["status"] = "refine_pending"
        session["phase"] = "refine_pending"
        session["pending_refinement"] = refinement
        session["pending_answer_text"] = answer_text
        session["pending_parsed"] = parsed
        session["pending_previous_question"] = previous_question
        session["updated_at"] = int(time.time())
        sessions[session_id] = session
        _save_sessions(sessions)
        return OuroIntakeResult(action="refine_pending", mutated=True, dispatched=False, session_id=session_id, message=_format_refine_gate(session_id, refinement))

    values = _merge_answer(base_values, parsed, str(parsed.get("goal") or ""), refinement=refinement)
    if (previous_question or {}).get("id") == "first_slice" and answer_text:
        values["first_slice"] = answer_text
    review = _seed_review(values)
    session["values"] = values
    session["updated_at"] = int(time.time())
    session.setdefault("turns", []).append({"at": int(time.time()), "question": previous_question, "answer": answer_text, "refined_answer": refinement, "review": review})
    session.setdefault("rounds", []).append({"at": int(time.time()), "answer": answer_text, "review": review})
    session["language"] = _detect_language(values)
    session["ambiguity_ledger"] = review["ambiguity_ledger"]
    session["seed"] = None
    if review["mode"] == "seed_ready_for_admission":
        session["status"] = "restate_pending"
        session["phase"] = "restate_pending"
        session["restate"] = _restate_goal(values)
        session["last_question"] = {"id": "restate_confirmation", "track": "restate", "text": session["restate"], "options": []}
        session["upstream_interview_state"] = _vendored_ouroboros_record_answer(
            session.get("upstream_interview_state"),
            answer_text,
            session["last_question"],
            ambiguity_score=review["ambiguity_score"],
            ambiguity_breakdown=review,
        )
        message = _format_restate(session_id, values, review)
    else:
        question = _hermes_gateway_fallback_question(values, review, previous_track=(previous_question or {}).get("track"))
        session["status"] = "interviewing"
        session["phase"] = "interviewing"
        session["track"] = question["track"]
        session["last_question"] = question
        session["upstream_interview_state"] = _vendored_ouroboros_record_answer(
            session.get("upstream_interview_state"),
            answer_text,
            question,
            ambiguity_score=review["ambiguity_score"],
            ambiguity_breakdown=review,
        )
        message = _format_next_question(session_id, review, question)
    sessions[session_id] = session
    _save_sessions(sessions)
    return OuroIntakeResult(action="interview_updated", mutated=True, dispatched=False, session_id=session_id, message=message)


def _show_or_seed(raw_args: str, *, actor: str | None) -> OuroIntakeResult:
    parsed = _parse_args(raw_args)
    session_id = str(parsed.get("session_id") or "").strip()
    sessions = _load_sessions()
    session = sessions.get(session_id)
    if not session_id or not isinstance(session, dict):
        return OuroIntakeResult(action="error", error="unknown_session", message="Missing or unknown session. No action was taken.")
    if session.get("status") != "seed_ready":
        restate = session.get("restate") or _restate_goal(dict(session.get("values") or {}))
        return OuroIntakeResult(
            action="seed_blocked",
            mutated=False,
            dispatched=False,
            session_id=session_id,
            message=(
                f"Seed is blocked for session {session_id}. Restate gate has not been approved.\n"
                f"Restate: {restate}\n"
                "맞으면 `승인`이라고 답해주세요. 아직 Kanban action은 하지 않았습니다."
            ),
        )
    values = dict(session.get("values") or {})
    seed = session.get("seed") if isinstance(session.get("seed"), dict) else _build_seed_contract(values, public_id=None, actor=actor or session.get("actor"), session_id=session_id)
    session["seed"] = seed
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
    if session.get("status") != "seed_ready":
        return OuroIntakeResult(
            action="admission_blocked",
            error="restate_not_approved",
            mutated=False,
            dispatched=False,
            session_id=session_id,
            message=f"Session {session_id} is not seed-ready. Approve the Restate gate before admission; no Kanban action was taken.",
        )
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


def handle_ouro_intake_command(raw_args: str = "", *, actor: str | None = None, origin: dict[str, Any] | None = None) -> OuroIntakeResult:
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
    rest = raw_args[len(raw_args.split(maxsplit=1)[0]) :].strip() if subcommand in {"start", "answer", "continue", "seed", "show", "admit", "cancel", "stop", "exit", "quit"} else raw_args

    if subcommand in {"answer", "continue"}:
        return _answer_interview(rest, actor=actor)
    if subcommand in {"seed", "show"}:
        return _show_or_seed(rest, actor=actor)
    if subcommand in {"cancel", "stop", "exit", "quit"}:
        return _cancel_interview(rest, actor=actor, origin=origin)
    if subcommand == "admit":
        return _admit_seed(rest)
    if subcommand == "start":
        return _start_interview(rest, actor=actor, origin=origin)
    return _start_interview(raw_args, actor=actor, origin=origin)

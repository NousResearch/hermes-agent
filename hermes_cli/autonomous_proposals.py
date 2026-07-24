"""Read-only autonomous proposal engine.

This module turns observable signals into candidate Team & Proposte records.
Version 2 is deliberately rules-based: it normalises explicit signal fields,
keeps supporter and critic views separate, and refuses any dispatch-oriented
state. It does not import Kanban, cron, worker, gateway, or messaging modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional
import hashlib
import re

_ALLOWED_CONFIDENCE = {"high", "medium", "low"}
_ALLOWED_KIND = {"operative", "evolution"}
_SAFE_GATE_STATES = {"review_required", "needs_revision", "parked"}
_FORBIDDEN_SIDE_EFFECT_FIELDS = {
    "auto_spawned",
    "cron_created",
    "external_send",
    "kanban_mutated",
    "subagent_spawned",
}
_FORBIDDEN_TASK_FIELDS = {
    "task_id",
    "plan_task_id",
    "plan_child_task_ids",
    "created_task_id",
    "created_task_ids",
    "dispatch_id",
    "worker_id",
    "cron_job_id",
}


class AutonomousProposalSafetyError(ValueError):
    """Raised when a candidate proposal contains dispatch-capable state."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or fallback


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = _text(value).lower()
    return text if text in allowed else fallback


def _profile_label(value: Any, fallback: str) -> str:
    text = _text(value, fallback)
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-") or fallback


def _view(actor: Any, rationale: Any, *, role: str, profile: Optional[Any] = None) -> Dict[str, str]:
    actor_text = _text(actor, f"Unknown {role}")
    return {
        "role": role,
        "actor": actor_text,
        "profile": _profile_label(profile or actor_text, f"unknown-{role}"),
        "rationale": _text(rationale, "No rationale supplied."),
        "method": "rules_based_v2",
    }


def _evidence_refs(signal: Mapping[str, Any]) -> List[str]:
    raw = signal.get("evidence_refs")
    if isinstance(raw, list):
        refs = [_text(item) for item in raw]
    elif raw:
        refs = [_text(raw)]
    else:
        refs = []
    if not refs:
        for key in ("source_ref", "source_key", "id", "title"):
            value = _text(signal.get(key))
            if value:
                refs.append(value)
                break
    return [ref for ref in refs if ref]


def _signal_contract(signal: Mapping[str, Any], summary: str, now: str) -> Dict[str, str]:
    return {
        "summary": summary,
        "source_type": _text(signal.get("source_type"), "registry"),
        "source_ref": _text(signal.get("source_ref") or signal.get("source_key") or signal.get("id") or signal.get("title"), "unknown"),
        "observed_at": _text(signal.get("observed_at"), now),
    }


def _interpretation_contract(signal: Mapping[str, Any], hypothesis: str) -> Dict[str, str]:
    return {
        "hypothesis": hypothesis,
        "expected_benefit": _choice(signal.get("benefit"), _ALLOWED_CONFIDENCE, "medium"),
        "effort": _choice(signal.get("effort"), _ALLOWED_CONFIDENCE, "medium"),
        "risk": _choice(signal.get("risk"), _ALLOWED_CONFIDENCE, "low"),
    }


def _chief_contract(signal: Mapping[str, Any], synthesis: str) -> Dict[str, Any]:
    return {
        "recommendation": _choice(signal.get("recommendation"), {"do_now", "prepare", "park", "reject"}, "prepare"),
        "synthesis": synthesis,
        "decision_needed": _text(signal.get("decision_needed"), "Daniele: approvare, modificare, indirizzare o scartare?"),
        "acceptance": _text(signal.get("acceptance") or synthesis, "Acceptance criteria pending human review."),
        "unresolved_questions": [str(item) for item in signal.get("unresolved_questions") or []],
    }


def _gate_contract() -> Dict[str, Any]:
    return {
        "state": "review_pending",
        "requires_daniele": True,
        "approved_by": None,
        "approved_at": None,
        "safe_actions_without_approval": [
            "show proposal",
            "edit proposal",
            "request chief review",
            "preview ready kanban plan",
        ],
        "forbidden_without_approval": [
            "dispatch task",
            "start worker",
            "start cron",
            "send external message",
            "modify production config",
            "create ready executable tasks",
        ],
    }


def _evidence_contract(refs: List[str], signal: Mapping[str, Any], fallback_excerpt: str, confidence: str) -> Dict[str, Any]:
    excerpts = signal.get("evidence_excerpts")
    if not isinstance(excerpts, list):
        excerpts = [_text(signal.get("evidence") or fallback_excerpt)[:180]]
    return {
        "refs": refs,
        "excerpts": [_text(item)[:180] for item in excerpts if _text(item)],
        "confidence": confidence,
    }


def assert_autonomous_proposal_record_safe(record: Mapping[str, Any]) -> None:
    """Fail closed if a candidate proposal tries to dispatch work.

    Autonomous proposal generation is allowed to prepare review/gated records
    only. Any task/cron/worker/message mutation belongs behind a later explicit
    human approval path, not in this engine.
    """

    gate_state = _text(record.get("gate_state"), "review_required")
    if gate_state not in _SAFE_GATE_STATES:
        raise AutonomousProposalSafetyError(f"unsafe autonomous proposal gate_state={gate_state!r}")

    for field in _FORBIDDEN_SIDE_EFFECT_FIELDS:
        if bool(record.get(field)):
            raise AutonomousProposalSafetyError(f"unsafe autonomous proposal side effect field {field}=true")

    if record.get("no_auto_dispatch") is not True:
        raise AutonomousProposalSafetyError("autonomous proposal must set no_auto_dispatch=true")

    gate = record.get("gate")
    if isinstance(gate, Mapping):
        if gate.get("requires_daniele") is not True:
            raise AutonomousProposalSafetyError("autonomous proposal gate must require Daniele")
        forbidden = set(gate.get("forbidden_without_approval") or [])
        if not {"dispatch task", "start cron", "create ready executable tasks"}.issubset(forbidden):
            raise AutonomousProposalSafetyError("autonomous proposal gate must forbid dispatch, cron, and ready task creation")

    for field in _FORBIDDEN_TASK_FIELDS:
        value = record.get(field)
        if value:
            raise AutonomousProposalSafetyError(f"unsafe autonomous proposal dispatch/task field {field} present")


def build_autonomous_proposal_records(
    signals: Iterable[Mapping[str, Any]],
    *,
    kind: Optional[str] = None,
    limit: Optional[int] = None,
    generated_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build candidate proposal records from observable signal dictionaries.

    The input is intentionally plain data so callers can feed registry counts,
    blocker summaries, radar findings, or future evidence snapshots without the
    engine learning how to dispatch or mutate external systems. V2 is
    rules-based/heuristic: caller-provided signal/supporter/critic fields are
    normalised into one auditable record shape.
    """

    now = generated_at or _utc_now_iso()
    records: List[Dict[str, Any]] = []
    max_items = max(0, limit) if limit is not None else None

    for raw in signals:
        if max_items is not None and len(records) >= max_items:
            break
        title = _text(raw.get("title"), "Autonomous proposal candidate")
        record_kind = _choice(kind or raw.get("kind"), _ALLOWED_KIND, "evolution")
        signal_summary = _text(raw.get("signal") or raw.get("source_signal") or raw.get("evidence") or raw.get("whyNow"))
        interpretation_text = _text(raw.get("interpretation") or raw.get("whyNow") or raw.get("acceptance"), "Interpretation pending human review.")
        supporter_actor = raw.get("supporter") or raw.get("supporter_agent") or raw.get("source_agent") or "ops"
        critic_actor = raw.get("critic") or raw.get("critic_agent") or "reliability"
        supporter_text = raw.get("supporter_view") or raw.get("support") or interpretation_text
        critic_text = raw.get("critic_view") or raw.get("challenge") or "Verify evidence, noise, and no-dispatch guardrails before approval."
        chief_synthesis_text = _text(raw.get("chief_synthesis") or raw.get("synthesis"), "Chief synthesis pending human review.")
        confidence = _choice(raw.get("confidence"), _ALLOWED_CONFIDENCE, "medium")
        refs = _evidence_refs(raw)
        source_key = _text(raw.get("source_key") or raw.get("id") or _slug(title))
        source_agent = _text(raw.get("source_agent"), "ops")
        recommendation = _choice(raw.get("recommendation"), {"do_now", "prepare", "park", "reject"}, "prepare")
        benefit = _choice(raw.get("benefit"), _ALLOWED_CONFIDENCE, "medium")
        effort = _choice(raw.get("effort"), _ALLOWED_CONFIDENCE, "medium")
        risk = _choice(raw.get("risk"), _ALLOWED_CONFIDENCE, "low")
        priority = _text(raw.get("priority"), "P2").upper()
        if priority not in {"P0", "P1", "P2", "P3"}:
            priority = "P2"

        supporter = _view(supporter_actor, supporter_text, role="supporter", profile=raw.get("supporter_profile"))
        critic = _view(critic_actor, critic_text, role="critic", profile=raw.get("critic_profile"))
        chief_synthesis = _chief_contract(raw, chief_synthesis_text)
        evidence = _evidence_contract(refs, raw, signal_summary, confidence)

        record: Dict[str, Any] = {
            "record_type": "autonomous_proposal_candidate",
            "schema_version": "autonomous_proposal.v2",
            "id": f"autonomous:{_slug(source_key)}",
            "title": title,
            "kind": record_kind,
            "origin": _text(raw.get("origin"), "Autonomous Proposal Engine: rules_based_v2"),
            "category": _text(raw.get("category")) or None,
            "status": "review_pending",
            "signal": _signal_contract(raw, signal_summary, now),
            "source_signal": signal_summary,
            "interpretation": _interpretation_contract(raw, interpretation_text),
            "supporter": supporter,
            "critic": critic,
            "supporter_view": supporter,
            "critic_view": critic,
            "chief_synthesis": chief_synthesis,
            "gate_state": _text(raw.get("gate_state"), "review_required"),
            "gate": _gate_contract(),
            "source_agent": source_agent,
            "source_agent_contract": {
                "profile": source_agent,
                "status": "pending_profile_verification",
                "legacy_label": None,
            },
            "confidence": confidence,
            "evidence_refs": refs,
            "evidence_contract": evidence,
            "evidence": _text(raw.get("evidence") or signal_summary),
            "whyNow": _text(raw.get("whyNow") or interpretation_text),
            "acceptance": chief_synthesis["acceptance"],
            "source_key": source_key,
            "suggested_next_action": _text(raw.get("suggested_next_action")) or None,
            "benefit": benefit,
            "effort": effort,
            "risk": risk,
            "priority": priority,
            "recommendation": recommendation,
            "challenge": {
                "supporter": _text(supporter_actor, "ops"),
                "support": _text(supporter_text, interpretation_text),
                "critic": _text(critic_actor, "reliability"),
                "challenge": _text(critic_text, "Review guardrails before approval."),
                "chief_synthesis": chief_synthesis_text,
                "veto_risk": _text(raw.get("veto_risk"), "none"),
                "method": "rules_based_v2",
            },
            "autonomy_level": "proposal_only",
            "autonomy_gate": "approval_required",
            "no_auto_dispatch": True,
            "external_send": False,
            "auto_spawned": False,
            "cron_created": False,
            "kanban_mutated": False,
            "subagent_spawned": False,
            "conversion": {
                "plan_task_id": None,
                "child_task_ids": [],
                "created_by": None,
                "board": None,
                "initial_status": None,
                "converted_at": None,
            },
            "engine": {
                "name": "autonomous_proposal_engine",
                "version": "v2",
                "method": "rules_based_v2",
                "generated_at": now,
                "limits": [
                    "heuristic/rules-based v2",
                    "no autonomous dispatch",
                    "candidate records require human review gate",
                ],
            },
        }
        assert_autonomous_proposal_record_safe(record)
        records.append(record)

    return records

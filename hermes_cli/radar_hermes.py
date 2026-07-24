"""Read-only Radar Hermes snapshot builder.

The dashboard endpoint uses this module as the UI-neutral source of truth for
ranking and normalising Hermes evolution proposals.  It intentionally has no
Kanban, cron, network, or messaging side effects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import re

_ALLOWED_ACTIONS = ["preview_spec", "approve_for_spec", "approve_for_kanban", "park", "reject"]
_SIDE_EFFECTS = {
    "kanban_mutated": False,
    "cron_created": False,
    "external_send": False,
    "subagent_spawned": False,
}
_APPROVAL_POLICY = {
    "read_only_first": True,
    "requires_explicit_approval_for_kanban": True,
    "preview_does_not_dispatch": True,
    "create_children_does_not_force_dispatch": True,
}

_PRIORITY_POINTS = {"P0": 40, "P1": 30, "P2": 18, "P3": 8}
_LEVEL_POINTS = {"high": 25, "medium": 14, "low": 5}
_CONFIDENCE_POINTS = {"high": 15, "medium": 9, "low": 3}
_EFFORT_PENALTY = {"low": 0, "medium": 6, "high": 14}
_RISK_PENALTY = {"low": 0, "medium": 8, "high": 18}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _choice(value: Any, allowed: Iterable[str], fallback: str) -> str:
    text = _text(value).lower()
    allowed_set = set(allowed)
    return text if text in allowed_set else fallback


def _priority(value: Any) -> str:
    text = _text(value).upper()
    return text if text in _PRIORITY_POINTS else "P2"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _excerpt(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", _text(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _source_excerpt(proposal: Dict[str, Any]) -> str:
    challenge = proposal.get("challenge") or {}
    return _excerpt(
        challenge.get("chief_synthesis")
        or challenge.get("challenge")
        or proposal.get("evidence")
        or proposal.get("whyNow")
        or proposal.get("acceptance")
    )


def _is_controversial(proposal: Dict[str, Any]) -> bool:
    challenge = proposal.get("challenge") or {}
    veto = _text(challenge.get("veto_risk")).lower()
    if challenge and veto not in {"", "none", "low"}:
        return True
    if _choice(proposal.get("risk"), {"high", "medium", "low"}, "low") == "high":
        return True
    if _text(proposal.get("recommendation")).lower() == "reject":
        return True
    return False


def _is_parkable(proposal: Dict[str, Any]) -> bool:
    return _text(proposal.get("status")).lower() == "parcheggiata" or _text(proposal.get("recommendation")).lower() == "park"


def _approval_state(proposal: Dict[str, Any]) -> str:
    status = _text(proposal.get("status")).lower()
    chief_status = _text(proposal.get("chief_review_status")).lower()
    if status == "trasformata_in_task":
        return "done"
    if status == "scartata" or chief_status == "rejected":
        return "rejected"
    if status == "parcheggiata" or chief_status == "deferred":
        return "parked"
    if status == "approvata":
        return "approved_for_spec"
    if status == "raccomandata" or chief_status == "shortlisted":
        return "needs_review"
    return "candidate"


def _suggested_assignee(proposal: Dict[str, Any]) -> str:
    for value in (proposal.get("suggested_assignee"), proposal.get("source_agent")):
        text = _text(value)
        if text:
            return text
    origin = _text(proposal.get("origin")).lower()
    if "reliability" in origin or "systems" in origin:
        return "reliability"
    if "evidence" in origin:
        return "evidence"
    if "legal" in origin or "claims" in origin:
        return "legal"
    if "mrv" in origin:
        return "mrv"
    if "ops" in origin:
        return "ops"
    return "needs_routing"


def _score_breakdown(proposal: Dict[str, Any]) -> Dict[str, int]:
    priority = _priority(proposal.get("priority"))
    impact = _choice(proposal.get("benefit"), {"high", "medium", "low"}, "medium")
    confidence = _choice(proposal.get("confidence"), {"high", "medium", "low"}, "medium")
    effort = _choice(proposal.get("effort"), {"high", "medium", "low"}, "medium")
    risk = _choice(proposal.get("risk"), {"high", "medium", "low"}, "low")
    evidence = 10 if _text(proposal.get("evidence") or proposal.get("source_signal") or proposal.get("acceptance")) else 3
    freshness = 8 if _text(proposal.get("last_signal_at") or proposal.get("updated_at") or proposal.get("created_at")) else 0
    park_penalty = 24 if _is_parkable(proposal) else 0
    return {
        "priority": _PRIORITY_POINTS[priority],
        "impact": _LEVEL_POINTS[impact],
        "confidence": _CONFIDENCE_POINTS[confidence],
        "freshness": freshness,
        "evidence": evidence,
        "effort_penalty": _EFFORT_PENALTY[effort],
        "risk_penalty": _RISK_PENALTY[risk],
        "park_penalty": park_penalty,
    }


def _ranking_score(proposal: Dict[str, Any]) -> int:
    explicit = proposal.get("chief_review_score")
    explicit_score: Optional[int]
    if explicit is None:
        explicit_score = None
    else:
        try:
            explicit_score = int(explicit)
        except (TypeError, ValueError):
            explicit_score = None
    if explicit_score is not None:
        base = max(0, min(100, explicit_score))
        if _is_parkable(proposal):
            base -= 24
        return max(0, base)
    parts = _score_breakdown(proposal)
    return max(
        0,
        min(
            100,
            parts["priority"]
            + parts["impact"]
            + parts["confidence"]
            + parts["freshness"]
            + parts["evidence"]
            - parts["effort_penalty"]
            - parts["risk_penalty"]
            - parts["park_penalty"],
        ),
    )


def _normalise_team_proposal(proposal: Dict[str, Any], *, block: str, rank: int, path: str, observed_at: str) -> Dict[str, Any]:
    raw_id = _text(proposal.get("id") or proposal.get("source_key") or proposal.get("title"), "proposal")
    title = _text(proposal.get("title"), "Proposta evolutiva Hermes")
    priority = _priority(proposal.get("priority"))
    impact = _choice(proposal.get("benefit"), {"high", "medium", "low"}, "medium")
    effort = _choice(proposal.get("effort"), {"high", "medium", "low"}, "medium")
    risk = _choice(proposal.get("risk"), {"high", "medium", "low"}, "low")
    confidence = _choice(proposal.get("confidence"), {"high", "medium", "low"}, "medium")
    score_breakdown = _score_breakdown(proposal)
    score = _ranking_score(proposal)
    excerpt = _source_excerpt(proposal)
    return {
        "id": f"radar:team_proposals:{_slug(raw_id)}",
        "source": {
            "kind": "team_proposals",
            "source_id": raw_id,
            "path": path,
            "source_key": proposal.get("source_key"),
            "excerpt": excerpt,
        },
        "title": title,
        "rationale": _excerpt(proposal.get("whyNow") or proposal.get("acceptance")),
        "evidence": [
            {
                "type": "team_proposal",
                "ref": raw_id,
                "summary": excerpt,
                "confidence": confidence,
            }
        ],
        "priority": {
            "label": priority,
            "score": score,
            "impact": impact,
            "effort": effort,
            "risk": risk,
            "confidence": confidence,
        },
        "flags": {
            "controversial": block == "controversial" or _is_controversial(proposal),
            "parkable": _is_parkable(proposal),
            "source_gap": not bool(excerpt),
            "already_done": _approval_state(proposal) == "done",
            "requires_review": True,
            "no_auto_dispatch": True,
        },
        "suggested_assignee": _suggested_assignee(proposal),
        "approval_state": _approval_state(proposal),
        "approval": {
            "allowed_actions": list(_ALLOWED_ACTIONS),
            "preview_available": True,
            "kanban_creation_available": False,
            "requires_explicit_confirmation": True,
        },
        "ranking": {
            "block": block,
            "rank": rank,
            "score": score,
            "score_breakdown": score_breakdown,
        },
        "timestamps": {
            "created_at": proposal.get("created_at"),
            "updated_at": proposal.get("updated_at"),
            "last_signal_at": proposal.get("last_signal_at"),
            "source_observed_at": observed_at,
        },
        "governance": {
            "read_only_surface": True,
            "no_cron_created": True,
            "no_external_send": True,
            "no_subagent_spawned": True,
            "kanban_mutation_requires_approval": True,
        },
    }


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for proposal in candidates:
        key = _text(proposal.get("source_key") or proposal.get("id"))
        if not key:
            key = _slug(_text(proposal.get("title"), "proposal"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(proposal)
    return unique


def build_radar_hermes_snapshot(
    team_proposals_data: Dict[str, Any],
    *,
    generated_at: Optional[str] = None,
    source_path: str = "team_proposals",
    top_limit: int = 5,
) -> Dict[str, Any]:
    """Build the read-only ``radar_hermes.v1`` dashboard snapshot.

    The first slice intentionally reads only the already-loaded Team & Proposte
    registry.  Supplemental JSONL sources can be merged later behind this same
    contract without changing React rendering logic.
    """
    now = generated_at or _utc_now_iso()
    observed_at = _text(team_proposals_data.get("updated_at"), now)
    proposals = team_proposals_data.get("proposals") or []
    evolution = [p for p in proposals if isinstance(p, dict) and p.get("kind") == "evolution"]
    evolution = [p for p in evolution if _approval_state(p) not in {"done", "rejected"}]
    evolution = _dedupe_candidates(evolution)
    ranked: List[Tuple[int, str, Dict[str, Any]]] = [(_ranking_score(p), _text(p.get("last_signal_at") or p.get("updated_at")), p) for p in evolution]
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    ranked_top = [item for item in ranked if not _is_parkable(item[2])]
    top_source = [item[2] for item in ranked_top[:top_limit]]
    top = [
        _normalise_team_proposal(proposal, block="top", rank=idx, path=source_path, observed_at=observed_at)
        for idx, proposal in enumerate(top_source, start=1)
    ]

    controversial_source = next((item[2] for item in ranked if _is_controversial(item[2])), None)
    controversial = (
        _normalise_team_proposal(controversial_source, block="controversial", rank=1, path=source_path, observed_at=observed_at)
        if controversial_source
        else None
    )
    controversy_state = {
        "status": "has_controversy" if controversial else "insufficient_controversy",
        "title": "Proposta controversa qualificata" if controversial else "Nessuna proposta controversa qualificata",
        "message": (
            "La card controversa deriva da challenge, veto risk materiale, rischio alto o raccomandazione di reject."
            if controversial
            else "Le fonti lette non contengono una proposta con challenge, veto risk materiale, rischio alto o raccomandazione di reject. Nessuna top proposta viene rilabelizzata come controversa."
        ),
    }
    parkable_ranked = [item for item in ranked if _is_parkable(item[2])]
    parkable_source = parkable_ranked[0][2] if parkable_ranked else None
    parkable = (
        _normalise_team_proposal(parkable_source, block="parkable", rank=1, path=source_path, observed_at=observed_at)
        if parkable_source
        else None
    )

    returned_ids = {item["id"] for item in top}
    if controversial:
        returned_ids.add(controversial["id"])
    if parkable:
        returned_ids.add(parkable["id"])

    empty_state = None
    if not returned_ids:
        empty_state = {
            "title": "Nessuna proposta Radar source-grounded pronta",
            "message": "Le fonti lette non contengono micro-slice Hermes con evidenza sufficiente. Nessuna card o automazione è stata creata.",
        }

    return {
        "version": "radar_hermes.v1",
        "generated_at": now,
        "source_summary": {
            "sources_read": ["team_proposals"],
            "proposals_seen": len(evolution),
            "proposals_returned": len(returned_ids),
            "side_effects": dict(_SIDE_EFFECTS),
        },
        "blocks": {
            "top": top,
            "controversial": controversial,
            "parkable": parkable,
        },
        "controversy_state": controversy_state,
        "approval_policy": dict(_APPROVAL_POLICY),
        "empty_state": empty_state,
    }

"""Turn-local context for an explicitly active StudyOS learning Session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plugins.study_os import tools as legacy
from plugins.study_os.runtime import (
    LearningRuntimeError,
    active_session_for_conversation,
    active_vault_for_conversation,
)


MAX_ACTIVE_CONTEXT_CHARS = 2800


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _candidate_vaults(conversation_session_id: str) -> list[Path]:
    candidates: list[Path] = []
    bound = active_vault_for_conversation(conversation_session_id)
    if bound is not None:
        candidates.append(bound)
    try:
        fallback = legacy.resolve_vault_path()
    except (FileNotFoundError, ValueError):
        fallback = None
    if fallback is not None and fallback not in candidates:
        candidates.append(fallback)
    return candidates


def _context_payload(session: dict[str, Any], *, include_details: bool = True) -> dict[str, Any]:
    contract_value = session.get("contract")
    contract: dict[str, Any] = contract_value if isinstance(contract_value, dict) else {}
    activity_value = session.get("current_activity")
    activity: dict[str, Any] = activity_value if isinstance(activity_value, dict) else {}
    current_activity: dict[str, Any] = {
        "activity_id": activity.get("activity_id"),
        "activity_adapter": activity.get("activity_adapter"),
            "kind": activity.get("kind"),
            "evidence_target": activity.get("evidence_target"),
            "assistance_level": activity.get("assistance_level"),
        "evidence_requirements": list(activity.get("evidence_requirements", [])),
        "instructions": _clip(activity.get("instructions"), 600),
        "response_policy": _clip(activity.get("response_policy"), 300),
        "reason": _clip(activity.get("reason"), 350),
    }
    payload: dict[str, Any] = {
        "session_id": session.get("session_id"),
        "project_id": session.get("project_id"),
        "mode": contract.get("mode"),
        "objective": _clip(contract.get("objective"), 600),
        "objective_ids": list(contract.get("objective_ids", []))[:12],
        "assistance_level": contract.get("assistance_level"),
        "required_evidence": list(contract.get("evidence_targets", [])),
        "recorded_evidence_ids": list(session.get("evidence_ids", []))[-20:],
        "current_activity": current_activity,
    }
    if include_details:
        current_activity["rubric_requirements"] = [
            _clip(item, 140) for item in activity.get("rubric_requirements", [])[:4]
        ]
        current_activity["source_anchors"] = [
            {
                "kind": anchor.get("kind"),
                "ref": _clip(anchor.get("ref"), 160),
                "locator": _clip(anchor.get("locator"), 100),
            }
            for anchor in activity.get("source_anchors", [])[:3]
            if isinstance(anchor, dict)
        ]
    return payload


def _render_context(session: dict[str, Any]) -> str:
    prefix = (
        "[StudyOS active learning session — turn-local context]\n"
        "This is workflow state, not proof of mastery. Follow the assistance level, collect the learner's "
        "own response before feedback, and record evaluated evidence with study_coach.advance.\n"
    )
    context = prefix + json.dumps(_context_payload(session), ensure_ascii=False, separators=(",", ":"))
    if len(context) > MAX_ACTIVE_CONTEXT_CHARS:
        context = prefix + json.dumps(
            _context_payload(session, include_details=False),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return context if len(context) <= MAX_ACTIVE_CONTEXT_CHARS else context[: MAX_ACTIVE_CONTEXT_CHARS - 1] + "…"


def active_learning_context(*, session_id: str = "", **_kwargs: Any) -> dict[str, str] | None:
    """Build cache-safe user-message context for one bound Hermes conversation."""

    conversation_id = str(session_id or "").strip()
    if not conversation_id:
        return None
    for vault in _candidate_vaults(conversation_id):
        try:
            session = active_session_for_conversation(vault, conversation_id)
        except (LearningRuntimeError, OSError, ValueError):
            continue
        if session is not None:
            return {"context": _render_context(session)}
    return None

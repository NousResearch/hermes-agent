"""Phase-C routing decision records and mixed-provider usage summaries."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

ROUTING_CONTRACT_VERSION = "phase-c-v1"
LEGACY_ROUTING_PROVENANCE = "legacy_unknown"
logger = logging.getLogger(__name__)

_FALLBACK_REASON_MAP = {
    "quota": "quota",
    "auth": "auth",
    "auth_permanent": "auth",
    "billing": "quota",
    "rate_limit": "rate_limit",
    "upstream_rate_limit": "rate_limit",
    "timeout": "timeout",
    "model_not_found": "model_unavailable",
    "content_policy_blocked": "policy_fallback",
    "provider_policy_blocked": "policy_fallback",
    "policy_fallback": "policy_fallback",
    "overloaded": "provider_error",
    "provider_error": "provider_error",
    "server_error": "provider_error",
    "ssl_cert_verification": "provider_error",
    "model_unavailable": "model_unavailable",
    "unknown": "unknown",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _optional_text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _decision_id(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "route_" + hashlib.sha256(encoded).hexdigest()[:24]


def normalize_fallback_reason(reason: Any) -> str:
    """Map the runtime classifier to the stable Phase-C fallback taxonomy."""
    raw = getattr(reason, "value", reason)
    key = _optional_text(raw)
    return _FALLBACK_REASON_MAP.get(key.lower() if key else "unknown", "unknown")


def build_routing_decision(
    *,
    task_id: Optional[str],
    board: Optional[str],
    task_type: Optional[str],
    capability: Optional[str],
    risk: Optional[str],
    code_change: Optional[bool],
    independent_review: Optional[bool],
    preferred_profile: Optional[str],
    reviewer_profile: Optional[str],
    selected_profile: Optional[str],
    selected_provider: Optional[str],
    selected_model: Optional[str],
    human_gate_required: Optional[bool],
    independence_valid: Optional[bool],
    policy_digest: Optional[str],
    selected_at: Optional[str] = None,
    selected_by: Optional[str] = None,
    run_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build one stable initial decision without guessing unavailable policy fields."""
    selected_at = _optional_text(selected_at) or _utc_now()
    identity = {
        "routing_contract_version": ROUTING_CONTRACT_VERSION,
        "task_id": _optional_text(task_id),
        "board": _optional_text(board),
        "run_id": run_id,
        "session_id": _optional_text(session_id),
        "selected_profile": _optional_text(selected_profile),
        "initial_provider": _optional_text(selected_provider),
        "initial_model": _optional_text(selected_model),
    }
    decision = {
        "routing_contract_version": ROUTING_CONTRACT_VERSION,
        "decision_id": _decision_id(identity),
        "task_id": identity["task_id"],
        "board": identity["board"],
        "task_type": _optional_text(task_type),
        "capability": _optional_text(capability),
        "risk": _optional_text(risk),
        "code_change": code_change if isinstance(code_change, bool) else None,
        "independent_review": independent_review if isinstance(independent_review, bool) else None,
        "preferred_profile": _optional_text(preferred_profile),
        "reviewer_profile": _optional_text(reviewer_profile),
        "selected_profile": identity["selected_profile"],
        "initial_provider": identity["initial_provider"],
        "initial_model": identity["initial_model"],
        "selected_provider": identity["initial_provider"],
        "selected_model": identity["initial_model"],
        "fallback_used": False,
        "fallback_from_provider": None,
        "fallback_from_model": None,
        "fallback_reason": None,
        "human_gate_required": human_gate_required if isinstance(human_gate_required, bool) else None,
        "independence_valid": independence_valid if isinstance(independence_valid, bool) else None,
        "policy_digest": _optional_text(policy_digest),
        "selected_at": selected_at,
        "selected_by": _optional_text(selected_by),
        "run_id": run_id,
        "session_id": identity["session_id"],
        "routing_history": [
            {
                "phase": "initial",
                "provider": identity["initial_provider"],
                "model": identity["initial_model"],
                "reason": None,
                "recorded_at": selected_at,
            }
        ],
    }
    return decision


def record_fallback(
    decision: Mapping[str, Any],
    *,
    from_provider: Optional[str],
    from_model: Optional[str],
    to_provider: Optional[str],
    to_model: Optional[str],
    reason: Any,
    recorded_at: Optional[str] = None,
) -> dict[str, Any]:
    """Return a decision updated with an append-only effective fallback transition."""
    updated = copy.deepcopy(dict(decision))
    category = normalize_fallback_reason(reason)
    from_provider = _optional_text(from_provider)
    from_model = _optional_text(from_model)
    to_provider = _optional_text(to_provider)
    to_model = _optional_text(to_model)
    if not updated.get("fallback_used"):
        updated["fallback_from_provider"] = from_provider
        updated["fallback_from_model"] = from_model
    updated["fallback_used"] = True
    updated["selected_provider"] = to_provider
    updated["selected_model"] = to_model
    updated["fallback_reason"] = category
    history = list(updated.get("routing_history") or [])
    history.append(
        {
            "phase": "fallback",
            "provider": to_provider,
            "model": to_model,
            "reason": category,
            "recorded_at": _optional_text(recorded_at) or _utc_now(),
            "from_provider": from_provider,
            "from_model": from_model,
        }
    )
    updated["routing_history"] = history
    return updated


def evaluate_reviewer_independence(
    *,
    implementation_profile: Optional[str],
    reviewer_profile: Optional[str],
    independent_review: bool,
    implementation_provider: Optional[str] = None,
    reviewer_provider: Optional[str] = None,
) -> bool:
    """Fail closed for required self-review while keeping implementation failover lane-local."""
    if not independent_review:
        return True
    implementer = (_optional_text(implementation_profile) or "").casefold()
    reviewer = (_optional_text(reviewer_profile) or "").casefold()
    if not implementer or not reviewer or implementer == reviewer:
        return False
    required_pair = {
        "rozmilo-codex": "rozmilo-claude",
        "rozmilo-claude": "rozmilo-codex",
    }
    if implementer in required_pair and reviewer != required_pair[implementer]:
        return False
    impl_provider = (_optional_text(implementation_provider) or "").casefold()
    review_provider = (_optional_text(reviewer_provider) or "").casefold()
    if impl_provider and review_provider and impl_provider == review_provider:
        return False
    return True


def routing_summary(
    decision: Optional[Mapping[str, Any]], *, final_provider: Optional[str], final_model: Optional[str]
) -> dict[str, Any]:
    """Return additive usage identity fields; legacy provider/model remain final-route fields."""
    if not isinstance(decision, Mapping):
        return {
            "routing_provenance": LEGACY_ROUTING_PROVENANCE,
            "routing_decision_id": None,
            "primary_profile": None,
            "initial_provider": None,
            "initial_model": None,
            "final_provider": _optional_text(final_provider),
            "final_model": _optional_text(final_model),
            "fallback_used": None,
            "fallback_reason": None,
        }
    return {
        "routing_provenance": decision.get("routing_contract_version") or LEGACY_ROUTING_PROVENANCE,
        "routing_decision_id": decision.get("decision_id"),
        "primary_profile": decision.get("selected_profile"),
        "initial_provider": decision.get("initial_provider"),
        "initial_model": decision.get("initial_model"),
        "final_provider": _optional_text(final_provider) or decision.get("selected_provider"),
        "final_model": _optional_text(final_model) or decision.get("selected_model"),
        "fallback_used": decision.get("fallback_used"),
        "fallback_reason": decision.get("fallback_reason"),
    }


def _history_extends(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if existing.get("decision_id") != candidate.get("decision_id"):
        return False
    mutable_fields = {
        "selected_provider",
        "selected_model",
        "fallback_used",
        "fallback_from_provider",
        "fallback_from_model",
        "fallback_reason",
        "routing_history",
    }
    existing_identity = {key: value for key, value in existing.items() if key not in mutable_fields}
    candidate_identity = {key: value for key, value in candidate.items() if key not in mutable_fields}
    if existing_identity != candidate_identity:
        return False
    existing_history = existing.get("routing_history")
    candidate_history = candidate.get("routing_history")
    if not isinstance(existing_history, list) or not isinstance(candidate_history, list):
        return False
    return candidate_history[:len(existing_history)] == existing_history


def _active_profile_name() -> Optional[str]:
    for key in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        value = _optional_text(os.environ.get(key))
        if value:
            return value
    try:
        from hermes_cli.profiles import get_active_profile_name
        return _optional_text(get_active_profile_name())
    except Exception:
        return None


def _environment_routing_context() -> dict[str, Any]:
    from agent.delegation_context import is_dispatcher_owned_worker_context
    if not is_dispatcher_owned_worker_context():
        return {}
    raw = os.environ.get("HERMES_ROUTING_CONTEXT")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid HERMES_ROUTING_CONTEXT")
    run_id = _optional_text(os.environ.get("HERMES_KANBAN_RUN_ID"))
    return {
        "task_id": _optional_text(os.environ.get("HERMES_KANBAN_TASK")),
        "board": _optional_text(os.environ.get("HERMES_KANBAN_BOARD")),
        "run_id": int(run_id) if run_id and run_id.isdigit() else None,
    }


def _persist_agent_decision(
    agent: Any, decision: Mapping[str, Any], event_kind: str
) -> dict[str, Any]:
    stored = copy.deepcopy(dict(decision))
    session_db = getattr(agent, "_session_db", None)
    get_session_decision = getattr(session_db, "get_session_model_config_value", None)
    if callable(get_session_decision):
        try:
            existing = get_session_decision(agent.session_id, "routing_decision")
        except Exception:
            existing = None
        if isinstance(existing, Mapping) and existing != stored:
            same_decision = existing.get("decision_id") == stored.get("decision_id")
            if same_decision and not _history_extends(existing, stored):
                stored = copy.deepcopy(dict(existing))

    task_id = _optional_text(stored.get("task_id"))
    run_id = stored.get("run_id")
    db_path = _optional_text(os.environ.get("HERMES_KANBAN_DB"))
    if task_id and isinstance(run_id, int) and db_path:
        try:
            from hermes_cli.kanban_db_connect import connect
            from hermes_cli.kanban_db_routing import (
                load_run_routing_decision,
                persist_run_routing_decision,
            )
            with connect(Path(db_path)) as conn:
                accepted = persist_run_routing_decision(
                    conn,
                    task_id=task_id,
                    run_id=run_id,
                    decision=stored,
                    event_kind=event_kind,
                )
                if not accepted:
                    authoritative = load_run_routing_decision(
                        conn, task_id=task_id, run_id=run_id
                    )
                    if isinstance(authoritative, Mapping):
                        stored = copy.deepcopy(dict(authoritative))
        except Exception:
            logger.warning("Could not persist Kanban routing decision", exc_info=True)

    agent.routing_decision = stored
    session_config = getattr(agent, "_session_init_model_config", None)
    if isinstance(session_config, dict):
        session_config["routing_decision"] = stored
    patch_session = getattr(session_db, "patch_session_model_config", None)
    if callable(patch_session):
        try:
            patch_session(agent.session_id, {"routing_decision": stored})
        except Exception:
            logger.debug("Could not persist session routing decision", exc_info=True)
    return stored


def _existing_session_decision(
    agent: Any, context: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    session_db = getattr(agent, "_session_db", None)
    getter = getattr(session_db, "get_session_model_config_value", None)
    if not callable(getter):
        return None
    try:
        existing = getter(agent.session_id, "routing_decision")
    except Exception:
        return None
    if not isinstance(existing, Mapping):
        return None
    if existing.get("routing_contract_version") != ROUTING_CONTRACT_VERSION:
        return None
    if existing.get("session_id") != _optional_text(getattr(agent, "session_id", None)):
        return None
    for key in ("task_id", "run_id"):
        expected = context.get(key)
        if expected is not None and existing.get(key) != expected:
            return None
    return copy.deepcopy(dict(existing))


def initialize_agent_routing_decision(
    agent: Any,
    *,
    routing_context: Optional[Mapping[str, Any]] = None,
    selected_at: Optional[str] = None,
) -> dict[str, Any]:
    """Capture the resolved profile/provider/model before the first model request."""
    context = _environment_routing_context()
    if routing_context is not None:
        context.update(dict(routing_context))
    existing = _existing_session_decision(agent, context)
    if existing is not None:
        agent.routing_decision = existing
        session_config = getattr(agent, "_session_init_model_config", None)
        if isinstance(session_config, dict):
            session_config["routing_decision"] = existing
        return existing
    selected_profile = _optional_text(context.get("selected_profile")) or _active_profile_name()
    independent_review = context.get("independent_review")
    reviewer_profile = _optional_text(context.get("reviewer_profile"))
    independence_valid = context.get("independence_valid")
    if not isinstance(independence_valid, bool) and isinstance(independent_review, bool):
        independence_valid = evaluate_reviewer_independence(
            implementation_profile=selected_profile,
            reviewer_profile=reviewer_profile,
            independent_review=independent_review,
        )
    decision = build_routing_decision(
        task_id=context.get("task_id"),
        board=context.get("board"),
        task_type=context.get("task_type"),
        capability=context.get("capability"),
        risk=context.get("risk"),
        code_change=context.get("code_change"),
        independent_review=independent_review,
        preferred_profile=context.get("preferred_profile") or selected_profile,
        reviewer_profile=reviewer_profile,
        selected_profile=selected_profile,
        selected_provider=getattr(agent, "provider", None),
        selected_model=getattr(agent, "model", None),
        human_gate_required=context.get("human_gate_required"),
        independence_valid=independence_valid,
        policy_digest=context.get("policy_digest"),
        selected_at=selected_at,
        selected_by=context.get("selected_by") or ("dispatcher" if context.get("task_id") else "operator"),
        run_id=context.get("run_id") if isinstance(context.get("run_id"), int) else None,
        session_id=getattr(agent, "session_id", None),
    )
    return _persist_agent_decision(agent, decision, "routing_selected")


def record_agent_fallback(
    agent: Any,
    *,
    from_provider: Optional[str],
    from_model: Optional[str],
    reason: Any,
    recorded_at: Optional[str] = None,
) -> dict[str, Any]:
    """Persist an activated fallback using the classifier category available at the boundary."""
    current = getattr(agent, "routing_decision", None)
    if not isinstance(current, Mapping):
        current = initialize_agent_routing_decision(agent)
    decision = record_fallback(
        current,
        from_provider=from_provider,
        from_model=from_model,
        to_provider=getattr(agent, "provider", None),
        to_model=getattr(agent, "model", None),
        reason=reason,
        recorded_at=recorded_at,
    )
    return _persist_agent_decision(agent, decision, "routing_fallback")


def record_agent_primary_restore(
    agent: Any,
    *,
    from_provider: Optional[str],
    from_model: Optional[str],
    recorded_at: Optional[str] = None,
) -> dict[str, Any]:
    """Append a primary-restored phase without clearing prior fallback provenance."""
    current = getattr(agent, "routing_decision", None)
    if not isinstance(current, Mapping):
        current = initialize_agent_routing_decision(agent)
    decision = copy.deepcopy(dict(current))
    decision["selected_provider"] = _optional_text(getattr(agent, "provider", None))
    decision["selected_model"] = _optional_text(getattr(agent, "model", None))
    history = list(decision.get("routing_history") or [])
    history.append(
        {
            "phase": "primary_restored",
            "provider": decision["selected_provider"],
            "model": decision["selected_model"],
            "reason": None,
            "recorded_at": _optional_text(recorded_at) or _utc_now(),
            "from_provider": _optional_text(from_provider),
            "from_model": _optional_text(from_model),
        }
    )
    decision["routing_history"] = history
    return _persist_agent_decision(agent, decision, "routing_primary_restored")


def add_routing_summary(result: Mapping[str, Any], agent: Any) -> dict[str, Any]:
    """Add structured route fields while retaining legacy provider/model final-route semantics."""
    enriched = dict(result)
    enriched.update(
        routing_summary(
            getattr(agent, "routing_decision", None),
            final_provider=enriched.get("provider") or getattr(agent, "provider", None),
            final_model=enriched.get("model") or getattr(agent, "model", None),
        )
    )
    return enriched

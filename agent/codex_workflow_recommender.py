from __future__ import annotations

from typing import Any

from agent import codex_workflow_verification as verification


_NON_GOALS = ["commit", "push", "deploy", "restart", "force-push"]
_FORBIDDEN_STAGE_TERMS = ("deploy", "restart", "force-push", "force_push", "forcepush")
_DEFAULT_WHY = "review and Hermes verification passed; next stage is recommendation-only"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _verification_commands(evidence: Any) -> list[dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    commands = evidence.get("hermes_verification_commands")
    if isinstance(commands, list):
        return [item for item in commands if isinstance(item, dict)]
    commands = evidence.get("verification_commands")
    if isinstance(commands, list):
        return [item for item in commands if isinstance(item, dict)]
    return []


def _verification_gate(evidence: Any) -> dict[str, Any]:
    if isinstance(evidence, dict) and isinstance(evidence.get("verification_gate"), dict):
        return evidence["verification_gate"]
    risk_classes: Any = []
    if isinstance(evidence, dict):
        risk_classes = evidence.get("risk_classes") or evidence.get("risk_class") or []
    return verification.validate_verification_results(risk_classes, _verification_commands(evidence))


def _candidate_from(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stage_is_forbidden(stage_id: Any) -> bool:
    lowered = str(stage_id or "").lower()
    return any(term in lowered for term in _FORBIDDEN_STAGE_TERMS)


def build_next_stage_recommendation(
    *,
    review_result: Any,
    verification_evidence: Any,
    candidate: Any = None,
    current_allowed_files: list[str] | None = None,
    current_verify_cmd_ids: list[str] | None = None,
    request_advance: bool = False,
) -> dict[str, Any]:
    """Return a declarative next-stage recommendation, never an action plan.

    This helper intentionally has no side effects. It does not run Codex,
    review, verification, git, deployment, or restart commands.
    """
    if not isinstance(review_result, dict) or review_result.get("status") != "passed":
        return {
            "status": "blocked",
            "reason": "review_unavailable",
            "recommendation": None,
        }

    gate = _verification_gate(verification_evidence)
    if gate.get("blocks_next_stage"):
        return {
            "status": "blocked",
            "reason": "verification_failed",
            "verification": gate,
            "recommendation": None,
        }

    candidate_dict = _candidate_from(candidate)
    stage_id = candidate_dict.get("stage_id") or "next-stage-candidate"
    if _stage_is_forbidden(stage_id):
        return {
            "status": "blocked",
            "reason": "forbidden_stage",
            "recommendation": None,
        }

    allowed_files = _string_list(candidate_dict.get("allowed_files")) or list(current_allowed_files or [])
    verify_cmd_ids = _string_list(candidate_dict.get("verify_cmd_ids")) or list(current_verify_cmd_ids or [])
    recommendation = {
        "stage_id": stage_id,
        "why": str(candidate_dict.get("why") or _DEFAULT_WHY),
        "allowed_files": allowed_files,
        "verify_cmd_ids": verify_cmd_ids,
        "authorization_required": True,
        "non_goals": list(_NON_GOALS),
    }
    if _string_list(candidate_dict.get("allowed_globs")):
        recommendation["allowed_globs"] = _string_list(candidate_dict.get("allowed_globs"))

    return {
        "status": "recommended",
        "recommendation": recommendation,
        "advance": {
            "requested": bool(request_advance),
            "status": "blocked",
            "reason": "authorization_required",
            "authorization_required": True,
            "executed": False,
        },
    }

from __future__ import annotations

from typing import Any


DEFAULT_MAX_FIX_ROUNDS = 2

_ROUND_CONTRACT = [
    "verify_review_claim",
    "add_or_update_focused_regression_test_if_practical",
    "bounded_implementation",
    "focused_verification",
    "packet_only_re_review",
]

_STOP_REASONS = {
    "review_unavailable",
    "verification_failed",
    "dirty_overlap_or_allowlist_escape",
    "new_secret_or_real_data_risk",
    "repeated_codex_flood_or_timeout",
    "max_fix_rounds_exhausted",
    "false_positive_requires_evidence",
    "true_positive_requires_regression_test",
}


def _coerce_max_rounds(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_MAX_FIX_ROUNDS
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_MAX_FIX_ROUNDS


def _coerce_current_round(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _review_payload(review_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(review_result, dict):
        return {}
    review = review_result.get("review")
    if isinstance(review, dict):
        return review
    return review_result


def _must_fix_items(review_result: dict[str, Any] | None) -> list[Any]:
    review = _review_payload(review_result)
    must_fix = review.get("must_fix")
    return list(must_fix) if isinstance(must_fix, list) else []


def _suggested_fixes(review_result: dict[str, Any] | None) -> list[Any]:
    review = _review_payload(review_result)
    suggested_fixes = review.get("suggested_fixes")
    return list(suggested_fixes) if isinstance(suggested_fixes, list) else []


def _is_review_unavailable(review_result: dict[str, Any] | None) -> bool:
    if not isinstance(review_result, dict):
        return True
    return review_result.get("status") in {"unavailable", "not_requested"} or review_result.get("verdict") == "unavailable"


def _verification_failed(verification_result: dict[str, Any] | None) -> bool:
    if not isinstance(verification_result, dict):
        return False
    if verification_result.get("status") == "failed":
        return True
    if verification_result.get("passed") is False:
        return True
    return bool(verification_result.get("blocks_next_stage"))


def _has_evidence(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _resolution_stop_reason(resolutions: Any) -> str | None:
    if not isinstance(resolutions, list):
        return None
    for resolution in resolutions:
        if not isinstance(resolution, dict):
            continue
        disposition = resolution.get("disposition")
        if disposition == "false_positive" and not _has_evidence(resolution.get("evidence")):
            return "false_positive_requires_evidence"
        if (
            disposition == "true_positive"
            and resolution.get("regression_test_practical") is True
            and not _has_evidence(resolution.get("regression_test"))
        ):
            return "true_positive_requires_regression_test"
    return None


def _stop_status(
    reason: str,
    *,
    current_round: int,
    max_fix_rounds: int,
    must_fix: list[Any],
    suggested_fixes: list[Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "status": "stopped",
        "reason": reason,
        "stop_reason": reason,
        "blocks_continuation": True,
        "authorization_required": False,
        "current_round": current_round,
        "max_fix_rounds": max_fix_rounds,
        "must_fix_count": len(must_fix),
        "must_fix": must_fix,
        "suggested_fixes_recorded": suggested_fixes,
        "auto_implements_suggested_fixes": False,
        "round_contract": list(_ROUND_CONTRACT),
        "next_actions": [],
    }
    if extra:
        result.update(extra)
    return result


def build_must_fix_loop_status(
    *,
    review_result: dict[str, Any] | None,
    authorized: bool = False,
    max_fix_rounds: int | None = None,
    current_round: int = 0,
    verification_result: dict[str, Any] | None = None,
    dirty_overlap: bool = False,
    allowlist_escape: bool = False,
    new_secret_or_real_data_risk: bool = False,
    codex_flood_timeout_count: int = 0,
    finding_resolutions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return declarative Phase 12E must-fix loop state.

    This helper intentionally does not run Codex, apply suggested fixes, mutate
    files, or invoke verification/re-review. It only states whether a bounded
    must-fix round may proceed and what the per-round contract is.
    """
    max_rounds = _coerce_max_rounds(max_fix_rounds)
    round_index = _coerce_current_round(current_round)
    must_fix = _must_fix_items(review_result)
    suggested_fixes = _suggested_fixes(review_result)

    base = {
        "current_round": round_index,
        "max_fix_rounds": max_rounds,
        "must_fix_count": len(must_fix),
        "must_fix": must_fix,
        "suggested_fixes_recorded": suggested_fixes,
        "auto_implements_suggested_fixes": False,
        "round_contract": list(_ROUND_CONTRACT),
        "stop_reasons": sorted(_STOP_REASONS),
    }

    if not authorized:
        return {
            **base,
            "status": "authorization_required",
            "reason": "must_fix_loop_authorization_required",
            "authorization_required": True,
            "blocks_continuation": True,
            "next_actions": ["request_explicit_must_fix_loop_authorization"],
        }

    if _is_review_unavailable(review_result):
        return _stop_status(
            "review_unavailable",
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
        )

    resolution_reason = _resolution_stop_reason(finding_resolutions)
    if resolution_reason is not None:
        return _stop_status(
            resolution_reason,
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
        )

    if _verification_failed(verification_result):
        return _stop_status(
            "verification_failed",
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
            extra={"verification_result": verification_result},
        )

    if dirty_overlap or allowlist_escape:
        return _stop_status(
            "dirty_overlap_or_allowlist_escape",
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
            extra={"dirty_overlap": bool(dirty_overlap), "allowlist_escape": bool(allowlist_escape)},
        )

    if new_secret_or_real_data_risk:
        return _stop_status(
            "new_secret_or_real_data_risk",
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
        )

    if codex_flood_timeout_count >= 2:
        return _stop_status(
            "repeated_codex_flood_or_timeout",
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
            extra={"codex_flood_timeout_count": codex_flood_timeout_count},
        )

    if not must_fix:
        return {
            **base,
            "status": "complete",
            "reason": "no_must_fix_remaining",
            "authorization_required": False,
            "blocks_continuation": False,
            "next_actions": [],
        }

    if round_index >= max_rounds:
        return _stop_status(
            "max_fix_rounds_exhausted",
            current_round=round_index,
            max_fix_rounds=max_rounds,
            must_fix=must_fix,
            suggested_fixes=suggested_fixes,
        )

    return {
        **base,
        "status": "ready_for_round",
        "reason": "must_fix_round_available",
        "authorization_required": False,
        "blocks_continuation": False,
        "next_round": round_index + 1,
        "remaining_rounds_after_next": max_rounds - round_index - 1,
        "next_actions": list(_ROUND_CONTRACT),
    }

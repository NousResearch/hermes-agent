"""Deterministic failure classifier.

Maps raw failure data (eval case results, error strings, metadata) into
normalized (type, subtype, severity) tuples using rule-based matching.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Optional

from .types import FailureType, FailureSubtype, Severity, NormalizedFailure


# ── Pattern-based rules ──────────────────────────────────────────────────

_TIMEOUT_PATTERNS = re.compile(
    r"timeout|timed?\s*out|deadline\s*exceeded|execution\s*expired",
    re.IGNORECASE,
)

_AUTH_PATTERNS = re.compile(
    r"auth(?:entication|orization)?\s*(?:fail|error|denied|invalid)"
    r"|401\s*unauthorized|403\s*forbidden|invalid\s*(?:api[_ ]?key|token|credential)",
    re.IGNORECASE,
)

_ENV_PATTERNS = re.compile(
    r"missing\s*(?:env|environment|credential|dependency|module|package)"
    r"|not\s*found.*(?:binary|executable|command)"
    r"|no\s*such\s*file|import\s*error|module\s*not\s*found",
    re.IGNORECASE,
)

_MALFORMED_OUTPUT_PATTERNS = re.compile(
    r"json\s*(?:parse|decode)\s*error|malformed\s*(?:output|response|json)"
    r"|schema\s*validation\s*(?:fail|error)|unexpected\s*(?:token|format)",
    re.IGNORECASE,
)

_APPROVAL_PATTERNS = re.compile(
    r"approval\s*(?:blocked|required|pending|denied)"
    r"|awaiting\s*approval|unsafe\s*action\s*refused",
    re.IGNORECASE,
)

_TOOL_ERROR_PATTERNS = re.compile(
    r"tool\s*(?:error|fail|exception)|command\s*(?:fail|error|non[- ]?zero)"
    r"|exit\s*code\s*[1-9]|execution\s*(?:fail|error)",
    re.IGNORECASE,
)


def _compute_fingerprint(
    failure_type: str,
    failure_subtype: str,
    tool_name: Optional[str],
    model: Optional[str],
    summary: str,
) -> str:
    """Deterministic fingerprint for clustering near-identical failures."""
    # Normalize summary: lowercase, collapse whitespace, strip numbers/ids
    norm = re.sub(r"[0-9a-f]{8,}", "<id>", summary.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    parts = [failure_type, failure_subtype, tool_name or "", model or "", norm]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return digest


# ── Public API ────────────────────────────────────────────────────────────

def classify_eval_case(
    case_result: dict[str, Any],
    run_id: str,
    prior_status: Optional[str] = None,
) -> NormalizedFailure:
    """Classify a failed eval case result into a NormalizedFailure.

    Args:
        case_result: Dict with keys from eval_case_results table.
        run_id: The eval run ID.
        prior_status: Status of the same case in the previous run, if known.
                      Used to detect regressions.
    """
    status = case_result.get("status", "")
    summary = case_result.get("failure_summary", "") or ""
    case_id = case_result.get("case_id", "")

    # Determine type/subtype
    if prior_status in ("passed",) and status in ("failed", "error"):
        ftype, fsubtype = FailureType.EVAL, FailureSubtype.REGRESSION
        severity = Severity.HIGH
    elif status == "timeout":
        ftype, fsubtype = FailureType.TOOL, FailureSubtype.TIMEOUT
        severity = Severity.MEDIUM
    elif _TIMEOUT_PATTERNS.search(summary):
        ftype, fsubtype = FailureType.TOOL, FailureSubtype.TIMEOUT
        severity = Severity.MEDIUM
    elif _AUTH_PATTERNS.search(summary):
        ftype, fsubtype = FailureType.INFRA, FailureSubtype.AUTH
        severity = Severity.HIGH
    elif _ENV_PATTERNS.search(summary):
        ftype, fsubtype = FailureType.INFRA, FailureSubtype.ENVIRONMENT
        severity = Severity.MEDIUM
    elif _MALFORMED_OUTPUT_PATTERNS.search(summary):
        ftype, fsubtype = FailureType.MODEL, FailureSubtype.OUTPUT_MALFORMED
        severity = Severity.MEDIUM
    elif _APPROVAL_PATTERNS.search(summary):
        ftype, fsubtype = FailureType.POLICY, FailureSubtype.APPROVAL_BLOCKED
        severity = Severity.LOW
    elif _TOOL_ERROR_PATTERNS.search(summary):
        ftype, fsubtype = FailureType.TOOL, FailureSubtype.EXECUTION
        severity = Severity.MEDIUM
    else:
        ftype, fsubtype = FailureType.EVAL, FailureSubtype.FAILED_CHECK
        severity = Severity.MEDIUM

    fingerprint = _compute_fingerprint(ftype, fsubtype, None, None, summary)

    return NormalizedFailure(
        id=uuid.uuid4().hex[:12],
        source_surface="eval",
        eval_run_id=run_id,
        case_id=case_id,
        failure_type=ftype.value,
        failure_subtype=fsubtype.value,
        severity=severity.value,
        summary=summary or f"Eval case {case_id} {status}",
        evidence_json=json.dumps({
            "status": status,
            "score": case_result.get("deterministic_score", 0),
            "category": case_result.get("category", ""),
        }),
        fingerprint=fingerprint,
    )


def classify_raw_error(
    error_text: str,
    source_surface: str = "tool",
    tool_name: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> NormalizedFailure:
    """Classify an arbitrary error string into a NormalizedFailure."""
    if _TIMEOUT_PATTERNS.search(error_text):
        ftype, fsubtype = FailureType.TOOL, FailureSubtype.TIMEOUT
        severity = Severity.MEDIUM
    elif _AUTH_PATTERNS.search(error_text):
        ftype, fsubtype = FailureType.INFRA, FailureSubtype.AUTH
        severity = Severity.HIGH
    elif _ENV_PATTERNS.search(error_text):
        ftype, fsubtype = FailureType.INFRA, FailureSubtype.ENVIRONMENT
        severity = Severity.MEDIUM
    elif _MALFORMED_OUTPUT_PATTERNS.search(error_text):
        ftype, fsubtype = FailureType.MODEL, FailureSubtype.OUTPUT_MALFORMED
        severity = Severity.MEDIUM
    elif _APPROVAL_PATTERNS.search(error_text):
        ftype, fsubtype = FailureType.POLICY, FailureSubtype.APPROVAL_BLOCKED
        severity = Severity.LOW
    elif _TOOL_ERROR_PATTERNS.search(error_text):
        ftype, fsubtype = FailureType.TOOL, FailureSubtype.EXECUTION
        severity = Severity.MEDIUM
    else:
        ftype, fsubtype = FailureType.UNKNOWN, FailureSubtype.UNKNOWN
        severity = Severity.MEDIUM

    fingerprint = _compute_fingerprint(ftype, fsubtype, tool_name, model, error_text)

    return NormalizedFailure(
        id=uuid.uuid4().hex[:12],
        source_surface=source_surface,
        session_id=session_id,
        task_id=task_id,
        failure_type=ftype.value,
        failure_subtype=fsubtype.value,
        severity=severity.value,
        tool_name=tool_name,
        model=model,
        provider=provider,
        summary=error_text[:500],
        evidence_json=json.dumps({"raw_error": error_text[:2000]}),
        fingerprint=fingerprint,
    )

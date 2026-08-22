"""ExecutiveResultAdapter — adapts Executive v2 results to user summaries.

Pure transformation. No LLM. No provider. No network. No subprocess.
No DB write. No commit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from .types import (
    ExecutiveLaunchRequest,
    ExecutiveUserSummary,
    LaunchStatus,
    SummaryKind,
    _now_iso8601,
)


# ──────────────────────────────────────────────────────────────────────
# ExecutiveResultAdapter
# ──────────────────────────────────────────────────────────────────────


class ExecutiveResultAdapter:
    """Pure facade. Adapts Executive v2 results into ExecutiveUserSummary."""

    SCHEMA_VERSION = "eil.v1"

    def adapt_pending(
        self,
        request: ExecutiveLaunchRequest,
    ) -> ExecutiveUserSummary:
        """Convert a pending request into a user summary."""
        body = (
            f"Your request has been classified as EXECUTIVE.\n\n"
            f"Risk level: {request.risk_level}.\n"
            f"Phases to run: {', '.join(request.expected_phases)}.\n"
            f"Please review and approve before launch."
        )
        next_steps = ("Approve", "Reject", "Modify")
        warnings = (
            "Risk level " + request.risk_level + " requires explicit approval.",
        ) if request.requires_human_approval else ()
        return _make_summary(
            request_id=request.request_id,
            summary_kind=SummaryKind.PENDING,
            title="Launch request pending",
            body=body,
            next_steps=next_steps,
            warnings=warnings,
            details_url=None,
            created_by="ExecutiveResultAdapter",
        )

    def adapt_executing(
        self,
        request: ExecutiveLaunchRequest,
        *,
        current_phase: str,
    ) -> ExecutiveUserSummary:
        """Convert an in-flight request into a user summary."""
        body = (
            f"Your request is being executed.\n\n"
            f"Current phase: {current_phase}.\n"
            f"Status: running."
        )
        return _make_summary(
            request_id=request.request_id,
            summary_kind=SummaryKind.EXECUTING,
            title="Launch request executing",
            body=body,
            next_steps=("Wait for completion",),
            warnings=(),
            details_url=None,
            created_by="ExecutiveResultAdapter",
        )

    def adapt_preview_ready(
        self,
        request: ExecutiveLaunchRequest,
        *,
        preview: Dict[str, Any],
    ) -> ExecutiveUserSummary:
        """Convert a contract-draft canary result into a preview summary."""
        body = (
            "OBJECTIVE_CREATED\n"
            "EXECUTION_PREVIEW_READY\n\n"
            f"Objective ID: {preview.get('objective_id')}.\n"
            f"Stop boundary: {preview.get('stop_state')}.\n"
            "Persisted: false.\n"
            "Runtime launched: false.\n"
            "Workers/Kanban/providers/subprocess: not invoked."
        )
        return _make_summary(
            request_id=request.request_id,
            summary_kind=SummaryKind.SUCCESS,
            title="Execution preview ready",
            body=body,
            next_steps=("Review contract draft", "Approve a future execution phase separately"),
            warnings=("Canary stopped at CONTRACT_DRAFT; no runtime was launched.",),
            details_url=None,
            created_by="ExecutiveResultAdapter",
        )

    def adapt_result(
        self,
        request: ExecutiveLaunchRequest,
        *,
        result: Dict[str, Any],
    ) -> ExecutiveUserSummary:
        """Convert a completed result into a user summary.

        ``result`` should be a dict with at least ``status``
        (``"success"`` or ``"failure"``) and optional
        ``phases_completed`` and ``failure_reason``.
        """
        status = str(result.get("status", "")).lower()
        phases_completed = tuple(result.get("phases_completed") or ())
        failure_reason = result.get("failure_reason") or ""

        if status == "success":
            body = (
                f"Executive v2 completed successfully.\n\n"
                f"Phases completed: {', '.join(phases_completed) or '(none)'}.\n"
                f"Final status: SUCCESS."
            )
            return _make_summary(
                request_id=request.request_id,
                summary_kind=SummaryKind.SUCCESS,
                title="Launch request succeeded",
                body=body,
                next_steps=("Review the report",),
                warnings=(),
                details_url=None,
                created_by="ExecutiveResultAdapter",
            )
        elif status == "partial":
            body = (
                f"Executive v2 completed with partial success.\n\n"
                f"Phases completed: {', '.join(phases_completed) or '(none)'}.\n"
                f"Status: PARTIAL."
            )
            return _make_summary(
                request_id=request.request_id,
                summary_kind=SummaryKind.PARTIAL,
                title="Launch request partially succeeded",
                body=body,
                next_steps=("Review the report",),
                warnings=(),
                details_url=None,
                created_by="ExecutiveResultAdapter",
            )
        else:
            body = (
                f"Executive v2 encountered a failure.\n\n"
                f"Phases completed: {', '.join(phases_completed) or '(none)'}.\n"
                f"Reason: {failure_reason}."
            )
            return _make_summary(
                request_id=request.request_id,
                summary_kind=SummaryKind.FAILED,
                title="Launch request failed",
                body=body,
                next_steps=("Review the failure", "Decide whether to retry"),
                warnings=("Failure detected.",),
                details_url=None,
                created_by="ExecutiveResultAdapter",
            )

    def adapt_rejected(
        self,
        request: ExecutiveLaunchRequest,
        *,
        reason: str,
    ) -> ExecutiveUserSummary:
        """Convert a rejected request into a user summary."""
        body = (
            f"The launch request was rejected.\n\n"
            f"Reason: {reason}.\n"
            f"No actions were taken."
        )
        return _make_summary(
            request_id=request.request_id,
            summary_kind=SummaryKind.REJECTED,
            title="Launch request rejected",
            body=body,
            next_steps=("Modify the request and retry",),
            warnings=(),
            details_url=None,
            created_by="ExecutiveResultAdapter",
        )

    def adapt_cancelled(
        self,
        request: ExecutiveLaunchRequest,
    ) -> ExecutiveUserSummary:
        """Convert a cancelled request into a user summary."""
        body = "The launch request was cancelled by the operator."
        return _make_summary(
            request_id=request.request_id,
            summary_kind=SummaryKind.CANCELLED,
            title="Launch request cancelled",
            body=body,
            next_steps=("Submit a new request if needed",),
            warnings=(),
            details_url=None,
            created_by="ExecutiveResultAdapter",
        )

    def adapt_blocked(
        self,
        request: ExecutiveLaunchRequest,
        *,
        reason: str,
    ) -> ExecutiveUserSummary:
        """Convert a blocked request (no approval granted) into a user summary."""
        body = (
            f"The launch request was blocked.\n\n"
            f"Reason: {reason}.\n"
            f"No actions were taken."
        )
        return _make_summary(
            request_id=request.request_id,
            summary_kind=SummaryKind.BLOCKED,
            title="Launch request blocked",
            body=body,
            next_steps=("Review the policy decision", "Decide: abort or replan"),
            warnings=("Approval required.",),
            details_url=None,
            created_by="ExecutiveResultAdapter",
        )


# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────


def _make_summary(
    *,
    request_id: str,
    summary_kind: SummaryKind,
    title: str,
    body: str,
    next_steps: Tuple[str, ...],
    warnings: Tuple[str, ...],
    details_url: Optional[str],
    created_by: str,
) -> ExecutiveUserSummary:
    payload = {
        "request_id": request_id,
        "summary_kind": summary_kind.value,
        "title": title,
        "body": body,
        "next_steps": sorted(next_steps),
        "warnings": sorted(warnings),
        "details_url": details_url,
        "schema_version": "eil.v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    return ExecutiveUserSummary(
        request_id=request_id,
        summary_kind=summary_kind,
        title=title,
        body=body,
        next_steps=next_steps,
        warnings=warnings,
        details_url=details_url,
        fingerprint=fingerprint,
        created_at=_now_iso8601(),
        created_by=created_by,
    )


__all__ = [
    "ExecutiveResultAdapter",
]

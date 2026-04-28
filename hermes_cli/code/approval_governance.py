#!/usr/bin/env python3
"""Persistent approval governance lifecycle for Hermes Code Mode."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_cli.code.event_bus import get_code_event_bus
from hermes_cli.code.execution_policy import RiskClass, redact_secrets
from hermes_cli.code.github_integration import redact_github_secrets
from hermes_state import SessionDB


class ApprovalStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    EXECUTED = "executed"
    FAILED = "failed"

    ALL = frozenset({PENDING, APPROVED, REJECTED, EXPIRED, CANCELLED, EXECUTED, FAILED})
    TERMINAL = frozenset({REJECTED, EXPIRED, CANCELLED, EXECUTED, FAILED})


class ApprovalKind:
    GITHUB_COMMENT = "github_comment"
    GITHUB_PR_PREPARE = "github_pr_prepare"
    GITHUB_CHECK_UPDATE = "github_check_update"
    GITHUB_STATUS_UPDATE = "github_status_update"
    COMMAND_EXECUTION = "command_execution"
    GIT_WRITE = "git_write"
    PRODUCTION_SENSITIVE = "production_sensitive"
    DESTRUCTIVE = "destructive"
    GENERIC = "generic"

    ALL = frozenset(
        {
            GITHUB_COMMENT,
            GITHUB_PR_PREPARE,
            GITHUB_CHECK_UPDATE,
            GITHUB_STATUS_UPDATE,
            COMMAND_EXECUTION,
            GIT_WRITE,
            PRODUCTION_SENSITIVE,
            DESTRUCTIVE,
            GENERIC,
        }
    )


RISK_CLASS_ALL = frozenset(
    {
        RiskClass.SAFE_READONLY,
        RiskClass.SAFE_LOCAL_WRITE,
        RiskClass.NETWORK,
        RiskClass.GIT_WRITE,
        RiskClass.SECRET_SENSITIVE,
        RiskClass.REMOTE_MUTATING,
        RiskClass.DESTRUCTIVE,
        RiskClass.PRODUCTION_SENSITIVE,
    }
)


_TRANSITIONS: dict[str, frozenset[str]] = {
    ApprovalStatus.PENDING: frozenset(
        {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.CANCELLED,
            ApprovalStatus.EXPIRED,
        }
    ),
    ApprovalStatus.APPROVED: frozenset(
        {
            ApprovalStatus.EXECUTED,
            ApprovalStatus.FAILED,
            ApprovalStatus.CANCELLED,
        }
    ),
    ApprovalStatus.REJECTED: frozenset(),
    ApprovalStatus.EXPIRED: frozenset(),
    ApprovalStatus.CANCELLED: frozenset(),
    ApprovalStatus.EXECUTED: frozenset(),
    ApprovalStatus.FAILED: frozenset(),
}


_EVENT_BY_STATUS: dict[str, str] = {
    ApprovalStatus.APPROVED: "code.approval.approved",
    ApprovalStatus.REJECTED: "code.approval.rejected",
    ApprovalStatus.CANCELLED: "code.approval.cancelled",
    ApprovalStatus.EXPIRED: "code.approval.expired",
    ApprovalStatus.EXECUTED: "code.approval.executed",
    ApprovalStatus.FAILED: "code.approval.failed",
}


_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(token|access_token|refresh_token|authorization|private_key|webhook_secret|client_secret|secret|password|HERMES_GITHUB_DEV_PAT)"
)


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sensitive_key(key: str) -> bool:
    return bool(_SECRET_KEY_PATTERN.search(str(key or "")))


def _redact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return redact_github_secrets(redact_secrets(value))
    return value


def redact_for_api(value: Any, *, key: Optional[str] = None) -> Any:
    if key and _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_for_api(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_api(v) for v in value]
    return _redact_scalar(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, separators=(",", ":"), sort_keys=True)


def _binding_fingerprint(
    *,
    kind: str,
    requested_action: str,
    resource_type: str | None,
    resource_id: str | None,
    github_repo_full_name: str | None,
    github_issue_number: int | None,
    github_pr_number: int | None,
    requested_payload: dict[str, Any] | None,
) -> str:
    payload = {
        "kind": kind,
        "requested_action": requested_action,
        "resource_type": resource_type or "",
        "resource_id": resource_id or "",
        "github_repo_full_name": github_repo_full_name or "",
        "github_issue_number": int(github_issue_number) if github_issue_number is not None else None,
        "github_pr_number": int(github_pr_number) if github_pr_number is not None else None,
        "requested_payload": requested_payload or {},
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_transition(current_status: str, next_status: str) -> tuple[bool, str]:
    if current_status not in ApprovalStatus.ALL:
        return False, f"Unknown current status: {current_status}"
    if next_status not in ApprovalStatus.ALL:
        return False, f"Unknown target status: {next_status}"
    if next_status not in _TRANSITIONS.get(current_status, frozenset()):
        return False, f"Invalid transition {current_status} -> {next_status}"
    return True, ""


class ApprovalGovernanceError(ValueError):
    """Raised when approval lifecycle operations fail safely."""


class ApprovalGovernanceService:
    DEFAULT_EXPIRY_SECONDS = 30 * 60

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path

    def _db(self) -> SessionDB:
        return SessionDB(db_path=self._db_path) if self._db_path else SessionDB()

    @staticmethod
    def _with_event_envelope(
        event_type: str,
        payload: dict[str, Any],
        *,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "type": event_type,
            "version": 1,
            "timestamp": _now_iso(),
            "payload": redact_for_api(payload),
        }
        if workspace_id:
            event["workspace_id"] = workspace_id
        if code_session_id:
            event["code_session_id"] = code_session_id
        return event

    def _emit(self, event_type: str, approval: dict[str, Any], *, extra_payload: dict[str, Any] | None = None, level: str = "info") -> None:
        payload = {
            "approval_id": approval.get("id"),
            "kind": approval.get("kind"),
            "status": approval.get("status"),
            "risk_class": approval.get("risk_class"),
            "resource_type": approval.get("resource_type"),
            "resource_id": approval.get("resource_id"),
            "github_repo_full_name": approval.get("github_repo_full_name"),
            "github_issue_number": approval.get("github_issue_number"),
            "github_pr_number": approval.get("github_pr_number"),
        }
        if extra_payload:
            payload.update(extra_payload)
        bus = get_code_event_bus(self._db_path)
        bus.publish(
            event_type,
            payload=payload,
            workspace_id=approval.get("workspace_id"),
            code_session_id=approval.get("code_session_id"),
            orchestrated_run_id=approval.get("orchestrated_run_id"),
            approval_id=approval.get("id"),
            github_repo_full_name=approval.get("github_repo_full_name"),
            metadata={"source": "approval_governance"},
            source="approval_governance",
            level=level,
        )

    def _serialize(self, approval: dict[str, Any]) -> dict[str, Any]:
        return redact_for_api(approval) if approval else {}

    def create_request(
        self,
        *,
        kind: str,
        risk_class: str,
        title: str,
        description: str | None = None,
        requested_action: str,
        requested_payload: dict[str, Any] | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        artifact_id: str | None = None,
        github_repo_full_name: str | None = None,
        github_issue_number: int | None = None,
        github_pr_number: int | None = None,
        requested_by: str | None = None,
        expires_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in ApprovalKind.ALL:
            raise ApprovalGovernanceError(f"Unknown approval kind: {kind}")
        if risk_class not in RISK_CLASS_ALL:
            raise ApprovalGovernanceError(f"Unknown risk_class: {risk_class}")

        now = _now_ts()
        payload = requested_payload or {}
        effective_expires_at = float(expires_at) if expires_at is not None else (now + self.DEFAULT_EXPIRY_SECONDS)
        metadata_value = dict(metadata or {})
        metadata_value["binding_fingerprint"] = _binding_fingerprint(
            kind=kind,
            requested_action=requested_action,
            resource_type=resource_type,
            resource_id=resource_id,
            github_repo_full_name=github_repo_full_name,
            github_issue_number=github_issue_number,
            github_pr_number=github_pr_number,
            requested_payload=payload,
        )
        approval = {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "risk_class": risk_class,
            "title": title or kind,
            "description": description or "",
            "requested_action": requested_action,
            "requested_payload": payload,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "workspace_id": workspace_id,
            "code_session_id": code_session_id,
            "orchestrated_run_id": orchestrated_run_id,
            "artifact_id": artifact_id,
            "github_repo_full_name": github_repo_full_name,
            "github_issue_number": int(github_issue_number) if github_issue_number is not None else None,
            "github_pr_number": int(github_pr_number) if github_pr_number is not None else None,
            "requested_by": requested_by or "local",
            "approved_by": None,
            "rejected_by": None,
            "status": ApprovalStatus.PENDING,
            "expires_at": effective_expires_at,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
            "metadata": metadata_value,
        }
        db = self._db()
        try:
            db.create_code_approval_request(approval)
            created = db.get_code_approval_request(approval["id"]) or approval
        finally:
            db.close()
        self._emit("code.approval.created", created)
        return self._serialize(created)

    def list_requests(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        risk_class: str | None = None,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        github_repo_full_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        db = self._db()
        try:
            items = db.list_code_approval_requests(
                status=status,
                kind=kind,
                risk_class=risk_class,
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                orchestrated_run_id=orchestrated_run_id,
                github_repo_full_name=github_repo_full_name,
                limit=max(1, min(int(limit), 500)),
                offset=max(0, int(offset)),
            )
        finally:
            db.close()
        return [self._serialize(item) for item in items]

    def get_request(self, approval_id: str) -> dict[str, Any] | None:
        db = self._db()
        try:
            item = db.get_code_approval_request(approval_id)
        finally:
            db.close()
        return self._serialize(item) if item else None

    def _transition(
        self,
        approval_id: str,
        next_status: str,
        *,
        actor: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._db()
        try:
            current = db.get_code_approval_request(approval_id)
            if not current:
                raise ApprovalGovernanceError(f"Approval not found: {approval_id}")
            ok, msg = _validate_transition(str(current.get("status")), next_status)
            if not ok:
                raise ApprovalGovernanceError(msg)

            now = _now_ts()
            updates: dict[str, Any] = {
                "status": next_status,
                "updated_at": now,
            }
            if next_status == ApprovalStatus.APPROVED:
                updates["approved_by"] = actor or "local"
            if next_status == ApprovalStatus.REJECTED:
                updates["rejected_by"] = actor or "local"
            if next_status in ApprovalStatus.TERMINAL:
                updates["resolved_at"] = now
            if metadata:
                merged_metadata = dict(current.get("metadata") or {})
                merged_metadata.update(metadata)
                updates["metadata"] = merged_metadata
            if reason:
                merged_metadata = dict(updates.get("metadata") or current.get("metadata") or {})
                merged_metadata["reason"] = reason
                updates["metadata"] = merged_metadata

            updated_ok = db.update_code_approval_request(approval_id, updates)
            if not updated_ok:
                raise ApprovalGovernanceError(f"Failed to update approval: {approval_id}")
            updated = db.get_code_approval_request(approval_id)
            if not updated:
                raise ApprovalGovernanceError(f"Approval not found after update: {approval_id}")
        finally:
            db.close()

        event_type = _EVENT_BY_STATUS.get(next_status)
        if event_type:
            self._emit(event_type, updated)
        return self._serialize(updated)

    def approve_request(self, approval_id: str, *, approved_by: str | None = None) -> dict[str, Any]:
        return self._transition(approval_id, ApprovalStatus.APPROVED, actor=approved_by)

    def reject_request(self, approval_id: str, *, rejected_by: str | None = None, reason: str | None = None) -> dict[str, Any]:
        return self._transition(approval_id, ApprovalStatus.REJECTED, actor=rejected_by, reason=reason)

    def cancel_request(self, approval_id: str, *, cancelled_by: str | None = None, reason: str | None = None) -> dict[str, Any]:
        return self._transition(approval_id, ApprovalStatus.CANCELLED, actor=cancelled_by, reason=reason)

    def mark_executed(self, approval_id: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._transition(approval_id, ApprovalStatus.EXECUTED, metadata=metadata)

    def mark_failed(self, approval_id: str, *, reason: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(metadata or {})
        if reason:
            merged["failure_reason"] = reason
        return self._transition(approval_id, ApprovalStatus.FAILED, reason=reason, metadata=merged)

    def expire_pending(self, *, now: float | None = None, limit: int = 500) -> dict[str, Any]:
        now_ts = float(now) if now is not None else _now_ts()
        db = self._db()
        try:
            pending = db.list_code_approval_requests(status=ApprovalStatus.PENDING, limit=max(1, min(limit, 2000)))
            expiring_ids = [
                item["id"]
                for item in pending
                if item.get("expires_at") is not None and float(item["expires_at"]) <= now_ts
            ]
        finally:
            db.close()

        expired: list[dict[str, Any]] = []
        for approval_id in expiring_ids:
            try:
                expired.append(self._transition(approval_id, ApprovalStatus.EXPIRED))
            except ApprovalGovernanceError:
                continue
        return {"expired": expired, "count": len(expired)}

    def summary(
        self,
        *,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        github_repo_full_name: str | None = None,
    ) -> dict[str, Any]:
        db = self._db()
        try:
            summary = db.summarize_code_approval_requests(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                orchestrated_run_id=orchestrated_run_id,
                github_repo_full_name=github_repo_full_name,
            )
        finally:
            db.close()
        safe = redact_for_api(summary)
        return {
            "status": safe.get("status", {}),
            "risk_class": safe.get("risk_class", {}),
            "kind": safe.get("kind", {}),
        }

    def validate_for_execution(
        self,
        approval_id: str,
        *,
        expected_kind: str,
        expected_requested_action: str,
        expected_resource_type: str | None = None,
        expected_resource_id: str | None = None,
        expected_github_repo_full_name: str | None = None,
        expected_github_issue_number: int | None = None,
        expected_github_pr_number: int | None = None,
        expected_requested_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._db()
        try:
            approval = db.get_code_approval_request(approval_id)
        finally:
            db.close()
        if not approval:
            raise ApprovalGovernanceError(f"Approval not found: {approval_id}")

        status = str(approval.get("status") or "")
        if status == ApprovalStatus.PENDING:
            raise ApprovalGovernanceError("Approval is still pending")
        if status == ApprovalStatus.REJECTED:
            raise ApprovalGovernanceError("Approval was rejected")
        if status == ApprovalStatus.CANCELLED:
            raise ApprovalGovernanceError("Approval was cancelled")
        if status == ApprovalStatus.EXPIRED:
            raise ApprovalGovernanceError("Approval expired")
        if status == ApprovalStatus.EXECUTED:
            raise ApprovalGovernanceError("Approval already executed")
        if status == ApprovalStatus.FAILED:
            raise ApprovalGovernanceError("Approval already failed")
        if status != ApprovalStatus.APPROVED:
            raise ApprovalGovernanceError(f"Approval status is not executable: {status}")

        expires_at = approval.get("expires_at")
        if expires_at is not None and float(expires_at) <= _now_ts():
            try:
                self._transition(approval_id, ApprovalStatus.EXPIRED)
            except ApprovalGovernanceError:
                pass
            raise ApprovalGovernanceError("Approval expired")

        if approval.get("kind") != expected_kind:
            raise ApprovalGovernanceError("Approval kind does not match requested action")
        if str(approval.get("requested_action") or "") != str(expected_requested_action or ""):
            raise ApprovalGovernanceError("Approval action does not match requested action")

        current_fingerprint = str((approval.get("metadata") or {}).get("binding_fingerprint") or "")
        expected_fingerprint = _binding_fingerprint(
            kind=expected_kind,
            requested_action=expected_requested_action,
            resource_type=expected_resource_type,
            resource_id=expected_resource_id,
            github_repo_full_name=expected_github_repo_full_name,
            github_issue_number=expected_github_issue_number,
            github_pr_number=expected_github_pr_number,
            requested_payload=expected_requested_payload or {},
        )
        if not current_fingerprint or current_fingerprint != expected_fingerprint:
            raise ApprovalGovernanceError("Approval binding does not match requested resource/payload")

        return self._serialize(approval)

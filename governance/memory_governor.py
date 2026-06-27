"""Memory governance admission schema and logging."""

from __future__ import annotations

from enum import Enum
import hashlib
import os
from typing import Any, Dict, Mapping, Optional

from .evidence import append_hash_chained_event
from .export_safety import SecretScanner


class MemoryAdmissionDecision(str, Enum):
    CANDIDATE_ONLY = "candidate_only"
    ADMIT = "admit"
    REJECT_SECURITY = "reject_security"
    REJECT_OVER_CAPACITY = "reject_over_capacity"
    REJECT_POLICY = "reject_policy"
    REMOVE = "remove"


class MemoryGovernor:
    """Record memory candidate and admission decisions.

    This class does not decide what the user prefers; it makes each durable
    memory write auditable with owner/scope/sensitivity/rollback metadata.
    Governance logs must not become a second raw secret store, so rejected or
    secret-looking content is represented by hashes, lengths, and redacted
    placeholders rather than echoed bytes.
    """

    def __init__(self, *, profile: str | None = None):
        self.profile = profile or os.getenv("HERMES_PROFILE") or os.getenv("HERMES_ACTIVE_PROFILE") or "default"
        self._secret_scanner = SecretScanner()

    @staticmethod
    def _sha256_content(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _redacted_placeholder(field_name: str, content: str) -> str:
        return f"[REDACTED:{field_name}:chars={len(content or '')}]"

    def _content_log_fields(self, field_name: str, content: str, *, redact: bool) -> Dict[str, Any]:
        content = content or ""
        findings = self._secret_scanner.find(content)
        should_redact = redact or bool(findings)
        return {
            field_name: self._redacted_placeholder(field_name, content) if should_redact else content,
            f"{field_name}_sha256": self._sha256_content(content),
            f"{field_name}_length": len(content),
            f"{field_name}_redacted": should_redact,
            f"{field_name}_secret_findings_count": len(findings),
        }

    def propose_candidate(
        self,
        *,
        candidate_content: str,
        source_ref: str,
        suggested_scope: str,
        suggested_owner: str | None = None,
        suggested_sensitivity: str = "general",
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        redact_candidate = suggested_sensitivity.lower() != "general"
        payload = {
            "event_type": "memory_candidate",
            "status": MemoryAdmissionDecision.CANDIDATE_ONLY.value,
            "source_ref": source_ref,
            "suggested_scope": suggested_scope,
            "suggested_owner": suggested_owner or self.profile,
            "suggested_sensitivity": suggested_sensitivity,
            "proposer_profile": self.profile,
            "metadata": dict(metadata or {}),
        }
        payload.update(self._content_log_fields("candidate_content", candidate_content, redact=redact_candidate))
        return append_hash_chained_event("memory_candidates", payload)

    def record_admission(
        self,
        *,
        operation: str,
        memory_target: str,
        proposed_content: str,
        normalized_content: str | None = None,
        source_ref: str = "tool:memory",
        decision: MemoryAdmissionDecision | str,
        decision_reason: str = "",
        owner: str | None = None,
        scope: str = "profile",
        sensitivity_level: str = "general",
        rollback_ref: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        decision_value = decision.value if isinstance(decision, MemoryAdmissionDecision) else str(decision)
        normalized = normalized_content if normalized_content is not None else proposed_content
        redact_content = decision_value != MemoryAdmissionDecision.ADMIT.value
        payload = {
            "event_type": "memory_admission",
            "operation": operation,
            "memory_target": memory_target,
            "memory_target_owner": owner or self.profile,
            "scope": scope,
            "sensitivity_level": sensitivity_level,
            "proposer_profile": self.profile,
            "source_ref": source_ref,
            "decision": decision_value,
            "decision_reason": decision_reason,
            "rollback_ref": rollback_ref,
            "metadata": dict(metadata or {}),
        }
        payload.update(self._content_log_fields("proposed_content", proposed_content, redact=redact_content))
        payload.update(self._content_log_fields("normalized_content", normalized, redact=redact_content))
        return append_hash_chained_event("memory_admission_log", payload)


def record_memory_tool_decision(
    *,
    operation: str,
    memory_target: str,
    proposed_content: str,
    decision: MemoryAdmissionDecision | str,
    decision_reason: str = "",
    normalized_content: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Best-effort admission logging for the existing memory tool."""
    try:
        MemoryGovernor().record_admission(
            operation=operation,
            memory_target=memory_target,
            proposed_content=proposed_content,
            normalized_content=normalized_content,
            decision=decision,
            decision_reason=decision_reason,
            metadata=metadata,
        )
    except Exception:
        # Memory writes must not fail solely because the governance ledger cannot
        # be appended; the Policy Gate logs a separate failure if strict mode is
        # enabled around the calling tool.
        pass

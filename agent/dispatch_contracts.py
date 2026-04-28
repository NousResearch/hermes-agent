"""Dispatch contract models for bounded worker orchestration.

These contracts intentionally describe receipts and verification evidence only.
They do not launch workers, choose providers, or alter model routing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "dispatch-contracts/v1"


class TaskComplexity(StrEnum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    HARD = "hard"


class Stakes(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IRREVERSIBLE = "irreversible"


class Reversibility(StrEnum):
    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"


class ToolReach(StrEnum):
    NONE = "none"
    LOCAL_FILES = "local_files"
    GITHUB = "github"
    BROWSER = "browser"
    WORKSPACE = "workspace"
    EXTERNAL_ACCOUNT = "external_account"


class MemoryDependency(StrEnum):
    NONE = "none"
    LOCAL_CONTEXT = "local_context"
    PROJECT_MEMORY = "project_memory"
    DURABLE_MEMORY_REQUIRED = "durable_memory_required"


class VerificationNeed(StrEnum):
    NONE = "none"
    LIGHTWEIGHT = "lightweight"
    DETERMINISTIC = "deterministic"
    INDEPENDENT_REVIEW = "independent_review"


class AttemptStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


class VerificationStatus(StrEnum):
    NOT_CHECKED = "not_checked"
    CLAIMED = "claimed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class EvidenceItemStatus(StrEnum):
    CLAIMED = "claimed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class EvidenceKind(StrEnum):
    FILE_READBACK = "file_readback"
    FILE_DIFF = "file_diff"
    GITHUB_READBACK = "github_readback"
    SOURCE_URL = "source_url"
    TEST_OUTPUT = "test_output"
    COMMAND_OUTPUT = "command_output"
    LOG_EXCERPT = "log_excerpt"
    ARTIFACT = "artifact"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class TaskClassification(ContractModel):
    task_complexity: TaskComplexity
    stakes: Stakes
    reversibility: Reversibility
    tool_reach: ToolReach
    memory_dependency: MemoryDependency
    verification_need: VerificationNeed


class BudgetPolicy(ContractModel):
    max_worker_sessions_per_issue: int = Field(ge=1)
    max_parallel_workers_default: int = Field(ge=1)
    max_worker_turns: int = Field(ge=1)
    max_worker_runtime_minutes: int = Field(ge=1)
    max_escalations_to_sota: int = Field(ge=0)
    stop_after_failed_attempts: int = Field(ge=1)
    require_ceo_approval_above_stakes: Literal["medium", "high"] = "high"

    @model_validator(mode="after")
    def _parallelism_cannot_exceed_total_sessions(self) -> BudgetPolicy:
        if self.max_parallel_workers_default > self.max_worker_sessions_per_issue:
            raise ValueError(
                "max_parallel_workers_default cannot exceed max_worker_sessions_per_issue"
            )
        return self


class WorkerAttemptMetadata(ContractModel):
    work_order_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    worker_session_id: str = Field(min_length=1)
    worker_role: str = Field(min_length=1)
    harness: str = Field(min_length=1)
    provider_model: str = Field(min_length=1)
    profile_or_toolset: str | None = None
    branch: str | None = None
    worktree: str | None = None
    assigned_scope: str = Field(min_length=1)
    allowed_paths: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    status: AttemptStatus
    touched_files: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    evidence_packet_path: str | None = None
    verification_status: VerificationStatus = VerificationStatus.NOT_CHECKED
    escalation_or_failure_reason: str | None = None
    external_approval_id: str | None = None

    @field_validator("allowed_paths", "allowed_tools", "touched_files", "commands_run")
    @classmethod
    def _dedupe_preserving_order(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            if not value:
                raise ValueError("list entries must be non-empty strings")
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    @model_validator(mode="after")
    def _fail_closed_metadata(self) -> WorkerAttemptMetadata:
        if self.status == AttemptStatus.SUCCEEDED and not self.evidence_packet_path:
            raise ValueError("evidence_packet_path is required when status is succeeded")
        if self.status in {AttemptStatus.FAILED, AttemptStatus.BLOCKED, AttemptStatus.ESCALATED}:
            if not self.escalation_or_failure_reason:
                raise ValueError(
                    "escalation_or_failure_reason is required for failed, blocked, or escalated attempts"
                )
        external_markers = {"external_account", "gmail", "gmail_send", "email", "slack", "x", "twitter"}
        if any(tool in external_markers for tool in self.allowed_tools):
            if not self.external_approval_id:
                raise ValueError("external_approval_id is required for external account tools")
        return self


class EvidenceItem(ContractModel):
    kind: EvidenceKind
    reference: str = Field(min_length=1)
    observed_at: str | None = None
    status: EvidenceItemStatus
    excerpt: str | None = None


class ConductorVerification(ContractModel):
    status: VerificationStatus
    commands: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvidencePacket(ContractModel):
    summary: str = Field(min_length=1)
    items: list[EvidenceItem] = Field(default_factory=list)
    conductor_verification: ConductorVerification


class DispatchReceipt(ContractModel):
    schema_version: Literal["dispatch-contracts/v1"] = SCHEMA_VERSION
    work_order_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    task_classification: TaskClassification
    budget_policy: BudgetPolicy
    worker_attempt: WorkerAttemptMetadata
    evidence: EvidencePacket

    @model_validator(mode="after")
    def _receipt_integrity(self) -> DispatchReceipt:
        if (
            self.worker_attempt.work_order_id != self.work_order_id
            or self.worker_attempt.attempt_id != self.attempt_id
        ):
            raise ValueError(
                "receipt and worker_attempt must share the same work_order_id and attempt_id"
            )
        if self.worker_attempt.status == AttemptStatus.SUCCEEDED and not self.evidence.items:
            raise ValueError("succeeded receipts require at least one evidence item")
        if self.worker_attempt.verification_status == VerificationStatus.VERIFIED:
            if self.evidence.conductor_verification.status != VerificationStatus.VERIFIED:
                raise ValueError("verified attempts require verified conductor_verification")
        evidence_refs = {item.reference for item in self.evidence.items}
        for touched_file in self.worker_attempt.touched_files:
            if touched_file not in evidence_refs:
                raise ValueError(f"touched file {touched_file!r} is missing from evidence references")
        return self


def validate_dispatch_receipt(payload: dict[str, Any]) -> DispatchReceipt:
    """Parse and validate a worker dispatch receipt with fail-closed semantics."""
    return DispatchReceipt.model_validate(payload)


def dispatch_contract_json_schemas() -> dict[str, dict[str, Any]]:
    """Return machine-readable JSON schemas for the Slice 1 dispatch contracts."""
    return {
        "TaskClassification": TaskClassification.model_json_schema(),
        "BudgetPolicy": BudgetPolicy.model_json_schema(),
        "WorkerAttemptMetadata": WorkerAttemptMetadata.model_json_schema(),
        "EvidencePacket": EvidencePacket.model_json_schema(),
        "DispatchReceipt": DispatchReceipt.model_json_schema(),
    }

"""Typed v1 request models and stable result construction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ArtifactRef(StrictModel):
    repository_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    path: str = Field(min_length=3)
    commit: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        if value.startswith("/") or "\\" in value:
            raise ValueError("artifact path must be repository-relative")
        return value


class ContractRef(ArtifactRef):
    contract_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{2,63}$")


class RepositoryRef(StrictModel):
    repository_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    canonical_remote: str = Field(min_length=8)


class TimeoutPolicy(StrictModel):
    max_runtime_seconds: int = Field(ge=60, le=43200)
    heartbeat_timeout_seconds: int = Field(ge=30, le=3600)


class RetryPolicy(StrictModel):
    max_attempts: int = Field(ge=1, le=3)
    retryable_terminal_states: list[
        Literal["CRASHED", "PROVIDER_UNAVAILABLE", "RATE_LIMITED"]
    ]


class DispatchRequest(StrictModel):
    """Untrusted caller intent.  It contains identities, never coordinates."""

    schema_version: Literal["1.0.0"]
    dispatch_id: str
    idempotency_key: str = Field(
        min_length=32, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    cycle_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{2,127}$")
    contract_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{2,63}$")
    repository_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    builder_role: Literal["primary_builder"]
    expected_cycle_revision: int = Field(ge=0)
    completion_schema_version: Literal["1.0.0"]

    @field_validator("dispatch_id")
    @classmethod
    def valid_dispatch_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("dispatch_id must be a UUID") from exc
        if str(parsed) != value:
            raise ValueError("dispatch_id must use canonical UUID text")
        return value


class ResolvedDispatchRequest(StrictModel):
    """Trusted runtime expansion consumed by the adapter and native worker."""

    schema_version: Literal["1.0.0"]
    dispatch_id: str
    idempotency_key: str = Field(
        min_length=32, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    cycle_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{2,127}$")
    contract: ContractRef
    repository: RepositoryRef
    worktree_path: str = Field(min_length=2)
    branch: str = Field(min_length=1, max_length=255)
    expected_head_sha: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    allowed_path_manifest: ArtifactRef
    validation_profile: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    builder_role: Literal["primary_builder"]
    timeout_policy: TimeoutPolicy
    retry_policy: RetryPolicy
    completion_schema_version: Literal["1.0.0"]

    @field_validator("dispatch_id")
    @classmethod
    def valid_dispatch_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("dispatch_id must be a UUID") from exc
        if str(parsed) != value:
            raise ValueError("dispatch_id must use canonical UUID text")
        return value

    @field_validator("worktree_path")
    @classmethod
    def absolute_worktree(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("worktree_path must be absolute")
        return value

    @field_validator("branch")
    @classmethod
    def valid_branch(cls, value: str) -> str:
        if value in {"HEAD", "-"}:
            raise ValueError("detached or option-like branch forbidden")
        return value


def observed_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def result_record(
    *,
    operation: str,
    dispatch_id: str,
    cycle_id: str,
    principal: str,
    request_sha256: str,
    status: str,
    side_effects_state: str,
    terminal: bool,
    attempt_count: int = 0,
    task_id: str | None = None,
    run_ids: list[str] | None = None,
    evidence: dict | None = None,
    errors: list[dict] | None = None,
    audit_refs: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "capability_id": "hermes.builder_dispatch.v1",
        "operation": operation,
        "dispatch_id": dispatch_id,
        "cycle_id": cycle_id,
        "caller_principal": principal,
        "request_sha256": request_sha256,
        "status": status,
        "side_effects_state": side_effects_state,
        "terminal": terminal,
        "attempt_count": attempt_count,
        "kanban_task_id": task_id,
        "kanban_run_ids": run_ids or [],
        "completion_evidence": evidence,
        "errors": errors or [],
        "audit_event_refs": audit_refs or [],
        "observed_at": observed_at(),
    }

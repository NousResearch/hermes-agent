"""Canonical Brain protocol boundary for Local Muncho runtime hooks."""

from __future__ import annotations

from typing import Protocol, Mapping, Any

from agent.local_muncho.types import (
    ApprovalRecord,
    AuditEvent,
    CodexTaskCreate,
    CodexTaskPatch,
    HeartbeatPayload,
    KnowledgeContext,
    LeaseState,
    LockHandle,
    RuntimeEvent,
    SupportCasePatch,
)


class CanonicalBrainUnavailable(RuntimeError):
    """Raised when an enabled runtime has no Canonical Brain backend."""


class CanonicalBrain(Protocol):
    def load_knowledge_context(self, scope: str, *, max_chars: int) -> KnowledgeContext:
        ...

    def read_active_lease(self) -> LeaseState | None:
        ...

    def refresh_active_lease(
        self,
        payload: HeartbeatPayload,
        ttl_seconds: int,
    ) -> LeaseState:
        ...

    def acquire_lock(
        self,
        key: str,
        value: Mapping[str, Any],
        ttl_seconds: int,
    ) -> LockHandle:
        ...

    def release_lock(self, handle: LockHandle) -> None:
        ...

    def write_runtime_event(self, event: RuntimeEvent) -> None:
        ...

    def write_audit_log(self, event: AuditEvent) -> None:
        ...

    def write_discord_event(self, event: Mapping[str, Any]) -> None:
        ...

    def upsert_support_case(self, case: SupportCasePatch) -> None:
        ...

    def record_approval(self, approval: ApprovalRecord) -> None:
        ...

    def create_codex_task(self, task: CodexTaskCreate) -> str:
        ...

    def update_codex_task(self, task_id: str, patch: CodexTaskPatch) -> None:
        ...

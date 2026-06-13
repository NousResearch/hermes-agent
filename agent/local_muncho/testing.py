"""In-memory Canonical Brain implementation for Local Muncho unit tests."""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

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
    utc_ts,
)


class InMemoryCanonicalBrain:
    def __init__(
        self,
        *,
        lease: LeaseState | None = None,
        lease_sequence: Sequence[LeaseState | None] | None = None,
        knowledge_records: Mapping[str, str] | None = None,
        knowledge_version: str = "in-memory",
    ) -> None:
        self.lease = lease
        self._lease_sequence = list(lease_sequence or [])
        self.knowledge_records = dict(knowledge_records or {})
        self.knowledge_version = knowledge_version
        self.runtime_events: list[RuntimeEvent] = []
        self.audit_events: list[AuditEvent] = []
        self.discord_events: list[Mapping[str, Any]] = []
        self.support_cases: list[SupportCasePatch] = []
        self.approvals: list[ApprovalRecord] = []
        self.codex_tasks: dict[str, dict[str, Any]] = {}
        self.locks: dict[str, LockHandle] = {}

    def load_knowledge_context(self, scope: str, *, max_chars: int) -> KnowledgeContext:
        selected: list[Mapping[str, Any]] = []
        chunks: list[str] = []
        for path, content in self.knowledge_records.items():
            if "FORBIDDEN_TO_LLM" in content or "PROTECTED_LIVE" in content:
                continue
            if "INTERNAL_ONLY" not in content and "CUSTOMER_SAFE_SHARED" not in content:
                continue
            selected.append({"path": path, "version": self.knowledge_version})
            chunks.append(f"# {path}\n{content}")
        text = "\n\n".join(chunks)[:max_chars]
        self.write_runtime_event(
            RuntimeEvent(
                event_type="knowledge_context_loaded",
                status="ok",
                context=None,  # type: ignore[arg-type]
                metadata={"scope": scope, "version": self.knowledge_version},
            )
        )
        return KnowledgeContext(
            scope=scope,
            text=text,
            version=self.knowledge_version,
            records=tuple(selected),
        )

    def read_active_lease(self) -> LeaseState | None:
        if self._lease_sequence:
            return self._lease_sequence.pop(0)
        return self.lease

    def refresh_active_lease(
        self,
        payload: HeartbeatPayload,
        ttl_seconds: int,
    ) -> LeaseState:
        expires_at = utc_ts() + max(1, int(ttl_seconds))
        self.lease = LeaseState(
            lease_owner=payload.runtime_id,
            active_runtime=payload.runtime_kind,
            expires_at=expires_at,
            metadata=dict(payload.metadata),
        )
        return self.lease

    def acquire_lock(
        self,
        key: str,
        value: Mapping[str, Any],
        ttl_seconds: int,
    ) -> LockHandle:
        existing = self.locks.get(key)
        now = utc_ts()
        if existing and existing.expires_at and existing.expires_at > now:
            return LockHandle(key=key, token="", acquired=False, expires_at=existing.expires_at)
        handle = LockHandle(
            key=key,
            token=str(value.get("token") or uuid.uuid4()),
            acquired=True,
            expires_at=now + max(1, int(ttl_seconds)),
        )
        self.locks[key] = handle
        return handle

    def release_lock(self, handle: LockHandle) -> None:
        existing = self.locks.get(handle.key)
        if existing and existing.token == handle.token:
            self.locks.pop(handle.key, None)

    def write_runtime_event(self, event: RuntimeEvent) -> None:
        self.runtime_events.append(event)

    def write_audit_log(self, event: AuditEvent) -> None:
        self.audit_events.append(event)

    def write_discord_event(self, event: Mapping[str, Any]) -> None:
        self.discord_events.append(dict(event))

    def upsert_support_case(self, case: SupportCasePatch) -> None:
        self.support_cases.append(case)

    def record_approval(self, approval: ApprovalRecord) -> None:
        self.approvals.append(approval)

    def create_codex_task(self, task: CodexTaskCreate) -> str:
        task_id = f"codex-{len(self.codex_tasks) + 1}"
        self.codex_tasks[task_id] = {"create": task, "patches": []}
        return task_id

    def update_codex_task(self, task_id: str, patch: CodexTaskPatch) -> None:
        self.codex_tasks.setdefault(task_id, {"patches": []})["patches"].append(patch)

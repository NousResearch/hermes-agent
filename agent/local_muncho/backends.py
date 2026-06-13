"""No-network backend boundary for the Local Muncho runtime.

Phase 2 intentionally stops at the adapter contract.  This module names the
Redis/Postgres boundary without importing client libraries or opening sockets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from agent.local_muncho.brain import CanonicalBrainUnavailable
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


@dataclass(frozen=True)
class BackendDescriptor:
    """Configuration-only description of a future Canonical Brain backend."""

    kind: str
    configured: bool
    reason: str
    env_var: str = ""
    schema: str = ""


class UnavailableCanonicalBrain:
    """CanonicalBrain-shaped placeholder that never performs I/O."""

    def __init__(
        self,
        reason: str = "Local Muncho backend is not implemented",
    ) -> None:
        self.reason = reason

    def _unavailable(self) -> None:
        raise CanonicalBrainUnavailable(self.reason)

    def load_knowledge_context(self, scope: str, *, max_chars: int) -> KnowledgeContext:
        self._unavailable()

    def read_active_lease(self) -> LeaseState | None:
        self._unavailable()

    def refresh_active_lease(
        self,
        payload: HeartbeatPayload,
        ttl_seconds: int,
    ) -> LeaseState:
        self._unavailable()

    def acquire_lock(
        self,
        key: str,
        value: Mapping[str, Any],
        ttl_seconds: int,
    ) -> LockHandle:
        self._unavailable()

    def release_lock(self, handle: LockHandle) -> None:
        self._unavailable()

    def write_runtime_event(self, event: RuntimeEvent) -> None:
        self._unavailable()

    def write_audit_log(self, event: AuditEvent) -> None:
        self._unavailable()

    def write_discord_event(self, event: Mapping[str, Any]) -> None:
        self._unavailable()

    def upsert_support_case(self, case: SupportCasePatch) -> None:
        self._unavailable()

    def record_approval(self, approval: ApprovalRecord) -> None:
        self._unavailable()

    def create_codex_task(self, task: CodexTaskCreate) -> str:
        self._unavailable()

    def update_codex_task(self, task_id: str, patch: CodexTaskPatch) -> None:
        self._unavailable()


def _brain_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    runtime_cfg = config.get("muncho_runtime")
    if isinstance(runtime_cfg, Mapping):
        config = runtime_cfg
    brain_cfg = config.get("brain")
    return brain_cfg if isinstance(brain_cfg, Mapping) else {}


def describe_configured_backends(
    config: Mapping[str, Any] | None = None,
) -> tuple[BackendDescriptor, ...]:
    """Return backend readiness from config/env presence without connecting."""

    brain_cfg = _brain_config(config)
    schema = str(brain_cfg.get("schema") or "muncho_internal_support")
    postgres_env = str(brain_cfg.get("postgres_dsn_env") or "MUNCHO_POSTGRES_DSN")
    redis_env = str(brain_cfg.get("redis_url_env") or "MUNCHO_REDIS_URL")

    postgres_present = bool(os.getenv(postgres_env, "").strip())
    redis_present = bool(os.getenv(redis_env, "").strip())
    return (
        BackendDescriptor(
            kind="postgres",
            configured=postgres_present,
            reason=(
                "dsn env var is present; adapter intentionally not connected in phase 2"
                if postgres_present
                else "dsn env var is not set"
            ),
            env_var=postgres_env,
            schema=schema,
        ),
        BackendDescriptor(
            kind="redis",
            configured=redis_present,
            reason=(
                "url env var is present; adapter intentionally not connected in phase 2"
                if redis_present
                else "url env var is not set"
            ),
            env_var=redis_env,
            schema=schema,
        ),
    )


def build_canonical_brain_backend(
    config: Mapping[str, Any] | None = None,
) -> UnavailableCanonicalBrain | None:
    """Return the no-network placeholder when backend config is present.

    The real Redis lease / Postgres audit and knowledge adapters belong here in
    a later phase.  Returning ``None`` for an unconfigured backend preserves the
    runtime's existing fail-closed "canonical brain unavailable" behavior.
    """

    descriptors = describe_configured_backends(config)
    configured = [item.kind for item in descriptors if item.configured]
    if not configured:
        return None
    return UnavailableCanonicalBrain(
        "Local Muncho backend adapter is configured but not implemented "
        f"in this phase: {', '.join(configured)}"
    )


# TODO(local-muncho): Implement Redis lease and Postgres audit/knowledge
# adapters here once live backend work is in scope.
# TODO(local-muncho): Route-back, approval mutation, kanban dashboard writes,
# and Codex task persistence should be wired through this boundary once their
# stable write seams are isolated.


__all__ = [
    "BackendDescriptor",
    "UnavailableCanonicalBrain",
    "build_canonical_brain_backend",
    "describe_configured_backends",
]

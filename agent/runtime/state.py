"""Serializable runtime state models for Hermes Agent.

These models are minimal by design. They are not a replacement for SessionDB,
Gateway sessions, or filesystem checkpoints. They provide a stable snapshot
shape that future capability layers can use for recovery, comparison, handoff,
and controlled self-improvement records.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RuntimeMutationType(StrEnum):
    """Mutation categories for controlled self-improvement records."""

    SKILL_PATCH = "skill_patch"
    MEMORY_ENTRY = "memory_entry"
    CRON_PROMPT_PATCH = "cron_prompt_patch"
    CONFIG_PATCH = "config_patch"
    IDENTITY_RULE_PATCH = "identity_rule_patch"
    CODE_PATCH = "code_patch"


class RuntimeMutationRisk(StrEnum):
    """Coarse mutation risk level for future gate/eval layers."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelRuntimeState(BaseModel):
    """Model/provider state captured for a runtime snapshot."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    api_mode: str | None = None
    fallback_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolRuntimeState(BaseModel):
    """Tool state captured for a runtime snapshot."""

    model_config = ConfigDict(extra="forbid")

    name: str
    toolset: str | None = None
    enabled: bool = True
    concurrency_safe: bool | None = None
    risk: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRuntimeState(BaseModel):
    """Agent identity/state subset needed to interpret a runtime snapshot."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    display_name: str | None = None
    platform: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeMutation(BaseModel):
    """Typed, auditable mutation proposal or applied change.

    Phase 0 only records mutation intent and rollback metadata. It deliberately
    does not implement gate hierarchy or automatic application; those should be
    based on real mutation history, not premature abstraction.
    """

    model_config = ConfigDict(extra="forbid")

    mutation_id: str
    mutation_type: RuntimeMutationType
    scope: str
    rationale: str
    risk: RuntimeMutationRisk = RuntimeMutationRisk.MEDIUM
    validator: str | None = None
    rollback_hint: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class RuntimeSessionState(BaseModel):
    """A minimal session snapshot for recovery, handoff, and comparison."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    agent: AgentRuntimeState
    model: ModelRuntimeState
    platform: str | None = None
    thread_id: str | None = None
    task_id: str | None = None
    tools: list[ToolRuntimeState] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    mutations: list[RuntimeMutation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

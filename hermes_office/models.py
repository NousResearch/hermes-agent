"""Pydantic v2 models for Hermes Digital Office.

These are the single source of truth for every JSON document persisted under
``$HERMES_HOME/office/`` and every payload exchanged over the REST/WebSocket API.
The TypeScript mirror lives in ``hermes_office/frontend/src/types.ts``.

See ``.kiro/specs/digital-office-ui/design.md`` §3 for the full schema rationale.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ────────────────────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────────────────────


class Activity(str, Enum):
    """A digital employee's runtime activity (drives the office sim animation)."""

    OFFLINE = "offline"
    RESTING = "resting"
    LEARNING = "learning"
    TALKING = "talking"
    WORKING = "working"


class Zone(str, Enum):
    """Office layout zones; each :class:`Activity` maps to exactly one zone."""

    REST = "rest"
    LEARN = "learn"
    TALK = "talk"
    WORK = "work"


ACTIVITY_TO_ZONE: dict[Activity, Zone] = {
    Activity.OFFLINE: Zone.REST,
    Activity.RESTING: Zone.REST,
    Activity.LEARNING: Zone.LEARN,
    Activity.TALKING: Zone.TALK,
    Activity.WORKING: Zone.WORK,
}


def zone_for(activity: Activity) -> Zone:
    return ACTIVITY_TO_ZONE[activity]


# Ids:    `dept_<10 hex>`, `emp_<10 hex>`, `task_<10 hex>`
_ID_RE = re.compile(r"^(emp|dept|task)_[a-z0-9]{6,32}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_SPRITE_IDS: tuple[str, ...] = (
    "robot-1",
    "robot-2",
    "cat",
    "fox",
    "panda",
    "wizard",
    "scientist",
    "writer",
    "designer",
    "analyst",
    "translator",
    "tutor",
)


def _utcnow() -> datetime:
    """UTC timestamp helper. Wrapped so tests can monkeypatch a single point."""
    return datetime.now(tz=timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


# ────────────────────────────────────────────────────────────────────────────
# Avatar / Employee
# ────────────────────────────────────────────────────────────────────────────


class AvatarStyle(BaseModel):
    """How the sprite is drawn on the office canvas."""

    model_config = ConfigDict(extra="forbid")

    sprite_id: str = "robot-1"
    hue: int = Field(default=200, ge=0, le=359)

    @field_validator("sprite_id")
    @classmethod
    def _validate_sprite(cls, value: str) -> str:
        if value not in _SPRITE_IDS:
            raise ValueError(
                f"sprite_id must be one of {_SPRITE_IDS}, got {value!r}"
            )
        return value


class Employee(BaseModel):
    """A configured digital employee.

    When the active runtime is ``hermes`` the configuration is materialised into a
    real :class:`run_agent.AIAgent` on each ``run_task`` call.  When the runtime is
    ``simulated`` no agent is created — only synthetic activity events.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _new_id("emp"))
    department_id: str
    name: str = Field(min_length=1, max_length=64)
    role: str = Field(default="Helper", min_length=1, max_length=80)
    avatar: AvatarStyle = Field(default_factory=AvatarStyle)

    # Brain
    model: str = Field(min_length=1, max_length=128)
    provider: str | None = None
    base_url: str | None = None

    # Skills & tools
    enabled_toolsets: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    system_prompt: str = ""

    runtime: Literal["simulated", "hermes"] = "simulated"
    activity: Activity = Activity.RESTING
    revision: int = 1

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("id", "department_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(f"id must match {_ID_RE.pattern}, got {value!r}")
        return value

    def with_revision_bumped(self) -> "Employee":
        return self.model_copy(
            update={"revision": self.revision + 1, "updated_at": _utcnow()}
        )


# ────────────────────────────────────────────────────────────────────────────
# Department
# ────────────────────────────────────────────────────────────────────────────


class Department(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _new_id("dept"))
    name: str = Field(min_length=1, max_length=50)
    mission: str = Field(default="", max_length=500)
    color: str = "#7c3aed"
    runtime_default: Literal["simulated", "hermes"] = "simulated"
    employee_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(f"id must match {_ID_RE.pattern}, got {value!r}")
        return value

    @field_validator("color")
    @classmethod
    def _validate_color(cls, value: str) -> str:
        if not _HEX_COLOR_RE.match(value):
            raise ValueError(f"color must match {_HEX_COLOR_RE.pattern}, got {value!r}")
        return value


# ────────────────────────────────────────────────────────────────────────────
# Task / ActivityEvent
# ────────────────────────────────────────────────────────────────────────────


TaskStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _new_id("task"))
    department_id: str | None = None
    employee_id: str | None = None
    text: str = Field(min_length=1, max_length=8_000)
    status: TaskStatus = "queued"
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_summary: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


ActivityKind = Literal[
    "state_change",
    "tool_call",
    "tool_result",
    "assistant",
    "clarify",
    "error",
]


class ActivityEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ts: datetime = Field(default_factory=_utcnow)
    employee_id: str
    department_id: str
    kind: ActivityKind
    text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Skill resolver
# ────────────────────────────────────────────────────────────────────────────


class ResolvedRole(BaseModel):
    """Output of :class:`SkillResolver.resolve`."""

    model_config = ConfigDict(extra="forbid")

    recommended_toolsets: list[str]
    recommended_skills: list[str]
    model_hint: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_md: str = ""
    matched_keywords: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Capacity
# ────────────────────────────────────────────────────────────────────────────


class GPU(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "unknown"
    vram_gb: float = Field(ge=0.0)


class HostProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cores: int = Field(ge=1)
    ram_gb: float = Field(ge=0.5)
    gpus: list[GPU] = Field(default_factory=list)
    os: str = "unknown"


class ModelProfile(BaseModel):
    """Inputs for the capacity calculation. All numbers are intentionally
    explicit (not autodetected) so the math is deterministic and auditable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider_kind: Literal["local", "api"] = "local"
    params_b: float = Field(ge=0.0, description="Billions of parameters")
    quant_bits: float = Field(default=4.0, ge=1.0, le=32.0)
    ctx_tokens: int = Field(ge=512)
    kv_bytes_per_token: float = Field(default=1024.0, ge=0.0)
    avg_tokens_per_response: int = Field(default=600, ge=1)
    tps_local: float = Field(default=30.0, ge=0.0, description="tokens/sec")
    api_p50_ms_per_token: float = Field(default=10.0, ge=0.0)
    usd_per_mtok_in: float = Field(default=0.0, ge=0.0)
    usd_per_mtok_out: float = Field(default=0.0, ge=0.0)
    rate_limit_rpm: int = Field(default=60, ge=1)


class CapacityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: HostProfile
    model: ModelProfile
    employee_count: int
    recommended_concurrency: int
    expected_p95_latency_ms: int
    est_usd_per_hour: float
    memory_headroom_gb: float
    notes: list[str] = Field(default_factory=list)

"""Request/response models for POST /v1/agent/execute_skill.

Wave-A primitive: synchronous, structured skill execution.  The
endpoint takes a ``skill_id``, a free-form ``inputs`` map, and
returns either ``status=ok`` with validated ``outputs`` or
``status=error`` with one or more structured ``errors``.

The endpoint is deliberately small: it owns input parsing, schema
validation against the per-skill input/output schemas declared in
each SKILL.md frontmatter, and surfaces errors in a way the desktop
shell (and future external orchestrators) can render without
guessing at error shapes.

Long-running / multi-step skill execution is out of scope for this
primitive — that lives behind ``POST /v1/runs`` and the jobs API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# Hard upper bounds.  These exist to keep the synchronous endpoint
# bounded so a misbehaving caller can't pin the gateway worker
# while waiting for an oversized payload to serialise.  Values are
# generous for normal office-skill workloads.
MAX_INPUT_BYTES = 256 * 1024            # 256 KiB request body cap
MAX_TIMEOUT_SECONDS = 120               # 2 minutes
DEFAULT_TIMEOUT_SECONDS = 30
MAX_CONTEXT_BUDGET_TOKENS = 200_000
DEFAULT_CONTEXT_BUDGET_TOKENS = 8_000


JobInputs = Dict[str, Any]
"""Free-form JSON object passed to the skill.

Validation against the skill's declared input schema (if any) is
performed by the executor, not by the request model.  We accept any
JSON object here so we can return a structured ``schema_violation``
error rather than a Pydantic 422 with a generic message — the
former is easier for the desktop UI to render.
"""


class AgentExecuteRequest(BaseModel):
    """POST /v1/agent/execute_skill request body."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description=(
            "Skill identifier: either a bare skill name "
            "('summarise-doc') or a qualified plugin skill "
            "('example-plugin:summarise-doc')."
        ),
    )
    inputs: JobInputs = Field(
        default_factory=dict,
        description="Skill-specific input payload. Validated against the skill's declared input schema.",
    )
    context_budget_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_CONTEXT_BUDGET_TOKENS,
        description="Soft cap on context tokens.  Falls back to DEFAULT_CONTEXT_BUDGET_TOKENS when omitted.",
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        le=MAX_TIMEOUT_SECONDS,
        description="Wall-clock cap on the synchronous execution.  Falls back to DEFAULT_TIMEOUT_SECONDS when omitted.",
    )

    def effective_timeout(self) -> float:
        return float(self.timeout_seconds) if self.timeout_seconds else float(DEFAULT_TIMEOUT_SECONDS)

    def effective_context_budget(self) -> int:
        return int(self.context_budget_tokens) if self.context_budget_tokens else int(DEFAULT_CONTEXT_BUDGET_TOKENS)


# Stable, machine-parseable error codes.  Keep this list small —
# every code is part of the public API and the desktop shell will
# branch on it.
ExecuteErrorCode = Literal[
    "skill_not_found",
    "schema_violation",
    "timeout",
    "execution_failed",
    "invalid_request",
    "executor_unavailable",
]


class ExecuteError(BaseModel):
    """One structured error from a failed skill execution."""

    model_config = ConfigDict(extra="forbid")

    code: ExecuteErrorCode
    message: str
    # Optional pointer into the offending field for schema_violation
    # (e.g. "inputs.recipient"); free-form for other codes.
    path: Optional[str] = None
    # Optional adapter for downstream telemetry — never user-facing.
    details: Optional[Dict[str, Any]] = None


class AgentExecuteResponse(BaseModel):
    """POST /v1/agent/execute_skill response body."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    skill_id: str
    outputs: Optional[Dict[str, Any]] = None
    errors: List[ExecuteError] = Field(default_factory=list)
    # Wall-clock duration in seconds (not billed tokens — that lives
    # on the runs API).  Useful for client-side budgets.
    duration_seconds: Optional[float] = None

"""Shadow task envelope — the versioned unit of work for a tool execution.

Phase 0 of the quad-rework contract is *observational*: the envelope records
what the tool seam can actually observe (which tool, which args, which cwd,
which session/task) and marks every safety field it cannot know as ``None``.
It never invents a permissive default — an unknown network policy is recorded
as unknown, not as "none".

:func:`validate_envelope` reports the resulting gaps. In this phase the verdict
is *recorded*, never enforced; enforcement is a later, separately gated phase.
"""

from __future__ import annotations

import hashlib
import math
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional


ENVELOPE_SCHEMA_VERSION = 1

_MAX_OBJECTIVE_CHARS = 2000

#: Safety fields an envelope must carry before it may ever gate execution.
#: A shadow envelope legitimately lacks most of them; :func:`validate_envelope`
#: names the gaps so the incompleteness is evidence rather than a silent
#: assumption.
MANDATORY_SAFETY_FIELDS: tuple[str, ...] = (
    "idempotency_key",
    "roots",
    "permissions_read",
    "permissions_write",
    "permissions_spawn",
    "network_policy",
    "exec_policy",
    "destructive_policy",
    "budget_usd_max",
    "budget_tokens_max",
    "budget_wall_clock_s_max",
    "deadline_utc",
    "lease_holder",
    "fence",
)


@dataclass(frozen=True)
class TaskEnvelope:
    """One sealed description of one execution surface's unit of work."""

    envelope_version: int
    envelope_id: str
    action_id: str
    session_id: Optional[str]
    task_id: Optional[str]
    tool_call_id: Optional[str]
    objective: str
    tool_name: str
    execution_surface: str
    lane: str

    cwd: Optional[str]
    roots: Optional[tuple[str, ...]]
    permissions_read: Optional[tuple[str, ...]]
    permissions_write: Optional[tuple[str, ...]]
    permissions_spawn: Optional[tuple[str, ...]]
    network_policy: Optional[str]
    exec_policy: Optional[str]
    destructive_policy: Optional[str]

    budget_usd_max: Optional[float]
    budget_tokens_max: Optional[int]
    budget_wall_clock_s_max: Optional[float]
    deadline_utc: Optional[str]

    retry_max_attempts: Optional[int]
    retry_eligible: Optional[bool]
    cancellation_mode: Optional[str]
    cancellation_grace_s: Optional[float]

    evidence_required_receipt_kinds: Optional[tuple[str, ...]]
    evidence_min_receipts: Optional[int]
    verifier_required: Optional[bool]
    verifier_lane_must_differ: Optional[bool]
    verifier_clean_workspace_required: Optional[bool]

    planner_hash: Optional[str]
    policy_hash: Optional[str]
    input_hash: str

    attempt_id: Optional[str]
    lease_holder: Optional[str]
    lease_granted_utc: Optional[str]
    lease_ttl_s: Optional[float]
    fence: Optional[int]

    idempotency_key: Optional[str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping (tuples become lists)."""
        payload = asdict(self)
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in payload.items()
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(payload: Any, *, depth: int = 0) -> Any:
    """Normalize to JSON values without invoking arbitrary representations."""
    if depth > 12:
        return "[truncated]"
    if payload is None or isinstance(payload, (str, bool, int)):
        return payload
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else f"[nonfinite:{payload.hex()}]"
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(key, str):
                key_text = key
            elif key is None:
                key_text = "null"
            elif isinstance(key, bool):
                key_text = "true" if key else "false"
            elif isinstance(key, int):
                key_text = str(key)
            elif isinstance(key, float):
                key_text = str(key) if math.isfinite(key) else f"[nonfinite-key:{key.hex()}]"
            else:
                key_text = f"[unsupported-key:{type(key).__name__}]"
            normalized[key_text] = _json_safe(value, depth=depth + 1)
        return normalized
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value, depth=depth + 1) for value in payload]
    return f"[unsupported:{type(payload).__name__}]"


def canonical_json(payload: Any) -> str:
    """Canonical JSON without calling user-defined ``str`` or ``repr``."""
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(payload: Any) -> str:
    """SHA-256 over the canonical JSON encoding of ``payload``."""
    return _sha256(canonical_json(payload))


def envelope_hash(env: TaskEnvelope) -> str:
    """SHA-256 over the envelope's canonical JSON bytes."""
    return canonical_hash(env.to_dict())


def build_shadow_envelope(
    *,
    tool_name: str,
    args: Any = None,
    cwd: Optional[str] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    objective: Optional[str] = None,
    envelope_id: Optional[str] = None,
    action_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> TaskEnvelope:
    """Build an envelope from what the seam can observe.

    Every field this function cannot honestly determine is left ``None``.
    That is the point: shadow mode records incompleteness so the gap shows up
    in :func:`validate_envelope` output instead of being papered over with a
    permissive default.
    """
    goal = objective if objective is not None else f"execute tool {tool_name}"

    return TaskEnvelope(
        envelope_version=ENVELOPE_SCHEMA_VERSION,
        envelope_id=envelope_id or f"e-{uuid.uuid4().hex}",
        action_id=action_id or f"a-{uuid.uuid4().hex}",
        session_id=session_id,
        task_id=task_id,
        tool_call_id=tool_call_id,
        objective=str(goal)[:_MAX_OBJECTIVE_CHARS],
        tool_name=str(tool_name),
        execution_surface="local_tool",
        lane="hermes",
        cwd=cwd,
        # ── Unknown in shadow mode: recorded as unknown, never guessed ──
        roots=None,
        permissions_read=None,
        permissions_write=None,
        permissions_spawn=None,
        network_policy=None,
        exec_policy=None,
        destructive_policy=None,
        budget_usd_max=None,
        budget_tokens_max=None,
        budget_wall_clock_s_max=None,
        deadline_utc=None,
        retry_max_attempts=None,
        retry_eligible=None,
        cancellation_mode=None,
        cancellation_grace_s=None,
        evidence_required_receipt_kinds=None,
        evidence_min_receipts=None,
        verifier_required=None,
        verifier_lane_must_differ=None,
        verifier_clean_workspace_required=None,
        planner_hash=None,
        policy_hash=None,
        input_hash=canonical_hash(args if args is not None else {}),
        attempt_id=None,
        lease_holder=None,
        lease_granted_utc=None,
        lease_ttl_s=None,
        fence=None,
        idempotency_key=None,
        created_at=created_at or _utc_now(),
    )


def validate_envelope(env: TaskEnvelope) -> list[str]:
    """Report absent mandatory safety fields. Never raises, never blocks.

    An unknown ``envelope_version`` is reported too — a consumer that would
    enforce this envelope must fail closed on it, but this function only names
    the problem.
    """
    missing: list[str] = []
    if getattr(env, "envelope_version", None) != ENVELOPE_SCHEMA_VERSION:
        missing.append("envelope_version")
    for field in MANDATORY_SAFETY_FIELDS:
        if getattr(env, field, None) is None:
            missing.append(field)
    return missing


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "MANDATORY_SAFETY_FIELDS",
    "TaskEnvelope",
    "build_shadow_envelope",
    "canonical_hash",
    "canonical_json",
    "envelope_hash",
    "validate_envelope",
]

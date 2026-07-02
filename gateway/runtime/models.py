"""Runtime API models for /v1/runs contract.

Isolated from the existing api_server.py adapter so both the current
aiohttp API server and future WebUI adapters can share a single
runtime run manager.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_SECRET_KEY_NAMES = frozenset({
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "authorization",
    "bearer",
    "api-key",
})


def _normalize_key(key: str) -> str:
    return str(key).strip().lower().replace("-", "_")


def _is_secret_key(key: str) -> bool:
    return _normalize_key(key) in _SECRET_KEY_NAMES


def redact_secrets(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively redact secret-like values from any dict or list.

    String values are passed through the existing ``redact_sensitive_text``
    from ``agent.redact``.  Dict keys whose *normalized* name matches a
    known secret key (``api_key``, ``token``, ``password``, etc.) have
    their values replaced with a mask.
    """
    if _depth > 20:
        return obj
    try:
        from agent.redact import redact_sensitive_text
    except ImportError:
        redact_sensitive_text = lambda s: s
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if _is_secret_key(k) and isinstance(v, str) and v:
                result[k] = "<<redacted>>"
            else:
                result[k] = redact_secrets(v, _depth=_depth + 1)
        return result
    if isinstance(obj, list):
        return [redact_secrets(item, _depth=_depth + 1) for item in obj]
    if isinstance(obj, str):
        return redact_sensitive_text(obj)
    return obj


RUN_STATUS_QUEUED = "queued"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_AWAITING_APPROVAL = "awaiting_approval"
RUN_STATUS_AWAITING_CLARIFY = "awaiting_clarify"
RUN_STATUS_PAUSED = "paused"
RUN_STATUS_CANCELLING = "cancelling"
RUN_STATUS_CANCELLED = "cancelled"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_EXPIRED = "expired"

TERMINAL_STATUSES = frozenset({
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_EXPIRED,
})

EVENT_RUN_STARTED = "run.started"
EVENT_RUN_STATUS = "run.status"
EVENT_TOKEN_DELTA = "token.delta"
EVENT_REASONING_DELTA = "reasoning.delta"
EVENT_REASONING_DONE = "reasoning.done"
EVENT_PROGRESS = "progress"
EVENT_TOOL_STARTED = "tool.started"
EVENT_TOOL_UPDATED = "tool.updated"
EVENT_TOOL_DONE = "tool.done"
EVENT_APPROVAL_REQUESTED = "approval.requested"
EVENT_APPROVAL_RESOLVED = "approval.resolved"
EVENT_CLARIFY_REQUESTED = "clarify.requested"
EVENT_CLARIFY_RESOLVED = "clarify.resolved"
EVENT_TITLE_UPDATED = "title.updated"
EVENT_USAGE_UPDATED = "usage.updated"
EVENT_USAGE_FINAL = "usage.final"
EVENT_ERROR = "error"
EVENT_DONE = "done"

TERMINAL_EVENT_TYPES = frozenset({EVENT_DONE, EVENT_ERROR})


@dataclass
class RuntimeEvent:
    """A single structured event in a run's lifecycle."""

    event_id: str
    seq: int
    run_id: str
    session_id: str
    type: str
    created_at: float = field(default_factory=time.time)
    terminal: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        result = {
            "event_id": self.event_id,
            "seq": self.seq,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "type": self.type,
            "created_at": self.created_at,
            "terminal": self.terminal,
            "payload": self.payload,
        }
        if redact:
            result = redact_secrets(result)
        return result


@dataclass
class RuntimeStatus:
    """Pollable status for a run."""

    run_id: str
    session_id: str
    status: str = RUN_STATUS_QUEUED
    last_event_id: Optional[str] = None
    last_seq: int = 0
    terminal: bool = False
    controls: List[str] = field(default_factory=list)
    pending_approval_ids: List[str] = field(default_factory=list)
    pending_clarify_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "last_event_id": self.last_event_id,
            "last_seq": self.last_seq,
            "terminal": self.terminal,
            "controls": self.controls,
            "pending_approval_ids": self.pending_approval_ids,
            "pending_clarify_ids": self.pending_clarify_ids,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if redact:
            result = redact_secrets(result)
        return result

"""
response_mode.py — Hermes output-pipeline submodule C§1.7

Phase 2 §4.2 reference: «Response Mode Selector takes the Dispatch descriptor
and (optionally) an upstream result and decides the shape of the final response:
brief acknowledgement, full detail, executive summary, approval request, or a
status-update ping.»

Event emitted: hermes.response.shaped
Wire-up: task C§1.9 (EventEmitter instance injected by turn_handler)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agent.modules.event_emitter import EventEmitter


# ---------------------------------------------------------------------------
# Input types (stubs matching C§1.6 Dispatch)
# ---------------------------------------------------------------------------

class Dispatch(BaseModel):
    """Routing-policy output (C§1.6)."""
    target: str
    payload: dict[str, Any] = Field(default_factory=dict)


class UpstreamResult(BaseModel):
    """
    Optional result from an upstream system (OpenClaw, MCP tool, state-engine).
    Present only after async completion; absent for synchronous/clarify paths.
    """
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

ResponseMode = Literal[
    "brief",
    "detail",
    "executive-summary",
    "approval-request",
    "status-update",
]

Tone = Literal["neutral", "executive", "technical", "conversational"]


class ResponseShape(BaseModel):
    """
    Descriptor consumed by the response renderer to format the final reply.

    mode:     high-level response style.
    tone:     language register.
    template: optional template key; None means free-form prose.
    """
    mode: ResponseMode
    tone: Tone
    template: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routing table: (target, has_result) → (mode, tone, template)
# ---------------------------------------------------------------------------

_SHAPE_TABLE: dict[
    tuple[str, bool],
    tuple[ResponseMode, Tone, Optional[str]],
] = {
    ("direct",        True):  ("brief",            "conversational", None),
    ("direct",        False): ("brief",            "conversational", None),
    ("openclaw",      True):  ("executive-summary","executive",      "mission_complete"),
    ("openclaw",      False): ("status-update",    "neutral",        "mission_queued"),
    ("state-engine",  True):  ("detail",           "technical",      None),
    ("state-engine",  False): ("status-update",    "neutral",        "workflow_queued"),
    ("mcp-tool",      True):  ("detail",           "technical",      None),
    ("mcp-tool",      False): ("status-update",    "neutral",        None),
    ("clarify",       False): ("brief",            "conversational", "clarification_request"),
    ("approve",       False): ("approval-request", "executive",      "approval_request"),
    ("escalate",      False): ("brief",            "conversational", "escalation_notice"),
}

_DEFAULT_SHAPE: tuple[ResponseMode, Tone, Optional[str]] = ("brief", "neutral", None)

# ---------------------------------------------------------------------------
# Module-level emitter (injected by turn_handler)
# ---------------------------------------------------------------------------

_emitter: Optional[EventEmitter] = None


def set_emitter(emitter: EventEmitter) -> None:
    """Inject the shared event emitter.

    Called by turn_handler.run_turn() before processing.
    """
    global _emitter
    _emitter = emitter


def select_response_mode(
    dispatch: Dispatch,
    upstream_result: Optional[UpstreamResult] = None,
) -> ResponseShape:
    """
    Select the response shape for the rendered reply.

    Decision matrix is keyed on (dispatch.target, upstream_result is not None).
    Falls back to ('brief', 'neutral', None) for unknown targets.

    Emits: hermes.response.shaped (stdout JSON line)

    Ref: Phase 2 §4.2 — Response Mode Selector.
    """
    has_result = upstream_result is not None
    key = (dispatch.target, has_result)

    mode, tone, template = _SHAPE_TABLE.get(key, _DEFAULT_SHAPE)

    meta: dict[str, Any] = {"dispatch_target": dispatch.target}
    if upstream_result is not None:
        meta["upstream_success"] = upstream_result.success
        if not upstream_result.success and upstream_result.error:
            meta["upstream_error"] = upstream_result.error
            # Override to brief on error so the user gets a concise failure notice
            mode = "brief"
            tone = "conversational"
            template = "error_notice"

    shape = ResponseShape(mode=mode, tone=tone, template=template, metadata=meta)

    if _emitter is not None:
        _emitter.emit("hermes.response.shaped", {
            "mode": shape.mode,
            "tone": shape.tone,
            "template": shape.template,
            "dispatch_target": dispatch.target,
        })

    return shape

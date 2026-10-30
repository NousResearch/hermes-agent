"""
routing_policy.py — Hermes output-pipeline submodule C§1.6

Phase 2 §4.2 reference: «Routing Policy converts a Route classification and
optional MissionContract into a concrete Dispatch descriptor that specifies
exactly which system should receive control and what payload to forward.»

Event emitted: hermes.route.dispatched
Wire-up: task C§1.9 (EventEmitter instance injected by turn_handler)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agent.modules.event_emitter import EventEmitter


# ---------------------------------------------------------------------------
# Input types (stubs; see mission_compiler.py for canonical definitions)
# ---------------------------------------------------------------------------

class Route(BaseModel):
    """Classifier output (C§1.4). Passed in from the input pipeline."""
    target: str
    payload: dict[str, Any] = Field(default_factory=dict)


class MissionContract(BaseModel):
    """
    Compiler output (C§1.5). Present when a formal contract was assembled;
    None for clarify/approve/escalate routes.
    """
    mission_id: str
    intent: str
    target: str
    priority: str
    context: dict[str, Any]
    payload: dict[str, Any]
    mode: str = "prose"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

DispatchTarget = Literal[
    "direct",
    "state-engine",
    "openclaw",
    "mcp-tool",
    "clarify",
    "approve",
    "escalate",
]


class Dispatch(BaseModel):
    """
    Concrete dispatch descriptor.

    target: system that should receive control.
    payload: forwarded payload (may include compiled MissionContract).
    """
    target: DispatchTarget
    payload: dict[str, Any] = Field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

def apply_routing_policy(
    route: Route,
    mission: Optional[MissionContract] = None,
) -> Dispatch:
    """
    Apply routing policy to produce a Dispatch.

    Rules (per Phase 2 §4.2 routing matrix):
    - 'openclaw'     → forward full MissionContract payload; reject if mission is None.
    - 'state-engine' → forward MissionContract + state-engine-specific metadata.
    - 'direct'       → forward payload as-is with optional mission summary.
    - 'mcp-tool'     → pass tool name + args from route.payload.
    - 'clarify'      → pass clarification prompt from route.payload.
    - 'approve'      → pass approval request metadata.
    - 'escalate'     → pass escalation reason + context.

    Emits: hermes.route.dispatched (stdout JSON line)

    Ref: Phase 2 §4.2 — Routing Policy.
    """
    target: DispatchTarget = route.target  # type: ignore[assignment]
    dispatch_payload: dict[str, Any] = {}

    if target == "openclaw":
        if mission is None:
            # Degenerate: no contract compiled — escalate instead
            target = "escalate"
            dispatch_payload = {
                "reason": "openclaw route requires a MissionContract; none compiled",
                "original_target": "openclaw",
                "route_payload": route.payload,
            }
        else:
            dispatch_payload = {
                "mission_id": mission.mission_id,
                "mission_contract": {
                    "intent": mission.intent,
                    "priority": mission.priority,
                    "context": mission.context,
                    "payload": mission.payload,
                    "mode": mission.mode,
                },
            }

    elif target == "state-engine":
        dispatch_payload = {
            "mission_id": mission.mission_id if mission else None,
            "graph_input": route.payload,
        }

    elif target == "direct":
        dispatch_payload = route.payload.copy()
        if mission:
            dispatch_payload["_mission_id"] = mission.mission_id

    elif target == "mcp-tool":
        dispatch_payload = {
            "tool": route.payload.get("tool"),
            "args": route.payload.get("args", {}),
        }

    elif target in ("clarify", "approve", "escalate"):
        dispatch_payload = route.payload.copy()

    dispatch = Dispatch(target=target, payload=dispatch_payload)

    if _emitter is not None:
        _emitter.emit("hermes.route.dispatched", {
            "target": dispatch.target,
            "mission_id": dispatch_payload.get("mission_id"),
        })

    return dispatch

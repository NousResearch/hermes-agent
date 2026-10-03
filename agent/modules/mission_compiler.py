"""
mission_compiler.py — Hermes output-pipeline submodule C§1.5

Phase 2 §4.2 reference: «Mission Compiler takes the Interpretation, ContextPackage,
and Route produced by the input pipeline and assembles a typed MissionContract
(per @agrv/mission-contract) ready for dispatch. Returns None when the intent
does not warrant a formal contract (clarification, approval-gate, escalation).»

Event emitted: hermes.mission.compiled
Wire-up: task C§1.9 (EventEmitter instance injected by turn_handler)
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.modules.event_emitter import EventEmitter


# ---------------------------------------------------------------------------
# Input types (lightweight stubs; replaced by typed imports in C§1.9)
# ---------------------------------------------------------------------------

class Interpretation(BaseModel):
    """Output of IntentInterpreter (C§1.2)."""
    intent: str
    confidence: float
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackage(BaseModel):
    """Output of ContextAssembler (C§1.3)."""
    session_id: str
    user_id: str
    history_summary: str
    kv: dict[str, Any] = Field(default_factory=dict)


class Route(BaseModel):
    """Output of RouteClassifier (C§1.4) / RoutingPolicy (C§1.6)."""
    target: str  # 'direct'|'state-engine'|'openclaw'|'mcp-tool'|'clarify'|'approve'|'escalate'
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class MissionContract(BaseModel):
    """
    Typed mission contract per @agrv/mission-contract.MissionContract.
    Fields aligned with Phase 2 §6 schema; wire-up task C§1.9 replaces this
    stub with the canonical Pydantic model from the monorepo package.
    """
    mission_id: str
    intent: str
    target: str
    priority: str  # 'low'|'medium'|'high'|'critical'
    context: dict[str, Any]
    payload: dict[str, Any]
    mode: str = "prose"  # 'prose'|'typed' — shim flag per Phase 3 §0.2


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

CONTRACTS_REQUIRING_MISSION = {"openclaw", "state-engine", "direct"}


def compile_mission(
    interpretation: Interpretation,
    context: ContextPackage,
    route: Route,
) -> Optional[MissionContract]:
    """
    Compile a MissionContract from input-pipeline outputs.

    Returns None when the route target is 'clarify', 'approve', or 'escalate'
    — those paths do not dispatch a formal mission.

    Emits: hermes.mission.compiled (stdout JSON line)

    Ref: Phase 2 §4.2 — Mission Compiler.
    """
    if route.target not in CONTRACTS_REQUIRING_MISSION:
        if _emitter is not None:
            _emitter.emit("hermes.mission.compiled", {
                "mission_contract": None,
                "reason": f"route.target={route.target!r} does not require a MissionContract",
                "session_id": context.session_id,
            })
        return None

    import uuid

    contract = MissionContract(
        mission_id=str(uuid.uuid4()),
        intent=interpretation.intent,
        target=route.target,
        priority=route.payload.get("priority", "medium"),
        context={
            "session_id": context.session_id,
            "user_id": context.user_id,
            "history_summary": context.history_summary,
            **context.kv,
        },
        payload=route.payload,
        mode="prose",  # shim: typed mode enabled in C§1.9 after @agrv/mission-contract import
    )

    if _emitter is not None:
        _emitter.emit("hermes.mission.compiled", {
            "mission_id": contract.mission_id,
            "intent": contract.intent,
            "target": contract.target,
            "mode": contract.mode,
            "session_id": context.session_id,
        })

    return contract

"""Central Hermes turn handler — C§1.9 wire-up.

Invokes the 8 input/output-pipeline submodules in declared order for every
inbound message turn:

    identity → context_loader → interpreter → intent_classifier →
    mission_compiler → routing_policy → response_mode → summarizer

The summarizer step is conditional: it only runs when an upstream system
has returned a ResultPackage (i.e. after async mission completion). For
synchronous turns it is skipped and ``summary`` is None in the result.

Behaviour is preserved from the inline logic that previously lived in
``gateway/run.py`` (_handle_message_with_agent) — this module wraps the
extracted submodules without redesigning the pipeline.

Event emitted per step: see individual submodule docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Event emitter
# ---------------------------------------------------------------------------
from agent.modules.event_emitter import EventEmitter

# ---------------------------------------------------------------------------
# Input-pipeline submodules (Pydantic types)
# ---------------------------------------------------------------------------
from agent.modules.identity import (
    IdentityPacket,
    SessionBootstrap,
    bootstrap_identity,
    set_emitter as set_identity_emitter,
)
from agent.modules.context_loader import (
    ContextPackage,
    UserMessage,
    assemble_context,
    set_emitter as set_context_emitter,
)
from agent.modules.interpreter import Interpretation, interpret, set_emitter as set_interp_emitter
from agent.modules.intent_classifier import ClassifiedIntent, Route, classify_intent, set_emitter as set_classifier_emitter

# ---------------------------------------------------------------------------
# Output-pipeline submodules (dataclass types — local stubs per C§1.5–C§1.8)
# ---------------------------------------------------------------------------
from agent.modules import mission_compiler as _mc
from agent.modules.mission_compiler import set_emitter as set_mc_emitter
from agent.modules.routing_policy import Dispatch, apply_routing_policy, set_emitter as set_routing_emitter
from agent.modules.response_mode import ResponseShape, UpstreamResult, select_response_mode, set_emitter as set_response_emitter
from agent.modules.summarizer import (
    CompanyKPIs,
    ExecutiveSummary,
    ResultPackage,
    summarize,
    set_emitter as set_summarizer_emitter,
)


# ---------------------------------------------------------------------------
# Turn input / result types
# ---------------------------------------------------------------------------


@dataclass
class TurnInput:
    """Everything needed to start a single Hermes turn.

    Fields mirror the gateway's inbound-message envelope; non-gateway callers
    (CLI, tests) can omit optional fields.
    """

    # Core
    session_id: str
    text: str
    source: str = "cli"  # e.g. "telegram", "slack", "cli"

    # Identity hints (may be absent for anonymous turns)
    raw_user_id: Optional[str] = None
    raw_company_id: Optional[str] = None

    # Optional extras forwarded from the gateway
    attachments: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    """Aggregate result of one completed Hermes turn.

    Every field maps to the output of one submodule so callers can inspect
    any stage of the pipeline.
    """

    identity: IdentityPacket
    context: ContextPackage
    interpretation: Interpretation
    classified: ClassifiedIntent
    mission: Optional[_mc.MissionContract]
    dispatch: Dispatch
    response_shape: ResponseShape
    summary: Optional[ExecutiveSummary]  # None unless upstream_result provided


# ---------------------------------------------------------------------------
# Type-bridge helpers
# ---------------------------------------------------------------------------


def _to_mc_interpretation(interp: Interpretation) -> _mc.Interpretation:
    """Bridge Pydantic Interpretation → mission_compiler's Pydantic model."""
    return _mc.Interpretation(
        intent=interp.intent,
        confidence=0.0,  # stub: classifier doesn't yet produce confidence
        raw_text=interp.raw_text,
        metadata=interp.metadata,
    )


def _to_mc_context(ctx: ContextPackage, identity: IdentityPacket) -> _mc.ContextPackage:
    """Bridge Pydantic ContextPackage → mission_compiler's dataclass."""
    return _mc.ContextPackage(
        session_id=identity.user_id,  # session_id proxied via user_id for stub
        user_id=identity.user_id,
        history_summary="",  # stub: full history summary deferred to C§2
        kv={
            "company_id": identity.company_id,
            "mode": identity.mode,
            "token_estimate": ctx.token_estimate,
        },
    )


def _to_mc_route(route: Route) -> _mc.Route:
    """Bridge intent_classifier.Route enum → mission_compiler's Route dataclass.

    The Route enum value maps to mission_compiler's target string by taking
    the lower-case, hyphen-separated form of the enum name where applicable.

    Mapping:
        ANSWER_DIRECTLY     → direct
        RECALL_MEMORY       → direct   (memory recall handled inline for stubs)
        INVOKE_TOOL         → mcp-tool
        DELEGATE_SPECIALIST → state-engine
        SUBMIT_OPENCLAW_JOB → openclaw
        CLARIFY_FIRST       → clarify
        DRAFT_FOR_APPROVAL  → approve
        ESCALATE_TO_ATTI    → escalate
    """
    _ROUTE_MAP: dict[str, str] = {
        "ANSWER_DIRECTLY": "direct",
        "RECALL_MEMORY": "direct",
        "INVOKE_TOOL": "mcp-tool",
        "DELEGATE_SPECIALIST": "state-engine",
        "SUBMIT_OPENCLAW_JOB": "openclaw",
        "CLARIFY_FIRST": "clarify",
        "DRAFT_FOR_APPROVAL": "approve",
        "ESCALATE_TO_ATTI": "escalate",
    }
    target = _ROUTE_MAP.get(route.value, "direct")
    return _mc.Route(target=target, payload={})


# ---------------------------------------------------------------------------
# Central turn handler
# ---------------------------------------------------------------------------


def run_turn(
    turn_input: TurnInput,
    upstream_result: Optional[UpstreamResult] = None,
    kpis: Optional[CompanyKPIs] = None,
) -> TurnResult:
    """Execute one complete Hermes turn through all 8 submodules.

    Parameters
    ----------
    turn_input:
        Inbound message envelope.
    upstream_result:
        Present when called after an async upstream system (OpenClaw,
        state-engine, MCP tool) has returned a result. Drives the
        response_mode selector and triggers the summarizer.
    kpis:
        Optional company KPI snapshot injected by the context layer.
        Passed through to the summarizer when upstream_result is present.

    Returns
    -------
    TurnResult
        Full pipeline output for this turn.
    """
    # ------------------------------------------------------------------
    # Inject shared EventEmitter into all 8 submodules
    # ------------------------------------------------------------------
    emitter = EventEmitter()
    set_identity_emitter(emitter)
    set_context_emitter(emitter)
    set_interp_emitter(emitter)
    set_classifier_emitter(emitter)
    set_mc_emitter(emitter)
    set_routing_emitter(emitter)
    set_response_emitter(emitter)
    set_summarizer_emitter(emitter)

    # ------------------------------------------------------------------
    # 1. Identity — SessionBootstrap → IdentityPacket
    # ------------------------------------------------------------------
    bootstrap = SessionBootstrap(
        raw_user_id=turn_input.raw_user_id,
        raw_company_id=turn_input.raw_company_id,
        session_id=turn_input.session_id,
        source=turn_input.source,
        metadata=turn_input.metadata,
    )
    identity: IdentityPacket = bootstrap_identity(bootstrap)

    # ------------------------------------------------------------------
    # 2. Context Loader — UserMessage + IdentityPacket → ContextPackage
    # ------------------------------------------------------------------
    user_msg = UserMessage(
        text=turn_input.text,
        attachments=turn_input.attachments,
        session_id=turn_input.session_id,
        turn_index=turn_input.turn_index,
    )
    ctx: ContextPackage = assemble_context(user_msg, identity)

    # ------------------------------------------------------------------
    # 3. Interpreter — UserMessage + ContextPackage → Interpretation
    # ------------------------------------------------------------------
    interp: Interpretation = interpret(user_msg, ctx)

    # ------------------------------------------------------------------
    # 4. Intent Classifier — Interpretation → ClassifiedIntent[Route]
    # ------------------------------------------------------------------
    classified: ClassifiedIntent = classify_intent(interp)

    # ------------------------------------------------------------------
    # 5. Mission Compiler — Interpretation + ContextPackage + Route → MissionContract?
    #    Bridges Pydantic types to mission_compiler's local dataclasses.
    # ------------------------------------------------------------------
    mc_interp = _to_mc_interpretation(interp)
    mc_ctx = _to_mc_context(ctx, identity)
    mc_route = _to_mc_route(classified.route)
    mission: Optional[_mc.MissionContract] = _mc.compile_mission(mc_interp, mc_ctx, mc_route)

    # ------------------------------------------------------------------
    # 6. Routing Policy — Route + MissionContract? → Dispatch
    #    routing_policy uses its own Route dataclass identical in shape to
    #    mission_compiler's; re-use mc_route (same type via same module).
    # ------------------------------------------------------------------
    from agent.modules.routing_policy import Route as _RPRoute  # noqa: PLC0415
    rp_route = _RPRoute(target=mc_route.target, payload=mc_route.payload)

    # Adapt mission_compiler.MissionContract → routing_policy.MissionContract
    # Both now Pydantic; use model_dump for a clean conversion.
    rp_mission = None
    if mission is not None:
        from agent.modules.routing_policy import MissionContract as _RPContract  # noqa: PLC0415
        rp_mission = _RPContract(**mission.model_dump())

    dispatch: Dispatch = apply_routing_policy(rp_route, rp_mission)

    # ------------------------------------------------------------------
    # 7. Response Mode Selector — Dispatch + UpstreamResult? → ResponseShape
    # ------------------------------------------------------------------
    # response_mode uses its own Dispatch dataclass; bridge from routing_policy.
    from agent.modules.response_mode import Dispatch as _RMDispatch  # noqa: PLC0415
    rm_dispatch = _RMDispatch(target=dispatch.target, payload=dispatch.payload)
    response_shape: ResponseShape = select_response_mode(rm_dispatch, upstream_result)

    # ------------------------------------------------------------------
    # 8. Summarizer — ResultPackage + CompanyKPIs? → ExecutiveSummary
    #    Only runs when an upstream result is present (completed missions).
    # ------------------------------------------------------------------
    summary: Optional[ExecutiveSummary] = None
    if upstream_result is not None:
        # TODO(C§2): replace with a real ResultPackage from the upstream system.
        result_pkg = ResultPackage(
            mission_id=mission.mission_id if mission else "unknown",
            success=upstream_result.success,
            outputs=upstream_result.data,
            errors=[upstream_result.error] if upstream_result.error else [],
        )
        summary = summarize(result_pkg, kpis)

    return TurnResult(
        identity=identity,
        context=ctx,
        interpretation=interp,
        classified=classified,
        mission=mission,
        dispatch=dispatch,
        response_shape=response_shape,
        summary=summary,
    )

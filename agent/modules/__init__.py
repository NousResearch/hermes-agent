"""Hermes input-pipeline submodules — Phase 3 §C§1.

Exposes the 4 named submodules extracted per the Phase-3 build plan.
Each module defines typed I/O (Pydantic BaseModel) and emits a
``hermes.*`` event via stdout JSON line on completion.

Wire-up to the central Hermes entrypoint is task C§1.9.

Submodules
----------
identity
    C§1.1 — SessionBootstrap → IdentityPacket (hermes.identity.bootstrap)
context_loader
    C§1.2 — UserMessage + IdentityPacket → ContextPackage (hermes.context.assembled)
interpreter
    C§1.3 — UserMessage + ContextPackage → Interpretation (hermes.interp.done)
intent_classifier
    C§1.4 — Interpretation → ClassifiedIntent[Route] (hermes.intent.classified)
"""

from agent.modules.identity import (
    IdentityPacket,
    SessionBootstrap,
    bootstrap_identity,
)
from agent.modules.context_loader import (
    ContextPackage,
    UserMessage,
    assemble_context,
)
from agent.modules.interpreter import (
    Interpretation,
    interpret,
)
from agent.modules.intent_classifier import (
    ClassifiedIntent,
    Route,
    classify_intent,
)

__all__ = [
    # identity
    "SessionBootstrap",
    "IdentityPacket",
    "bootstrap_identity",
    # context_loader
    "UserMessage",
    "ContextPackage",
    "assemble_context",
    # interpreter
    "Interpretation",
    "interpret",
    # intent_classifier
    "Route",
    "ClassifiedIntent",
    "classify_intent",
]

# ------ C§1.5-C§1.8 output pipeline ------
from agent.modules.mission_compiler import MissionContract, compile_mission
from agent.modules.routing_policy import Dispatch, apply_routing_policy
from agent.modules.response_mode import ResponseShape, UpstreamResult, select_response_mode
from agent.modules.summarizer import CompanyKPIs, ExecutiveSummary, ResultPackage, summarize

__all__.extend([
    "MissionContract", "compile_mission",
    "Dispatch", "apply_routing_policy",
    "UpstreamResult", "ResponseShape", "select_response_mode",
    "ResultPackage", "CompanyKPIs", "ExecutiveSummary", "summarize",
])

# ------ C§1.9 central turn handler ------
from agent.modules.turn_handler import TurnInput, TurnResult, run_turn

__all__.extend([
    "TurnInput",
    "TurnResult",
    "run_turn",
])

"""Hardened typed runtime for hermes-agent.

Lives under ``agent/runtime/``. See ``agent/runtime/README.md`` for the
integration guide and ``docs/plans/2026-05-15-smolagents-runtime-upgrade.md``
for the design rationale.

Goal alignment (governance-ready execution kernel):

    Prompt → Typed Step Machine → Validated Transition → Governed Tool Action → Replayable Runtime Trace

  * **Typed**     — every step is a frozen dataclass; failures are typed.
  * **Validated** — ``TransitionGuard`` rejects illegal step orderings.
  * **Governed**  — every tool call passes through ``GovernanceGate``; default
                    policy is deny-all.
  * **Replayable** — ``replay.run_from_trace(jsonl)`` reproduces a run from
                     its recorded JSONL with a deterministic clock + ids.
  * **Fail-closed** — the loop's default ``continue_on_error=False`` makes
                      every typed failure terminate the run.
  * **Audited**    — ``RunState`` records every mutation with a reason.

Public surface kept narrow on purpose — callers should import from
``agent.runtime`` rather than reaching into submodules.
"""

from .acgs_governance import (
    ACGSClient,
    ACGSDecisionReceipt,
    ACGSGovernance,
    ACGSRule,
    ACGSVerdict,
    Constitution,
    LocalACGSClient,
    Severity,
    build_acgs_governance_from_config,
)
from .callbacks import CallbackRegistry, StepCallback
from .governance import AllowAllGovernance, AllowListGovernance, DenyAllGovernance, GovernanceGate
from .interfaces import (
    FINAL_ANSWER_TOOL,
    Clock,
    FrozenClock,
    GovernanceContext,
    GovernanceDecision,
    GovernanceProtocol,
    GovernanceVerdict,
    IdSource,
    ModelOutput,
    ModelProtocol,
    SequentialIdSource,
    SystemClock,
    ToolHandlerProtocol,
    UuidIdSource,
    failure_from_exception,
)
from .loop import FinalAnswerCheck, MultiStepLoop
from .memory import AgentMemory
from .replay import (
    RecordedModel,
    RecordedToolHandler,
    ReplayExhausted,
    ReplayMissingToolOutput,
    ScriptedGovernance,
    build_replay_loop,
    run_from_trace,
)
from .result import RunResult, Timing, TokenUsage
from .state import RunState, StateFrozenError, StateMutation
from .steps import (
    ActionStep,
    FailureKind,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    StepFailure,
    TaskStep,
    ToolCall,
    ToolOutput,
)
from .transitions import LEGAL_TRANSITIONS, TransitionGuard

__all__ = [
    # steps
    "ActionStep",
    "FailureKind",
    "FinalAnswerStep",
    "MemoryStep",
    "PlanningStep",
    "StepFailure",
    "TaskStep",
    "ToolCall",
    "ToolOutput",
    # memory
    "AgentMemory",
    # result
    "RunResult",
    "Timing",
    "TokenUsage",
    # state
    "RunState",
    "StateFrozenError",
    "StateMutation",
    # callbacks
    "CallbackRegistry",
    "StepCallback",
    # interfaces
    "Clock",
    "FrozenClock",
    "FINAL_ANSWER_TOOL",
    "GovernanceContext",
    "GovernanceDecision",
    "GovernanceProtocol",
    "GovernanceVerdict",
    "IdSource",
    "ModelOutput",
    "ModelProtocol",
    "SequentialIdSource",
    "SystemClock",
    "ToolHandlerProtocol",
    "UuidIdSource",
    "failure_from_exception",
    # governance
    "AllowAllGovernance",
    "AllowListGovernance",
    "DenyAllGovernance",
    "GovernanceGate",
    # acgs-lite constitutional governance
    "ACGSClient",
    "ACGSDecisionReceipt",
    "ACGSGovernance",
    "ACGSRule",
    "ACGSVerdict",
    "Constitution",
    "LocalACGSClient",
    "Severity",
    "build_acgs_governance_from_config",
    # transitions
    "LEGAL_TRANSITIONS",
    "TransitionGuard",
    # loop
    "FinalAnswerCheck",
    "MultiStepLoop",
    # replay
    "RecordedModel",
    "RecordedToolHandler",
    "ReplayExhausted",
    "ReplayMissingToolOutput",
    "ScriptedGovernance",
    "build_replay_loop",
    "run_from_trace",
]

# -*- coding: utf-8 -*-
"""Public API for the bounded model-counterpoint workflow."""

from ._contracts import Artifact, Critique, Decision, GateResult, JudgeVerdict
from ._dispatcher import (
    DispatchConflict,
    DispatchDecision,
    DispatchOutcome,
    RequestEnvelope,
    SingleEntryDispatcher,
)
from ._hermes import (
    HermesAgentClient,
    HermesCallError,
    HermesCompletion,
    HermesCounterpointCallbacks,
    HermesRuntime,
)
from ._policy import (
    CounterpointDecision,
    NoIndependentRoute,
    RouteIdentity,
    RoutePolicyError,
    decide_counterpoint,
    eligible_routes,
)
from ._workflow import (
    CounterpointWorkflow,
    WorkflowContractError,
    WorkflowResult,
    WorkflowSpec,
)

__all__ = [
    "Artifact",
    "CounterpointDecision",
    "CounterpointWorkflow",
    "Critique",
    "Decision",
    "DispatchConflict",
    "DispatchDecision",
    "DispatchOutcome",
    "GateResult",
    "HermesAgentClient",
    "HermesCallError",
    "HermesCompletion",
    "HermesCounterpointCallbacks",
    "HermesRuntime",
    "JudgeVerdict",
    "NoIndependentRoute",
    "RequestEnvelope",
    "RouteIdentity",
    "RoutePolicyError",
    "SingleEntryDispatcher",
    "WorkflowContractError",
    "WorkflowResult",
    "WorkflowSpec",
    "decide_counterpoint",
    "eligible_routes",
]

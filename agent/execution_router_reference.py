"""Deterministic, credential-free reference fixture for execution-router providers."""

from __future__ import annotations

from dataclasses import dataclass

from agent.execution_router import (
    ExecutionRouteDecisionV1,
    ExecutionRouteRequestV1,
    ExecutionRouterCancellationSignalV1,
    ExecutionRouterProviderDescriptorV1,
)


@dataclass(frozen=True)
class FirstEligibleExecutionRouterProviderV1:
    """Select the first host-issued eligible candidate, or pass through when empty."""

    descriptor: ExecutionRouterProviderDescriptorV1

    def resolve_execution_route(
        self,
        request: ExecutionRouteRequestV1,
        _cancellation: ExecutionRouterCancellationSignalV1,
    ) -> ExecutionRouteDecisionV1:
        if not request.eligible_candidates:
            return ExecutionRouteDecisionV1.pass_through(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
            )
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id=request.eligible_candidates[0].candidate_id,
        )

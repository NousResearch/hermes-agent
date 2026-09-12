from __future__ import annotations


def _request(candidates):
    from agent.execution_router import (
        CONTRACT_VERSION,
        ExecutionKind,
        ExecutionRouteInstructionV1,
        ExecutionRoutePinsV1,
        ExecutionRouteRequestV1,
    )

    return ExecutionRouteRequestV1(
        contract_version=CONTRACT_VERSION,
        request_id="request-1",
        root_id="root-1",
        task_id=None,
        execution_id="execution-1",
        attempt_id="attempt-1",
        execution_kind=ExecutionKind.MAIN_TURN,
        surface_class="test",
        instruction=ExecutionRouteInstructionV1.absent(),
        pins=ExecutionRoutePinsV1(),
        native_candidate_id=None,
        eligibility_revision="revision-1",
        eligible_candidates=tuple(candidates),
        previous_attempt=None,
        request_digest=None,
    ).with_computed_digest()


def test_reference_provider_selects_only_the_first_host_issued_candidate():
    from agent.execution_router import (
        ExecutionKind,
        ExecutionRouteCandidateV1,
        ExecutionRouteDecisionKind,
        ExecutionRouterProviderDescriptorV1,
    )
    from agent.execution_router_reference import FirstEligibleExecutionRouterProviderV1

    candidates = (
        ExecutionRouteCandidateV1("host-candidate-2", "provider-b", "model-b", None),
        ExecutionRouteCandidateV1("host-candidate-1", "provider-a", "model-a", "high"),
    )
    provider = FirstEligibleExecutionRouterProviderV1(
        descriptor=ExecutionRouterProviderDescriptorV1(
            plugin_id="reference-fixture",
            plugin_version="1.0.0",
            provider_id="first-eligible",
            contract_version="1.0",
            supported_execution_kinds=tuple(ExecutionKind),
        )
    )

    decision = provider.resolve_execution_route(_request(candidates), object())
    empty = provider.resolve_execution_route(_request(()), object())

    assert decision.kind is ExecutionRouteDecisionKind.ROUTE
    assert decision.candidate_id == candidates[0].candidate_id
    assert empty.kind is ExecutionRouteDecisionKind.PASS_THROUGH
    assert empty.candidate_id is None

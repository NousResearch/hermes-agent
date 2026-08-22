from __future__ import annotations

import pytest

from gateway.hermes_tag import (
    ContextBundle,
    ContinuityMode,
    ContinuityStore,
    DecisionOutcome,
    HermesTagConfig,
    HermesTagKernel,
    LeaseAuthority,
    LeaseError,
    LeaseTampered,
    PolicyRule,
    RuntimeAuthority,
    SurfaceRef,
    TurnAdmission,
    bind_admission,
    bind_authority,
    bind_decision,
    bind_lease,
    capture_authority,
    clear_runtime_context,
    current_admission,
    current_decision,
    current_lease,
    new_id,
    scope_from_surface,
)


def _admission(principal, surface, scope) -> TurnAdmission:
    assert scope.continuity_id is not None
    return TurnAdmission(
        admission_id=new_id("admission"),
        principal=principal,
        surface=surface,
        scope=scope,
        continuity_id=scope.continuity_id,
        context=ContextBundle(facts=()),
        shadow=False,
    )


def _kernel(ledger) -> HermesTagKernel:
    return HermesTagKernel(
        ledger,
        HermesTagConfig.from_mapping({"enabled": True}),
        rules=(
            PolicyRule(
                "admin-may-grant-approval",
                DecisionOutcome.ALLOW,
                capabilities=("approval.grant",),
                roles=("admin",),
                reason="authenticated admin may grant exact approval",
            ),
            PolicyRule(
                "operator-may-spawn",
                DecisionOutcome.ALLOW,
                capabilities=("process.spawn",),
                roles=("operator",),
                reason="operator process policy matched",
            ),
            PolicyRule(
                "principal-may-read-context",
                DecisionOutcome.ALLOW,
                capabilities=("context.read",),
                reason="context read allowed",
            ),
        ),
        lease_authority=LeaseAuthority(b"a" * 32),
    )


def _second_admission(ledger, identity_store):
    principal = identity_store.create_principal(
        "Second operator",
        roles=("operator",),
    )
    surface = SurfaceRef(
        platform="slack",
        profile="default",
        scope_id="T_WORKSPACE",
        chat_id="C_SECOND",
        thread_id="1712345.200",
    )
    continuity = ContinuityStore(ledger).resolve_or_create(
        principal,
        surface,
        mode=ContinuityMode.ISOLATED,
    )
    scope = scope_from_surface(
        surface,
        principal_id=principal.principal_id,
        continuity_id=continuity.continuity_id,
    )
    return _admission(principal, surface, scope)


def _read_authority(kernel, admission):
    authorization = kernel.authorize(
        admission.principal,
        capability="context.read",
        action="read",
        resource="context",
        arguments={},
        scope=admission.scope,
    )
    assert authorization.decision.outcome is DecisionOutcome.ALLOW
    assert authorization.lease is not None
    return RuntimeAuthority(
        admission=admission,
        decision=authorization.decision,
        lease=authorization.lease,
    )


def test_nested_authority_clears_stale_decision_and_lease(
    ledger,
    identity_store,
    bound_principal,
    surface,
    scope,
):
    kernel = _kernel(ledger)
    first = _admission(bound_principal, surface, scope)
    second = _second_admission(ledger, identity_store)
    outer = _read_authority(kernel, first)

    clear_runtime_context()
    with bind_authority(outer):
        assert capture_authority() == outer

        with bind_authority(RuntimeAuthority(admission=second)):
            assert current_admission(required=True) == second
            assert current_decision() is None
            assert current_lease() is None
            assert capture_authority() == RuntimeAuthority(admission=second)

        assert capture_authority() == outer

        with bind_admission(second):
            assert current_admission(required=True) == second
            assert current_decision() is None
            assert current_lease() is None

        assert capture_authority() == outer

    assert current_admission() is None
    assert current_decision() is None
    assert current_lease() is None


def test_incremental_binding_validates_and_restores_complete_tuple(
    ledger,
    bound_principal,
    surface,
    scope,
):
    kernel = _kernel(ledger)
    admission = _admission(bound_principal, surface, scope)
    authority = _read_authority(kernel, admission)
    assert authority.decision is not None
    assert authority.lease is not None

    clear_runtime_context()
    with bind_admission(admission):
        assert current_decision() is None
        assert current_lease() is None
        with bind_decision(authority.decision):
            assert current_decision(required=True) == authority.decision
            assert current_lease() is None
            with bind_lease(authority.lease):
                assert capture_authority() == authority
            assert current_lease() is None
        assert current_decision() is None
    assert current_admission() is None


def test_runtime_authority_rejects_cross_scope_tuple(
    ledger,
    identity_store,
    bound_principal,
    surface,
    scope,
):
    kernel = _kernel(ledger)
    first = _admission(bound_principal, surface, scope)
    second = _second_admission(ledger, identity_store)
    authority = _read_authority(kernel, first)

    with pytest.raises(LeaseTampered, match="scope"):
        RuntimeAuthority(
            admission=second,
            decision=authority.decision,
            lease=authority.lease,
        )


def test_runtime_authority_rejects_lease_without_decision(
    ledger,
    bound_principal,
    surface,
    scope,
):
    kernel = _kernel(ledger)
    admission = _admission(bound_principal, surface, scope)
    authority = _read_authority(kernel, admission)

    with pytest.raises(LeaseTampered, match="without its policy decision"):
        RuntimeAuthority(admission=admission, lease=authority.lease)


def test_public_policy_facade_cannot_mint_raw_approval(ledger):
    kernel = _kernel(ledger)

    assert not hasattr(kernel.policy.approvals, "grant")
    assert not hasattr(kernel.policy.approvals, "consume_exact")


def test_governed_approval_grant_unlocks_one_exact_high_risk_effect(
    ledger,
    bound_principal,
    surface,
    scope,
):
    kernel = _kernel(ledger)
    admission = _admission(bound_principal, surface, scope)

    pending = kernel.authorize(
        bound_principal,
        capability="process.spawn",
        action="spawn",
        resource="process:worker",
        arguments={"command": ["python", "worker.py"]},
        scope=scope,
    )
    assert pending.decision.outcome is DecisionOutcome.REQUIRE_APPROVAL
    assert pending.lease is None

    grant_authority = kernel.authorize_approval_grant(
        admission,
        principal_id=bound_principal.principal_id,
        intent_digest=pending.intent.digest,
        scope_digest=scope.digest,
        ttl_seconds=300,
    )
    assert grant_authority.decision.outcome is DecisionOutcome.ALLOW
    assert grant_authority.lease is not None
    assert grant_authority.token is not None

    grant = kernel.grant_approval(
        admission,
        grant_authority,
        principal_id=bound_principal.principal_id,
        intent_digest=pending.intent.digest,
        scope_digest=scope.digest,
        ttl_seconds=300,
    )
    assert grant.approver_id == admission.principal.principal_id
    assert kernel.policy.approvals.get(grant.approval_id) == grant

    allowed = kernel.authorize(
        bound_principal,
        capability="process.spawn",
        action="spawn",
        resource="process:worker",
        arguments={"command": ["python", "worker.py"]},
        scope=scope,
    )
    assert allowed.decision.outcome is DecisionOutcome.ALLOW
    assert allowed.decision.approval_id == grant.approval_id
    assert allowed.lease is not None

    consumed = kernel.authorize(
        bound_principal,
        capability="process.spawn",
        action="spawn",
        resource="process:worker",
        arguments={"command": ["python", "worker.py"]},
        scope=scope,
    )
    assert consumed.decision.outcome is DecisionOutcome.REQUIRE_APPROVAL


def test_unprivileged_self_nomination_cannot_mint_approval(
    ledger,
    identity_store,
):
    kernel = _kernel(ledger)
    operator_admission = _second_admission(ledger, identity_store)
    target_digest = "a" * 64

    denied = kernel.authorize_approval_grant(
        operator_admission,
        principal_id=operator_admission.principal.principal_id,
        intent_digest=target_digest,
        scope_digest=operator_admission.scope.digest,
    )
    assert denied.decision.outcome is DecisionOutcome.DENY
    assert denied.lease is None

    with pytest.raises(LeaseError, match="not allowed"):
        kernel.grant_approval(
            operator_admission,
            denied,
            principal_id=operator_admission.principal.principal_id,
            intent_digest=target_digest,
            scope_digest=operator_admission.scope.digest,
        )

    connection = ledger.connection()
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM hermes_tag_approvals"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 0


def test_approval_authority_is_bound_to_exact_target(
    ledger,
    bound_principal,
    surface,
    scope,
):
    kernel = _kernel(ledger)
    admission = _admission(bound_principal, surface, scope)
    original_digest = "b" * 64
    changed_digest = "c" * 64

    authorization = kernel.authorize_approval_grant(
        admission,
        principal_id=bound_principal.principal_id,
        intent_digest=original_digest,
        scope_digest=scope.digest,
    )
    assert authorization.decision.outcome is DecisionOutcome.ALLOW

    with pytest.raises(LeaseTampered, match="does not match"):
        kernel.grant_approval(
            admission,
            authorization,
            principal_id=bound_principal.principal_id,
            intent_digest=changed_digest,
            scope_digest=scope.digest,
        )

    connection = ledger.connection()
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM hermes_tag_approvals"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 0

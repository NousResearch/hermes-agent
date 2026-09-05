from __future__ import annotations

import concurrent.futures
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gateway.hermes_tag import (
    ApprovalRequired,
    BudgetLimits,
    CapabilityRegistry,
    DecisionOutcome,
    HermesTagConfig,
    HermesTagKernel,
    LeaseAuthority,
    LeaseError,
    LeaseExpired,
    LeaseReplay,
    LeaseTampered,
    ObligationError,
    ObligationPhase,
    ObligationRegistry,
    PolicyDenied,
    PolicyEngine,
    PolicyRule,
    RiskLevel,
    ScopeRef,
    StorageError,
    arguments_digest,
)


def _intent(registry, scope, capability="message.send", **overrides):
    values = dict(
        capability=capability,
        action="send",
        resource="slack:C_ENGINEERING",
        arguments_digest=arguments_digest({"text": "hello"}),
        scope=scope,
    )
    values.update(overrides)
    return registry.normalize_intent(**values)


def test_builtin_registry_has_expected_horizontal_capabilities():
    names = CapabilityRegistry().names()
    assert len(names) == 20
    for expected in (
        "message.send",
        "tool.execute",
        "process.spawn",
        "network.egress",
        "admin.policy.write",
    ):
        assert expected in names


def test_unknown_capability_is_denied(scope):
    with pytest.raises(PolicyDenied, match="unknown capability"):
        _intent(CapabilityRegistry(), scope, capability="unknown.effect")


def test_registry_escalates_caller_risk_and_effect_flags(scope):
    intent, definition = _intent(
        CapabilityRegistry(),
        scope,
        capability="network.egress",
        risk=RiskLevel.LOW,
        external_effect=False,
        network_egress=False,
    )
    assert intent.risk is RiskLevel.HIGH
    assert intent.external_effect is True
    assert intent.network_egress is True
    assert definition.name == "network.egress"


def test_thread_manage_requires_thread_scope(bound_principal, continuity):
    scope = ScopeRef(
        profile="default",
        platform="slack",
        scope_id="T1",
        chat_id="C1",
        principal_id=bound_principal.principal_id,
        continuity_id=continuity.continuity_id,
    )
    with pytest.raises(Exception, match="thread_id"):
        _intent(CapabilityRegistry(), scope, capability="thread.manage")


def test_default_policy_is_deny(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    result = PolicyEngine(ledger).evaluate(bound_principal, intent, definition)
    assert result.decision.outcome is DecisionOutcome.DENY
    assert "default deny" in result.decision.reasons[0]


def test_principal_cannot_authorize_another_principals_scope(
    ledger, identity_store, scope
):
    other = identity_store.create_principal("Other", roles=("operator",))
    intent, definition = _intent(CapabilityRegistry(), scope)
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    result = engine.evaluate(other, intent, definition)
    assert result.decision.outcome is DecisionOutcome.DENY
    assert result.decision.reasons == (
        "principal does not own the action scope",
    )


def test_allow_rule_allows_medium_effect(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    engine = PolicyEngine(
        ledger,
        rules=(
            PolicyRule(
                "allow-slack-send",
                DecisionOutcome.ALLOW,
                capabilities=("message.send",),
                platforms=("slack",),
                roles=("operator",),
                reason="operator may send Slack messages",
            ),
        ),
    )
    result = engine.evaluate(bound_principal, intent, definition)
    assert result.decision.outcome is DecisionOutcome.ALLOW
    assert result.decision.matched_rules == ("allow-slack-send",)


def test_deny_overrides_higher_priority_allow(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    engine = PolicyEngine(
        ledger,
        rules=(
            PolicyRule(
                "allow",
                DecisionOutcome.ALLOW,
                capabilities=("message.*",),
                reason="allowed",
                priority=100,
            ),
            PolicyRule(
                "deny-channel",
                DecisionOutcome.DENY,
                capabilities=("message.send",),
                chat_ids=(scope.chat_id,),
                reason="channel frozen",
            ),
        ),
    )
    result = engine.evaluate(bound_principal, intent, definition)
    assert result.decision.outcome is DecisionOutcome.DENY
    assert result.decision.reasons == ("channel frozen",)


def test_guest_cannot_use_non_guest_capability(ledger, identity_store, scope):
    guest = identity_store.create_principal("Guest", guest=True, roles=("guest",))
    guest_scope = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id=scope.chat_id,
        thread_id=scope.thread_id,
        principal_id=guest.principal_id,
        continuity_id=None,
    )
    intent, definition = _intent(CapabilityRegistry(), guest_scope)
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    assert engine.evaluate(guest, intent, definition).decision.outcome is DecisionOutcome.DENY


def test_guest_can_use_guest_eligible_read(ledger, identity_store, scope):
    guest = identity_store.create_principal("Guest", guest=True, roles=("guest",))
    guest_scope = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id=scope.chat_id,
        principal_id=guest.principal_id,
    )
    intent, definition = _intent(
        CapabilityRegistry(),
        guest_scope,
        capability="context.read",
        action="read",
        resource="context",
    )
    engine = PolicyEngine(
        ledger,
        rules=(
            PolicyRule(
                "allow-context",
                DecisionOutcome.ALLOW,
                capabilities=("context.read",),
                reason="guest read",
            ),
        ),
    )
    assert engine.evaluate(guest, intent, definition).decision.outcome is DecisionOutcome.ALLOW


def test_high_risk_requires_exact_approval(ledger, bound_principal, scope):
    intent, definition = _intent(
        CapabilityRegistry(), scope, capability="process.spawn", action="spawn"
    )
    engine = PolicyEngine(
        ledger,
        rules=(
            PolicyRule(
                "allow-process",
                DecisionOutcome.ALLOW,
                capabilities=("process.spawn",),
                reason="process permitted",
            ),
        ),
    )
    first = engine.evaluate(bound_principal, intent, definition)
    assert first.decision.outcome is DecisionOutcome.REQUIRE_APPROVAL
    grant = engine.approvals.grant(
        principal_id=bound_principal.principal_id,
        approver_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    second = engine.evaluate(bound_principal, intent, definition)
    assert second.decision.outcome is DecisionOutcome.ALLOW
    assert second.decision.approval_id == grant.approval_id
    third = engine.evaluate(bound_principal, intent, definition)
    assert third.decision.outcome is DecisionOutcome.REQUIRE_APPROVAL


def test_approval_requires_known_principal_and_approver(
    ledger, bound_principal, scope
):
    engine = PolicyEngine(ledger)
    with pytest.raises(StorageError, match="must already exist"):
        engine.approvals.grant(
            principal_id=bound_principal.principal_id,
            approver_id="principal_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            intent_digest="a" * 64,
            scope_digest=scope.digest,
        )


def test_budget_denial_does_not_consume_exact_approval(
    ledger, bound_principal, scope
):
    intent, definition = _intent(
        CapabilityRegistry(), scope, capability="process.spawn", action="spawn"
    )
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        budget_limits=BudgetLimits(hourly_cost_usd=1),
    )
    grant = engine.approvals.grant(
        principal_id=bound_principal.principal_id,
        approver_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    denied = engine.evaluate(
        bound_principal,
        intent,
        definition,
        projected_cost_usd=None,
    )
    assert denied.decision.outcome is DecisionOutcome.DENY
    assert engine.approvals.get(grant.approval_id).used_at is None
    allowed = engine.evaluate(
        bound_principal,
        intent,
        definition,
        projected_cost_usd=0.25,
    )
    assert allowed.decision.outcome is DecisionOutcome.ALLOW
    assert allowed.decision.approval_id == grant.approval_id


def test_changed_arguments_cannot_reuse_approval(ledger, bound_principal, scope):
    registry = CapabilityRegistry()
    intent, definition = _intent(registry, scope, capability="process.spawn")
    changed, _ = _intent(
        registry,
        scope,
        capability="process.spawn",
        arguments_digest=arguments_digest({"command": "different"}),
    )
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    engine.approvals.grant(
        principal_id=bound_principal.principal_id,
        approver_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    assert engine.evaluate(bound_principal, changed, definition).decision.outcome is DecisionOutcome.REQUIRE_APPROVAL


def test_changed_scope_cannot_reuse_approval(ledger, bound_principal, scope):
    registry = CapabilityRegistry()
    intent, definition = _intent(registry, scope, capability="process.spawn")
    other_scope = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id="C_OTHER",
        principal_id=scope.principal_id,
        continuity_id=scope.continuity_id,
    )
    changed, _ = _intent(registry, other_scope, capability="process.spawn")
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    engine.approvals.grant(
        principal_id=bound_principal.principal_id,
        approver_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    assert engine.evaluate(bound_principal, changed, definition).decision.outcome is DecisionOutcome.REQUIRE_APPROVAL


def test_one_approval_is_consumed_once_under_concurrency(
    ledger, bound_principal, scope
):
    registry = CapabilityRegistry()
    intent, definition = _intent(registry, scope, capability="process.spawn")
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    engine.approvals.grant(
        principal_id=bound_principal.principal_id,
        approver_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )

    def evaluate():
        return engine.evaluate(bound_principal, intent, definition).decision.outcome

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: evaluate(), range(2)))
    assert sorted(item.value for item in outcomes) == ["allow", "require_approval"]


def test_one_approval_and_budget_reservation_are_atomic_under_concurrency(
    ledger, bound_principal, scope
):
    registry = CapabilityRegistry()
    intent, definition = _intent(registry, scope, capability="process.spawn")
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        budget_limits=BudgetLimits(hourly_tokens=100),
    )
    engine.approvals.grant(
        principal_id=bound_principal.principal_id,
        approver_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )

    def evaluate():
        return engine.evaluate(
            bound_principal,
            intent,
            definition,
            projected_tokens=60,
            projected_cost_usd=0,
        ).decision

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(pool.map(lambda _: evaluate(), range(2)))

    assert sorted(item.outcome.value for item in decisions) == [
        "allow",
        "require_approval",
    ]
    allowed = next(item for item in decisions if item.outcome is DecisionOutcome.ALLOW)
    assert allowed.approval_id is not None
    assert allowed.budget_reservation_id is not None
    assert ledger.budget_usage(scope.digest)["hour"]["tokens"] == 60


def test_budget_projection_unknown_fails_closed(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        budget_limits=BudgetLimits(hourly_cost_usd=1),
    )
    result = engine.evaluate(
        bound_principal,
        intent,
        definition,
        projected_tokens=10,
        projected_cost_usd=None,
    )
    assert result.decision.outcome is DecisionOutcome.DENY
    assert "unknown" in result.decision.reasons[0]


def test_budget_reservation_is_linked_to_allow_decision(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        budget_limits=BudgetLimits(hourly_tokens=100),
    )
    result = engine.evaluate(
        bound_principal,
        intent,
        definition,
        projected_tokens=40,
        projected_cost_usd=0,
    )
    assert result.decision.outcome is DecisionOutcome.ALLOW
    assert result.decision.budget_reservation_id is not None
    assert ledger.budget_usage(scope.digest)["hour"]["tokens"] == 40


def test_require_allow_raises_typed_errors(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope, capability="process.spawn")
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("approval", DecisionOutcome.REQUIRE_APPROVAL, reason="human"),),
    )
    evaluation = engine.evaluate(bound_principal, intent, definition)
    with pytest.raises(ApprovalRequired):
        engine.require_allow(evaluation)


def _allowed_decision(ledger, principal, intent, definition):
    engine = PolicyEngine(
        ledger,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    evaluation = engine.evaluate(principal, intent, definition)
    assert evaluation.decision.outcome is DecisionOutcome.ALLOW
    return evaluation.decision


def test_lease_round_trip(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    decision = _allowed_decision(ledger, bound_principal, intent, definition)
    authority = LeaseAuthority(b"s" * 32)
    lease, token = authority.issue(bound_principal, intent, decision)
    assert authority.verify(token, intent).lease_id == lease.lease_id


def test_wrong_secret_rejects_lease(ledger, bound_principal, scope):
    intent, definition = _intent(CapabilityRegistry(), scope)
    decision = _allowed_decision(ledger, bound_principal, intent, definition)
    _, token = LeaseAuthority(b"a" * 32).issue(bound_principal, intent, decision)
    with pytest.raises(LeaseTampered, match="signature"):
        LeaseAuthority(b"b" * 32).verify(token, intent)


def test_changed_intent_rejects_lease(ledger, bound_principal, scope):
    registry = CapabilityRegistry()
    intent, definition = _intent(registry, scope)
    decision = _allowed_decision(ledger, bound_principal, intent, definition)
    _, token = LeaseAuthority(b"a" * 32).issue(bound_principal, intent, decision)
    changed, _ = _intent(
        registry,
        scope,
        arguments_digest=arguments_digest({"text": "changed"}),
    )
    with pytest.raises(LeaseTampered, match="exact arguments"):
        LeaseAuthority(b"a" * 32).verify(token, changed)


def test_expired_lease_is_rejected(ledger, bound_principal, scope):
    registry = CapabilityRegistry()
    intent, definition = _intent(registry, scope)
    decision = _allowed_decision(ledger, bound_principal, intent, definition)
    issued = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    authority = LeaseAuthority(b"a" * 32, ttl_seconds=5, clock_skew_seconds=0)
    _, token = authority.issue(bound_principal, intent, decision, now=issued)
    with pytest.raises(LeaseExpired):
        authority.verify(token, intent, now=issued + timedelta(seconds=6))


@pytest.mark.parametrize("token", ["", "bad", "ht1.bad.bad", "ht2.x.y", "x" * 20000])
def test_malformed_lease_is_rejected(token, scope):
    intent, _ = _intent(CapabilityRegistry(), scope)
    with pytest.raises(LeaseError):
        LeaseAuthority(b"a" * 32).verify(token, intent)


def test_obligation_registry_rejects_unknown():
    with pytest.raises(ObligationError, match="unknown obligation"):
        ObligationRegistry().verify(
            ("unknown.obligation",), {}, phase=ObligationPhase.PRE_EFFECT
        )


def test_obligation_registry_requires_evidence():
    with pytest.raises(ObligationError, match="target_verified"):
        ObligationRegistry().verify(
            ("target.exact",), {}, phase=ObligationPhase.PRE_EFFECT
        )
    assert ObligationRegistry().verify(
        ("target.exact",),
        {"target_verified": True},
        phase=ObligationPhase.PRE_EFFECT,
    ) == ("target.exact",)


def test_kernel_issues_verifies_and_completes_effect(
    ledger, bound_principal, scope
):
    config = HermesTagConfig.from_mapping(
        {
            "enabled": True,
            "budgets": {"hourly_tokens": 100},
        }
    )
    kernel = HermesTagKernel(
        ledger,
        config,
        rules=(
            PolicyRule(
                "allow-message",
                DecisionOutcome.ALLOW,
                capabilities=("message.send",),
                reason="allow",
            ),
        ),
        lease_authority=LeaseAuthority(b"k" * 32),
    )
    authorization = kernel.authorize(
        bound_principal,
        capability="message.send",
        action="send",
        resource="slack:C_ENGINEERING",
        arguments={"text": "hello"},
        scope=scope,
        projected_tokens=60,
    )
    assert authorization.token is not None
    assert authorization.lease is not None
    assert (
        authorization.lease.budget_reservation_id
        == authorization.decision.budget_reservation_id
    )
    pre = {
        "target_verified": True,
        "payload_redacted": True,
        "delivery_obligation_id": "delivery_123",
    }
    kernel.verify_effect(
        authorization.token,
        authorization.intent,
        evidence=pre,
        decision_id=authorization.decision.decision_id,
    )
    completion = kernel.complete_effect(
        authorization.token,
        authorization.intent,
        success=True,
        evidence={},
        actual_tokens=25,
    )
    assert completion.receipt.kind == "action.completed"
    assert completion.budget_state == "settled"
    assert ledger.budget_usage(scope.digest)["hour"]["tokens"] == 25
    assert ledger.verify_receipt_chain()[0] >= 4


def _kernel_authorization(ledger, bound_principal, scope):
    kernel = HermesTagKernel(
        ledger,
        HermesTagConfig.from_mapping({"enabled": True}),
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        lease_authority=LeaseAuthority(b"z" * 32),
    )
    authorization = kernel.authorize(
        bound_principal,
        capability="message.send",
        action="send",
        resource="slack:C_ENGINEERING",
        arguments={"text": "hello"},
        scope=scope,
    )
    assert authorization.token is not None
    return kernel, authorization


def _pre_effect_evidence():
    return {
        "target_verified": True,
        "payload_redacted": True,
        "delivery_obligation_id": "delivery_123",
    }


def test_one_shot_lease_cannot_be_verified_twice(
    ledger, bound_principal, scope
):
    kernel, authorization = _kernel_authorization(
        ledger, bound_principal, scope
    )
    kernel.verify_effect(
        authorization.token,
        authorization.intent,
        evidence=_pre_effect_evidence(),
    )
    with pytest.raises(LeaseReplay, match="already presented"):
        kernel.verify_effect(
            authorization.token,
            authorization.intent,
            evidence=_pre_effect_evidence(),
        )


def test_effect_cannot_complete_without_pre_effect_verification(
    ledger, bound_principal, scope
):
    kernel, authorization = _kernel_authorization(
        ledger, bound_principal, scope
    )
    with pytest.raises(LeaseReplay, match="must be verified"):
        kernel.complete_effect(
            authorization.token,
            authorization.intent,
            success=True,
            evidence={},
        )


def test_completed_lease_cannot_complete_twice(
    ledger, bound_principal, scope
):
    kernel, authorization = _kernel_authorization(
        ledger, bound_principal, scope
    )
    kernel.verify_effect(
        authorization.token,
        authorization.intent,
        evidence=_pre_effect_evidence(),
    )
    kernel.complete_effect(
        authorization.token,
        authorization.intent,
        success=True,
        evidence={},
    )
    with pytest.raises(LeaseReplay, match="already completed"):
        kernel.complete_effect(
            authorization.token,
            authorization.intent,
            success=True,
            evidence={},
        )


def test_completion_rejects_fabricated_budget_linkage(
    ledger, bound_principal, scope
):
    config = HermesTagConfig.from_mapping(
        {"enabled": True, "budgets": {"hourly_tokens": 100}}
    )
    kernel = HermesTagKernel(
        ledger,
        config,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        lease_authority=LeaseAuthority(b"f" * 32),
    )
    authorization = kernel.authorize(
        bound_principal,
        capability="message.send",
        action="send",
        resource="slack:C_ENGINEERING",
        arguments={"text": "hello"},
        scope=scope,
        projected_tokens=10,
    )
    kernel.verify_effect(
        authorization.token,
        authorization.intent,
        evidence=_pre_effect_evidence(),
    )
    fabricated = replace(
        authorization.decision,
        budget_reservation_id="budget_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    with pytest.raises(LeaseTampered, match="does not match"):
        kernel.complete_effect(
            authorization.token,
            authorization.intent,
            success=True,
            evidence={},
            decision=fabricated,
        )


def test_missing_signing_authority_restores_consumed_approval(
    ledger, bound_principal, scope, external_identity, surface
):
    config = HermesTagConfig.from_mapping({"enabled": True})
    rules = (PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),)
    issuer = HermesTagKernel(
        ledger,
        config,
        rules=rules,
        lease_authority=LeaseAuthority(b"g" * 32),
    )
    kernel = HermesTagKernel(ledger, config, rules=rules)
    intent, _ = _intent(
        kernel.capabilities, scope, capability="process.spawn", action="spawn"
    )
    admission = issuer.admit_turn(external_identity, surface).admission
    grant_authority = issuer.authorize_approval_grant(
        admission,
        principal_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    grant = issuer.grant_approval(
        admission,
        grant_authority,
        principal_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    with pytest.raises(LeaseError, match="no lease"):
        kernel.authorize(
            bound_principal,
            capability="process.spawn",
            action="spawn",
            resource="slack:C_ENGINEERING",
            arguments={"text": "hello"},
            scope=scope,
        )
    assert kernel.policy.approvals.get(grant.approval_id).used_at is None


def test_lease_issuance_failure_releases_budget_and_restores_approval(
    ledger, bound_principal, scope, external_identity, surface, monkeypatch
):
    authority = LeaseAuthority(b"q" * 32)
    kernel = HermesTagKernel(
        ledger,
        HermesTagConfig.from_mapping(
            {"enabled": True, "budgets": {"hourly_tokens": 100}}
        ),
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        lease_authority=authority,
    )
    intent, _ = _intent(
        kernel.capabilities, scope, capability="process.spawn", action="spawn"
    )
    admission = kernel.admit_turn(external_identity, surface).admission
    grant_authority = kernel.authorize_approval_grant(
        admission,
        principal_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )
    grant = kernel.grant_approval(
        admission,
        grant_authority,
        principal_id=bound_principal.principal_id,
        intent_digest=intent.digest,
        scope_digest=scope.digest,
    )

    def fail_issue(*_args, **_kwargs):
        raise LeaseError("synthetic issuance failure")

    monkeypatch.setattr(authority, "issue", fail_issue)
    with pytest.raises(LeaseError, match="synthetic"):
        kernel.authorize(
            bound_principal,
            capability="process.spawn",
            action="spawn",
            resource="slack:C_ENGINEERING",
            arguments={"text": "hello"},
            scope=scope,
            projected_tokens=40,
        )
    assert ledger.budget_usage(scope.digest)["hour"]["tokens"] == 0
    assert kernel.policy.approvals.get(grant.approval_id).used_at is None


def test_kernel_without_signing_authority_releases_budget(
    ledger, bound_principal, scope
):
    config = HermesTagConfig.from_mapping(
        {"enabled": True, "budgets": {"hourly_tokens": 100}}
    )
    kernel = HermesTagKernel(
        ledger,
        config,
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
    )
    with pytest.raises(LeaseError, match="no lease"):
        kernel.authorize(
            bound_principal,
            capability="message.send",
            action="send",
            resource="slack:C",
            arguments={"text": "x"},
            scope=scope,
            projected_tokens=40,
        )
    assert ledger.budget_usage(scope.digest)["hour"]["tokens"] == 0

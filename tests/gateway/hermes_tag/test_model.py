from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from gateway.hermes_tag import (
    ActionIntent,
    ApprovalGrant,
    CapabilityDefinition,
    ContextBundle,
    ContinuityEnvelope,
    ExternalIdentity,
    Fact,
    Principal,
    RiskLevel,
    ScopeRef,
    Sensitivity,
    SurfaceRef,
    TurnAdmission,
    arguments_digest,
    canonical_digest,
    canonical_json,
    new_id,
    parse_utc,
    utc_text,
)


@dataclass(frozen=True)
class Example:
    z: int
    a: str


def test_canonical_json_is_order_and_dataclass_stable():
    left = {"b": 2, "a": Example(z=3, a="x")}
    right = {"a": Example(a="x", z=3), "b": 2}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_digest(left) == canonical_digest(right)


def test_canonical_sets_are_sorted():
    assert canonical_json({"values": {"z", "a"}}) == '{"values":["a","z"]}'


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_canonical_json_rejects_non_finite(value):
    with pytest.raises(ValueError):
        canonical_json({"value": value})


def test_utc_round_trip():
    value = datetime(2026, 8, 20, 12, 30, 4, 55, tzinfo=timezone.utc)
    assert parse_utc(utc_text(value)) == value


def test_parse_utc_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        parse_utc("2026-08-20T12:30:00")


@pytest.mark.parametrize("prefix", ["A", "1bad", "a-b", "x" * 40])
def test_new_id_rejects_invalid_prefix(prefix):
    with pytest.raises(ValueError):
        new_id(prefix)


def test_surface_key_is_tenant_qualified():
    a = SurfaceRef("slack", "default", "T1", "C1")
    b = SurfaceRef("slack", "default", "T2", "C1")
    assert a.key != b.key


def test_identity_key_ignores_display_name_but_not_scope():
    a = ExternalIdentity("slack", "default", "T1", "U1", "Old")
    b = ExternalIdentity("slack", "default", "T1", "U1", "New")
    c = ExternalIdentity("slack", "default", "T2", "U1", "New")
    assert a.key == b.key
    assert a.key != c.key


def test_scope_digest_changes_with_thread():
    base = dict(
        profile="default",
        platform="slack",
        scope_id="T1",
        chat_id="C1",
        principal_id="principal_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    assert ScopeRef(**base).digest != ScopeRef(**base, thread_id="TH1").digest


@pytest.mark.parametrize(
    ("value", "expected"),
    [("low", RiskLevel.LOW), ("HIGH", RiskLevel.HIGH), (40, RiskLevel.CRITICAL)],
)
def test_risk_coercion(value, expected):
    assert RiskLevel.coerce(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("public", Sensitivity.PUBLIC),
        ("CONFIDENTIAL", Sensitivity.CONFIDENTIAL),
        (40, Sensitivity.SECRET),
    ],
)
def test_sensitivity_coercion(value, expected):
    assert Sensitivity.coerce(value) is expected


def test_capability_definition_rejects_unknown_scope_field():
    with pytest.raises(ValueError, match="scope"):
        CapabilityDefinition(
            "example.capability",
            RiskLevel.LOW,
            required_scope_fields=("profile", "made_up"),
        )


def test_action_intent_digest_binds_metadata_and_scope(scope):
    common = dict(
        capability="tool.execute",
        action="run",
        resource="tool:example",
        arguments_digest=arguments_digest({"x": 1}),
        scope=scope,
    )
    assert ActionIntent(**common).digest != ActionIntent(
        **common, metadata={"tool": "other"}
    ).digest


def test_approval_requires_future_expiry(principal, scope):
    issued = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="expiry"):
        ApprovalGrant(
            approval_id="approval_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            principal_id=principal.principal_id,
            approver_id=principal.principal_id,
            intent_digest="a" * 64,
            scope_digest=scope.digest,
            issued_at=utc_text(issued),
            expires_at=utc_text(issued - timedelta(seconds=1)),
        )


def test_continuity_envelope_hop_count_must_match(surface, continuity):
    with pytest.raises(ValueError, match="hop_count"):
        ContinuityEnvelope(
            event_id="event_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            continuity_id=continuity.continuity_id,
            origin=surface,
            payload_digest="a" * 64,
            propagation_path=("slack",),
            hop_count=0,
        )


def test_fact_rejects_invalid_confidence(scope):
    with pytest.raises(ValueError, match="confidence"):
        Fact(
            fact_id="fact_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            subject="project",
            predicate="status",
            value="active",
            scope=scope,
            source_type="api",
            source_id="github",
            source_revision="sha",
            confidence=1.1,
            authority=10,
        )


def test_fact_content_hash_includes_source_revision(scope):
    common = dict(
        subject="project",
        predicate="status",
        value="active",
        scope=scope,
        source_type="api",
        source_id="github",
        confidence=1.0,
        authority=10,
    )
    a = Fact(
        fact_id="fact_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        source_revision="sha1",
        **common,
    )
    b = Fact(
        fact_id="fact_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        source_revision="sha2",
        **common,
    )
    assert a.content_hash != b.content_hash


def test_canonical_json_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json({1: "value"})


def test_arguments_digest_is_shape_sensitive():
    assert arguments_digest({"a": 1}) != arguments_digest(["a", 1])


def test_action_intent_freezes_nested_metadata_and_digest(scope):
    metadata = {"labels": ["alpha"], "nested": {"enabled": True}}
    intent = ActionIntent(
        capability="tool.execute",
        action="run",
        resource="tool:example",
        arguments_digest=arguments_digest({"x": 1}),
        scope=scope,
        metadata=metadata,
    )
    digest = intent.digest
    metadata["labels"].append("mutated")
    metadata["nested"]["enabled"] = False
    assert intent.digest == digest
    assert intent.metadata["labels"] == ("alpha",)
    assert intent.metadata["nested"]["enabled"] is True
    with pytest.raises(TypeError):
        intent.metadata["other"] = "value"


def test_fact_freezes_nested_value(scope):
    value = {"items": [{"state": "open"}]}
    fact = Fact(
        fact_id="fact_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        subject="project",
        predicate="state",
        value=value,
        scope=scope,
        source_type="api",
        source_id="github",
        source_revision="sha",
        confidence=1.0,
        authority=10,
    )
    content_hash = fact.content_hash
    value["items"][0]["state"] = "closed"
    assert fact.content_hash == content_hash
    assert fact.value["items"][0]["state"] == "open"
    with pytest.raises(TypeError):
        fact.value["other"] = "value"


def test_continuity_envelope_freezes_caller_path(surface, continuity):
    path = ["slack"]
    envelope = ContinuityEnvelope(
        event_id="event_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        continuity_id=continuity.continuity_id,
        origin=surface,
        payload_digest="a" * 64,
        propagation_path=path,
        hop_count=1,
    )
    path.append("desktop")
    assert envelope.propagation_path == ("slack",)


def test_policy_and_lease_collections_normalize_to_tuples(scope, principal):
    from gateway.hermes_tag import CapabilityLease, DecisionOutcome, PolicyDecision

    decision = PolicyDecision(
        decision_id="decision_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        outcome=DecisionOutcome.ALLOW,
        capability="tool.execute",
        intent_digest="a" * 64,
        scope_digest=scope.digest,
        reasons=["allow"],
        obligations=["receipt.append", "receipt.append"],
        matched_rules=["rule-a", "rule-a"],
    )
    assert decision.reasons == ("allow",)
    assert decision.obligations == ("receipt.append",)
    assert decision.matched_rules == ("rule-a",)

    now = datetime.now(timezone.utc)
    lease = CapabilityLease(
        lease_id="lease_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        principal_id=principal.principal_id,
        continuity_id=scope.continuity_id,
        capability="tool.execute",
        intent_digest="a" * 64,
        scope_digest=scope.digest,
        decision_id=decision.decision_id,
        obligations=["receipt.append", "receipt.append"],
        budget_reservation_id=None,
        approval_id=None,
        issued_at=utc_text(now),
        expires_at=utc_text(now + timedelta(seconds=30)),
        nonce="nonce",
    )
    assert lease.obligations == ("receipt.append",)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (
            lambda scope: ActionIntent(
                capability="tool.execute",
                action="run",
                resource="tool:example",
                arguments_digest=arguments_digest({}),
                scope=scope,
                metadata={"payload": "x" * 65537},
            ),
            "metadata",
        ),
        (
            lambda scope: Fact(
                fact_id="fact_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                subject="project",
                predicate="payload",
                value="x" * 1_000_001,
                scope=scope,
                source_type="api",
                source_id="github",
                source_revision="sha",
                confidence=1.0,
                authority=10,
            ),
            "fact value",
        ),
    ],
)
def test_structured_payload_bounds_fail_closed(scope, constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor(scope)


def test_arguments_digest_rejects_oversized_payload():
    with pytest.raises(ValueError, match="arguments exceed"):
        arguments_digest({"payload": "x" * 1_000_001})


def test_context_bundle_normalizes_and_bounds(scope):
    fact = Fact(
        fact_id="fact_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        subject="project",
        predicate="state",
        value="open",
        scope=scope,
        source_type="api",
        source_id="github",
        source_revision="sha",
        confidence=1.0,
        authority=10,
    )
    bundle = ContextBundle(
        facts=[fact],
        conflicts=[["fact_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", fact.fact_id]],
        omitted_count=1,
        rendered_text="context",
    )
    assert bundle.facts == (fact,)
    assert bundle.conflicts == ((
        fact.fact_id,
        "fact_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),)
    with pytest.raises(ValueError, match="non-negative"):
        ContextBundle(facts=(), omitted_count=-1)
    with pytest.raises(ValueError, match="at least two"):
        ContextBundle(facts=(), conflicts=((fact.fact_id,),))
    with pytest.raises(ValueError, match="one megabyte"):
        ContextBundle(facts=(), rendered_text="x" * 1_000_001)


def test_turn_admission_rejects_inconsistent_identity_boundaries(
    principal, surface, scope
):
    context = ContextBundle(facts=())
    common = dict(
        admission_id="admission_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        principal=principal,
        surface=surface,
        scope=scope,
        continuity_id=scope.continuity_id,
        context=context,
    )
    TurnAdmission(**common)

    other = Principal(
        principal_id="principal_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        display_name="Other",
    )
    with pytest.raises(ValueError, match="principal"):
        TurnAdmission(**{**common, "principal": other})

    mismatched_surface = SurfaceRef(
        platform=surface.platform,
        profile=surface.profile,
        scope_id=surface.scope_id,
        chat_id="C_OTHER",
        thread_id=surface.thread_id,
    )
    with pytest.raises(ValueError, match="chat_id"):
        TurnAdmission(**{**common, "surface": mismatched_surface})

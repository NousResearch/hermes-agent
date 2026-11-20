from __future__ import annotations

from tests.agent._dynamic_orchestration_support import (
    AttemptState,
    AttemptStateEvent,
    CredentialState,
    CredentialStateEvent,
    DomainValidationError,
    FrozenInstanceError,
    ReservationState,
    ReservationStateEvent,
    ReviewState,
    ReviewStateEvent,
    RouteState,
    RouteStateEvent,
    TaskState,
    TaskStateEvent,
    UNSELECTED_ROUTE_ID,
    audit,
    breaker_payload,
    capacity_reservation,
    orchestration,
    pytest,
    reservation_payload,
    route,
    validate_attempt_transition,
    validate_credential_transition,
    validate_reservation_transition,
    validate_review_transition,
    validate_route_transition,
    validate_task_transition,
)
@pytest.mark.parametrize(
    ("validator", "event_type", "from_state", "to_state", "identity_field"),
    [
        (validate_task_transition, TaskStateEvent, TaskState.NEW, TaskState.PLANNED, "task_id"),
        (
            validate_attempt_transition,
            AttemptStateEvent,
            AttemptState.CREATED,
            AttemptState.RESERVED,
            "attempt_id",
        ),
        (
            validate_route_transition,
            RouteStateEvent,
            RouteState.DISCOVERED,
            RouteState.ELIGIBLE,
            "route_id",
        ),
        (
            validate_credential_transition,
            CredentialStateEvent,
            CredentialState.AVAILABLE,
            CredentialState.COOLDOWN,
            "credential_id",
        ),
        (
            validate_reservation_transition,
            ReservationStateEvent,
            ReservationState.PENDING,
            ReservationState.HELD,
            "reservation_id",
        ),
        (
            validate_review_transition,
            ReviewStateEvent,
            ReviewState.PENDING,
            ReviewState.IN_PROGRESS,
            "review_id",
        ),
    ],
)
def test_each_state_domain_has_an_independent_audited_legal_transition(
    validator: object,
    event_type: type[object],
    from_state: object,
    to_state: object,
    identity_field: str,
):
    event = validator(  # type: ignore[operator]
        from_state=from_state,
        to_state=to_state,
        **audit(identity_field),
    )
    assert type(event) is event_type
    assert getattr(event, identity_field) == "entity-1"
    assert event.actor == "policy-engine"  # type: ignore[attr-defined]
    assert event.timestamp.endswith("Z")  # type: ignore[attr-defined]
    assert event.reason == "contract transition"  # type: ignore[attr-defined]
    assert event.correlation_id == "corr-1"  # type: ignore[attr-defined]

def test_transition_validators_reject_illegal_edges_missing_audit_and_cross_domain_inference():
    with pytest.raises(DomainValidationError, match="state.transition_illegal"):
        validate_task_transition(
            from_state=TaskState.NEW,
            to_state=TaskState.COMPLETED,
            **audit("task_id"),
        )
    with pytest.raises(DomainValidationError, match="state.audit_metadata_required"):
        validate_attempt_transition(
            from_state=AttemptState.CREATED,
            to_state=AttemptState.RESERVED,
            **{**audit("attempt_id"), "actor": ""},
        )
    with pytest.raises(DomainValidationError, match="state.domain_mismatch"):
        validate_task_transition(
            from_state=AttemptState.CREATED,  # type: ignore[arg-type]
            to_state=TaskState.PLANNED,
            **audit("task_id"),
        )

    with pytest.raises(DomainValidationError, match="state.transition_illegal"):
        TaskStateEvent(
            task_id="task-1",
            from_state=TaskState.NEW,
            to_state=TaskState.COMPLETED,
            actor="policy-engine",
            timestamp="2026-07-26T17:00:00Z",
            reason="illegal direct construction",
            correlation_id="corr-1",
        )

def test_state_domains_are_distinct_even_when_values_overlap():
    assert type(TaskState.NEW) is not type(AttemptState.CREATED)
    assert type(RouteState.COOLDOWN) is not type(CredentialState.COOLDOWN)
    assert type(ReservationState.HELD) is not type(ReviewState.PENDING)

def test_capacity_reservation_is_exact_immutable_and_route_bound():
    bound_route = route(model="reservation-route", quota_pool_id="reservation-quota")
    reservation = capacity_reservation(bound_route)

    assert set(reservation.canonical_object()) == set(reservation_payload(bound_route))
    assert (
        orchestration.CapacityReservationV1.from_mapping(
            reservation.canonical_object(),
            trusted_route=bound_route,
        ).canonical_object()
        == reservation.canonical_object()
    )
    with pytest.raises(FrozenInstanceError):
        reservation.version = 2
    with pytest.raises(DomainValidationError, match="reservation.schema_invalid"):
        orchestration.CapacityReservationV1.from_mapping(
            {**reservation.canonical_object(), "schema_version": "reservation/v1"},
            trusted_route=bound_route,
        )
    with pytest.raises(DomainValidationError, match="reservation.trusted_route_required"):
        orchestration.CapacityReservationV1.from_mapping(reservation.canonical_object())

    past_expiry = capacity_reservation(
        bound_route,
        expires_at="2020-01-01T00:00:00Z",
    )
    assert past_expiry.expires_at == "2020-01-01T00:00:00Z"

def test_capacity_reservation_is_bounded_secret_free_and_detached_from_route_mutation():
    bound_route = route(model="reservation-safety", quota_pool_id="reservation-safety")
    reservation = capacity_reservation(bound_route)
    canonical_before = reservation.canonical_object()
    object.__setattr__(bound_route, "quota_pool_id", "mutated-after-validation")

    assert reservation.canonical_object() == canonical_before
    with pytest.raises(DomainValidationError):
        orchestration.CapacityReservationV1.from_mapping(
            {
                **canonical_before,
                "reservation_id": "r" * 8193,
            },
            trusted_route=route(
                model="reservation-safety",
                quota_pool_id="reservation-safety",
            ),
        )
    with pytest.raises(DomainValidationError, match="sensitive"):
        orchestration.CapacityReservationV1.from_mapping(
            {
                **canonical_before,
                "unit": "api_key=must-not-survive",
            },
            trusted_route=route(
                model="reservation-safety",
                quota_pool_id="reservation-safety",
            ),
        )

@pytest.mark.parametrize(
    ("overrides", "error_code"),
    (
        ({"quota_pool_id": "wrong-pool"}, "reservation.route_binding_invalid"),
        ({"billing_pool_id": "wrong-billing"}, "reservation.route_binding_invalid"),
        ({"route_id": UNSELECTED_ROUTE_ID}, "reservation.route_binding_invalid"),
        ({"estimated_amount": True}, "reservation.amount_invalid"),
        ({"held_amount": float("inf")}, "reservation.amount_invalid"),
        ({"status": "PENDING", "held_amount": 1}, "reservation.state_invariant"),
        ({"status": "HELD", "held_amount": 0}, "reservation.state_invariant"),
        ({"status": "HELD", "held_amount": 11}, "reservation.state_invariant"),
        ({"status": "CONSUMED", "held_amount": 0}, "reservation.state_invariant"),
        ({"status": "RELEASED", "held_amount": 1}, "reservation.state_invariant"),
        ({"status": "INVALID"}, "reservation.status_invalid"),
        ({"expires_at": "2026-07-26T18:00:00"}, "reservation.timestamp_invalid"),
        ({"version": True}, "reservation.version_invalid"),
    ),
)
def test_capacity_reservation_rejects_invalid_binding_amount_state_and_version(
    overrides: dict[str, object],
    error_code: str,
):
    bound_route = route(model="reservation-invalid", quota_pool_id="reservation-invalid")
    payload = reservation_payload(bound_route)
    payload.update(overrides)

    with pytest.raises(DomainValidationError) as error:
        orchestration.CapacityReservationV1.from_mapping(
            payload,
            trusted_route=bound_route,
        )
    assert error.value.code == error_code

def test_circuit_breaker_is_exact_immutable_and_route_scoped():
    bound_route = route(model="breaker-route", quota_pool_id="breaker-quota")
    closed = orchestration.CircuitBreakerV1.from_mapping(
        breaker_payload(bound_route),
        trusted_route=bound_route,
    )
    opened = orchestration.CircuitBreakerV1.from_mapping(
        breaker_payload(
            bound_route,
            status="OPEN",
            cooldown_until="2026-07-26T18:00:00Z",
            reason_code="route_capacity_exhausted",
        ),
        trusted_route=bound_route,
    )
    half_open = orchestration.CircuitBreakerV1.from_mapping(
        breaker_payload(bound_route, status="HALF_OPEN", probe_budget=1),
        trusted_route=bound_route,
    )

    assert set(closed.canonical_object()) == set(breaker_payload(bound_route))
    assert (
        orchestration.CircuitBreakerV1.from_mapping(
            opened.canonical_object(),
            trusted_route=bound_route,
        ).canonical_object()
        == opened.canonical_object()
    )
    assert opened.status.value == "OPEN"
    assert half_open.probe_budget == 1
    with pytest.raises(FrozenInstanceError):
        closed.probe_budget = 1
    with pytest.raises(DomainValidationError, match="breaker.schema_invalid"):
        orchestration.CircuitBreakerV1.from_mapping(
            {**closed.canonical_object(), "schema_version": "breaker/v1"},
            trusted_route=bound_route,
        )
    with pytest.raises(DomainValidationError, match="breaker.trusted_route_required"):
        orchestration.CircuitBreakerV1.from_mapping(closed.canonical_object())

def test_circuit_breaker_is_bounded_secret_free_and_detached_from_route_mutation():
    bound_route = route(model="breaker-safety", quota_pool_id="breaker-safety")
    breaker = orchestration.CircuitBreakerV1.from_mapping(
        breaker_payload(bound_route),
        trusted_route=bound_route,
    )
    canonical_before = breaker.canonical_object()
    object.__setattr__(bound_route, "quota_pool_id", "mutated-after-validation")

    assert breaker.canonical_object() == canonical_before
    trusted_route = route(model="breaker-safety", quota_pool_id="breaker-safety")
    with pytest.raises(DomainValidationError):
        orchestration.CircuitBreakerV1.from_mapping(
            {**canonical_before, "breaker_id": "b" * 8193},
            trusted_route=trusted_route,
        )
    with pytest.raises(DomainValidationError, match="sensitive"):
        orchestration.CircuitBreakerV1.from_mapping(
            {
                **canonical_before,
                "reason_code": "authorization=must-not-survive",
            },
            trusted_route=trusted_route,
        )

@pytest.mark.parametrize(
    ("overrides", "error_code"),
    (
        ({"route_id": UNSELECTED_ROUTE_ID}, "breaker.route_binding_invalid"),
        ({"quota_pool_id": "wrong-pool"}, "breaker.route_binding_invalid"),
        ({"status": "OPEN"}, "breaker.state_invariant"),
        (
            {
                "status": "CLOSED",
                "cooldown_until": "2026-07-26T18:00:00Z",
            },
            "breaker.state_invariant",
        ),
        ({"status": "CLOSED", "probe_budget": 1}, "breaker.state_invariant"),
        ({"status": "HALF_OPEN", "probe_budget": 0}, "breaker.state_invariant"),
        ({"status": "INVALID"}, "breaker.status_invalid"),
        ({"probe_budget": True}, "breaker.probe_budget_invalid"),
        ({"version": True}, "breaker.version_invalid"),
    ),
)
def test_circuit_breaker_rejects_invalid_route_state_and_versions(
    overrides: dict[str, object],
    error_code: str,
):
    bound_route = route(model="breaker-invalid", quota_pool_id="breaker-invalid")
    payload = breaker_payload(bound_route)
    payload.update(overrides)

    with pytest.raises(DomainValidationError) as error:
        orchestration.CircuitBreakerV1.from_mapping(
            payload,
            trusted_route=bound_route,
        )
    assert error.value.code == error_code

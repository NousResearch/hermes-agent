from __future__ import annotations

from tests.agent._dynamic_orchestration_support import (
    AcceptanceThresholdV1,
    Any,
    AuditedModelJustification,
    Callable,
    CandidateEvaluation,
    CapacityConflictDisposition,
    CapacityObservationV1,
    CapacityValueV1,
    CapacityViewV1,
    CompensationEscalationV1,
    DecisionRelation,
    DeepcopyBomb,
    DomainValidationError,
    EscalationAction,
    ExecutionEnvelopeV1,
    ExecutionIdentityAttestationV1,
    ExplosiveNumber,
    ExplosiveString,
    HOSTILE_MAPPING_HOOKS,
    HOSTILE_MAPPING_MARKER,
    HashBomb,
    HostileMapping,
    HugePrefixMapping,
    INDEPENDENT_REVIEW_ROUTE,
    INDEPENDENT_REVIEW_ROUTE_ID,
    IndependentReviewAttestationV1,
    InitialSelectionTriggerV1,
    IterationBombList,
    PoolCapacityV1,
    QualityCompensationPlanV1,
    RouteDecisionV1,
    RouteV1,
    RuntimeErrorClassificationV1,
    TaskEnvelope,
    TaskState,
    TaskStateEvent,
    _MAX_GRAPH_DEPTH,
    _MAX_GRAPH_NODES,
    _MAX_MAPPING_FIELDS,
    _MAX_TEXT_COLLECTION_ITEMS,
    _invoke_public_integer_boundary,
    _invoke_public_numeric_boundary,
    _invoke_public_status_boundary,
    _invoke_untrusted_label_boundary,
    _mapping_snapshot,
    _reject_sensitive,
    _task_with_context_bounds,
    asdict,
    audit,
    benchmark,
    breaker_payload,
    capacity_observation,
    capacity_reservation,
    capacity_view,
    classification,
    decision_metadata,
    execution_authority,
    execution_payload,
    json,
    nested_dicts,
    nested_lists,
    orchestration,
    pytest,
    replace,
    replan,
    reservation_payload,
    route,
    scored_candidate,
    task_payload,
    valid_plan,
    validate_task_transition,
)
def test_public_contracts_reject_oversized_and_cyclic_payloads_with_domain_errors():
    oversized_collection = task_payload(
        deliverables=[f"deliverable-{index}" for index in range(129)]
    )
    with pytest.raises(DomainValidationError, match="task.collection_invalid"):
        TaskEnvelope.from_mapping(oversized_collection)

    with pytest.raises(DomainValidationError, match="task.scalar_invalid"):
        TaskEnvelope.from_mapping(task_payload(objective="x" * 8193))

    cyclic = task_payload()
    cyclic_context: dict[str, object] = {}
    cyclic_context["loop"] = cyclic_context
    cyclic["context"] = cyclic_context
    with pytest.raises(DomainValidationError, match="contract.payload_too_complex"):
        TaskEnvelope.from_mapping(cyclic)

    deep = task_payload()
    deep_context: dict[str, object] = {}
    cursor = deep_context
    for _ in range(34):
        child: dict[str, object] = {}
        cursor["nested"] = child
        cursor = child
    deep["context"] = deep_context
    with pytest.raises(DomainValidationError, match="contract.payload_too_complex"):
        TaskEnvelope.from_mapping(deep)

    sensitive_key = "github_pat_REDACTED"
    leaked_key = task_payload()
    leaked_key[sensitive_key] = "value"
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited") as exc:
        TaskEnvelope.from_mapping(leaked_key)
    assert sensitive_key not in str(exc.value)

@pytest.mark.parametrize(
    ("field_count", "include_sensitive_key", "expected_code"),
    (
        (128, False, "route.unexpected_field"),
        (128, True, "decision.sensitive_field_prohibited"),
        (129, False, "contract.payload_too_complex"),
        (129, True, "decision.sensitive_field_prohibited"),
    ),
)
def test_public_route_mapping_sensitive_precedence_at_field_count_boundary(
    field_count: int,
    include_sensitive_key: bool,
    expected_code: str,
):
    payload: dict[str, object] = route(model="mapping-boundary").canonical_object()
    filler_count = field_count - len(payload) - int(include_sensitive_key)
    payload.update(
        (f"unexpected_{index}", "value")
        for index in range(filler_count)
    )
    if include_sensitive_key:
        payload["api_key"] = "redacted"
    assert len(payload) == field_count

    with pytest.raises(DomainValidationError) as error:
        RouteV1.from_mapping(payload)

    assert error.value.code == expected_code

@pytest.mark.parametrize(
    "parser",
    (
        RouteV1.from_mapping,
        AuditedModelJustification.from_mapping,
        InitialSelectionTriggerV1.from_mapping,
        CandidateEvaluation.from_mapping,
        AcceptanceThresholdV1.from_mapping,
        QualityCompensationPlanV1.from_mapping,
        CapacityValueV1.from_mapping,
        PoolCapacityV1.from_mapping,
        ExecutionIdentityAttestationV1.from_mapping,
    ),
)
def test_public_mapping_peers_preserve_sensitive_key_precedence_on_n_plus_one(
    parser: Callable[[Any], object],
):
    payload = {f"field_{index}": "value" for index in range(128)}
    payload["api_key"] = "redacted"

    with pytest.raises(DomainValidationError) as error:
        parser(payload)

    assert error.value.code == "decision.sensitive_field_prohibited"

def test_huge_public_mapping_scans_only_bounded_keys_and_never_values():
    payload = HugePrefixMapping()

    with pytest.raises(DomainValidationError) as error:
        RouteV1.from_mapping(payload)

    assert error.value.code == "contract.payload_too_complex"
    assert payload.keys_yielded == 129
    assert payload.values_read == 0

def test_benchmark_cli_accepts_conservative_upper_bounds_without_real_workload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[tuple[int, int, int]] = []

    def record_call(
        cardinality: int,
        *,
        warmups: int,
        sample_count: int,
    ) -> dict[str, object]:
        calls.append((cardinality, warmups, sample_count))
        return {
            "cardinality": cardinality,
            "warmups": warmups,
            "sample_count": sample_count,
        }

    monkeypatch.setattr(benchmark, "_benchmark_cardinality", record_call)
    monkeypatch.setattr(benchmark, "_environment_metadata", lambda: {"test": True})

    assert benchmark.main(["--samples", "100", "--warmups", "100"]) == 0
    assert calls == [(cardinality, 100, 100) for cardinality in benchmark.CARDINALITIES]
    assert json.loads(capsys.readouterr().out)["environment"] == {"test": True}

@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--samples", "101"),
        ("--samples", "1000000000"),
        ("--warmups", "101"),
        ("--warmups", "1000000000"),
    ),
)
def test_benchmark_cli_rejects_unbounded_counts_before_any_work(
    option: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_work(*args: object, **kwargs: object) -> object:
        raise AssertionError("benchmark work started before argument rejection")

    monkeypatch.setattr(benchmark, "_environment_metadata", unexpected_work)
    monkeypatch.setattr(benchmark, "_benchmark_cardinality", unexpected_work)

    with pytest.raises(SystemExit) as error:
        benchmark.main([option, value])

    assert error.value.code == 2

def test_public_contracts_bound_all_textual_audit_and_threshold_fields():
    with pytest.raises(DomainValidationError, match="quality.threshold_invalid"):
        AcceptanceThresholdV1(metric="quality", operator="==", value="x" * 8193)

    with pytest.raises(DomainValidationError, match="state.audit_metadata_required"):
        TaskStateEvent(
            task_id="task-1",
            from_state=TaskState.NEW,
            to_state=TaskState.PLANNED,
            actor="policy-engine",
            timestamp="2026-07-27T09:00:00Z",
            reason="x" * 8193,
            correlation_id="corr-1",
        )

def test_runtime_classification_bounds_quota_pool_text_direct_and_persisted():
    attempted = route(model="bounded-quota-pool")

    accepted = classification(attempted, quota_pool_id="q" * 8192)
    assert len(accepted.quota_pool_id) == 8192

    with pytest.raises(DomainValidationError, match="classification.capacity_scope_required"):
        classification(attempted, quota_pool_id="q" * 8193)

    persisted = asdict(classification(attempted))
    persisted["quota_pool_id"] = "q" * 8193
    with pytest.raises(DomainValidationError, match="classification.capacity_scope_required"):
        RuntimeErrorClassificationV1.from_mapping(persisted)

def test_task_identity_claims_are_bounded_before_iteration_and_claim_injection():
    base_justification = {
        "policy_version": "policy/v1",
        "reason": "bounded identity",
        "evidence_refs": ["evidence:test"],
        "author": "policy-router",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    accepted = TaskEnvelope.from_mapping(
        task_payload(
            model="bounded-model",
            audited_model_justification={
                **base_justification,
                "identity_claims": [
                    (f"claim-{index}", "value") for index in range(127)
                ],
            },
        )
    )
    assert accepted.audited_model_justification is not None
    assert len(accepted.audited_model_justification.identity_claims) == 128

    bomb = IterationBombList([("claim", "value")] * 129)
    with pytest.raises(DomainValidationError):
        TaskEnvelope.from_mapping(
            task_payload(
                model="bounded-model",
                audited_model_justification={
                    **base_justification,
                    "identity_claims": bomb,
                },
            )
        )

def test_persisted_decision_and_replan_bound_collections_before_iteration():
    selected = route(model="pre-materialization-bound")
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-pre-materialization-bound",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    metadata = decision_metadata()
    metadata["verification"] = "V0"
    valid_decision = RouteDecisionV1(
        decision_id="decision-pre-materialization-bound",
        task_id=task.task_id,
        attempt_id="attempt-pre-materialization-bound",
        **metadata,
        fallback=False,
        relation=DecisionRelation.INITIAL,
        candidates=(scored_candidate(selected),),
        selected_route_id=selected.route_id,
        trigger=InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy-router",
            "2026-07-27T09:00:00Z",
        ),
        reason_codes=("initial_selection",),
        trusted_task=task,
        trusted_routes={selected.route_id: selected},
    )
    persisted = asdict(valid_decision)
    persisted["reason_codes"] = IterationBombList(["bounded"] * 129)
    with pytest.raises(DomainValidationError):
        RouteDecisionV1.from_mapping(
            persisted,
            trusted_task=task,
            trusted_routes={selected.route_id: selected},
        )

    failed = route(model="failed-pre-materialization", quota_pool_id="failed-pre-bound")
    with pytest.raises(DomainValidationError, match="task.collection_invalid"):
        replan(
            decision_id="decision-recheck-pre-materialization",
            task_id="task-recheck-pre-materialization",
            attempt_id="attempt-recheck-pre-materialization",
            failed_route=failed,
            classification=classification(failed),
            candidates=(),
            recheck_evidence=IterationBombList(["bounded"] * 129),
        )

def test_public_contracts_bound_collections_before_normalizing_or_sorting():
    claims = tuple((f"claim-{index}", "value") for index in range(129))
    with pytest.raises(DomainValidationError, match="task.justification_invalid"):
        AuditedModelJustification(
            policy_version="policy/v1",
            reason="bounded",
            evidence_refs=("evidence:test",),
            author="policy-router",
            expires_at="2099-01-01T00:00:00Z",
            identity_claims=claims,
        )

    threshold = AcceptanceThresholdV1(metric="quality", operator=">=", value=1)
    prior = route(model="prior-bounded-plan")
    selected = route(model="selected-bounded-plan")
    with pytest.raises(DomainValidationError, match="quality.threshold_invalid"):
        valid_plan(
            "decision-bounded-plan",
            prior,
            selected,
            acceptance_thresholds=(threshold,) * 129,
        )

def test_sensitive_task_and_decision_content_is_rejected_without_token_name_false_positive():
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        TaskEnvelope.from_mapping({"prompt": "private"})
    attempted = route()
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        replan(
            task_id="task-1",
            attempt_id="attempt-1",
            decision_id="decision-2",
            failed_route=attempted,
            classification=classification(attempted),
            candidates=(),
            recheck_evidence=("Bearer abc",),
        )

    for leaked_value in (
        "password=hunter2",
        "api_key=not-safe",
        "token=not-safe",
        "prompt=private-content",
        "github_pat_exampletoken",
        "-----BEGIN PRIVATE KEY----- material",
    ):
        with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
            replan(
                task_id="task-1",
                attempt_id="attempt-1",
                decision_id="decision-2",
                failed_route=attempted,
                classification=classification(attempted),
                candidates=(),
                recheck_evidence=(leaked_value,),
            )

@pytest.mark.parametrize(
    "parser",
    (
        RouteV1.from_mapping,
        AuditedModelJustification.from_mapping,
        TaskEnvelope.from_mapping,
        InitialSelectionTriggerV1.from_mapping,
        RuntimeErrorClassificationV1.from_mapping,
        CandidateEvaluation.from_mapping,
        AcceptanceThresholdV1.from_mapping,
        CompensationEscalationV1.from_mapping,
        IndependentReviewAttestationV1.from_mapping,
        QualityCompensationPlanV1.from_mapping,
        RouteDecisionV1.from_mapping,
    ),
)
@pytest.mark.parametrize("malformed", (None, 7, [], "not-a-mapping"))
def test_public_mapping_parsers_reject_non_mappings_with_domain_errors(
    parser: Callable[[Any], object], malformed: object
):
    with pytest.raises(DomainValidationError):
        parser(malformed)

def test_endpoint_and_justification_reject_noncanonical_unicode_or_time():
    with pytest.raises(DomainValidationError, match="route.endpoint_invalid"):
        route(endpoint="https://api.example.com/\ud800")
    base = {
        "policy_version": "policy/v1",
        "reason": "audited exception",
        "evidence_refs": ("evidence:audit",),
        "author": "policy-owner",
    }
    for expires_at in (
        "2099-08-01T00:00:00+01:00",
        "2099-W31-5T00:00:00Z",
        "20990801T000000Z",
    ):
        with pytest.raises(DomainValidationError, match="task.justification_expiry_invalid"):
            TaskEnvelope.from_mapping(
                task_payload(
                    model="audited-model",
                    provider="audited-provider",
                    audited_model_justification={**base, "expires_at": expires_at},
                )
            )

def test_sensitive_scan_covers_complete_public_graphs_and_bytearray():
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        TaskEnvelope.from_mapping(
            task_payload(
                model="audited-model",
                provider="audited-provider",
                audited_model_justification={
                    "policy_version": "policy/v1",
                    "reason": "audited exception",
                    "evidence_refs": ("evidence:audit",),
                    "author": "github_pat_REDACTED",
                    "expires_at": "2099-08-01T00:00:00Z",
                },
            )
        )
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        InitialSelectionTriggerV1(
            "initial-selection-trigger/v1", "initial_selection", "policy", "gho_REDACTED"
        )
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        CompensationEscalationV1(EscalationAction.BLOCK_DISPATCH, "xoxb-REDACTED")
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        IndependentReviewAttestationV1(
            "sk-proj-REDACTED",
            INDEPENDENT_REVIEW_ROUTE_ID,
            "review-quota-pool",
            "review-billing-pool",
            "attempt-1",
            "review-execution",
            "evidence:test-run",
        )
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        validate_task_transition(
            from_state=TaskState.NEW,
            to_state=TaskState.PLANNED,
            **{**audit("task_id"), "actor": "github_pat_REDACTED"},
        )
    attempted = route()
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        RuntimeErrorClassificationV1.from_mapping(
            {
                "kind": "capacity_exhausted",
                "source": "runtime",
                "attempted_route_id": attempted.route_id,
                "quota_pool_id": attempted.quota_pool_id,
                "billing_pool_id": bytearray(b"gho_REDACTED"),
                "classified_at": "2026-07-26T17:00:00Z",
            }
        )

def test_capacity_public_boundaries_normalize_malformed_registries_and_nested_dtos():
    with pytest.raises(DomainValidationError, match="capacity.freshness_invalid"):
        capacity_observation(freshness=[])  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError, match="capacity.confidence_invalid"):
        capacity_observation(confidence=[])  # type: ignore[call-arg]
    with pytest.raises(DomainValidationError, match="capacity.metric_invalid"):
        CapacityObservationV1.from_mapping({
            **asdict(capacity_observation()),
            "metric": [],
        })
    with pytest.raises(DomainValidationError, match="capacity.unit_registry_required"):
        capacity_view(canonical_units={"remaining": []})  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError, match="capacity.authority_required"):
        capacity_view(
            conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
            authoritative_sources={"pool-a": []},  # type: ignore[dict-item]
        )

    forged = CapacityValueV1(known=True, value=77, unit="requests")
    object.__setattr__(forged, "unit", None)
    with pytest.raises(DomainValidationError, match="capacity.unit_invalid"):
        PoolCapacityV1(
            schema_version="capacity-pool/v1",
            quota_pool_id="pool-a",
            request=forged,
            token=CapacityValueV1(known=False),
            concurrency=CapacityValueV1(known=False),
        )

    valid_pool = PoolCapacityV1(
        schema_version="capacity-pool/v1",
        quota_pool_id="pool-a",
        request=CapacityValueV1(known=True, value=1, unit="requests"),
        token=CapacityValueV1(known=False),
        concurrency=CapacityValueV1(known=False),
    )
    object.__setattr__(valid_pool.request, "unit", None)
    with pytest.raises(DomainValidationError, match="capacity.unit_invalid"):
        CapacityViewV1(
            view_id="forged-view",
            built_at="2026-07-26T17:30:00Z",
            pools=(valid_pool,),
            source_observation_ids=(),
        )

def test_typed_capacity_and_execution_rehydration_rejects_cycles_with_domain_errors():
    cyclic_observation = capacity_observation()
    object.__setattr__(cyclic_observation, "source", cyclic_observation)
    with pytest.raises(DomainValidationError) as observation_error:
        capacity_view(cyclic_observation)
    assert observation_error.value.code == "contract.payload_too_complex"

    cyclic_pool = PoolCapacityV1(
        schema_version="capacity-pool/v1",
        quota_pool_id="cyclic-pool",
        request=CapacityValueV1(known=True, value=1, unit="requests"),
        token=CapacityValueV1(known=False),
        concurrency=CapacityValueV1(known=False),
    )
    object.__setattr__(cyclic_pool.request, "value", cyclic_pool)
    with pytest.raises(DomainValidationError) as pool_error:
        CapacityViewV1(
            view_id="cyclic-view",
            built_at="2026-07-26T17:30:00Z",
            pools=(cyclic_pool,),
            source_observation_ids=(),
        )
    assert pool_error.value.code == "contract.payload_too_complex"

    task, decision, routes = execution_authority()
    payload = execution_payload(task, decision)
    object.__setattr__(task.budget, "currency", task.budget)
    with pytest.raises(DomainValidationError) as task_error:
        ExecutionEnvelopeV1.from_mapping(
            payload,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    assert task_error.value.code == "contract.payload_too_complex"

    task, decision, routes = execution_authority()
    payload = execution_payload(task, decision)
    object.__setattr__(decision.trigger, "source", decision.trigger)
    with pytest.raises(DomainValidationError) as decision_error:
        ExecutionEnvelopeV1.from_mapping(
            payload,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    assert decision_error.value.code == "contract.payload_too_complex"

@pytest.mark.parametrize("identity_value", ([], {"unexpected": "shape"}, [0] * 9000))
def test_task_identity_fields_require_exact_strings_without_container_coercion(
    identity_value: object,
):
    with pytest.raises(DomainValidationError) as error:
        TaskEnvelope.from_mapping(
            task_payload(
                model=identity_value,
                audited_model_justification={
                    "policy_version": "policy/v1",
                    "reason": "audited identity",
                    "evidence_refs": ["evidence:test"],
                    "author": "policy-owner",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
        )
    assert error.value.code == "task.scalar_invalid"

def test_task_identity_field_rejects_explosive_string_without_invoking_it():
    explosive = ExplosiveString()
    with pytest.raises(DomainValidationError) as error:
        TaskEnvelope.from_mapping(
            task_payload(
                model=explosive,
                audited_model_justification={
                    "policy_version": "policy/v1",
                    "reason": "audited identity",
                    "evidence_refs": ["evidence:test"],
                    "author": "policy-owner",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )
        )
    assert error.value.code == "task.scalar_invalid"
    assert explosive.called is False

@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("billing_pool_id", []),
        ("billing_pool_id", {"unexpected": "shape"}),
        ("evidence_code", []),
        ("evidence_code", {"unexpected": "shape"}),
        ("evidence_code", [0] * 9000),
    ),
)
def test_runtime_classification_optional_text_requires_exact_strings(
    field_name: str,
    value: object,
):
    attempted = route(model=f"classification-{field_name}")
    payload = asdict(classification(attempted))
    payload[field_name] = value
    with pytest.raises(DomainValidationError) as error:
        RuntimeErrorClassificationV1.from_mapping(payload)
    assert error.value.code == "classification.scalar_invalid"

def test_runtime_classification_rejects_explosive_string_without_invoking_it():
    attempted = route(model="classification-explosive")
    explosive = ExplosiveString()
    payload = asdict(classification(attempted))
    payload["evidence_code"] = explosive
    with pytest.raises(DomainValidationError) as error:
        RuntimeErrorClassificationV1.from_mapping(payload)
    assert error.value.code == "classification.scalar_invalid"
    assert explosive.called is False

def test_quality_plan_direct_reconstruction_rehydrates_forged_nested_dtos():
    failed = route(model="direct-plan-failed", quota_pool_id="direct-plan-failed")
    selected = route(model="direct-plan-selected", quota_pool_id="direct-plan-selected")
    plan = valid_plan("decision-direct-plan", failed, selected)
    object.__setattr__(plan.acceptance_thresholds[0], "operator", "ALWAYS")

    assert plan.valid_for(
        decision_id=plan.decision_id,
        prior_route_id=failed.route_id,
        selected_route_id=selected.route_id,
        attempt_id="attempt-1",
        task_verification_minimum="V2",
        task_independence_required=True,
        task_human_gate_required=False,
        decision_policy_version=plan.policy_version,
        decision_verification="V2",
        trusted_reviewer_routes={
            INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE,
        },
        trusted_human_approval_refs=frozenset(),
        trusted_execution_routes={"review-execution": INDEPENDENT_REVIEW_ROUTE},
        trusted_execution_evidence={
            "review-execution": frozenset({"evidence:test-run"}),
        },
        trusted_evidence_refs=frozenset({"evidence:test-run"}),
        trusted_threshold_results={"evidence:test-run": True},
    ) is False
    with pytest.raises(DomainValidationError) as error:
        replace(plan)
    assert error.value.code == "quality.threshold_invalid"

def test_pure_phase_forged_review_evidence_and_threshold_maps_never_satisfy_quality():
    failed = route(model="forged-authority-failed", quota_pool_id="forged-authority-failed")
    selected = route(
        model="forged-authority-selected",
        quota_pool_id="forged-authority-selected",
    )
    task = TaskEnvelope.from_mapping(task_payload(task_id="task-forged-authority"))
    plan = valid_plan(
        "decision-forged-authority",
        failed,
        selected,
        reviewed_execution_id="attempt-forged-authority",
    )
    candidates = (
        CandidateEvaluation(failed, False, ("failed_route",)),
        scored_candidate(selected),
    )
    forged_authorities = {
        "trusted_reviewer_routes": {
            INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE,
        },
        "trusted_execution_routes": {
            "review-execution": INDEPENDENT_REVIEW_ROUTE,
        },
        "trusted_execution_evidence": {
            "review-execution": ("evidence:test-run",),
        },
        "trusted_evidence_refs": ("evidence:test-run",),
        "trusted_threshold_results": {"evidence:test-run": True},
    }

    fallback = replan(
        trusted_task=task,
        task_id=task.task_id,
        attempt_id="attempt-forged-authority",
        decision_id="decision-forged-authority",
        failed_route=failed,
        classification=classification(failed),
        candidates=candidates,
        quality_compensation_plan=plan,
        task_verification_minimum="V2",
        **forged_authorities,
    )
    direct = RouteDecisionV1(
        decision_id="decision-forged-authority",
        task_id=task.task_id,
        attempt_id="attempt-forged-authority",
        fallback=True,
        relation=DecisionRelation.FALLBACK,
        candidates=candidates,
        selected_route_id=selected.route_id,
        trigger=classification(
            failed,
            billing_pool_id=failed.billing_pool_id,
        ),
        reason_codes=("route_capacity_exhausted",),
        prior_route_id=failed.route_id,
        parent_decision_id="parent-forged-authority",
        quality_compensation_plan=plan,
        trusted_task=task,
        trusted_routes={failed.route_id: failed, selected.route_id: selected},
        trusted_prior_route=failed,
        **decision_metadata(),
        **forged_authorities,
    )
    persisted = RouteDecisionV1.from_mapping(
        asdict(fallback),
        trusted_task=task,
        trusted_routes={failed.route_id: failed, selected.route_id: selected},
        **forged_authorities,
    )

    for decision in (fallback, direct, persisted):
        assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
        assert decision.activation_block_reason == "quality_compensation_insufficient"
        assert "quality_compensation_insufficient" in decision.reason_codes
        assert decision.dispatchable is False

def test_initial_decision_ignores_forged_quality_authority_maps():
    selected = route(model="initial-forged-authority", quota_pool_id="initial-forged-authority")
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-initial-forged-authority",
            verification={
                "minimum": "V2",
                "independent_required": True,
                "human_gate_required": False,
            },
        )
    )
    values = {
        "decision_id": "decision-initial-forged-authority",
        "task_id": task.task_id,
        "attempt_id": "attempt-initial-forged-authority",
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "candidates": (scored_candidate(selected),),
        "selected_route_id": selected.route_id,
        "trigger": InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy",
            "2026-07-27T12:00:00Z",
        ),
        "reason_codes": ("initial_selection",),
        "trusted_task": task,
        "trusted_routes": {selected.route_id: selected},
        **decision_metadata(),
    }
    blocked = RouteDecisionV1(**values)
    forged = RouteDecisionV1(
        **values,
        trusted_reviewer_routes={
            INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE,
        },
        trusted_execution_routes={"review-execution": INDEPENDENT_REVIEW_ROUTE},
        trusted_execution_evidence={"review-execution": ("evidence:test-run",)},
        trusted_evidence_refs=("evidence:test-run",),
        trusted_threshold_results={"evidence:test-run": True},
    )

    assert blocked.policy_status == "ACTIVATION_BLOCKED_INDEPENDENT_REVIEW"
    assert forged.policy_status == blocked.policy_status
    assert forged.activation_block_reason == blocked.activation_block_reason
    assert forged.dispatchable is False

def test_safe_typed_rehydration_contains_deepcopy_runtime_errors_without_message_leak():
    observed = capacity_observation(observation_id="deepcopy-observation")
    object.__setattr__(observed, "source", DeepcopyBomb())

    bound_route = route(model="deepcopy-reservation", quota_pool_id="deepcopy-reservation")
    reservation = capacity_reservation(bound_route)
    object.__setattr__(reservation, "unit", DeepcopyBomb())

    failed = route(model="deepcopy-quality-failed", quota_pool_id="deepcopy-quality-failed")
    selected = route(
        model="deepcopy-quality-selected",
        quota_pool_id="deepcopy-quality-selected",
    )
    plan = valid_plan("decision-deepcopy-quality", failed, selected)
    object.__setattr__(plan.acceptance_thresholds[0], "metric", DeepcopyBomb())

    for value, location in (
        (observed, "capacity observation"),
        (reservation, "capacity reservation"),
        (plan, "quality compensation plan"),
    ):
        with pytest.raises(DomainValidationError) as error:
            orchestration._safe_asdict(value, location)
        assert error.value.code == "contract.payload_too_complex"
        assert "ATTACKER_DEEPCOPY_MESSAGE" not in str(error.value)

@pytest.mark.parametrize(
    ("case", "error_code"),
    (
        ("task_verification", "task.verification_invalid"),
        ("task_effort", "task.effort_invalid"),
        ("quality_verification", "quality.verification_invalid"),
        ("decision_effort", "decision.effort_invalid"),
        ("execution_verification", "execution.verification_invalid"),
    ),
)
@pytest.mark.parametrize("invalid_kind", ("hash", "list", "dict", "explosive"))
def test_untrusted_rank_and_membership_labels_fail_without_hash_or_string_hooks(
    case: str,
    error_code: str,
    invalid_kind: str,
):
    value: object
    if invalid_kind == "hash":
        value = HashBomb()
    elif invalid_kind == "list":
        value = ["V2"]
    elif invalid_kind == "dict":
        value = {"label": "V2"}
    else:
        value = ExplosiveString()

    with pytest.raises(DomainValidationError) as error:
        _invoke_untrusted_label_boundary(case, value)
    assert error.value.code == error_code
    assert "ATTACKER_HASH_MESSAGE" not in str(error.value)
    if isinstance(value, HashBomb):
        assert value.called is False
    if isinstance(value, ExplosiveString):
        assert value.called is False

@pytest.mark.parametrize(
    ("case", "error_code"),
    (
        ("capacity_value", "capacity.value_invalid"),
        ("observation_value", "capacity.value_invalid"),
        ("observation_confidence", "capacity.confidence_invalid"),
        ("observation_price", "capacity.value_invalid"),
        ("task_budget", "task.budget_invalid"),
        ("candidate_score", "candidate.invalid"),
        ("score", "score.invalid"),
        ("threshold", "quality.threshold_invalid"),
        ("reservation_estimated", "reservation.amount_invalid"),
        ("reservation_held", "reservation.amount_invalid"),
        ("execution_timeout", "execution.timeout_invalid"),
    ),
)
def test_every_public_numeric_conversion_rejects_huge_int_without_raw_exception(
    case: str,
    error_code: str,
):
    with pytest.raises(DomainValidationError) as error:
        _invoke_public_numeric_boundary(case, 10**10000)
    assert error.value.code == error_code

@pytest.mark.parametrize("value", (True, "1", float("inf"), float("-inf"), float("nan")))
def test_exact_finite_number_boundary_rejects_bool_non_number_and_non_finite(
    value: object,
):
    with pytest.raises(DomainValidationError) as error:
        CapacityValueV1(known=True, value=value, unit="requests")  # type: ignore[arg-type]
    assert error.value.code == "capacity.value_invalid"

def test_exact_finite_number_boundary_never_invokes_attacker_conversion():
    explosive = ExplosiveNumber()
    with pytest.raises(DomainValidationError) as error:
        CapacityValueV1(known=True, value=explosive, unit="requests")  # type: ignore[arg-type]
    assert error.value.code == "capacity.value_invalid"
    assert explosive.called is False

@pytest.mark.parametrize(
    ("case", "error_code"),
    (
        ("candidate", "candidate.invalid"),
        ("relation", "decision.relation_invalid"),
        ("escalation", "quality.escalation_invalid"),
        ("reservation", "reservation.status_invalid"),
        ("breaker", "breaker.status_invalid"),
        ("outcome", "execution.outcome_invalid"),
        ("dispatch", "execution.dispatch_state_invalid"),
    ),
)
@pytest.mark.parametrize("invalid_kind", ("list", "dict", "explosive"))
def test_public_status_and_enum_parsers_require_exact_scalars(
    case: str,
    error_code: str,
    invalid_kind: str,
):
    value: object
    if invalid_kind == "list":
        value = ["PENDING"]
    elif invalid_kind == "dict":
        value = {"status": "PENDING"}
    else:
        value = ExplosiveString()

    with pytest.raises(DomainValidationError) as error:
        _invoke_public_status_boundary(case, value)
    assert error.value.code == error_code
    if isinstance(value, ExplosiveString):
        assert value.called is False

def test_integer_domain_bounds_accept_documented_maxima():
    context = _task_with_context_bounds(
        max_tokens=orchestration._MAX_CONTEXT_TOKENS,
        token_count=orchestration._MAX_CONTEXT_TOKENS,
    )
    assert context.context.max_tokens == orchestration._MAX_CONTEXT_TOKENS

    observation = CapacityObservationV1.from_mapping(
        {
            **asdict(capacity_observation()),
            "max_concurrency": orchestration._MAX_CAPACITY_CONCURRENCY,
        }
    )
    assert observation.max_concurrency == orchestration._MAX_CAPACITY_CONCURRENCY

    bound_route = route(model="integer-maxima", quota_pool_id="integer-maxima")
    reservation = orchestration.CapacityReservationV1.from_mapping(
        {
            **reservation_payload(bound_route),
            "version": orchestration._MAX_CONTRACT_VERSION,
        },
        trusted_route=bound_route,
    )
    breaker = orchestration.CircuitBreakerV1.from_mapping(
        breaker_payload(
            bound_route,
            status="HALF_OPEN",
            probe_budget=orchestration._MAX_BREAKER_PROBE_BUDGET,
            version=orchestration._MAX_CONTRACT_VERSION,
        ),
        trusted_route=bound_route,
    )
    assert reservation.version == breaker.version == orchestration._MAX_CONTRACT_VERSION
    assert breaker.probe_budget == orchestration._MAX_BREAKER_PROBE_BUDGET

@pytest.mark.parametrize(
    ("case", "limit", "error_code"),
    (
        ("context_max_tokens", "_MAX_CONTEXT_TOKENS", "task.context_bounds_invalid"),
        ("context_token_count", "_MAX_CONTEXT_TOKENS", "task.context_bounds_invalid"),
        (
            "max_concurrency",
            "_MAX_CAPACITY_CONCURRENCY",
            "capacity.max_concurrency_invalid",
        ),
        ("reservation_version", "_MAX_CONTRACT_VERSION", "reservation.version_invalid"),
        (
            "breaker_probe_budget",
            "_MAX_BREAKER_PROBE_BUDGET",
            "breaker.probe_budget_invalid",
        ),
        ("breaker_version", "_MAX_CONTRACT_VERSION", "breaker.version_invalid"),
    ),
)
@pytest.mark.parametrize("excess", (1, 10**10000), ids=("n-plus-one", "giant-int"))
def test_integer_domain_bounds_reject_n_plus_one_and_giant_int(
    case: str,
    limit: str,
    error_code: str,
    excess: int,
):
    maximum = getattr(orchestration, limit)
    value = maximum + excess if excess == 1 else excess
    with pytest.raises(DomainValidationError) as error:
        _invoke_public_integer_boundary(case, value)
    assert error.value.code == error_code

@pytest.mark.parametrize(
    ("case", "error_code"),
    (
        ("context_max_tokens", "task.context_bounds_invalid"),
        ("context_token_count", "task.context_bounds_invalid"),
        ("max_concurrency", "capacity.max_concurrency_invalid"),
        ("reservation_version", "reservation.version_invalid"),
        ("breaker_probe_budget", "breaker.probe_budget_invalid"),
        ("breaker_version", "breaker.version_invalid"),
    ),
)
@pytest.mark.parametrize("value", (True, -1), ids=("bool", "negative"))
def test_integer_domain_bounds_reject_bool_and_negative(
    case: str,
    error_code: str,
    value: object,
):
    with pytest.raises(DomainValidationError) as error:
        _invoke_public_integer_boundary(case, value)
    assert error.value.code == error_code

# ---------------------------------------------------------------------------
# L1 — hostile custom Mapping adapters are contained at from_mapping boundaries.
# The pure slice's supported input is decoded mappings; a custom Mapping whose
# container hooks raise must surface a stable DomainValidationError instead of
# leaking its own RuntimeError/TypeError/ValueError.
# ---------------------------------------------------------------------------

# Every public from_mapping that takes a single decoded ``payload`` argument.
_SNAPSHOT_PARSERS = (
    RouteV1.from_mapping,
    AuditedModelJustification.from_mapping,
    TaskEnvelope.from_mapping,
    InitialSelectionTriggerV1.from_mapping,
    RuntimeErrorClassificationV1.from_mapping,
    CandidateEvaluation.from_mapping,
    AcceptanceThresholdV1.from_mapping,
    CompensationEscalationV1.from_mapping,
    IndependentReviewAttestationV1.from_mapping,
    QualityCompensationPlanV1.from_mapping,
    CapacityValueV1.from_mapping,
    CapacityObservationV1.from_mapping,
    PoolCapacityV1.from_mapping,
    CapacityViewV1.from_mapping,
    orchestration.ExecutionOutcomeV1.from_mapping,
    ExecutionIdentityAttestationV1.from_mapping,
    orchestration.ExecutionVerificationAttestationV1.from_mapping,
)

def _assert_contained(error: DomainValidationError) -> None:
    assert HOSTILE_MAPPING_MARKER not in str(error)
    assert "ATTACKER" not in str(error)

@pytest.mark.parametrize(
    "parser",
    _SNAPSHOT_PARSERS,
    ids=lambda parser: parser.__self__.__name__,
)
@pytest.mark.parametrize("mode", HOSTILE_MAPPING_HOOKS)
def test_public_from_mapping_snapshots_contain_hostile_mapping_adapters(
    parser: Callable[[Any], object],
    mode: str,
):
    # A single benign unknown key means the ``items``/``get`` bypass modes still
    # reach a downstream schema rejection, while ``len``/``iter``/``getitem`` are
    # contained inside the snapshot itself. Every path stays a domain error.
    hostile = HostileMapping(mode, base={"unexpected_probe_field": "value"})
    with pytest.raises(DomainValidationError) as error:
        parser(hostile)
    _assert_contained(error.value)

@pytest.mark.parametrize("mode", HOSTILE_MAPPING_HOOKS)
@pytest.mark.parametrize("exc", (RuntimeError, TypeError, ValueError))
def test_hostile_mapping_exception_types_are_all_contained(
    mode: str,
    exc: type[BaseException],
):
    hostile = HostileMapping(mode, exc=exc, base={"unexpected_probe_field": "value"})
    with pytest.raises(DomainValidationError) as error:
        RouteV1.from_mapping(hostile)
    _assert_contained(error.value)

def test_multi_authority_from_mapping_boundaries_contain_hostile_mappings():
    task, decision, routes = execution_authority()
    bound = route(model="hostile-lease", quota_pool_id="hostile-lease")
    base = {"unexpected_probe_field": "value"}
    for mode in HOSTILE_MAPPING_HOOKS:
        with pytest.raises(DomainValidationError) as decision_error:
            RouteDecisionV1.from_mapping(
                HostileMapping(mode, base=base),
                trusted_task=task,
                trusted_routes=routes,
            )
        _assert_contained(decision_error.value)

        with pytest.raises(DomainValidationError) as execution_error:
            ExecutionEnvelopeV1.from_mapping(
                HostileMapping(mode, base=base),
                trusted_task=task,
                trusted_decision=decision,
                trusted_routes=routes,
            )
        _assert_contained(execution_error.value)

        with pytest.raises(DomainValidationError) as reservation_error:
            orchestration.CapacityReservationV1.from_mapping(
                HostileMapping(mode, base=base),
                trusted_route=bound,
            )
        _assert_contained(reservation_error.value)

        with pytest.raises(DomainValidationError) as breaker_error:
            orchestration.CircuitBreakerV1.from_mapping(
                HostileMapping(mode, base=base),
                trusted_route=bound,
            )
        _assert_contained(breaker_error.value)

@pytest.mark.parametrize("mode", HOSTILE_MAPPING_HOOKS)
def test_nested_hostile_mapping_inside_decoded_payload_is_contained(mode: str):
    # A hostile Mapping nested inside an otherwise-valid decoded payload — the
    # persisted decision trigger, a persisted candidate, and an audited
    # justification — must also be snapshotted before its keys are read.
    selected = route(model="nested-hostile", quota_pool_id="nested-hostile")
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-nested-hostile",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    metadata = decision_metadata()
    metadata["verification"] = "V0"
    decision = RouteDecisionV1(
        decision_id="decision-nested-hostile",
        task_id=task.task_id,
        attempt_id="attempt-nested-hostile",
        **metadata,
        fallback=False,
        relation=DecisionRelation.INITIAL,
        candidates=(scored_candidate(selected),),
        selected_route_id=selected.route_id,
        trigger=InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy",
            "2026-07-27T09:00:00Z",
        ),
        reason_codes=("initial_selection",),
        trusted_task=task,
        trusted_routes={selected.route_id: selected},
    )
    routes = {selected.route_id: selected}
    persisted = asdict(decision)

    trigger_payload = dict(persisted)
    trigger_payload["trigger"] = HostileMapping(
        mode, base={"schema_version": "initial-selection-trigger/v1"}
    )
    with pytest.raises(DomainValidationError) as trigger_error:
        RouteDecisionV1.from_mapping(
            trigger_payload, trusted_task=task, trusted_routes=routes
        )
    _assert_contained(trigger_error.value)

    candidate_payload = dict(persisted)
    candidate_payload["candidates"] = (
        HostileMapping(mode, base={"route_id": selected.route_id}),
    )
    with pytest.raises(DomainValidationError) as candidate_error:
        RouteDecisionV1.from_mapping(
            candidate_payload, trusted_task=task, trusted_routes=routes
        )
    _assert_contained(candidate_error.value)

    with pytest.raises(DomainValidationError) as justification_error:
        TaskEnvelope.from_mapping(
            task_payload(
                model="nested-hostile-model",
                provider="nested-hostile-provider",
                audited_model_justification=HostileMapping(
                    mode, base={"policy_version": "policy/v1"}
                ),
            )
        )
    _assert_contained(justification_error.value)

@pytest.mark.parametrize("mode", ("len", "iter", "items", "getitem"))
@pytest.mark.parametrize("complete_graph", (False, True), ids=("default", "complete"))
def test_sensitive_scanner_contains_hostile_nested_container_adapters(
    mode: str,
    complete_graph: bool,
):
    # The shared secret scanner walks nested containers via len()/items(); a
    # hostile nested Mapping must be contained, not leaked. ``get`` is not a
    # graph-walk vector (the scanner never calls it) so it is covered only at
    # the from_mapping boundary above.
    hostile = HostileMapping(mode, base={"k": "v"})
    with pytest.raises(DomainValidationError) as error:
        _reject_sensitive(
            {"outer": {"inner": hostile}},
            "sensitive probe",
            validate_complete_graph=complete_graph,
        )
    assert error.value.code == "contract.payload_too_complex"
    _assert_contained(error.value)

@pytest.mark.parametrize(
    "parser",
    _SNAPSHOT_PARSERS,
    ids=lambda parser: parser.__self__.__name__,
)
def test_oversized_hostile_mapping_scans_only_bounded_keys_across_all_peers(
    parser: Callable[[Any], object],
):
    payload = HugePrefixMapping()
    with pytest.raises(DomainValidationError) as error:
        parser(payload)
    assert error.value.code == "contract.payload_too_complex"
    assert payload.keys_yielded == 129
    assert payload.values_read == 0

def test_snapshot_preserves_ordinary_decoded_mapping_semantics():
    from collections import OrderedDict
    from types import MappingProxyType

    base = route(model="snapshot-equivalence").canonical_object()
    expected = RouteV1.from_mapping(dict(base)).route_id
    # Ordinary decoded dict/JSON behaviour is unchanged; a non-dict Mapping is
    # snapshotted into an equivalent plain dict without coercing any value.
    assert RouteV1.from_mapping(OrderedDict(base)).route_id == expected
    assert RouteV1.from_mapping(MappingProxyType(dict(base))).route_id == expected
    # The snapshot copies via iteration + __getitem__, never via items(), so a
    # Mapping whose only hostile hook is items() still builds the same route.
    assert RouteV1.from_mapping(HostileMapping("items", base=base)).route_id == expected
    assert RouteV1.from_mapping(HostileMapping("get", base=base)).route_id == expected

# ---------------------------------------------------------------------------
# L2 — exact graph-budget edges: depth/node N versus N+1, benign DAG aliases,
# and hostile containers as part of the supported decoded-object contract.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", (nested_lists, nested_dicts), ids=("lists", "dicts"))
def test_graph_depth_budget_exact_edge_accepts_n_and_rejects_n_plus_one(
    build: Callable[[int], object],
):
    # A decoded structure nested to exactly _MAX_GRAPH_DEPTH is accepted; one
    # level deeper is rejected. No timing assertion, no unbounded recursion.
    _reject_sensitive(build(_MAX_GRAPH_DEPTH), "depth probe")

    with pytest.raises(DomainValidationError) as error:
        _reject_sensitive(build(_MAX_GRAPH_DEPTH + 1), "depth probe")
    assert error.value.code == "contract.payload_too_complex"

    # The budget never suppresses sensitive-value detection at the deepest node.
    with pytest.raises(DomainValidationError) as sensitive_error:
        _reject_sensitive(build(_MAX_GRAPH_DEPTH, "gho_REDACTED"), "depth probe")
    assert sensitive_error.value.code == "decision.sensitive_field_prohibited"

def test_graph_depth_budget_enforced_through_public_decoded_object_boundary():
    # The exact depth edge is wired into a real public from_mapping. A context
    # nested to _MAX_GRAPH_DEPTH exceeds the budget (its leaf sits one past the
    # limit); one level shallower is rejected only by later schema validation,
    # never by the graph budget.
    over = TaskEnvelope
    with pytest.raises(DomainValidationError) as depth_error:
        over.from_mapping(task_payload(context=nested_dicts(_MAX_GRAPH_DEPTH)))
    assert depth_error.value.code == "contract.payload_too_complex"

    with pytest.raises(DomainValidationError) as schema_error:
        TaskEnvelope.from_mapping(
            task_payload(context=nested_dicts(_MAX_GRAPH_DEPTH - 1))
        )
    assert schema_error.value.code != "contract.payload_too_complex"

def test_graph_node_budget_exact_edge_in_complete_graph_scan():
    # In the complete-graph rehydration scan a flat list of _MAX_GRAPH_NODES - 1
    # leaves is exactly at the node budget (the list plus its leaves); one more
    # leaf exceeds it. This is the bound used by trusted-DTO rehydration.
    at_limit = ["node"] * (_MAX_GRAPH_NODES - 1)
    _reject_sensitive(at_limit, "node probe", validate_complete_graph=True)

    over_limit = ["node"] * _MAX_GRAPH_NODES
    with pytest.raises(DomainValidationError) as error:
        _reject_sensitive(over_limit, "node probe", validate_complete_graph=True)
    assert error.value.code == "contract.payload_too_complex"

@pytest.mark.parametrize("complete_graph", (False, True), ids=("default", "complete"))
def test_benign_dag_aliases_are_accepted_and_cycles_are_rejected(
    complete_graph: bool,
):
    shared_mapping = {"detail": "value"}
    shared_sequence = ["a", "b"]
    # A benign DAG: the same object is reachable by several sibling paths. This
    # is not a cycle and must be accepted (and de-duplicated, not re-walked).
    dag = {
        "first": shared_mapping,
        "second": shared_mapping,
        "third": [shared_sequence, shared_sequence, {"again": shared_mapping}],
    }
    _reject_sensitive(dag, "alias probe", validate_complete_graph=complete_graph)

    # A real self-cycle through the same alias is still rejected.
    cyclic: dict[str, object] = {"child": shared_mapping}
    cyclic["loop"] = cyclic
    with pytest.raises(DomainValidationError) as cycle_error:
        _reject_sensitive(cyclic, "alias probe", validate_complete_graph=complete_graph)
    assert cycle_error.value.code == "contract.payload_too_complex"

    # Aliasing does not hide a sensitive value inside the shared subtree.
    sensitive_shared = {"author": "gho_REDACTED"}
    with pytest.raises(DomainValidationError) as sensitive_error:
        _reject_sensitive(
            {"left": sensitive_shared, "right": sensitive_shared},
            "alias probe",
            validate_complete_graph=complete_graph,
        )
    assert sensitive_error.value.code == "decision.sensitive_field_prohibited"

def test_dag_alias_fanout_cannot_escape_the_complete_graph_node_budget():
    # A single small subtree aliased _MAX_GRAPH_NODES times still counts one node
    # per reference, so the node budget rejects the fan-out rather than being
    # bypassed by the visited-node de-duplication.
    shared = {"detail": "value"}
    fanned = {"root": [shared for _ in range(_MAX_GRAPH_NODES)]}
    with pytest.raises(DomainValidationError) as error:
        _reject_sensitive(fanned, "alias probe", validate_complete_graph=True)
    assert error.value.code == "contract.payload_too_complex"

def test_graph_collection_scan_exact_edge_defers_oversized_to_field_validator():
    # The default decoded-object scan walks a collection at exactly
    # _MAX_TEXT_COLLECTION_ITEMS and catches a sensitive value inside it; one
    # element more, the scanner defers to the owning field validator (which
    # rejects the oversized collection) instead of scanning unbounded content.
    at_limit = {"refs": ["gho_REDACTED"] + ["safe"] * (_MAX_TEXT_COLLECTION_ITEMS - 1)}
    with pytest.raises(DomainValidationError) as scanned_error:
        _reject_sensitive(at_limit, "collection probe")
    assert scanned_error.value.code == "decision.sensitive_field_prohibited"

    over_limit = {"refs": ["gho_REDACTED"] + ["safe"] * _MAX_TEXT_COLLECTION_ITEMS}
    _reject_sensitive(over_limit, "collection probe")

    # The complete-graph rehydration scan never defers: it always walks the whole
    # collection, so the same oversized alias is still caught.
    with pytest.raises(DomainValidationError) as complete_error:
        _reject_sensitive(over_limit, "collection probe", validate_complete_graph=True)
    assert complete_error.value.code == "decision.sensitive_field_prohibited"

def test_mapping_snapshot_primitive_preserves_dicts_and_rejects_non_mappings():
    from collections import OrderedDict

    # An ordinary decoded dict is returned unchanged (same object): identical
    # JSON/dict behaviour and error precedence downstream.
    decoded = {"a": 1, "b": 2}
    assert _mapping_snapshot(decoded, code="c", location="thing") is decoded

    # A non-dict Mapping becomes an equivalent plain dict without coercion.
    snapshot = _mapping_snapshot(OrderedDict(decoded), code="c", location="thing")
    assert type(snapshot) is dict
    assert snapshot == decoded

    # A non-Mapping is a stable domain error carrying the caller's code.
    with pytest.raises(DomainValidationError) as error:
        _mapping_snapshot(7, code="route.unexpected_field", location="thing")
    assert error.value.code == "route.unexpected_field"

    # A non-dict Mapping past the field limit is rejected without reading values.
    oversized = HugePrefixMapping()
    with pytest.raises(DomainValidationError) as oversized_error:
        _mapping_snapshot(oversized, code="c", location="thing")
    assert oversized_error.value.code == "contract.payload_too_complex"
    assert oversized.keys_yielded == _MAX_MAPPING_FIELDS + 1
    assert oversized.values_read == 0

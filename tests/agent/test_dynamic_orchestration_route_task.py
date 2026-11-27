from __future__ import annotations

from tests.agent._dynamic_orchestration_support import (
    CandidateEvaluation,
    CandidateScoreV1,
    DecisionRelation,
    DomainValidationError,
    ELIGIBILITY_GATES,
    EligibilityDisposition,
    ErrorKind,
    FIXTURES,
    FrozenInstanceError,
    Path,
    QualityCompensationPlanV1,
    RouteEligibilityFactsV1,
    RouteV1,
    RuntimeErrorClassificationV1,
    TaskEnvelope,
    asdict,
    classification,
    datetime,
    eligibility_facts,
    evaluate_route_eligibility,
    json,
    os,
    permutations,
    pytest,
    replan,
    route,
    score_eligible_candidates,
    subprocess,
    sys,
    task_payload,
    timezone,
    valid_plan,
)
@pytest.mark.parametrize("fixture_path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.name)
def test_every_route_fixture_declares_and_matches_canonical_expectations(fixture_path: Path):
    fixture = json.loads(fixture_path.read_text())
    routes = [RouteV1.from_mapping(item) for item in fixture["inputs"]]

    assert fixture["fixture_version"] == "route-v1-fixture/1"
    if "expected_canonical_json" in fixture:
        assert {item.canonical_json for item in routes} == {fixture["expected_canonical_json"]}
    if "expected_route_id" in fixture:
        assert {item.route_id for item in routes} == {fixture["expected_route_id"]}
    else:
        assert [item.route_id for item in routes] == fixture["expected_route_ids"]

def test_endpoint_path_case_is_identity_significant():
    upper = route(endpoint="https://API.EXAMPLE.COM:443/V1/Responses")
    lower = route(endpoint="https://api.example.com/v1/responses")

    assert upper.endpoint == "https://api.example.com/V1/Responses"
    assert lower.endpoint == "https://api.example.com/v1/responses"
    assert upper.route_id != lower.route_id

def test_endpoint_percent_equivalents_share_identity_and_reserved_escape_is_uppercase():
    escaped = route(endpoint="https://api.example.com/v1/%7eresponses/%2f")
    literal = route(endpoint="https://api.example.com/v1/~responses/%2F")

    assert escaped.endpoint == "https://api.example.com/v1/~responses/%2F"
    assert escaped.route_id == literal.route_id

def test_endpoint_duplicate_slashes_are_identity_significant():
    duplicate = route(endpoint="https://api.example.com/a//b")
    single = route(endpoint="https://api.example.com/a/b")
    with_parent_segment = route(endpoint="https://api.example.com/a//../b")

    assert duplicate.endpoint == "https://api.example.com/a//b"
    assert duplicate.route_id != single.route_id
    assert with_parent_segment.endpoint == "https://api.example.com/a/b"

def test_endpoint_equivalent_idna_hosts_share_identity():
    unicode_host = route(endpoint="https://café.example/v1")
    ascii_host = route(endpoint="https://xn--caf-dma.example/v1")

    assert unicode_host.endpoint == "https://xn--caf-dma.example/v1"
    assert unicode_host.route_id == ascii_host.route_id

def test_endpoint_equivalent_ipv6_spellings_share_identity():
    expanded = route(endpoint="https://[2001:0db8:0000:0000:0000:0000:0000:0001]/v1")
    compressed = route(endpoint="https://[2001:db8::1]/v1")

    assert expanded.endpoint == "https://[2001:db8::1]/v1"
    assert expanded.route_id == compressed.route_id

def test_route_rejects_unknown_canonicalization_version():
    payload = json.loads(route().canonical_json)
    payload["canonicalization_version"] = "route-v2"
    with pytest.raises(DomainValidationError, match="route.canonicalization_unknown"):
        RouteV1.from_mapping(payload)

def test_route_rejects_missing_identity_query_fragment_and_malformed_percent():
    with pytest.raises(DomainValidationError, match="route.identity_required"):
        route(model="")
    for endpoint in (
        "https://api.example.com/v1?x=1",
        "https://api.example.com/v1#frag",
        "https://api.example.com/v1/%GG",
        "https://example.com/a b",
        "https://exa%mple.com/",
        "https://example.com/\\evil",
        "https://example.com/\x01x",
    ):
        with pytest.raises(DomainValidationError, match="route.endpoint_invalid"):
            route(endpoint=endpoint)

def test_route_identity_rejects_unicode_surrogates_with_stable_domain_error():
    with pytest.raises(DomainValidationError, match="route.value_invalid"):
        route(provider="openai\ud800")

def test_route_missing_field_error_is_hash_seed_invariant():
    script = (
        "from agent.dynamic_orchestration import RouteV1, DomainValidationError\n"
        "try:\n"
        "    RouteV1.from_mapping({})\n"
        "except DomainValidationError as exc:\n"
        "    print(str(exc))\n"
    )
    outputs = []
    for seed in ("0", "1", "2", "99"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[2],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())
    assert outputs == ["route.identity_required: provider is required"] * 4

def test_non_string_mapping_keys_fail_with_stable_domain_errors():
    route_mapping: dict[object, object] = {}
    route_mapping.update(json.loads(route().canonical_json))
    route_mapping[7] = "unexpected"
    with pytest.raises(DomainValidationError, match="route.unexpected_field"):
        RouteV1.from_mapping(route_mapping)  # type: ignore[arg-type]

    task_mapping: dict[object, object] = {}
    task_mapping.update(task_payload())
    task_mapping[7] = "unexpected"
    with pytest.raises(DomainValidationError, match="task.unexpected_field"):
        TaskEnvelope.from_mapping(task_mapping)  # type: ignore[arg-type]

    failed = route(model="failed-key-test", quota_pool_id="failed-key-q")
    selected = route(model="selected-key-test", quota_pool_id="selected-key-q")
    plan_mapping: dict[object, object] = {}
    plan_mapping.update(asdict(valid_plan("decision-key-test", failed, selected)))
    plan_mapping[7] = "unexpected"
    with pytest.raises(DomainValidationError, match="quality.schema_invalid"):
        QualityCompensationPlanV1.from_mapping(plan_mapping)  # type: ignore[arg-type]

def test_task_is_closed_validated_and_deeply_immutable():
    justification = {
        "policy_version": "policy/v1",
        "reason": "approved exception",
        "evidence_refs": ["evidence:approval"],
        "author": "policy-owner",
        "expires_at": "2099-08-01T00:00:00Z",
    }
    source = task_payload(audited_model_justification=justification)
    deliverables = source["deliverables"]
    task = TaskEnvelope.from_mapping(source)

    assert task.deliverables == ("module",)
    assert task.capabilities_required == ("filesystem.write",)
    assert task.tools_allowed == ("patch",)
    assert task.permissions_required == ("repository.write",)
    assert task.context.max_tokens == 900
    assert task.context.token_count == 12
    assert task.schema_version == "task-envelope/v1"
    assert task.privacy.outbound_allowed is False
    assert task.risk.reversibility == "reversible"
    assert task.budget.hard_cap == 0
    assert task.verification.minimum == "V2"
    assert task.audited_model_justification is not None
    assert task.audited_model_justification.evidence_refs == ("evidence:approval",)
    assert isinstance(deliverables, list)
    deliverables.append("mutated")
    justification_evidence = justification["evidence_refs"]
    assert isinstance(justification_evidence, list)
    justification_evidence.append("mutated")
    assert task.deliverables == ("module",)
    assert task.audited_model_justification.evidence_refs == ("evidence:approval",)
    with pytest.raises(FrozenInstanceError):
        task.task_id = "changed"  # type: ignore[misc]

@pytest.mark.parametrize(
    ("override", "code"),
    [
        (
            {
                "context": {
                    "classification": "internal",
                    "max_tokens": 0,
                    "token_count": 0,
                    "allowed_sources": [],
                }
            },
            "task.context_bounds_invalid",
        ),
        (
            {
                "context": {
                    "classification": "internal",
                    "max_tokens": 10,
                    "token_count": 11,
                    "allowed_sources": [],
                }
            },
            "task.context_bounds_invalid",
        ),
        ({"deliverables": ["ok", 1]}, "task.collection_invalid"),
        ({"unexpected": "value"}, "task.unexpected_field"),
    ],
)
def test_task_rejects_invalid_scalars_bounds_collections_and_unknown_fields(
    override: dict[str, object], code: str
):
    with pytest.raises(DomainValidationError, match=code):
        TaskEnvelope.from_mapping(task_payload(**override))

def test_task_rejects_unaudited_model_identity_but_valid_token_fields_are_not_sensitive():
    with pytest.raises(DomainValidationError, match="task.unaudited_model_identity"):
        TaskEnvelope.from_mapping(task_payload(model="gpt"))

    task = TaskEnvelope.from_mapping(task_payload(objective="verify token_count and max_tokens bounds"))
    assert task.task_id == "task-1"
    assert not hasattr(task, "model")

def test_task_unknown_policy_identifiers_and_secret_bearing_objective_fail_closed():
    for field_name in (
        "capabilities_required",
        "tools_allowed",
        "permissions_required",
    ):
        with pytest.raises(DomainValidationError, match="task.unknown_policy_identifier"):
            TaskEnvelope.from_mapping(
                task_payload(**{field_name: ["totally.unknown.policy.identifier"]})
            )
    for leaked_objective in (
        "password=hunter2",
        "token=private-token",
        "-----BEGIN PRIVATE KEY----- material",
    ):
        with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
            TaskEnvelope.from_mapping(task_payload(objective=leaked_objective))

    private_browser = TaskEnvelope.from_mapping(
        task_payload(
            capabilities_required=["browser.private"],
            tools_allowed=["computer-use"],
            permissions_required=["browser.private"],
            privacy={
                "classification": "private",
                "outbound_allowed": False,
                "retention": "ephemeral",
            },
            budget={
                "currency": "USD",
                "paid_allowed": True,
                "soft_cap": 10,
                "hard_cap": 20,
            },
            verification={
                "minimum": "V3",
                "independent_required": True,
                "human_gate_required": False,
            },
        )
    )
    assert private_browser.capabilities_required == ("browser.private",)

def test_classification_source_is_secret_filtered():
    attempted = route()
    for source in ("password=hunter2", "token=private-token", "-----BEGIN PRIVATE KEY-----"):
        with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
            RuntimeErrorClassificationV1(
                kind=ErrorKind.CAPACITY_EXHAUSTED,
                source=source,
                attempted_route_id=attempted.route_id,
                quota_pool_id=attempted.quota_pool_id,
                classified_at="2026-07-26T17:00:00Z",
            )

def test_task_accepts_audited_identity_without_retaining_model_fields():
    payload = task_payload(
        provider="OpenAI",
        model="GPT-5",
        audited_model_justification={
            "policy_version": "policy/v1",
            "reason": "explicit migration exception",
            "evidence_refs": ["evidence:approval-7"],
            "author": "policy-owner",
            "expires_at": "2099-08-01T00:00:00Z",
            "identity_claims": [
                ["model", "GPT-5"],
                ["provider", "OpenAI"],
            ],
        },
    )

    task = TaskEnvelope.from_mapping(payload)

    assert not hasattr(task, "provider")
    assert not hasattr(task, "model")
    assert task.audited_model_justification is not None
    assert task.audited_model_justification.identity_claims == (
        ("model", "GPT-5"),
        ("provider", "OpenAI"),
    )

    for policy_version, expires_at, expected_error in (
        ("policy/attacker", "2099-08-01T00:00:00Z", "task.justification_policy_mismatch"),
        ("policy/v1", "not-a-timestamp", "task.justification_expiry_invalid"),
        ("policy/v1", "2000-01-01T00:00:00Z", "task.justification_expired"),
    ):
        invalid = dict(payload)
        justification_source = payload["audited_model_justification"]
        assert isinstance(justification_source, dict)
        invalid_justification: dict[str, object] = dict(justification_source)
        invalid_justification["policy_version"] = policy_version
        invalid_justification["expires_at"] = expires_at
        invalid["audited_model_justification"] = invalid_justification
        with pytest.raises(DomainValidationError, match=expected_error):
            TaskEnvelope.from_mapping(invalid)

@pytest.mark.parametrize("gate", ELIGIBILITY_GATES)
@pytest.mark.parametrize(
    ("disposition", "suffix"),
    (
        (EligibilityDisposition.REJECT, "rejected"),
        (EligibilityDisposition.UNKNOWN, "unknown"),
    ),
)
def test_eligibility_emits_stable_fail_closed_code_for_every_gate(
    gate: str,
    disposition: EligibilityDisposition,
    suffix: str,
):
    candidate_route = route(model=f"{gate}-{suffix}", quota_pool_id=f"{gate}-{suffix}")

    (candidate,) = evaluate_route_eligibility(
        (eligibility_facts(candidate_route, **{gate: disposition}),)
    )

    assert candidate.route == candidate_route
    assert candidate.deterministic_status == "REJECTED"
    assert candidate.rejection_codes == (f"{gate}_{suffix}",)
    assert candidate.score is None
    assert candidate.score_factors == ()

def test_eligibility_records_multiple_failures_in_canonical_gate_order_and_unknown_fails_closed():
    candidate_route = route(model="multi-failure", quota_pool_id="multi-failure")
    facts = eligibility_facts(
        candidate_route,
        concurrency_reservation=EligibilityDisposition.REJECT,
        identity_policy=EligibilityDisposition.UNKNOWN,
        context=EligibilityDisposition.REJECT,
        budget=EligibilityDisposition.UNKNOWN,
    )

    (candidate,) = evaluate_route_eligibility((facts,))

    assert candidate.rejection_codes == (
        "identity_policy_unknown",
        "context_rejected",
        "budget_unknown",
        "concurrency_reservation_rejected",
    )
    assert candidate.eligible is False

def test_eligibility_is_route_sorted_permutation_invariant_and_isolated_from_mutable_input():
    routes = (
        route(model="permutation-a", quota_pool_id="permutation-a"),
        route(model="permutation-b", quota_pool_id="permutation-b"),
        route(model="permutation-c", quota_pool_id="permutation-c"),
    )
    source = [eligibility_facts(candidate_route) for candidate_route in routes]
    expected = evaluate_route_eligibility(source)

    for permutation in permutations(source):
        assert evaluate_route_eligibility(permutation) == expected
    assert tuple(candidate.route_id for candidate in expected) == tuple(
        sorted(candidate_route.route_id for candidate_route in routes)
    )

    source.reverse()
    assert evaluate_route_eligibility(source) == expected
    with pytest.raises(FrozenInstanceError):
        source[0].budget = EligibilityDisposition.REJECT  # type: ignore[misc]

def test_eligibility_rejects_duplicate_route_ids_and_untyped_dispositions():
    candidate_route = route(model="duplicate-facts", quota_pool_id="duplicate-facts")
    facts = eligibility_facts(candidate_route)
    with pytest.raises(DomainValidationError, match="eligibility.duplicate_route"):
        evaluate_route_eligibility((facts, facts))
    with pytest.raises(DomainValidationError, match="eligibility.invalid"):
        RouteEligibilityFactsV1(
            route=candidate_route,
            identity_policy="PASS",  # type: ignore[arg-type]
            privacy_permission=EligibilityDisposition.PASS,
            capability_tool=EligibilityDisposition.PASS,
            context=EligibilityDisposition.PASS,
            freshness_confidence=EligibilityDisposition.PASS,
            budget=EligibilityDisposition.PASS,
            breaker_cooldown=EligibilityDisposition.PASS,
            concurrency_reservation=EligibilityDisposition.PASS,
        )

def test_eligibility_stops_consuming_a_generic_iterable_at_the_candidate_bound():
    requested = 0

    def unbounded_facts():
        nonlocal requested
        while True:
            requested += 1
            if requested == 258:
                raise AssertionError("eligibility requested an item beyond the bound")
            yield object()

    with pytest.raises(DomainValidationError, match="eligibility.invalid"):
        evaluate_route_eligibility(unbounded_facts())  # type: ignore[arg-type]

    assert requested == 257

def test_scoring_requires_exact_validated_scores_and_never_scores_rejected_candidates():
    eligible_route = route(model="score-eligible", quota_pool_id="score-eligible")
    rejected_route = route(model="score-rejected", quota_pool_id="score-rejected")
    candidates = evaluate_route_eligibility(
        (
            eligibility_facts(rejected_route, budget=EligibilityDisposition.REJECT),
            eligibility_facts(eligible_route),
        )
    )
    factors = ["quality", "healthy_capacity"]
    score = CandidateScoreV1(score=7, score_factors=factors)

    scored = score_eligible_candidates(candidates, {eligible_route.route_id: score})
    assert score_eligible_candidates(
        tuple(reversed(candidates)),
        {eligible_route.route_id: score},
    ) == scored

    factors.append("mutated")
    assert tuple(candidate.route_id for candidate in scored) == tuple(
        sorted((eligible_route.route_id, rejected_route.route_id))
    )
    eligible = next(candidate for candidate in scored if candidate.route_id == eligible_route.route_id)
    rejected = next(candidate for candidate in scored if candidate.route_id == rejected_route.route_id)
    assert eligible.route == eligible_route
    assert eligible.score == 7.0
    assert eligible.score_factors == ("quality", "healthy_capacity")
    assert rejected.score is None
    assert rejected.score_factors == ()

    invalid_maps: tuple[dict[object, object], ...] = (
        {},
        {
            eligible_route.route_id: score,
            rejected_route.route_id: CandidateScoreV1(1, ("forbidden",)),
        },
        {rejected_route.route_id: CandidateScoreV1(1, ("forbidden",))},
        {eligible_route.route_id: 7.0},
        {7: score},
    )
    for invalid in invalid_maps:
        with pytest.raises(DomainValidationError, match="score.invalid"):
            score_eligible_candidates(candidates, invalid)  # type: ignore[arg-type]

def test_score_contract_rejects_malformed_or_nonfinite_values_and_duplicate_candidates():
    for invalid_score in (True, "7", float("nan"), float("inf"), float("-inf")):
        with pytest.raises(DomainValidationError, match="score.invalid"):
            CandidateScoreV1(invalid_score, ("quality",))  # type: ignore[arg-type]
    for invalid_factors in ((), ("",), ("quality", 1)):
        with pytest.raises(DomainValidationError, match="score.invalid"):
            CandidateScoreV1(1, invalid_factors)  # type: ignore[arg-type]

    candidate_route = route(model="duplicate-score", quota_pool_id="duplicate-score")
    candidate = evaluate_route_eligibility((eligibility_facts(candidate_route),))[0]
    with pytest.raises(DomainValidationError, match="score.duplicate_candidate"):
        score_eligible_candidates(
            (candidate, candidate),
            {candidate_route.route_id: CandidateScoreV1(1, ("quality",))},
        )

def test_scoring_bounds_candidates_and_score_registry_before_validation():
    class OversizedCandidates(list[object]):
        def __iter__(self):
            raise AssertionError("candidate validation must not start above the bound")

    class OversizedScores(dict[str, object]):
        def items(self):
            raise AssertionError("score validation must not start above the bound")

    oversized_candidates = OversizedCandidates([object()] * 257)
    with pytest.raises(DomainValidationError, match="score.invalid"):
        score_eligible_candidates(oversized_candidates, {})  # type: ignore[arg-type]

    candidate_route = route(model="bounded-score", quota_pool_id="bounded-score")
    candidate = CandidateEvaluation(candidate_route, True)
    oversized_scores = OversizedScores(
        {f"route-{index}": object() for index in range(513)}
    )
    with pytest.raises(DomainValidationError, match="score.invalid"):
        score_eligible_candidates(
            (candidate,),
            oversized_scores,  # type: ignore[arg-type]
        )

@pytest.mark.parametrize(
    "invalid_identifiers",
    (
        ("user prompt contains confidential merger terms",),
        ("AWS_SECRET_ACCESS_KEY=example-secret-value",),
        ("a" * 65,),
        tuple(f"factor_{index}" for index in range(33)),
        ("quality", "quality"),
    ),
)
def test_candidate_identifiers_reject_free_text_secrets_and_unbounded_collections(
    invalid_identifiers: tuple[str, ...],
):
    candidate_route = route(model="identifier-candidate", quota_pool_id="identifier-candidate")

    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(candidate_route, False, invalid_identifiers)
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(
            candidate_route,
            True,
            score=1,
            score_factors=invalid_identifiers,
        )

@pytest.mark.parametrize(
    "invalid_factors",
    (
        ("user prompt contains confidential merger terms",),
        ("AWS_SECRET_ACCESS_KEY=example-secret-value",),
        ("a" * 65,),
        tuple(f"factor_{index}" for index in range(33)),
        ("quality", "quality"),
    ),
)
def test_candidate_score_identifiers_reject_free_text_secrets_and_unbounded_collections(
    invalid_factors: tuple[str, ...],
):
    with pytest.raises(DomainValidationError, match="score.invalid"):
        CandidateScoreV1(1, invalid_factors)

def test_huge_integer_scores_are_normalized_to_domain_errors_in_all_candidate_paths():
    candidate_route = route(model="huge-score", quota_pool_id="huge-score")
    huge_score = 10**10000

    with pytest.raises(DomainValidationError, match="score.invalid"):
        CandidateScoreV1(huge_score, ("quality",))
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(
            candidate_route,
            True,
            score=huge_score,
            score_factors=("quality",),
        )
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation.from_mapping(
            {
                "route_id": candidate_route.route_id,
                "deterministic_status": "ELIGIBLE",
                "rejection_codes": (),
                "score": huge_score,
                "score_factors": ("quality",),
            },
            route_context=candidate_route,
        )

def test_eligibility_and_scoring_revalidate_forged_frozen_inputs_and_reject_sensitive_factors():
    forged_route = route(model="forged-route", quota_pool_id="forged-route")
    forged_facts = eligibility_facts(forged_route)
    object.__setattr__(forged_facts.route, "endpoint", "not-an-absolute-url")
    with pytest.raises(DomainValidationError, match="route.endpoint_invalid"):
        evaluate_route_eligibility((forged_facts,))

    eligible_route = route(model="forged-score", quota_pool_id="forged-score")
    candidate = evaluate_route_eligibility((eligibility_facts(eligible_route),))[0]
    forged_score = CandidateScoreV1(1, ("quality",))
    object.__setattr__(forged_score, "score", float("nan"))
    with pytest.raises(DomainValidationError, match="score.invalid"):
        score_eligible_candidates(
            (candidate,),
            {eligible_route.route_id: forged_score},
        )

    with pytest.raises(DomainValidationError, match="score.invalid"):
        CandidateScoreV1(1, ("Bearer do-not-persist-this-value",))

def test_scored_and_rejected_evaluations_remain_persisted_contract_compatible():
    eligible_route = route(model="persisted-eligible", quota_pool_id="persisted-eligible")
    rejected_route = route(model="persisted-rejected", quota_pool_id="persisted-rejected")
    scored = score_eligible_candidates(
        evaluate_route_eligibility(
            (
                eligibility_facts(eligible_route),
                eligibility_facts(
                    rejected_route,
                    freshness_confidence=EligibilityDisposition.UNKNOWN,
                ),
            )
        ),
        {eligible_route.route_id: CandidateScoreV1(3.5, ("quality",))},
    )

    restored = tuple(
        CandidateEvaluation.from_mapping(
            asdict(candidate),
            route_context=(eligible_route if candidate.route_id == eligible_route.route_id else rejected_route),
        )
        for candidate in scored
    )

    assert restored == scored
    assert tuple(candidate.route for candidate in restored) == tuple(
        candidate.route for candidate in scored
    )

def test_full_pure_domain_eligibility_score_fallback_and_sole_wait_flow():
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(task_id="task-t103-flow", objective="exercise pure routing flow")
    )
    attempted_opus = route(
        provider="anthropic",
        product="claude",
        model="claude-opus",
        account_id="anthropic-team-a",
        billing_pool_id="anthropic-team-a",
        quota_pool_id="anthropic-team-a",
    )
    gpt = route(
        provider="openai",
        product="api",
        model="gpt-5",
        account_id="openai-team-a",
        billing_pool_id="openai-team-a",
        quota_pool_id="openai-team-a",
    )
    policy_rejected = route(
        provider="zai",
        product="api",
        model="glm",
        account_id="zai-team-a",
        billing_pool_id="zai-team-a",
        quota_pool_id="zai-team-a",
    )
    facts = (
        eligibility_facts(
            attempted_opus,
            freshness_confidence=EligibilityDisposition.REJECT,
            concurrency_reservation=EligibilityDisposition.REJECT,
        ),
        eligibility_facts(gpt),
        eligibility_facts(
            policy_rejected,
            identity_policy=EligibilityDisposition.REJECT,
        ),
    )
    evaluated = evaluate_route_eligibility(reversed(facts))
    scored = score_eligible_candidates(
        evaluated,
        {gpt.route_id: CandidateScoreV1(100, ("quality", "capacity"))},
    )
    plan = valid_plan(
        "decision-t103-fallback",
        attempted_opus,
        gpt,
        reviewed_execution_id="attempt-t103-flow",
    )

    fallback = replan(
        trusted_task=trusted_task,
        task_id=trusted_task.task_id,
        attempt_id="attempt-t103-flow",
        decision_id="decision-t103-fallback",
        failed_route=attempted_opus,
        classification=classification(attempted_opus),
        candidates=scored,
        quality_compensation_plan=plan,
        task_verification_minimum="V2",
    )

    assert not hasattr(trusted_task, "model")
    assert fallback.relation is DecisionRelation.FALLBACK
    assert fallback.selected_route_id == gpt.route_id
    assert fallback.reason_codes == (
        "route_capacity_exhausted",
        "quality_compensation_insufficient",
    )
    assert fallback.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert fallback.activation_block_reason == "quality_compensation_insufficient"
    assert fallback.dispatchable is False
    assert fallback.recheck_evidence == ()
    assert next(
        candidate for candidate in fallback.candidates if candidate.route_id == attempted_opus.route_id
    ).rejection_codes[:2] == (
        "freshness_confidence_rejected",
        "concurrency_reservation_rejected",
    )

    no_capacity = evaluate_route_eligibility(
        (
            facts[0],
            eligibility_facts(
                gpt,
                breaker_cooldown=EligibilityDisposition.REJECT,
            ),
            facts[2],
        )
    )
    rejected_only = score_eligible_candidates(no_capacity, {})
    evidence = ("breaker:openai-team-a:cooldown-until-2026-07-26T17:05:00Z",)
    waiting = replan(
        trusted_task=trusted_task,
        task_id=trusted_task.task_id,
        attempt_id="attempt-t103-flow",
        decision_id="decision-t103-wait",
        failed_route=attempted_opus,
        classification=classification(attempted_opus),
        candidates=rejected_only,
        recheck_evidence=evidence,
        task_verification_minimum="V2",
    )

    assert waiting.relation is DecisionRelation.WAITING
    assert waiting.policy_status == "WAITING_FOR_CAPACITY"
    assert waiting.selected_route_id is None
    assert waiting.dispatchable is False
    assert waiting.recheck_evidence == evidence
    assert waiting.recheck_evidence[0]

def test_task_justification_expiry_uses_explicit_reference_time():
    payload = task_payload(
        model="audited-model",
        audited_model_justification={
            "policy_version": "policy/v1",
            "reason": "time-bounded audited exception",
            "evidence_refs": ("evidence:approval",),
            "author": "policy-owner",
            "expires_at": "2026-07-27T00:00:00Z",
        },
    )
    accepted = TaskEnvelope.from_mapping(
        payload,
        reference_time=datetime(2026, 7, 26, 23, 59, tzinfo=timezone.utc),
    )
    assert accepted.audited_model_justification is not None

    with pytest.raises(DomainValidationError, match="task.justification_expired"):
        TaskEnvelope.from_mapping(
            payload,
            reference_time=datetime(2026, 7, 27, 0, 1, tzinfo=timezone.utc),
        )

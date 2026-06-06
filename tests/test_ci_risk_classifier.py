from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci_risk_classifier.py"
spec = importlib.util.spec_from_file_location("ci_risk_classifier", MODULE_PATH)
assert spec is not None
ci_risk_classifier = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ci_risk_classifier
spec.loader.exec_module(ci_risk_classifier)


def body(**overrides: str) -> str:
    fields = {
        "Risk class": "R0",
        "Complexity class": "C0",
        "Impacted surfaces": "docs_only",
        "RuntimePayloadContract present": "no",
        "protected_surface": "false",
        "runtime_authority_change": "false",
        "customer_data_or_finance_impact": "false",
        "governance_or_merge_authority_change": "false",
        "model_tier_required": "0",
        "cc_review_required": "false",
        "opposite_frontier_required": "false",
        "escalation_reason": "none",
        "Blocker exemption, if any": "N/A",
        "Secondary review required": "no",
        "Adversarial review required": "no",
        "Opposite-provider adversarial required": "no",
        "Human/protected review required": "no",
        "Founder review required": "no",
        "Required CI lanes": "pr_body_contract, diff_check, docs_impact",
        "Skipped CI lanes and rationale": "runtime lanes skipped for docs-only change",
        "Token class": "S",
        "Expected state change": "PR body/docs repair",
        "Stop condition": "classification emitted",
    }
    fields.update(overrides)
    lines = ["## Risk, Complexity, Review, and CI Classification", ""]
    lines.extend(f"- {key}: {value}" for key, value in fields.items())
    return "\n".join(lines) + "\n"


def runtime_payload_contract(**overrides: str) -> str:
    fields = {
        "user_or_operator_visible_outcome": "Operators see deterministic CI classification artifacts.",
        "runtime_surface_touched": "Classifier script used by the PR workflow.",
        "product_or_platform_capability_advanced": "Risk-appropriate CI and review routing foundation.",
        "why_this_is_not_only_docs_or_scaffolding": "The classifier executes and blocks weaker PR bodies.",
        "tests_that_prove_runtime_behavior": "scripts/run_tests.sh tests/test_ci_risk_classifier.py",
        "acceptance_gate": "Classifier tests and live workflow pass.",
        "rollback": "Revert the classifier workflow and script changes.",
        "protected_non_claims": "No production, launch, customer-data, or protected approval readiness is claimed.",
    }
    fields.update(overrides)
    lines = ["", "## RuntimePayloadContract", ""]
    lines.extend(f"- {key}: {value}" for key, value in fields.items())
    return "\n".join(lines) + "\n"


def test_docs_only_pr_is_low_risk_and_ready() -> None:
    result = ci_risk_classifier.classify(
        ["docs/usage.md"],
        body(),
        additions=12,
        pr_number="1",
        repo="neoengine/hermes-agent",
        title="docs: update usage",
    )

    assert result.risk_class == "R0"
    assert result.complexity_class == "C0"
    assert result.token_class == "S"
    assert result.allowed_to_mark_ready is True
    assert result.required_reviews == ["no_secondary_review_required"]


def test_runtime_change_without_contract_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
            }
        ),
        additions=80,
    )

    assert result.risk_class == "R2"
    assert result.secondary_review_required is True
    assert result.adversarial_review_required is True
    assert "runtime_surface_without_runtimePayloadContract" in result.merge_blocking_conditions
    assert result.allowed_to_mark_ready is False


def test_protected_surface_requires_human_and_non_claims() -> None:
    result = ci_risk_classifier.classify(
        ["gateway/auth/token_vault.py"],
        body(
            **{
                "Risk class": "R4",
                "Complexity class": "C3",
                "Impacted surfaces": "secrets_tokens, auth_identity",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build, governance_required, security_required, protected_claim_gate, human_gate_required, privacy_data_gate, rollback_proof, audit_log_validation, auth_tests, session_isolation_tests, secret_scan, token_storage_tests, encryption_tests, protected_human_review",
            }
        )
        + runtime_payload_contract(protected_non_claims=""),
        additions=120,
    )

    assert result.risk_class == "R4"
    assert result.human_gate_required is True
    assert result.opposite_provider_required is True
    assert "protected_surface_without_protected_non_claims" in result.merge_blocking_conditions



def test_runtime_contract_yes_without_any_contract_block_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
            }
        ),
        additions=80,
    )

    assert result.runtime_payload_contract_present is False
    assert "runtime_surface_without_runtimePayloadContract" in result.merge_blocking_conditions
    assert result.allowed_to_mark_ready is False


def test_runtime_contract_yes_without_contract_block_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
            }
        )
        + "\nThis prose mentions runtimePayloadContract without providing a contract block.\n",
        additions=80,
    )

    assert result.runtime_payload_contract_present is False
    assert "runtime_surface_without_runtimePayloadContract" in result.merge_blocking_conditions
    assert result.allowed_to_mark_ready is False


def test_runtime_contract_block_missing_tests_field_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
            }
        )
        + runtime_payload_contract(tests_that_prove_runtime_behavior=""),
        additions=80,
    )

    assert result.runtime_payload_contract_present is False
    assert "runtimePayloadContract_missing_required_fields:tests_that_prove_runtime_behavior" in result.merge_blocking_conditions
    assert result.allowed_to_mark_ready is False


def test_runtime_contract_block_with_blank_rollback_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
            }
        )
        + runtime_payload_contract(rollback=""),
        additions=80,
    )

    assert result.runtime_payload_contract_present is False
    assert "runtimePayloadContract_missing_required_fields:rollback" in result.merge_blocking_conditions
    assert result.allowed_to_mark_ready is False


def test_runtime_contract_full_block_allows_ready() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
                "model_tier_required": "2",
                "escalation_reason": "standard_bounded_engineering",
            }
        )
        + runtime_payload_contract(),
        additions=80,
    )

    assert result.runtime_payload_contract_present is True
    assert result.merge_blocking_conditions == []
    assert result.allowed_to_mark_ready is True

def test_declared_lanes_weaker_than_classifier_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        [".github/workflows/tests.yml"],
        body(
            **{
                "Risk class": "R3",
                "Complexity class": "C2",
                "Impacted surfaces": "ci_workflow",
                "Required CI lanes": "pr_body_contract, diff_check",
            }
        ),
        additions=20,
    )

    assert result.risk_class == "R3"
    assert any(item.startswith("declared_ci_lanes_weaker_than_classifier:") for item in result.merge_blocking_conditions)
    assert "workflow_lint" in result.required_ci_lanes
    assert "pr_impact_classifier_tests" in result.required_ci_lanes


def test_missing_classification_section_blocks_ready() -> None:
    result = ci_risk_classifier.classify(["docs/usage.md"], "plain body without required section", additions=3)

    assert result.allowed_to_mark_ready is False
    assert "missing_required_pr_body_classification_section" in result.merge_blocking_conditions
    assert "missing_or_invalid_declared_risk_class" in result.merge_blocking_conditions
    assert "missing_or_invalid_declared_complexity_class" in result.merge_blocking_conditions


def test_invalid_risk_and_complexity_declarations_block_ready() -> None:
    result = ci_risk_classifier.classify(
        ["docs/usage.md"],
        body(**{"Risk class": "medium", "Complexity class": "large"}),
        additions=3,
    )

    assert result.allowed_to_mark_ready is False
    assert "missing_or_invalid_declared_risk_class" in result.merge_blocking_conditions
    assert "missing_or_invalid_declared_complexity_class" in result.merge_blocking_conditions


def test_missing_required_ci_lanes_blocks_ready() -> None:
    result = ci_risk_classifier.classify(
        ["docs/usage.md"],
        body(**{"Required CI lanes": "N/A"}),
        additions=3,
    )

    assert result.allowed_to_mark_ready is False
    assert "missing_required_ci_lanes" in result.merge_blocking_conditions


def test_readiness_split_and_downstream_matrix_are_dry_run_only() -> None:
    result = ci_risk_classifier.classify(
        ["agent/system_prompt.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C2",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
                "model_tier_required": "2",
                "escalation_reason": "standard_bounded_engineering",
            }
        )
        + runtime_payload_contract(),
        additions=80,
        pr_number="8",
        repo="neoengine-ai-org/hermes-agent",
    )

    data = result.as_dict()
    matrix = ci_risk_classifier.downstream_matrix(data)

    assert data["body_and_classification_ready"] is True
    assert data["review_ready"] is True
    assert data["merge_ready"] is True
    assert matrix["enforced"] is False
    assert matrix["readiness"] == {
        "body_and_classification_ready": True,
        "review_ready": True,
        "merge_ready": True,
    }
    assert {entry["lane"] for entry in matrix["include"]} >= {"pr_body_contract", "diff_check"}
    assert all(entry["dry_run_only"] is True for entry in matrix["include"])
    assert "not_downstream_ci_matrix_enforced" in matrix["non_claims"]
    assert "not_cross_repo_propagated" in matrix["non_claims"]
    assert "not_support_or_marketing_classifier_implemented" in matrix["non_claims"]


def test_pull_request_template_contains_classifier_policy_and_non_claim_blocks() -> None:
    template = (Path(__file__).resolve().parents[1] / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    assert "## Risk, Complexity, Review, and CI Classification" in template
    assert "- Risk class:" in template
    assert "- Complexity class:" in template
    assert "- Required CI lanes:" in template
    assert "## Policy decision output" in template
    assert "## Non-claims" in template
    assert "does not claim production readiness" in template
    assert "does not claim launch readiness" in template
    assert "does not bypass branch protection" in template



def test_cc_review_tier3_uses_codex_without_opposite_frontier_for_complex_bounded_engineering() -> None:
    result = ci_risk_classifier.classify(
        [
            "agent/context_engine.py",
            "agent/prompt_builder.py",
            "agent/tool_executor.py",
            "tests/agent/test_prompt_builder.py",
            "tests/agent/test_tool_result_classification.py",
            "docs/runtime-review.md",
        ],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C3",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
                "model_tier_required": "3",
                "cc_review_required": "true",
                "opposite_frontier_required": "false",
                "escalation_reason": "bounded_complex_engineering",
            }
        )
        + runtime_payload_contract(),
        additions=900,
    )

    assert result.model_tier_required == 3
    assert result.model_reviewer == "codex_engineering_review"
    assert result.cc_review_required is True
    assert result.opposite_frontier_required is False
    assert result.opposite_provider_required is False
    assert "codex_engineering_review_required" in result.required_reviews
    assert "opposite_provider_adversarial_required" not in result.required_reviews


def test_protected_surface_routes_tier4_opposite_frontier_authority_review() -> None:
    result = ci_risk_classifier.classify(
        ["gateway/auth/token_vault.py"],
        body(
            **{
                "Risk class": "R4",
                "Complexity class": "C3",
                "Impacted surfaces": "secrets_tokens, auth_identity",
                "RuntimePayloadContract present": "yes",
                "protected_surface": "true",
                "runtime_authority_change": "true",
                "governance_or_merge_authority_change": "true",
                "model_tier_required": "4",
                "cc_review_required": "true",
                "opposite_frontier_required": "true",
                "escalation_reason": "protected_surface",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build, governance_required, security_required, protected_claim_gate, human_gate_required, privacy_data_gate, rollback_proof, audit_log_validation, auth_tests, session_isolation_tests, secret_scan, token_storage_tests, encryption_tests, protected_human_review, conductor_tests, executive_gate_tests, permissions_tests, rollback_tests, storage_conformance_tests",
            }
        )
        + runtime_payload_contract(),
        additions=120,
    )

    assert result.protected_surface is True
    assert result.model_tier_required == 4
    assert result.model_reviewer == "opposite_frontier_cc_review"
    assert result.cc_review_required is True
    assert result.opposite_frontier_required is True
    assert result.opposite_provider_required is True
    assert "opposite_frontier_cc_review_required" in result.required_reviews


def test_docs_only_review_policy_is_not_tier0_mechanical() -> None:
    result = ci_risk_classifier.classify(
        ["docs/governance/review-routing-policy.md"],
        body(
            **{
                "Risk class": "R3",
                "Complexity class": "C1",
                "Impacted surfaces": "governance_review_policy",
                "governance_or_merge_authority_change": "true",
                "model_tier_required": "4",
                "cc_review_required": "true",
                "opposite_frontier_required": "true",
                "escalation_reason": "governance_or_merge_authority_change",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, governance_required, review_policy_guard",
            }
        ),
        additions=30,
    )

    assert result.governance_or_merge_authority_change is True
    assert result.model_tier_required == 4
    assert result.cc_review_required is True
    assert result.opposite_frontier_required is True


def test_missing_routing_decision_fields_block_ready() -> None:
    incomplete_body = "\n".join(
        [
            "## Risk, Complexity, Review, and CI Classification",
            "",
            "- Risk class: R0",
            "- Complexity class: C0",
            "- Required CI lanes: pr_body_contract, diff_check, docs_impact",
        ]
    )

    result = ci_risk_classifier.classify(["docs/usage.md"], incomplete_body, additions=5)

    assert "missing_or_invalid_declared_model_tier_required" in result.merge_blocking_conditions
    assert "missing_or_invalid_declared_opposite_frontier_required" in result.merge_blocking_conditions
    assert "missing_or_invalid_declared_protected_surface" in result.merge_blocking_conditions
    assert "missing_or_invalid_declared_cc_review_required" in result.merge_blocking_conditions


def test_tier3_requires_declared_cc_review_true() -> None:
    result = ci_risk_classifier.classify(
        ["agent/context_engine.py", "agent/prompt_builder.py", "agent/tool_executor.py"],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C3",
                "Impacted surfaces": "runtime_backend",
                "RuntimePayloadContract present": "yes",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
                "model_tier_required": "3",
                "cc_review_required": "false",
                "opposite_frontier_required": "false",
                "escalation_reason": "bounded_complex_engineering",
            }
        )
        + runtime_payload_contract(),
        additions=900,
    )

    assert result.cc_review_required is True
    assert "declared_cc_review_required_weaker_than_classifier" in result.merge_blocking_conditions


def test_tier4_rejects_none_escalation_reason_as_weaker_than_classifier() -> None:
    result = ci_risk_classifier.classify(
        ["gateway/auth/token_vault.py"],
        body(
            **{
                "Risk class": "R4",
                "Complexity class": "C3",
                "Impacted surfaces": "secrets_tokens, auth_identity",
                "RuntimePayloadContract present": "yes",
                "protected_surface": "true",
                "runtime_authority_change": "true",
                "governance_or_merge_authority_change": "true",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build, governance_required, security_required, protected_claim_gate, human_gate_required, privacy_data_gate, rollback_proof, audit_log_validation, auth_tests, session_isolation_tests, secret_scan, token_storage_tests, encryption_tests, protected_human_review, conductor_tests, executive_gate_tests, permissions_tests, rollback_tests, storage_conformance_tests",
                "model_tier_required": "4",
                "cc_review_required": "true",
                "opposite_frontier_required": "true",
                "escalation_reason": "none",
            }
        )
        + runtime_payload_contract(),
        additions=120,
    )

    assert result.escalation_reason != "mechanical_no_semantic_risk"
    assert "declared_escalation_reason_weaker_than_classifier" in result.merge_blocking_conditions


def test_ci_repair_without_merge_authority_routes_tier3_codex_not_opposite_frontier() -> None:
    result = ci_risk_classifier.classify(
        [
            ".github/workflows/tests.yml",
            "scripts/ci/repair_flaky_test_selection.py",
            "tests/ci/test_repair_flaky_test_selection.py",
        ],
        body(
            **{
                "Risk class": "R3",
                "Complexity class": "C3",
                "Impacted surfaces": "ci_workflow",
                "model_tier_required": "3",
                "cc_review_required": "true",
                "opposite_frontier_required": "false",
                "escalation_reason": "bounded_complex_engineering",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, workflow_lint, pr_impact_classifier_tests, governance_required",
            }
        ),
        additions=500,
    )

    assert result.model_tier_required == 3
    assert result.model_reviewer == "codex_engineering_review"
    assert result.cc_review_required is True
    assert result.opposite_frontier_required is False
    assert result.opposite_provider_required is False


def test_classifier_self_change_routes_tier4_opposite_frontier() -> None:
    result = ci_risk_classifier.classify(
        ["scripts/ci_risk_classifier.py", "tests/test_ci_risk_classifier.py"],
        body(
            **{
                "Risk class": "R3",
                "Complexity class": "C2",
                "Impacted surfaces": "classifier_change, ci_workflow",
                "governance_or_merge_authority_change": "true",
                "model_tier_required": "4",
                "cc_review_required": "true",
                "opposite_frontier_required": "true",
                "escalation_reason": "classifier_change",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, workflow_lint, pr_impact_classifier_tests, governance_required, review_policy_guard",
            }
        ),
        additions=80,
    )

    assert "classifier_change" in result.impacted_surfaces
    assert result.governance_or_merge_authority_change is True
    assert result.model_tier_required == 4
    assert result.model_reviewer == "opposite_frontier_cc_review"
    assert result.opposite_frontier_required is True


def test_c4_systemic_complexity_routes_tier4_even_without_extra_protected_flags() -> None:
    result = ci_risk_classifier.classify(
        [f"docs/architecture/module_{index}.md" for index in range(25)],
        body(
            **{
                "Risk class": "R2",
                "Complexity class": "C4",
                "model_tier_required": "4",
                "cc_review_required": "true",
                "opposite_frontier_required": "true",
                "escalation_reason": "complexity=C4",
                "Required CI lanes": "pr_body_contract, diff_check, docs_impact, typecheck, targeted_runtime_tests, backend_runtime, build",
            }
        ),
        additions=1800,
    )

    assert result.complexity_class == "C4"
    assert result.model_tier_required == 4
    assert result.opposite_frontier_required is True


def test_head_classifier_cannot_weaken_model_tier_or_frontier_requirements() -> None:
    base = {
        "risk_class": "R4",
        "complexity_class": "C3",
        "model_tier_required": 4,
        "protected_surface": True,
        "runtime_authority_change": True,
        "customer_data_or_finance_impact": True,
        "governance_or_merge_authority_change": True,
        "cc_review_required": True,
        "opposite_frontier_required": True,
        "required_ci_lanes": ["governance_required"],
        "required_reviews": ["opposite_frontier_cc_review_required"],
        "protected_flags": ["secrets_tokens"],
        "merge_blocking_conditions": [],
        "secondary_review_required": True,
        "adversarial_review_required": True,
        "opposite_provider_required": True,
        "human_gate_required": True,
        "founder_review_required": False,
        "allowed_to_mark_ready": False,
    }
    head = {
        **base,
        "risk_class": "R3",
        "model_tier_required": 3,
        "protected_surface": False,
        "runtime_authority_change": False,
        "customer_data_or_finance_impact": False,
        "governance_or_merge_authority_change": False,
        "cc_review_required": False,
        "opposite_frontier_required": False,
        "opposite_provider_required": False,
        "human_gate_required": False,
        "allowed_to_mark_ready": True,
    }

    weakenings = ci_risk_classifier.detect_head_classifier_weakenings(base, head)

    assert "lower_model_tier_required" in weakenings
    assert "missing_protected_surface" in weakenings
    assert "missing_runtime_authority_change" in weakenings
    assert "missing_customer_data_or_finance_impact" in weakenings
    assert "missing_governance_or_merge_authority_change" in weakenings
    assert "missing_cc_review_required" in weakenings
    assert "missing_opposite_frontier_required" in weakenings

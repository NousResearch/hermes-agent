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

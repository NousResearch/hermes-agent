from __future__ import annotations

import json

import pytest

from hermes_cli.kanban_budget import (
    TaskBudgetAssessment,
    assess_task,
    canonical_review_idempotency_key,
)


def test_narrow_bugfix_is_ok() -> None:
    assessment = assess_task(
        "Fix None handling in parser",
        "Add one regression test, patch parser.py, and run that test.",
        max_turns=60,
    )

    assert isinstance(assessment, TaskBudgetAssessment)
    assert assessment.verdict == "ok"
    assert assessment.score < assessment.split_threshold
    assert "implementation" in assessment.action_families
    assert "verification" in assessment.action_families


def test_compound_security_lifecycle_requires_split() -> None:
    body = """
    Inspect the current authority implementation and diagnose the bypass.
    Implement the remediation across installer.py and bootstrap.py.
    Run focused RED/GREEN tests, dual-runtime provisioning tests, Ruff,
    compilation, and the full project suite. Freeze hashes and create receipts.
    Launch one independent adversarial review over the immutable postimage.
    After PASS, run a separate controller and prepare the sudo gate without
    activating it.
    """

    assessment = assess_task("Close S3R7 remediation, review, and controller", body)

    assert assessment.verdict == "split"
    assert {
        "discovery",
        "implementation",
        "verification",
        "evidence",
        "review",
        "controller",
    }.issubset(set(assessment.action_families))
    assert "compound_lifecycle" in assessment.reasons
    assert assessment.suggested_shards == (
        "discovery",
        "implementation",
        "verification_and_evidence",
        "independent_review",
        "controller",
    )


def test_spanish_scope_markers_are_detected() -> None:
    assessment = assess_task(
        "Investigar, corregir, verificar y revisar el gate",
        """
        Investiga la causa, diseña la arquitectura, implementa la corrección,
        ejecuta pruebas focales y la matriz completa, congela hashes y recibos,
        realiza una revisión adversarial independiente y después un controller.
        """,
    )

    assert assessment.verdict == "split"
    assert "discovery" in assessment.action_families
    assert "review" in assessment.action_families
    assert "controller" in assessment.action_families


def test_many_acceptance_items_raise_scope_without_model_call() -> None:
    body = "\n".join(f"{n}. Acceptance criterion {n}" for n in range(1, 13))
    assessment = assess_task("Validate migration", body)

    assert assessment.acceptance_items >= 12
    assert "many_acceptance_items" in assessment.reasons
    assert assessment.verdict in {"caution", "split"}


def test_estimated_turns_are_compared_with_the_actual_budget() -> None:
    assessment = assess_task(
        "Implement and verify one parser correction",
        "Patch parser.py and run its focused pytest regression.",
        max_turns=8,
    )

    assert assessment.estimated_turns > assessment.max_turns
    assert assessment.verdict in {"caution", "split"}
    assert "estimated_turns_exceed_budget" in assessment.reasons


def test_four_distinct_lifecycle_families_always_require_split() -> None:
    assessment = assess_task(
        "Investigate, implement, verify, and independently review",
        "Inspect the bug, patch parser.py, run focused tests, then audit the postimage.",
        max_turns=120,
    )

    assert len(assessment.action_families) == 4
    assert assessment.verdict == "split"
    assert assessment.score >= assessment.split_threshold


@pytest.mark.parametrize(
    ("title", "body"),
    [
        (
            "Release version 12 to production",
            "Ship the approved build now.",
        ),
        (
            "Promote version 12 to production",
            "Move the approved build into the live environment.",
        ),
        (
            "Roll out the production release",
            "Make the new build live for all users now.",
        ),
        (
            "Poner la versión en producción",
            "Lanza la nueva versión para todos los usuarios.",
        ),
        (
            "Liberar la versión 12 en producción",
            "Promover el build aprobado al entorno vivo.",
        ),
        (
            "Submit the approved package",
            "Send it to the external registry now.",
        ),
        (
            "Publica la aplicación en producción ahora",
            "Hazla disponible para todos los usuarios externos.",
        ),
    ],
)
def test_activation_euphemisms_are_classified_fail_closed(title, body) -> None:
    assessment = assess_task(title, body)

    assert "activation" in assessment.action_families


def test_public_english_adjective_is_not_an_activation_verb() -> None:
    assessment = assess_task(
        "Document the public API",
        "Write local documentation for the public interface.",
    )

    assert "activation" not in assessment.action_families


def test_assessment_serialization_is_deterministic() -> None:
    first = assess_task("Implement and test", "Patch x.py and run pytest.")
    second = assess_task("Implement and test", "Patch x.py and run pytest.")

    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_review_key_maps_to_existing_idempotency_surface() -> None:
    digest = "a" * 64
    claims = "b" * 64
    assert (
        canonical_review_idempotency_key(f"  {digest.upper()}:{claims.upper()}  ")
        == f"review:v1:{digest}:{claims}"
    )
    assert canonical_review_idempotency_key(f"{digest}:{claims}") == (
        f"review:v1:{digest}:{claims}"
    )
    assert canonical_review_idempotency_key(f"{digest}:{claims}:round-2") == (
        f"review:v1:{digest}:{claims}:round-2"
    )


@pytest.mark.parametrize(
    "value",
    ["", "../escape", "contains spaces", "x" * 201, "review:already-prefixed"],
)
def test_review_key_rejects_ambiguous_or_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_review_idempotency_key(value)


def test_review_key_requires_postimage_and_claim_contract_digests() -> None:
    with pytest.raises(ValueError, match="postimage.*claims"):
        canonical_review_idempotency_key("a" * 64)

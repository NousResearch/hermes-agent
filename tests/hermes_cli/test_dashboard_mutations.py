from __future__ import annotations

import pytest

from hermes_cli.dashboard_mutations import (
    MUTATION_SPECS,
    MutationRequestError,
    MutationRisk,
    validate_mutation_request,
)


def test_risk_registry_covers_required_dashboard_mutations():
    assert {spec.risk for spec in MUTATION_SPECS.values()} == {
        MutationRisk.DESTRUCTIVE,
        MutationRisk.SECRET_SENSITIVE,
        MutationRisk.SERVICE_INTERRUPTING,
    }
    assert MUTATION_SPECS["gateway-restart"].confirmation == "RESTART"
    assert MUTATION_SPECS["hermes-update"].confirmation == "UPDATE"
    assert MUTATION_SPECS["session-delete-all"].confirmation == "DELETE ALL"


def test_rollout_accepts_legacy_empty_body():
    request = validate_mutation_request("gateway-restart", {})
    assert request.confirmation is None
    assert request.idempotency_key is None


def test_rollout_rejects_incorrect_confirmation_when_supplied():
    with pytest.raises(MutationRequestError, match="exactly match"):
        validate_mutation_request(
            "gateway-restart",
            {"confirmation": "restart", "idempotency_key": "request-00000001"},
        )


def test_rollout_rejects_non_object_body():
    with pytest.raises(MutationRequestError, match="body must be an object"):
        validate_mutation_request("gateway-restart", ["RESTART"])


def test_enforcement_requires_confirmation_and_idempotency_key():
    with pytest.raises(MutationRequestError, match="confirmation"):
        validate_mutation_request(
            "hermes-update",
            {"idempotency_key": "request-00000001"},
            require_confirmation=True,
        )
    with pytest.raises(MutationRequestError, match="idempotency_key"):
        validate_mutation_request(
            "hermes-update",
            {"confirmation": "UPDATE"},
            require_confirmation=True,
        )


def test_enforcement_accepts_exact_contract():
    request = validate_mutation_request(
        "hermes-update",
        {
            "confirmation": "UPDATE",
            "idempotency_key": "request-00000001",
        },
        require_confirmation=True,
    )
    assert request.confirmation == "UPDATE"
    assert request.idempotency_key == "request-00000001"

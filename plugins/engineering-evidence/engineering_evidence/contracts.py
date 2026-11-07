"""Role ownership and typed event names for the engineering evidence lane."""

from __future__ import annotations

from typing import Any


RECEIPT_OWNERS = {
    "codebase_snapshot": {
        "producer": "architecture-steward",
        "reviewer": "review-verification-steward",
    },
    "test_discovery": {
        "producer": "test-contract-steward",
        "failure_route": "ci-repair-steward",
    },
    "experience_candidate": {
        "producer": "memory-intake",
        "validator": "memory-validator",
    },
    "deployment_correlation": {
        "producer": "operations-steward",
        "reviewer": "release-steward",
    },
}

EVENT_NAMES = (
    "engineering_evidence.snapshot.created",
    "engineering_evidence.tests.recorded",
    "engineering_evidence.experience.proposed",
    "engineering_evidence.experience.validated",
    "engineering_evidence.deployment.correlated",
)


def workflow_contract() -> dict[str, Any]:
    """Return a defensive, read-only description for agent routing."""

    return {
        "schema_name": "engineering_evidence_v1",
        "authority": "diagnostic-only",
        "receipt_owners": RECEIPT_OWNERS,
        "event_names": list(EVENT_NAMES),
        "automatic_authority": [],
        "requires_exact_head": True,
    }

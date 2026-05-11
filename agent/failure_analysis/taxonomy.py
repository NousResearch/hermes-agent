"""Canonical failure taxonomy for Hermes.

Defines the valid (type, subtype) pairs and helpers for validation.
"""

from __future__ import annotations

from .types import FailureType, FailureSubtype

# Valid (type, subtype) pairs — the canonical taxonomy.
VALID_PAIRS: frozenset[tuple[str, str]] = frozenset({
    (FailureType.EVAL, FailureSubtype.REGRESSION),
    (FailureType.EVAL, FailureSubtype.FAILED_CHECK),
    (FailureType.TOOL, FailureSubtype.TIMEOUT),
    (FailureType.TOOL, FailureSubtype.EXECUTION),
    (FailureType.POLICY, FailureSubtype.APPROVAL_BLOCKED),
    (FailureType.INFRA, FailureSubtype.AUTH),
    (FailureType.INFRA, FailureSubtype.ENVIRONMENT),
    (FailureType.MODEL, FailureSubtype.OUTPUT_MALFORMED),
    (FailureType.UNKNOWN, FailureSubtype.UNKNOWN),
})


def is_valid_pair(failure_type: str, failure_subtype: str) -> bool:
    """Check whether (type, subtype) is a recognized taxonomy pair."""
    return (failure_type, failure_subtype) in VALID_PAIRS


def all_types() -> list[str]:
    """Return all valid top-level failure types."""
    return sorted({t for t, _ in VALID_PAIRS})


def subtypes_for(failure_type: str) -> list[str]:
    """Return valid subtypes for a given type."""
    return sorted({s for t, s in VALID_PAIRS if t == failure_type})

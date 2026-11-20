"""Typed exceptions for the Phase 2 durable authority contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class AuthorityError(RuntimeError):
    """A requested authority transition is invalid or unavailable."""


class AuthorityMigrationRequired(AuthorityError):
    """A durable store predates an integrity invariant its data now violates.

    Raised instead of creating a unique index that existing authoritative rows
    would violate. Authoritative events are never deleted or rewritten to make
    an index fit, so the store stays readable for inspection
    (:meth:`Phase2AuthorityStore.migration_report`,
    :meth:`Phase2AuthorityStore.read_events`) while every authority-serving path
    fails closed with this typed, actionable error.
    """

    def __init__(self, violations: list[dict[str, Any]], db_path: Path | str) -> None:
        self.violations = tuple(violations)
        self.db_path = str(db_path)
        summary = "; ".join(
            "{invariant}: {groups} violating group(s), sample {sample} ({remediation})".format(
                invariant=violation["invariant"],
                groups=violation["violating_groups"],
                sample=[
                    f"{item['key']} x{item['events']}" for item in violation["sample"]
                ],
                remediation=violation["remediation"],
            )
            for violation in self.violations
        )
        super().__init__(
            f"authority store {self.db_path} requires explicit migration before use: "
            f"{summary}. Authoritative events are never auto-deleted or rewritten; "
            "inspect with Phase2AuthorityStore(db_path).migration_report() and "
            "read_events(graph_id), then quarantine this database and re-seal, or "
            "resolve the listed rows out of band."
        )


class ResultRejection(AuthorityError):
    """A completion was rejected; exactly one RESULT_REJECTED event was appended.

    The rejection is recorded durably in the same decision transaction, so this
    exception carries the bounded machine-readable ``reason`` and the
    ``result_hash`` that was refused.
    """

    def __init__(self, reason: str, *, result_hash: str | None = None) -> None:
        super().__init__(f"result rejected: {reason}")
        self.reason = reason
        self.result_hash = result_hash


class MalformedEnvelopeFence(ResultRejection):
    """A completion presented a fence that is not a durably bindable integer.

    Contract v2 requires ``fence`` to be a positive ``int``. A caller-supplied
    ``"1"``, ``1.0``, ``True``, ``[1]``, or ``{}`` is not a fence: it can neither
    be compared against the authoritative fence nor bound to the INTEGER column
    without SQLite type affinity silently rewriting it into a value that no
    longer matches the integrity hash computed over it. An ``int`` outside the
    signed 64-bit range (``claimed_fence_reason == "out_of_range"``) is equally
    unusable — SQLite cannot bind it at all. Such a completion is rejected
    fail-closed, and the claim survives only as bounded audit metadata on the
    ``RESULT_REJECTED`` event (whose ``fence`` column is ``NULL``).

    Subclasses :class:`ResultRejection` so existing fail-closed handlers keep
    working unchanged.
    """

    def __init__(
        self,
        reason: str,
        *,
        result_hash: str | None = None,
        claimed_fence_reason: str | None = None,
        claimed_fence_type: str | None = None,
        claimed_fence_repr: str | None = None,
    ) -> None:
        super().__init__(reason, result_hash=result_hash)
        self.claimed_fence_reason = claimed_fence_reason
        self.claimed_fence_type = claimed_fence_type
        self.claimed_fence_repr = claimed_fence_repr

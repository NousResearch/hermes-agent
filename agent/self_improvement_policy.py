"""T3 PR97786 — Self-improvement Decision class + evaluate() factory.

A ``Decision`` is the canonical authority record for whether a turn (or sub-agent /
background-review fork) is allowed to mutate the filesystem / terminal state. The
``evaluate`` factory takes the SessionWritePolicy and any provenance signal and returns
a Decision; this is called ONCE during AIAgent init and retained on the agent — never
re-derived from the current environment at mutation time.

Invariants (C5, C7, C8, C9):

  * ``evaluate`` is the ONLY legitimate producer of a Decision instance for a session.
  * Provenance lookup failure => fail closed (Decision(allowed=False, reason="...")).
  * Mutation-time environment changes cannot elevate a Decision: tools/delegates read
    only the retained Decision on the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Decision:
    """Authorization for self-improvement activity on behalf of this turn/session.

    Attributes:
        allowed: Whether mutations are permitted.
        reason: Human-readable provenance label (logged for audit, never trusted).
        source: Identifier of the authority that produced this decision (e.g.
            "session_init", "background_review_fork", "subagent_delegate"). Set at
            construction to disambiguate the origin of the decision.
    """
    allowed: bool = False
    reason: str = ""
    source: str = "unspecified"

    def is_allowed(self) -> bool:
        return bool(self.allowed)


# Sentinel returned by ``get_self_improvement_decision()`` when no Decision has been
# bound (i.e. outside any explicit turn scope, or before retention completes).
# Consumers MUST treat this as deny and refuse mutations.
MISSING_DECISION: Decision = Decision(
    allowed=False,
    reason="no_retained_decision_outside_turn_scope",
    source="default_sentinel",
)


def allow(reason: str, source: str) -> Decision:
    """Construct an allow-decision with explicit reason and source."""
    return Decision(allowed=True, reason=reason, source=source)


def deny(reason: str, source: str) -> Decision:
    """Construct a deny-decision with explicit reason and source."""
    return Decision(allowed=False, reason=reason, source=source)


def evaluate(
    *,
    policy_protected: bool,
    provenance_lookup_ok: bool = True,
    source: str = "session_init",
) -> Decision:
    """Produce a single Decision for this AIAgent's lifetime.

    Rules (C5, C8, C9):
      - If ``policy_protected`` is True, the session is in protected mode and self-
        improvement is DENIED (fail-closed for protected evaluation).
      - If ``provenance_lookup_ok`` is False, the provenance lookup failed and the
        Decision is DENIED with explicit provenance-failure reason (fail-closed).
      - Otherwise, ALLOW with the given source label.

    This is called ONCE at AIAgent initialization; the result is retained on the
    agent. Subsequent mutations read ``agent.self_improvement_decision`` directly.
    """
    if not provenance_lookup_ok:
        return Decision(
            allowed=False,
            reason="provenance_lookup_failure_fail_closed",
            source=source,
        )
    if policy_protected:
        return Decision(
            allowed=False,
            reason="protected_session_self_improvement_disallowed",
            source=source,
        )
    return Decision(
        allowed=True,
        reason="unprotected_session_default_allow",
        source=source,
    )


def provenance_lookup() -> bool:
    """Canonical provenance lookup used during AIAgent initialization.

    Returns True when the provenance signal is valid; False when lookup fails.
    The default implementation is intentionally minimal: it returns True unless the
    environment explicitly sets ``HERMES_PROVENANCE_DISABLED=1``, in which case the
    Decision must be denied (C9: provenance lookup failure => fail closed).

    Subclasses or test fixtures may monkeypatch this symbol during AIAgent init to
    simulate specific provenance conditions without modifying the global env.
    """
    import os
    if os.environ.get("HERMES_PROVENANCE_DISABLED") == "1":
        return False
    return True


__all__ = [
    "Decision",
    "MISSING_DECISION",
    "allow",
    "deny",
    "evaluate",
    "provenance_lookup",
]

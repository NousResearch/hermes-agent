"""T3 PR97786 — Self-improvement Decision ContextVar scope.

Mirrors the SessionWritePolicy ContextVar contract: a single per-turn ``Decision`` is
retained on the AIAgent at initialization time and bound at the actual turn-execution
seam in ``agent/turn_facade.py``. Reading outside the bound scope returns the default
fallback (``MISSING_DECISION`` sentinel) which fails closed at every consumer.

This module is intentionally a thin re-export of the same ContextVar + token-reset scope
pattern used by ``session_write_policy_scope`` so that the two contracts can be reasoned
about identically.

Invariants (C5, C6, C7, C8, C9, C14, C15, C18):

  * ``Decision`` is evaluated ONCE from initialization authority (the agent retains it).
  * Mutation-time environment changes cannot become authority: only the retained value
    on ``agent.self_improvement_decision`` is read by tools/delegates/cron.
  * Missing / invalid retained Decision fails closed — every consumer raises explicitly
    rather than allowing an implicit normal/allow.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

from agent.self_improvement_policy import Decision, MISSING_DECISION


_default_decision: Decision = MISSING_DECISION

_self_improvement_decision: ContextVar[Decision] = ContextVar(
    "hermes_self_improvement_decision", default=_default_decision
)


def get_self_improvement_decision() -> Decision:
    """Return the bound per-turn Decision (or the ``MISSING_DECISION`` sentinel if no
    decision has been retained). Consumers must refuse mutations when this returns the
    sentinel — see ``Decision.is_allowed()``."""
    return _self_improvement_decision.get()


@contextmanager
def self_improvement_decision_scope(decision: Decision) -> Iterator[Decision]:
    """Bind ``decision`` as the active per-turn Decision. Restoration via ContextVar
    token reset in a guaranteed finally path (C11/C12/C13)."""
    if decision is None:
        # C8: refuse to bind None — must be an explicit ``MISSING_DECISION`` sentinel.
        raise ValueError("Decision: cannot bind None (fail-closed C8)")
    token: Token = _self_improvement_decision.set(decision)
    try:
        yield decision
    finally:
        _self_improvement_decision.reset(token)


__all__ = [
    "get_self_improvement_decision",
    "self_improvement_decision_scope",
]

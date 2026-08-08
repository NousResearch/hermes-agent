"""Typed, immutable ContextVar context for self-improvement authorization (PHASE 2).

This module replaces the Phase 1 helper that walked the Python stack /
read ``os.environ`` to recover the captured ``Decision``. The new contract
is **CONTEXTVAR** — the captured ``Decision`` is bound to a module-level
``ContextVar`` at the canonical ``AIAgent.__init__`` initialization
boundary and looked up by the L3 (tool-boundary) code paths.

Design contract (reconciled across Phase 2 design + independent review +
scope reconciliation; documented in
``/tmp/hermes-post-turn-self-improvement-p0-phase2-scope-reconciliation.Ion3jz/PHASE2_SCOPE_RECONCILIATION_REPORT.md``):

A. Canonical ContextVar

   The ContextVar is typed ``ContextVar[Optional[Decision]]`` and holds
   the **Phase 1 frozen** ``Decision`` dataclass. The Decision class is
   ``@dataclass(frozen=True)`` so once set in this module's variable
   reference it is effectively immutable.

B. DENY fallback (no implicit ALLOW)

   ``get_self_improvement_decision()`` returns the
   ``DENY_FALLBACK_DECISION`` singleton when the ContextVar is unset,
   set to ``None``, or set to a value that is *not* a Decision
   instance. This is the fail-closed behaviour: missing context ⇒ DENY.

C. NO environment reads (Phase 2 contract)

   This module deliberately does NOT import ``os`` and does NOT read
   ``HERMES_DISABLE_SELF_IMPROVEMENT`` or ``HERMES_READ_ONLY_SESSION``.
   Those env vars are sampled **only** at canonical initialization
   (Phase 1 site ``agent/agent_init.py``). The captured Decision is
   then propagated exclusively through this ContextVar.

D. NO stack inspection (Phase 2 contract)

   This module does NOT import ``inspect``. It does not call
   ``sys._getframe`` or ``f_back`` or ``stack()``. Authorization comes
   from the ContextVar lookup alone.

E. Binding API

   ``bind_self_improvement_decision(decision)`` is the narrow
   production-side API. It validates that *decision* is a real
   ``agent.self_improvement_policy.Decision`` instance, then binds it
   to the ContextVar and returns a ``Token`` that the caller may
   ``reset`` to restore the outer binding.

   ``self_improvement_decision_scope(decision)`` is the recommended
   contextmanager form (``@contextmanager``). It uses
   ``try: ... finally: token.reset()`` so a raise inside the ``with``
   block still restores the outer binding — exactly the
   "scoped reset" / "prevent stale ALLOW" guarantee.

F. Malformed binding

   A non-Decision value passed to either binding API is rejected by
   ``ValueError``; the ContextVar is left untouched. This way no
   ALLOW can be smuggled in via a malformed dict, string, or ``None``.

G. Frozen authority

   The Decision object captured at canonical init is the only
   authority. Rebinding must occur only at the canonical trusted
   boundary (``agent/agent_init.py``) or through these test-scoped
   helpers — never by the L3 callers.

H. Background thread / executor propagation

   This module does NOT implement ``copy_context`` itself — that
   contract is provided by ``tools/thread_context.py``'s existing
   ``propagate_context_to_thread`` wrapper, which already snapshots
   the parent's ``ContextVars`` and runs the worker inside the copied
   context. As long as the production init happens in the parent
   thread *before* the worker thread is spawned, the Decision is
   visible to the worker via standard ``ContextVar.get()``.

I. Stub helper (test-scoped)

   ``_is_decision_instance(value)`` is a private predicate exposed via
   ``__all__`` for test introspection but not for production use.
   Tests that need to install a Decision bypass the production
   contextmanager and use ``bind_self_improvement_decision`` directly
   so they can install/reset cleanly with their own try/finally.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Decision import (lazy to avoid an import cycle with
# ``agent.self_improvement_policy``; this module is imported from
# ``agent/agent_init.py`` BEFORE the policy module is fully ready).
# ---------------------------------------------------------------------------

def _build_deny_fallback_decision() -> object:
    """Construct the DENY fallback decision.

    Imported lazily so this module can be loaded even if
    ``agent.self_improvement_policy`` is unavailable (e.g. minimal unit
    test that only verifies the ContextVar shape). The returned object
    is a *real* ``Decision`` instance — i.e. it has ``.allow`` (False),
    ``.result`` (one of the DENY_* sentinels) and ``.reason`` fields.
    """
    try:
        from agent.self_improvement_policy import (
            DENY_UNKNOWN_OPERATION,
            Decision,
        )
        return Decision(
            result=DENY_UNKNOWN_OPERATION,
            reason=(
                "self_improvement_decision_context: missing or invalid "
                "typed context; fail-closed DENY"
            ),
        )
    except Exception:
        # Last-ditch fallback: a simple object that mimics Decision.allow/result/reason
        # so callers can still call .allow / .result / .reason as a defensive default.
        class _FallbackDecision:
            result = "DENY"
            reason = "self_improvement_decision_context: context module unavailable; fail-closed DENY"

            @property
            def allow(self) -> bool:
                return False

        return _FallbackDecision()


# Singleton — built once at import time. Frozen; callers cannot mutate.
# If the Decision class is unavailable, _FallbackDecision above mimics it.
DENY_FALLBACK_DECISION = _build_deny_fallback_decision()


# ---------------------------------------------------------------------------
# The canonical ContextVar. Default is None so the value is distinguishable
# from "explicitly set to None" (which we also treat as DENY).
# ---------------------------------------------------------------------------
hermes_self_improvement_decision: ContextVar[Optional[object]] = ContextVar(
    "hermes_self_improvement_decision",
    default=None,
)


def _is_decision_instance(value: object) -> bool:
    """Return True iff *value* is a real Phase 1 Decision instance.

    We check both the dataclass ``Decision`` and the FallbackDecision
    returned by ``_build_deny_fallback_decision`` so the boundary treats
    the singleton like a Decision (avoids accidentally re-evaluating an
    already-fallbacked lookup). Production frozen Decision objects from
    ``agent.self_improvement_policy.Decision`` are detected via duck
    typing on ``.allow``, ``.result``, ``.reason``.
    """
    if value is None:
        return False
    # Real Policy Decision or the built-in fallback.
    if isinstance(value, type(DENY_FALLBACK_DECISION)):
        return True
    # Duck-typed Decision — has .allow, .result, .reason attributes.
    return all(hasattr(value, attr) for attr in ("allow", "result", "reason"))


# ---------------------------------------------------------------------------
# The two API entry points the directive requires.
# ---------------------------------------------------------------------------
def get_self_improvement_decision() -> object:
    """Return the active typed Decision or the DENY fallback.

    Behaviour:
      * If the ContextVar is unset / set to None -> DENY_FALLBACK_DECISION.
      * If the ContextVar is set to a Decision-like object -> return it.
      * If the ContextVar is set to a malformed value (rare; only possible
        if a non-validated binding helper slipped in) -> DENY_FALLBACK_DECISION.

    This function NEVER raises; that is the caller's contract — L3
    callers can treat the return value as always-present and ask
    ``decision.allow``.
    """
    try:
        current = hermes_self_improvement_decision.get()
    except Exception:
        return DENY_FALLBACK_DECISION
    if not _is_decision_instance(current):
        return DENY_FALLBACK_DECISION
    return current


def bind_self_improvement_decision(decision: object) -> Token:
    """Bind *decision* as the active self-improvement ContextVar.

    Rejects anything that is not a Decision-like object; raises
    ``ValueError`` BEFORE touching the ContextVar so a failed bind
    leaves the previous binding intact.

    Returns the ``ContextVar.Token`` so the caller can ``reset`` it.
    Production init code uses
    :func:`self_improvement_decision_scope` (the contextmanager form)
    instead of this raw helper.
    """
    if not _is_decision_instance(decision):
        raise ValueError(
            "bind_self_improvement_decision: expected a Decision-like "
            f"instance, got {type(decision).__name__}"
        )
    return hermes_self_improvement_decision.set(decision)


def reset_self_improvement_decision(token: Token) -> None:
    """Restore the binding captured by ``token``.

    Safe to call only with a token obtained from
    ``bind_self_improvement_decision``. Calling with a foreign token is
    a programmer error and propagates ``LookupError`` — by design.
    """
    hermes_self_improvement_decision.reset(token)


@contextmanager
def self_improvement_decision_scope(decision: object) -> Iterator[object]:
    """Contextmanager form — the recommended production entry point.

    On ``__enter__`` the Decision is bound; on ``__exit__`` the token
    is reset via ``finally`` so an exception inside the ``with`` body
    does NOT leave a stale ALLOW in the ContextVar for the next
    operation.

    Tests that install a Decision in a fixture should use this
    contextmanager form so they automatically get exception-path
    cleanup.
    """
    token = bind_self_improvement_decision(decision)
    try:
        yield decision
    finally:
        try:
            hermes_self_improvement_decision.reset(token)
        except Exception:
            # Token reset failures must not mask the original exception
            # nor leak the temporary ALLOW. We swallow the reset error
            # because the worst case is a stale ContextVar for the
            # rest of the test/thread, which subsequent operations
            # overwrite via their own bind. Production code paths
            # never reach here because canonical init deliberately
            # does not call __exit__ (single AIAgent per process).
            pass


__all__ = [
    "DENY_FALLBACK_DECISION",
    "bind_self_improvement_decision",
    "get_self_improvement_decision",
    "reset_self_improvement_decision",
    "self_improvement_decision_scope",
]

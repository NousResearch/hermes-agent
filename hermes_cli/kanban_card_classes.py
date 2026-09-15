"""Runtime-gated isolation card classes.

Blind cards are deliberately kept separate from the task database: a label is not
an isolation boundary.  Construction and execution both require evidence that the
ceiling resolver is wired into the runtime path and is enforcing a ceiling.

Phase 2 is currently absent in this repository (see the isolation work verdict).
Consequently the default path fails closed; callers must not replace this guard
with an import-only feature check or a full-worktree fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class ResolverWiringError(RuntimeError):
    """Raised when a blind card cannot prove an enforced runtime ceiling."""


class CeilingResolver(Protocol):
    def verify_runtime_wiring(self) -> bool: ...

    def resolve_ceiling(
        self, *, role: str, card_class: str, requested_scope: Mapping[str, Any]
    ) -> Any: ...


@dataclass(frozen=True)
class BlindCard:
    """A single-blind card whose dev execution is ceiling-gated.

    ``resolver`` is intentionally an object, rather than a module name.  This
    makes the runtime dependency explicit and permits a real integration adapter
    to prove wiring before a worker is started.  The resolver's probe is run at
    construction and again immediately before execution to prevent a stale or
    replaced runtime dependency from becoming an unscoped spawn.
    """

    title: str
    requested_scope: Mapping[str, Any]
    resolver: CeilingResolver
    role: str = "junior-dev"
    card_class: str = "single_blind"

    def __post_init__(self) -> None:
        if self.card_class != "single_blind":
            raise ValueError("BlindCard requires card_class='single_blind'")
        _require_runtime_wiring(self.resolver)

    def execute(self, operation: Any) -> Any:
        """Resolve an enforced ceiling, then run ``operation(ceiling)``.

        No operation is invoked if the probe or resolution is unavailable,
        malformed, or reports that enforcement is not active.
        """
        _require_runtime_wiring(self.resolver)
        ceiling = _resolve_enforced_ceiling(self.resolver, self.role, self.card_class, self.requested_scope)
        if not callable(operation):
            raise TypeError("blind card operation must be callable")
        return operation(ceiling)


def _require_runtime_wiring(resolver: Any) -> None:
    """Require an affirmative live probe; importability alone is insufficient."""
    probe = getattr(resolver, "verify_runtime_wiring", None)
    resolve = getattr(resolver, "resolve_ceiling", None)
    if not callable(probe) or not callable(resolve):
        raise ResolverWiringError(
            "blind cards require an end-to-end ceiling resolver with a runtime wiring probe; "
            "no resolver is wired (importability is insufficient)"
        )
    try:
        wired = probe()
    except Exception as exc:
        raise ResolverWiringError(
            "blind-card ceiling resolver runtime wiring could not be verified"
        ) from exc
    if wired is not True:
        raise ResolverWiringError(
            "blind-card ceiling resolver is not enforced end-to-end; refusing unscoped execution"
        )


def _resolve_enforced_ceiling(
    resolver: Any, role: str, card_class: str, requested_scope: Mapping[str, Any]
) -> Any:
    try:
        ceiling = resolver.resolve_ceiling(
            role=role, card_class=card_class, requested_scope=requested_scope
        )
    except Exception as exc:
        raise ResolverWiringError(
            "blind-card ceiling resolution failed; refusing unscoped execution"
        ) from exc
    if ceiling is None:
        raise ResolverWiringError(
            "blind-card ceiling resolver returned no enforced ceiling; refusing unscoped execution"
        )
    # The adapter must attest that the returned scope is enforced by the runtime,
    # not merely calculated.  This prevents a cosmetic resolver implementation.
    if not bool(getattr(ceiling, "enforced", False)):
        raise ResolverWiringError(
            "blind-card ceiling was resolved without runtime enforcement; refusing execution"
        )
    return ceiling


__all__ = ["BlindCard", "CeilingResolver", "ResolverWiringError"]

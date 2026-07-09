"""Capability registry for the thin Memory Router.

The registry is the single source of truth for *what memory capabilities are
available*. The Router only ever routes to capabilities that are both
registered AND available. Optional layers (L2, L3, L4, L7, Holographic)
simply do not register when absent — so graceful degradation is automatic and
centralized, with no scattered availability checks elsewhere in the codebase.

The Router contains NO project-specific knowledge; it only knows how to find a
registered handler for an intent. Handlers are opaque callables obeying the
contract ``handler(method, **kwargs) -> result``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .intents import Intent


class Capability:
    """A registered memory capability the Router can dispatch to.

    Beyond routing (intent -> handler), a capability may declare that it
    participates in cross-layer *context* assembly. That participation is
    OPT-IN and reviewed: a capability contributes to ``context()`` ONLY if a
    human (or a reviewed product decision) sets ``contributes_to_context`` to
    True at registration time. Merely existing in the registry never implies
    context participation — this is the guardrail that stops "more backends"
    from silently changing what Hermes loads by default.
    """

    def __init__(
        self,
        name: str,
        intents: Iterable[Intent],
        available: bool,
        handler: Callable[..., Any],
        *,
        provider: Any = None,
        contributes_to_context: bool = False,
        context_category: str = "",
    ) -> None:
        self.name = name
        self.intents = list(intents)
        self.available = available
        self.handler = handler
        # Optional reference to the single backend instance that owns this
        # capability's storage. Lets the Router expose exactly ONE concrete
        # instance per capability without re-constructing it.
        self.provider = provider
        # Declarative context participation (Invariant B). Defaults to FALSE:
        # a newly registered capability is invisible to context() until a
        # human explicitly opts it in.
        self.contributes_to_context = contributes_to_context
        # Target slot in the ContextBundle this capability feeds, e.g.
        # "identity" / "recent" / "decision" / "project". Empty when not
        # opted in.
        self.context_category = context_category or ""

    def serves(self, intent: Intent) -> bool:
        return self.available and intent in self.intents

    def call(self, method: str, **kwargs: Any) -> Any:
        return self.handler(method, **kwargs)

    def in_context(self) -> bool:
        """True iff this capability should be folded into context() output."""
        return self.available and self.contributes_to_context and bool(self.context_category)


class CapabilityRegistry:
    """Tracks available memory capabilities keyed by intent."""

    def __init__(self) -> None:
        self._caps: list[Capability] = []

    def register(
        self,
        name: str,
        intents: Iterable[Intent],
        available: bool,
        handler: Callable[..., Any],
        *,
        provider: Any = None,
        contributes_to_context: bool = False,
        context_category: str = "",
    ) -> Capability:
        cap = Capability(
            name,
            intents,
            available,
            handler,
            provider=provider,
            contributes_to_context=contributes_to_context,
            context_category=context_category,
        )
        self._caps.append(cap)
        return cap

    def set_availability(self, name: str, value: bool) -> None:
        for cap in self._caps:
            if cap.name == name:
                cap.available = value

    def get_available(self, intent: Intent) -> Optional[Capability]:
        for cap in self._caps:
            if cap.serves(intent):
                return cap
        return None

    def list(self) -> list[Capability]:
        return list(self._caps)

    def names(self) -> list[str]:
        return [c.name for c in self._caps]

    def by_name(self, name: str) -> Optional[Capability]:
        """Return the registered capability with the given name, or None."""
        for cap in self._caps:
            if cap.name == name:
                return cap
        return None

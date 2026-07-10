"""Thin Memory Router (Phase 1).

The Router is the SINGLE ENTRY POINT for all memory access. It:

1. CLASSIFIES a request by intent (rule-based, deterministic, no LLM).
2. SELECTS an available registered capability for that intent.
3. DISPATCHES the request to the capability's handler.
4. LOGS every routing decision (intent -> capability) for auditability.

It does NOT interpret memory contents, generate memories, call an LLM, or hold
project-specific knowledge. Those concerns live behind the capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .classify import classify
from .intents import Intent
from .provenance import SearchResult
from .registry import Capability, CapabilityRegistry

_LOGGER = logging.getLogger("hermes.memory_router")


@dataclass
class RouteResult:
    """Outcome of a routed memory request."""

    intent: Any
    capability: Optional[str]
    ok: bool
    results: list = field(default_factory=list)
    note: str = ""


@dataclass
class RouteRecord:
    """Auditable record of one routing decision."""

    intent: str
    capability: Optional[str]
    method: str
    fallback_used: bool


class MemoryRouter:
    """Thin dispatcher over registered memory capabilities."""

    def __init__(self) -> None:
        self.registry = CapabilityRegistry()
        self.last_routes: list[RouteRecord] = []
        # Provider self-registration (Invariant A): importing the registrations
        # package runs each capability's @_registrar, which builds its single
        # provider instance and registers it here. The Router holds NO concrete
        # provider imports at routing level — only the package boundary below.
        from .registrations import load_default_capabilities

        load_default_capabilities(self.registry)
        # Convenience single-instance handles, sourced from the registry (not a
        # separately-constructed object) so there is exactly ONE provider
        # instance per capability. None if the corresponding capability is not
        # registered in this Router.
        self._adr_provider = self._single_provider("L4-adr")
        self._project_provider = self._single_provider("L2-project")

    def _single_provider(self, name: str):
        """Return the single backend instance for a capability, or None.

        Sourced from the registry (self.registration) so there is exactly ONE
        provider instance per capability — the Router is its sole owner.
        """
        cap = self.registry.by_name(name)
        if cap is None:
            return None
        return cap.provider

    # -- wiring -----------------------------------------------------------
    # Provider self-registration now lives in
    # hermes_cli.memory_router.registrations; the Router imports only that
    # package boundary and never a concrete provider class.

    def register_capability(
        self,
        name: str,
        intents: list[Intent],
        available: bool,
        handler: Callable[..., Any],
    ) -> Capability:
        return self.registry.register(name, intents, available, handler)

    # -- core dispatch ----------------------------------------------------
    def route(self, intent: Intent, method: str = "search", **kwargs: Any) -> RouteResult:
        cap = self.registry.get_available(intent)
        if cap is None:
            _LOGGER.info("NO CAPABILITY for intent=%s method=%s", intent, method)
            self.last_routes.append(
                RouteRecord(
                    intent=str(intent), capability=None, method=method, fallback_used=True
                )
            )
            return RouteResult(
                intent=intent,
                capability=None,
                ok=False,
                results=[],
                note="capability unavailable",
            )
        results = cap.call(method, **kwargs)
        _LOGGER.info(
            "route intent=%s capability=%s method=%s fallback=%s",
            intent,
            cap.name,
            method,
            False,
        )
        self.last_routes.append(
            RouteRecord(
                intent=str(intent), capability=cap.name, method=method, fallback_used=False
            )
        )
        return RouteResult(
            intent=intent, capability=cap.name, ok=True, results=results or [], note="ok"
        )

    # -- intent-based convenience ----------------------------------------
    def search(
        self,
        query: str,
        intent_hint: Optional[str] = None,
        limit: int = 10,
        scope: Optional[str] = None,
    ) -> RouteResult:
        intent = classify(query, intent_hint)
        return self.route(intent, "search", query=query, limit=limit, scope=scope)

    def project(self, *, method: str = "get", **kwargs: Any) -> RouteResult:
        return self.route(Intent.PROJECT_STATE, method, **kwargs)

    def draft_decision(self, title: str, **kwargs: Any) -> RouteResult:
        return self.route(Intent.DECISION, "draft", title=title, **kwargs)

    def accept_decision(self, id: str, **kwargs: Any) -> RouteResult:
        return self.route(Intent.DECISION, "accept", id=id, **kwargs)

    def remember(self, content: str, *, layer: str, **meta: Any) -> RouteResult:
        return self.route(Intent.REMEMBER, "remember", content=content, layer=layer, **meta)

    def draft_remember(self, content: str, *, topic: str = "", proposed_by: str = "hermes", **meta: Any) -> RouteResult:
        return self.route(Intent.REMEMBER, "draft", content=content, topic=topic, proposed_by=proposed_by, **meta)

    def accept_remember(self, id: str, *, approved_by: str, **meta: Any) -> RouteResult:
        return self.route(Intent.REMEMBER, "accept", id=id, approved_by=approved_by, **meta)

    def capability_status(self) -> dict[str, str]:
        """Snapshot of every registered capability's readiness (graceful degrad.)."""
        return {c.name: ("available" if c.available else "unavailable") for c in self.registry.list()}

    def decision(self, **kwargs: Any) -> RouteResult:
        return self.route(Intent.DECISION, "decision", **kwargs)

    def recent(self, **kwargs: Any) -> RouteResult:
        return self.route(Intent.RECENT, "recent", **kwargs)

    def archive(self, query: str = "", limit: int = 10, session_id: Optional[str] = None) -> RouteResult:
        return self.route(Intent.ARCHIVE, "archive", query=query, limit=limit, session_id=session_id)

    def context(self, query: str, **kwargs: Any) -> RouteResult:
        # Phase 1 fan-out: only L5 participates. As later layers come online
        # they register for CONTEXT and are gathered here automatically.
        intent = Intent.CONTEXT
        cap = self.registry.get_available(intent)
        if cap is None:
            _LOGGER.info("NO CAPABILITY for intent=%s method=%s", intent, "search")
            self.last_routes.append(
                RouteRecord(
                    intent=str(intent), capability=None, method="search", fallback_used=True
                )
            )
            return RouteResult(
                intent=intent,
                capability=None,
                ok=False,
                results=[],
                note="capability unavailable",
            )
        results: list[SearchResult] = []
        for c in self.registry.list():
            if c.serves(intent):
                r = c.call("search", query=query, **kwargs)
                if isinstance(r, list):
                    results.extend(r)
        _LOGGER.info(
            "route intent=%s capability=%s method=%s fallback=%s",
            intent,
            cap.name,
            "search",
            False,
        )
        self.last_routes.append(
            RouteRecord(
                intent=str(intent), capability=cap.name, method="search", fallback_used=False
            )
        )
        return RouteResult(
            intent=intent, capability=cap.name, ok=True, results=results, note="ok"
        )

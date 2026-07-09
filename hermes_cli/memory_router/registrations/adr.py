"""Self-registration of the L4-ADR decision capability (Invariant A)."""

from __future__ import annotations

from ..registry import CapabilityRegistry
from ...memory_api.adr import AdrProvider
from ..intents import Intent
from . import _registrar


@_registrar
def register(registry: CapabilityRegistry) -> None:
    # Exactly ONE AdrProvider instance owns ADR storage resolution.
    adr_provider = AdrProvider()

    def adr_handle(method: str, **kw):
        if method == "decision":
            return adr_provider.decision(
                id=kw.get("id"), topic=kw.get("topic"), project=kw.get("project")
            )
        if method == "search":
            return adr_provider.search(kw.get("query", ""), project=kw.get("project"))
        if method == "project_decisions":
            return adr_provider.by_project(kw.get("project", ""))
        if method == "recent_decisions":
            return adr_provider.recent(project=kw.get("project"), limit=kw.get("limit", 10))
        if method == "draft":
            return adr_provider.draft(
                kw["title"],
                context=kw.get("context", ""),
                decision=kw.get("decision", ""),
                alternatives=kw.get("alternatives", ""),
                reasoning=kw.get("reasoning", ""),
                consequences=kw.get("consequences", ""),
                related_components=kw.get("related_components"),
                project=kw.get("project", "_system"),
                proposed_by=kw.get("proposed_by", "hermes"),
                tags=kw.get("tags"),
            )
        if method == "accept":
            return adr_provider.accept(
                kw["id"],
                approved_by=kw["approved_by"],
                supersedes=kw.get("supersedes"),
                status=kw.get("status", "accepted"),
            )
        raise RuntimeError(f"unknown ADR method {method!r}")

    registry.register(
        "L4-adr",
        [Intent.DECISION],
        True,
        adr_handle,
        provider=adr_provider,
        # Opted in by a reviewed product decision: ADRs feed the "decision"
        # slot of the context bundle (accepted decisions only).
        contributes_to_context=True,
        context_category="decision",
    )

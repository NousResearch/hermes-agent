"""Self-registration of the L1-remember capability (Invariant A).

Replaces the old stub (which registered ``available=False`` and raised
``RuntimeError`` on any REMEMBER write). The capability is now LIVE: a single
``RememberProvider`` instance owns L1 markdown memory storage and is registered
here with ``available=True``. The Router holds no concrete provider import at
routing level — only this package boundary.

Authority model (approved option B): Hermes drafts PROPOSED memories; only a
human ``accept`` flips them to accepted. The curated identity files
(MEMORY.md / USER.md / SOUL.md / IDENTITY.md) are never touched.

Context guardrail: ``contributes_to_context`` is LEFT False (the Invariant-B
default). A newly registered capability contributes nothing to context()
until a human explicitly opts it in. L1-remember is intentionally NOT opted in.
"""

from __future__ import annotations

from ..registry import CapabilityRegistry
from ...memory_api.remember import RememberProvider
from ..intents import Intent
from . import _registrar


@_registrar
def register(registry: CapabilityRegistry) -> None:
    # Exactly ONE RememberProvider instance owns L1 storage resolution.
    remember_provider = RememberProvider()

    def remember_handle(method: str, **kw):
        if method in ("remember", "draft"):
            # L1 write = a PROPOSED (non-authoritative) memory. The curated
            # identity files are never touched; a human accept() later makes
            # it established. `remember()` and `draft()` are the same proposal
            # entry point for L1.
            return remember_provider.draft(
                kw["content"],
                topic=kw.get("topic", ""),
                proposed_by=kw.get("proposed_by", "hermes"),
            )
        if method == "accept":
            approved_by = kw.get("approved_by")
            if not approved_by:
                raise RuntimeError("accept requires approved_by (human authority)")
            return remember_provider.accept(kw["id"], approved_by=approved_by)
        if method == "list":
            return remember_provider.list(status=kw.get("status"))
        if method == "established":
            return remember_provider.remember_established()
        raise RuntimeError(f"unknown L1-remember method {method!r}")

    registry.register(
        "L1-remember",
        [Intent.REMEMBER],
        True,  # NOW available — the writer is implemented.
        remember_handle,
        provider=remember_provider,
        # Guardrail: NOT opted into context(). A human must explicitly flip
        # contributes_to_context=True to make L1 memories load by default.
        contributes_to_context=False,
        context_category="",
    )

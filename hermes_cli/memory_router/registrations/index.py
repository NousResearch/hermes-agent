"""Self-registration of the shared MemoryIndex capability (Invariant A).

One :class:`IndexCapability` instance backs BOTH the L5 broad FTS index and
the L3 archive/recent retrieval. It is a derived index only (never source of
truth), so it does NOT participate in declarative context — its content is
already reachable via the opted-in L1/L2/L4 capabilities and the historical
search path. Opting it into context would be a separate, reviewed product
decision; for now it stays ``contributes_to_context=False`` by design.
"""

from __future__ import annotations

from ..registry import CapabilityRegistry
from ...memory_index.capability import IndexCapability
from ..intents import Intent
from . import _registrar


@_registrar
def register(registry: CapabilityRegistry) -> None:
    index_cap = IndexCapability()

    registry.register(
        "L5-index",
        [Intent.HISTORICAL, Intent.CONTEXT],
        True,
        lambda method, **kw: index_cap.handle(method, **kw),
        provider=index_cap,
        contributes_to_context=False,
    )
    registry.register(
        "L3-archive",
        [Intent.ARCHIVE, Intent.RECENT],
        True,
        lambda method, **kw: index_cap.handle(method, **kw),
        provider=index_cap,
        # Recently-archived activity feeds the "recent" slot of the context
        # bundle (opted in by a reviewed product decision).
        contributes_to_context=True,
        context_category="recent",
    )

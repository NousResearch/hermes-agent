"""Self-registration of the L1-identity capability (Invariant A)."""

from __future__ import annotations

from ..registry import CapabilityRegistry
from ..capabilities import IdentityCapability
from ..intents import Intent
from . import _registrar


@_registrar
def register(registry: CapabilityRegistry) -> None:
    # Exactly ONE IdentityCapability instance (no storage; file pointers only).
    identity_cap = IdentityCapability()

    registry.register(
        "L1-identity",
        [Intent.IDENTITY],
        True,
        lambda method, **kw: identity_cap.handle(method, **kw),
        provider=identity_cap,
        # Opted in by a reviewed product decision: identity file pointers feed
        # the "identity" slot of the context bundle.
        contributes_to_context=True,
        context_category="identity",
    )

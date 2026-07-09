"""Self-registration of the L1-remember capability (Invariant A).

No writer is wired today, so it stays UNAVAILABLE. Routing it through the
Router (rather than a concrete provider call) keeps the facade backend-agnostic;
when an L1 writer is added it registers here with no facade change.
``remember()`` therefore degrades to a clean "capability unavailable" rather
than a silent no-op. It does NOT participate in declarative context.
"""

from __future__ import annotations

from ..registry import CapabilityRegistry
from ..intents import Intent
from . import _registrar


@_registrar
def register(registry: CapabilityRegistry) -> None:
    registry.register(
        "L1-remember",
        [Intent.REMEMBER],
        False,
        lambda method, **kw: (_ for _ in ()).throw(RuntimeError("no REMEMBER writer")),
        contributes_to_context=False,
    )

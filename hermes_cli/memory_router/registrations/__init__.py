"""Provider self-registration package (Invariant A).

The Router owns its registry and, at construction, imports THIS package and
calls ``load_default_capabilities(self.registry)``. Each sub-module declares
itself via the ``@_registrar`` decorator, which records a registration
function that builds the capability's SINGLE provider instance and registers
it (with a ``provider=`` reference and its ``contributes_to_context`` /
``context_category`` context-participation flags).

The Router therefore imports ONLY this package — never ``AdrProvider``,
``ProjectProvider``, ``IndexCapability`` or ``IdentityCapability``. Removing a
default capability is a single edit here (drop its import + registrar), and
nothing in the routing logic changes.

Each Router gets its own fresh registry + fresh provider instances (so test
isolation / HERMES_HOME redirection keeps working), but within a Router there
is exactly ONE instance per capability — the Router is the sole owner, sourced
from the registry via ``registry.by_name(...)``.
"""

from __future__ import annotations

from typing import Callable

from ..registry import CapabilityRegistry

# Registration functions collected from the sub-modules at import time.
_REGISTRARS: list[Callable[[CapabilityRegistry], None]] = []


def _registrar(fn: Callable[[CapabilityRegistry], None]) -> Callable[[CapabilityRegistry], None]:
    """Decorator: record a capability's registration function."""
    _REGISTRARS.append(fn)
    return fn


def load_default_capabilities(registry: CapabilityRegistry) -> None:
    """Run every collected registrar against ``registry`` (idempotent)."""
    for register in _REGISTRARS:
        register(registry)


# Importing the sub-modules populates ``_REGISTRARS`` via their @_registrar
# decorators. Order is irrelevant — registration is keyed by capability name.
from . import adr  # noqa: E402,F401
from . import identity  # noqa: E402,F401
from . import index  # noqa: E402,F401
from . import project  # noqa: E402,F401
from . import remember  # noqa: E402,F401

__all__ = ["load_default_capabilities", "adr", "identity", "index", "project", "remember"]

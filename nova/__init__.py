"""NOVA — the platform layer built around an agent runtime.

NOVA is runtime-agnostic by construction: everything in this package is expressed in
NOVA's own vocabulary (agents, identity, tenants), and a runtime adapter under
``nova.runtime`` translates that vocabulary into whatever the underlying runtime calls
it. Today the only adapter is Hermes, where a NOVA *agent* becomes a Hermes *profile*.

Nothing under ``nova/`` may be imported by runtime code. The dependency arrow points one
way; see ``docs/platform/ARCHITECTURE_BOUNDARIES.md``.

Dependencies are deliberately limited to the standard library plus PyYAML, so the
platform layer loads and is testable without installing the runtime.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

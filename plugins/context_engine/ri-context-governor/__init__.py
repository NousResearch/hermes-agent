"""RecursiveIntell context-governor — Rust-backed context engine plugin.

Activate in config.yaml::

    context:
      engine: ri-context-governor
      governor:
        token_budget: 8000

This directory is the sole discoverable owner for the configured engine name.
The shared adapter uses the Rust CLI so installation does not depend on an
unrelated PyO3 extension and persists receipt-bound exact fallback.
"""

from __future__ import annotations

from plugins.context_engine._context_governor import ContextGovernorEngine


class RiContextGovernorEngine(ContextGovernorEngine):
    """Configured Ares engine identity backed by the canonical CLI adapter."""

    @property
    def name(self) -> str:
        return "ri-context-governor"


def register(ctx):
    """Plugin contract: register the RI context engine."""
    ctx.register_context_engine(RiContextGovernorEngine())

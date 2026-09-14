"""Native JIT Context Engine for Hermes Agent.

Provides active wire context pruning, lean working set management,
and local model enablement.
"""

from __future__ import annotations

from .engine import JitContextEngine


def register(ctx):
    """Plugin registration entrypoint."""
    ctx.register_context_engine(JitContextEngine())


__all__ = ["JitContextEngine", "register"]

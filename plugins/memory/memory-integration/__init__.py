"""Bundled memory-integration provider package."""

from __future__ import annotations

from .provider import MemoryIntegrationProvider


def register(ctx):
    """Register the memory-integration MemoryProvider."""
    ctx.register_memory_provider(MemoryIntegrationProvider())


__all__ = ["MemoryIntegrationProvider", "register"]

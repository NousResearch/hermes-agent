"""Buzz platform plugin entry point."""

from __future__ import annotations

__all__ = ["register"]


def register(ctx) -> None:
    """Materialize and register the deferred Buzz platform adapter."""
    from .adapter import register as register_adapter

    register_adapter(ctx)

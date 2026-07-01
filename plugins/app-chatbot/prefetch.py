"""Backward-compatible prefetch entrypoint — delegates to intent router."""

from __future__ import annotations

from .router import format_router_context

__all__ = ["format_router_context", "prefetch_database_context"]


def prefetch_database_context(user_message: str, *, database: str = "", default_user_id: str = "", **_kwargs) -> str:
    """Legacy name used by tests; database kwarg ignored (queries use config)."""
    return format_router_context(user_message, default_user_id=default_user_id)

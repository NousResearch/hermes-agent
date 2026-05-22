"""Parallel.ai web search + extract plugin — bundled, auto-loaded.

First plugin in this repo to expose an async :meth:`extract` — Parallel's
SDK is async-native (``AsyncParallel.beta.extract``). The web_extract_tool
dispatcher detects coroutines via :func:`inspect.iscoroutinefunction` and
awaits.
"""

from __future__ import annotations

from plugins.web.parallel.provider import ParallelWebSearchProvider


def register(ctx) -> None:
    """Register the Parallel provider with the plugin context."""
    ctx.register_web_search_provider(ParallelWebSearchProvider())

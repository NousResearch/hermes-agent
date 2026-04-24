"""ThreadDataMiddleware — lazily acquire per-thread sandbox on first hook.

Inspired by DeerFlow's ThreadDataMiddleware (§1.B #1). For each new
`thread_id` seen in ctx, acquire a LocalSandbox so directories exist and
the agent's virtual paths can be translated.

Stores sandbox reference in `ctx.metadata["sandbox"]` so downstream hooks
(e.g., Slice 7's tool dispatch) can reach it without re-acquiring.

Order: 20 (after tracing, before everything else that might touch files).
"""

from __future__ import annotations

import logging

from agent_bus.middleware import BaseMiddleware, MiddlewareContext
from agent_bus.sandbox.local import get_default_provider

logger = logging.getLogger(__name__)


class ThreadDataMiddleware(BaseMiddleware):
    """Lazy-acquire per-thread sandbox on first hook invocation."""

    name = "thread-data"

    def _ensure(self, ctx: MiddlewareContext, hook: str) -> MiddlewareContext:
        if not ctx.thread_id:
            return ctx
        if ctx.metadata.get("sandbox") is not None:
            return ctx
        provider = get_default_provider()
        sb = provider.acquire(ctx.thread_id)
        ctx.metadata["sandbox"] = sb
        ctx.metadata["sandbox_id"] = sb.id
        ctx.record(
            self.name, hook, "acquired",
            f"workspace={sb.workspace_dir}",
        )
        return ctx

    def before_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._ensure(ctx, "before_model")

    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._ensure(ctx, "after_model")

    def before_tool(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._ensure(ctx, "before_tool")

    def on_session_end(self, ctx: MiddlewareContext) -> MiddlewareContext:
        # Nothing to release here — sandbox cache persists; explicit wipe
        # via LocalSandboxProvider.wipe_thread_dir for destructive cleanup.
        return ctx

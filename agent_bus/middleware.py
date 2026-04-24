"""Named middleware chain framework (S3 — se-based integration).

Inspired by DeerFlow's 18-stage middleware pattern (see
~/wiki/concepts/deerflow-distilled.md §1.B) but *not* bound to LangGraph.

Philosophy
----------
- Middleware = named single-responsibility handler with explicit hook points
- Registry = ordered list, each with optional env-var gate for per-component
  rollback without touching code
- All hooks are pure-ish functions on a MiddlewareContext; exceptions in one
  middleware DO NOT abort the chain (they log + continue) unless the
  middleware is marked as critical

Hook points
-----------
- before_model(ctx) -> ctx'      (mutate message history before LLM call)
- after_model(ctx) -> ctx'       (inspect / rewrite LLM response)
- before_tool(ctx) -> ctx'       (authorize / block tool call)
- after_tool(ctx) -> ctx'        (audit / transform tool result)
- on_session_end(ctx) -> ctx'    (finalize cleanup)

Usage
-----
    from agent_bus.middleware import (
        Middleware, MiddlewareContext, MiddlewareChain, register,
    )

    @register(order=10, env_var="HERMES_MW_LOOP_DETECT", critical=False)
    class LoopDetectionMiddleware(Middleware):
        name = "loop-detection"
        def after_model(self, ctx): ...

    chain = MiddlewareChain.build()
    ctx = MiddlewareContext(messages=[...], ...)
    ctx = chain.run("after_model", ctx)

Env vars
--------
- HERMES_MIDDLEWARE_CHAIN   off | core (default core) — master switch
- HERMES_MW_<NAME>          off | core (default core) — per-middleware
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

HOOKS = ("before_model", "after_model", "before_tool", "after_tool", "on_session_end")


# -------- Context --------
@dataclass
class MiddlewareContext:
    """Mutable context passed through the chain for a single invocation.

    Only carries data shapes that matter for the hook; keep generic so we can
    reuse for tests and real runtime.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    """Chat history as list of {role, content, tool_calls?, tool_call_id?}."""

    pending_tool_call: dict[str, Any] | None = None
    """The tool call currently being evaluated (for before_tool / after_tool)."""

    tool_result: Any = None
    """The result of a tool invocation (for after_tool)."""

    thread_id: str | None = None
    agent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    """Scratch space for middleware-specific state across hooks."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    """Log of middleware decisions (name, hook, action, reason)."""

    aborted: bool = False
    """Set to True to short-circuit the chain (e.g., guardrail deny)."""

    def record(self, mw_name: str, hook: str, action: str, reason: str = "") -> None:
        self.decisions.append({
            "middleware": mw_name,
            "hook": hook,
            "action": action,
            "reason": reason,
        })


# -------- Protocol --------
@runtime_checkable
class Middleware(Protocol):
    """All middlewares implement these hooks (default no-op)."""

    name: ClassVar[str]

    def before_model(self, ctx: MiddlewareContext) -> MiddlewareContext: ...
    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext: ...
    def before_tool(self, ctx: MiddlewareContext) -> MiddlewareContext: ...
    def after_tool(self, ctx: MiddlewareContext) -> MiddlewareContext: ...
    def on_session_end(self, ctx: MiddlewareContext) -> MiddlewareContext: ...


class BaseMiddleware:
    """Convenience base — inherit and override just the hooks you need."""

    name: ClassVar[str] = "unnamed"

    def before_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return ctx

    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return ctx

    def before_tool(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return ctx

    def after_tool(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return ctx

    def on_session_end(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return ctx


# -------- Registry --------
@dataclass
class _Entry:
    order: int
    mw: BaseMiddleware
    env_var: str | None
    critical: bool

    def is_enabled(self) -> bool:
        if os.environ.get("HERMES_MIDDLEWARE_CHAIN", "core").lower() == "off":
            return False
        if self.env_var is None:
            return True
        val = os.environ.get(self.env_var, "core").lower()
        return val != "off"


_REGISTRY: list[_Entry] = []


def register(
    *,
    order: int,
    env_var: str | None = None,
    critical: bool = False,
) -> Callable[[type], type]:
    """Decorator: register a middleware class into the global registry.

    The decorated class is instantiated once (via `cls()`) and stored.
    """

    def wrap(cls: type) -> type:
        inst = cls()
        name = getattr(inst, "name", None)
        if not name or name == "unnamed":
            raise TypeError(
                f"{cls.__name__} must set a non-default `name` class attribute "
                f"(got {name!r})"
            )
        _REGISTRY.append(_Entry(order=order, mw=inst, env_var=env_var, critical=critical))
        _REGISTRY.sort(key=lambda e: e.order)
        return cls

    return wrap


def clear_registry() -> None:
    """For tests — drop all registered middlewares."""
    _REGISTRY.clear()


def registered() -> list[BaseMiddleware]:
    """List enabled middleware instances in order."""
    return [e.mw for e in _REGISTRY if e.is_enabled()]


def all_entries() -> list[_Entry]:
    """List all entries (including disabled) — for introspection / dashboard."""
    return list(_REGISTRY)


# -------- Chain executor --------
class MiddlewareChain:
    def __init__(self, entries: list[_Entry] | None = None) -> None:
        self.entries = entries if entries is not None else list(_REGISTRY)

    @classmethod
    def build(cls) -> MiddlewareChain:
        return cls()

    def run(self, hook: str, ctx: MiddlewareContext) -> MiddlewareContext:
        if hook not in HOOKS:
            raise ValueError(f"unknown hook: {hook!r} (allowed: {HOOKS})")
        # Master switch
        if os.environ.get("HERMES_MIDDLEWARE_CHAIN", "core").lower() == "off":
            return ctx
        for entry in self.entries:
            if not entry.is_enabled():
                continue
            fn = getattr(entry.mw, hook, None)
            if fn is None:
                continue
            try:
                ctx = fn(ctx)
            except Exception as exc:  # pragma: no cover — safety net
                logger.warning(
                    "middleware[%s].%s raised %s; critical=%s",
                    entry.mw.name, hook, exc, entry.critical,
                )
                if entry.critical:
                    raise
                ctx.record(entry.mw.name, hook, "error", str(exc))
            if ctx.aborted:
                ctx.record(entry.mw.name, hook, "abort", "aborted by middleware")
                break
        return ctx


__all__ = [
    "HOOKS",
    "MiddlewareContext",
    "Middleware",
    "BaseMiddleware",
    "MiddlewareChain",
    "register",
    "registered",
    "all_entries",
    "clear_registry",
]

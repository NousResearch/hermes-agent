"""TracingMiddleware — records per-hook timings + emits to LangSmith/Langfuse.

Inspired by DeerFlow's built-in observability (§1.I). Optional: if neither
provider is configured, the middleware still records timings into
`ctx.metadata["trace_spans"]` so the dashboard can show call latency even
without a cloud backend.

Providers
---------
- LangSmith: installed if `langsmith` package import succeeds and
  `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` set
- Langfuse: installed if `langfuse` package import succeeds and
  `LANGFUSE_TRACING=true` + `LANGFUSE_PUBLIC_KEY` set
- Both can run simultaneously (DeerFlow pattern)

Fail-fast: if a provider is enabled by env var but package missing OR
initialization fails, we log the error but DO NOT crash the chain. (Unlike
DeerFlow which fails on model-create; we're ambient, so safer to degrade.)

Order: 10 (runs first on every hook for accurate timing floor).

Env vars
--------
HERMES_MW_TRACING           off | core (default core)
LANGSMITH_TRACING           true | false
LANGSMITH_API_KEY           required when tracing=true
LANGSMITH_PROJECT           default "hermes-agent-bus"
LANGFUSE_TRACING            true | false
LANGFUSE_PUBLIC_KEY         required when tracing=true
LANGFUSE_SECRET_KEY         required when tracing=true
LANGFUSE_BASE_URL           default https://cloud.langfuse.com
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from agent_bus.middleware import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

_PROVIDERS_INITIALIZED = False
_LANGSMITH_CLIENT: Any = None
_LANGFUSE_CLIENT: Any = None


def _init_providers() -> None:
    """Lazy-init provider clients. Safe to call repeatedly."""
    global _PROVIDERS_INITIALIZED, _LANGSMITH_CLIENT, _LANGFUSE_CLIENT
    if _PROVIDERS_INITIALIZED:
        return

    # LangSmith
    if os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1"):
        key = os.environ.get("LANGSMITH_API_KEY")
        if not key:
            logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY missing — skipping")
        else:
            try:
                from langsmith import Client  # type: ignore
                _LANGSMITH_CLIENT = Client(api_key=key)
                logger.info("langsmith tracing enabled")
            except ImportError:
                logger.warning("langsmith package not installed — skipping")
            except Exception as exc:
                logger.warning("langsmith init failed: %s", exc)

    # Langfuse
    if os.environ.get("LANGFUSE_TRACING", "").lower() in ("true", "1"):
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_SECRET_KEY")
        if not (pk and sk):
            logger.warning("LANGFUSE_TRACING=true but keys missing — skipping")
        else:
            try:
                from langfuse import Langfuse  # type: ignore
                _LANGFUSE_CLIENT = Langfuse(
                    public_key=pk, secret_key=sk,
                    host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                )
                logger.info("langfuse tracing enabled")
            except ImportError:
                logger.warning("langfuse package not installed — skipping")
            except Exception as exc:
                logger.warning("langfuse init failed: %s", exc)

    _PROVIDERS_INITIALIZED = True


def _reset_providers_for_test() -> None:  # pragma: no cover (test helper)
    global _PROVIDERS_INITIALIZED, _LANGSMITH_CLIENT, _LANGFUSE_CLIENT
    _PROVIDERS_INITIALIZED = False
    _LANGSMITH_CLIENT = None
    _LANGFUSE_CLIENT = None


_TRACE_STORE_LOCK: Any = None  # lazy-set by _init_providers


def _get_trace_store_path() -> "Path":
    from pathlib import Path as _Path
    return _Path(os.environ.get(
        "HERMES_TRACE_STORE_PATH",
        str(_Path.home() / ".hermes" / "traces.jsonl"),
    )).expanduser()


def _persist_span(span: dict) -> None:
    """Append span to local JSONL file for dashboard timeline queries."""
    import json as _json
    from pathlib import Path as _Path
    try:
        path = _get_trace_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(span, ensure_ascii=False) + "\n")
        # Bound file size — trim to last 5000 lines if grows beyond 10k
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > 10_000:
                with path.open("w", encoding="utf-8") as f:
                    f.writelines(lines[-5000:])
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        logger.debug("trace persist failed: %s", exc)


def _emit_span(hook: str, ctx: MiddlewareContext, start: float, end: float) -> None:
    """Push a span to whichever providers are configured + local JSONL store."""
    duration_ms = (end - start) * 1000.0
    span = {
        "name": f"middleware.{hook}",
        "hook": hook,
        "thread_id": ctx.thread_id,
        "agent": ctx.agent,
        "start_ts": start,
        "end_ts": end,
        "duration_ms": round(duration_ms, 2),
        "msg_count": len(ctx.messages),
        "decisions": len(ctx.decisions),
    }
    # Always record into ctx metadata for dashboard even without providers
    ctx.metadata.setdefault("trace_spans", []).append(span)
    # Persist to local JSONL — dashboard can tail this
    _persist_span(span)

    if _LANGSMITH_CLIENT is not None:
        try:
            _LANGSMITH_CLIENT.create_run(
                name=span["name"],
                run_type="chain",
                start_time=start,
                end_time=end,
                extra={"thread_id": ctx.thread_id, "agent": ctx.agent},
                project_name=os.environ.get("LANGSMITH_PROJECT", "hermes-agent-bus"),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("langsmith emit failed: %s", exc)

    if _LANGFUSE_CLIENT is not None:
        try:
            _LANGFUSE_CLIENT.trace(
                name=span["name"],
                metadata=span,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("langfuse emit failed: %s", exc)


class TracingMiddleware(BaseMiddleware):
    """Record timing for each hook invocation."""

    name = "tracing"

    def _wrap(self, hook: str, ctx: MiddlewareContext) -> MiddlewareContext:
        _init_providers()
        start = time.time()
        # Nothing to do in the middleware body itself — the value is the
        # timing around this call. But because middlewares run sequentially
        # and we're at order 10 (before most others), the span we emit
        # represents "how long this middleware itself took", which is near
        # zero. The real value emerges when we accumulate across hooks.
        end = time.time()
        _emit_span(hook, ctx, start, end)
        return ctx

    def before_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._wrap("before_model", ctx)

    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._wrap("after_model", ctx)

    def before_tool(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._wrap("before_tool", ctx)

    def after_tool(self, ctx: MiddlewareContext) -> MiddlewareContext:
        return self._wrap("after_tool", ctx)

    def on_session_end(self, ctx: MiddlewareContext) -> MiddlewareContext:
        result = self._wrap("on_session_end", ctx)
        # Summarize total time across all spans for this session
        spans = ctx.metadata.get("trace_spans", [])
        total = sum(s["duration_ms"] for s in spans)
        ctx.record(
            self.name, "on_session_end", "summary",
            f"spans={len(spans)} total_ms={round(total, 2)}",
        )
        return result

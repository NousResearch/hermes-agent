"""mq9 Hermes plugin entrypoint."""

from __future__ import annotations

import atexit
import logging
from typing import Any

from . import schemas, tools
from .runtime import MQ9HermesRuntime

logger = logging.getLogger(__name__)

_RUNTIME = MQ9HermesRuntime()


_TOOLS: tuple[tuple[str, dict[str, Any], Any, str], ...] = (
    ("mq9_register_self", schemas.MQ9_REGISTER_SELF, tools.mq9_register_self, "🛰️"),
    ("mq9_unregister_self", schemas.MQ9_UNREGISTER_SELF, tools.mq9_unregister_self, "🧹"),
    ("mq9_discover", schemas.MQ9_DISCOVER, tools.mq9_discover, "🔎"),
    ("mq9_call", schemas.MQ9_CALL, tools.mq9_call, "📨"),
    ("mq9_status", schemas.MQ9_STATUS, tools.mq9_status, "📊"),
)


def _on_session_start(session_id: str = "", **kwargs: Any) -> None:
    del kwargs
    _RUNTIME.start_background(reason=f"on_session_start:{session_id}")


def _on_session_reset(session_id: str = "", **kwargs: Any) -> None:
    del kwargs
    _RUNTIME.start_background(reason=f"on_session_reset:{session_id}")


def _on_session_finalize(session_id: str | None = None, **kwargs: Any) -> None:
    del session_id, kwargs
    _RUNTIME.stop_background(reason="on_session_finalize", unregister=True)


def _on_process_exit() -> None:
    try:
        _RUNTIME.stop_background(reason="process_exit", unregister=True)
    except Exception:
        # Interpreter shutdown can leave dependencies half-torn-down.
        pass


def register(ctx: Any) -> None:
    """Register tools + hooks into Hermes plugin manager."""
    tools.bind_runtime(_RUNTIME)
    _RUNTIME.attach_context(ctx)

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="mq9",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_reset", _on_session_reset)
    ctx.register_hook("on_session_finalize", _on_session_finalize)

    # Closest behavior to historical "post_setup": bring runtime up eagerly
    # so this Hermes instance becomes discoverable before first remote call.
    try:
        _RUNTIME.start_background(reason="plugin_register")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("mq9 plugin runtime eager start failed: %s", exc)


atexit.register(_on_process_exit)

"""Tool handlers for mq9 Hermes plugin."""

from __future__ import annotations

import json
import logging
from typing import Any

from .runtime import MQ9HermesRuntime

logger = logging.getLogger(__name__)

_RUNTIME: MQ9HermesRuntime | None = None

try:  # pragma: no cover - only active inside Hermes runtime
    from tools.registry import tool_error as _tool_error
    from tools.registry import tool_result as _tool_result
except Exception:  # pragma: no cover - local fallback
    def _tool_error(message: Any, **extra: Any) -> str:
        payload = {"error": str(message)}
        payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)

    def _tool_result(data: Any = None, **kwargs: Any) -> str:
        if data is not None:
            return json.dumps(data, ensure_ascii=False)
        return json.dumps(kwargs, ensure_ascii=False)


def bind_runtime(runtime: MQ9HermesRuntime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def _require_runtime() -> MQ9HermesRuntime:
    if _RUNTIME is None:
        raise RuntimeError("mq9 runtime is not initialized")
    return _RUNTIME


def mq9_register_self(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    runtime = _require_runtime()
    try:
        result = runtime.register_self(args)
        return _tool_result(result)
    except Exception as exc:
        logger.exception("mq9_register_self failed: %s", exc)
        return _tool_error(f"mq9_register_self failed: {exc}")


def mq9_unregister_self(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    runtime = _require_runtime()
    try:
        result = runtime.unregister_self(args)
        return _tool_result(result)
    except Exception as exc:
        logger.exception("mq9_unregister_self failed: %s", exc)
        return _tool_error(f"mq9_unregister_self failed: {exc}")


def mq9_discover(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    runtime = _require_runtime()
    try:
        result = runtime.discover(args)
        return _tool_result(result)
    except Exception as exc:
        logger.exception("mq9_discover failed: %s", exc)
        return _tool_error(f"mq9_discover failed: {exc}")


def mq9_call(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    runtime = _require_runtime()
    try:
        result = runtime.call(args)
        return _tool_result(result)
    except Exception as exc:
        logger.exception("mq9_call failed: %s", exc)
        return _tool_error(f"mq9_call failed: {exc}")


def mq9_status(args: dict[str, Any], **kwargs: Any) -> str:
    del args, kwargs
    runtime = _require_runtime()
    try:
        return _tool_result(runtime.status())
    except Exception as exc:
        logger.exception("mq9_status failed: %s", exc)
        return _tool_error(f"mq9_status failed: {exc}")

"""Core NeMo Relay adapter for Hermes tool execution."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from agent import relay_runtime


def execute(
    tool_name: str,
    args: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one tool call through Relay and return its final arguments."""
    runtime, session, parent = relay_runtime.resolve_execution_context(session_id)
    if runtime is None or session is None:
        return callback(args), args

    observed_args = args
    raw_result: dict[str, Any] = {}
    callback_error: BaseException | None = None

    def invoke(next_args: Any) -> Any:
        nonlocal callback_error, observed_args
        observed_args = next_args if isinstance(next_args, dict) else args
        try:
            result = callback(observed_args)
        except BaseException as exc:
            callback_error = exc
            raise
        raw_result["value"] = result
        raw_result["json"] = _jsonable(result)
        return raw_result["json"]

    try:
        managed = _run_awaitable(
            runtime.run_in_session_async(
                session,
                runtime.relay.tools.execute,
                tool_name,
                _jsonable(args),
                invoke,
                handle=parent,
                metadata=_jsonable(metadata or {}),
            )
        )
    except BaseException as exc:
        if (
            callback_error is not None
            and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error)
        ):
            raise callback_error
        raise

    if "value" in raw_result and _json_equal(managed, raw_result["json"]):
        return raw_result["value"], observed_args
    return managed, observed_args


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except Exception:
            pass
    try:
        return _jsonable(vars(value))
    except (TypeError, AttributeError):
        return str(value)


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            _jsonable(left), sort_keys=True, separators=(",", ":")
        ) == json.dumps(_jsonable(right), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return left == right


def _run_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "Synchronous Hermes Relay tool execution cannot run on an active event-loop thread"
    )

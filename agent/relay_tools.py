"""Core NeMo Relay adapter for Hermes tool execution."""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from agent import relay_llm, relay_runtime

logger = logging.getLogger(__name__)


def execute(
    tool_name: str, args: dict[str, Any], callback: Callable[[dict[str, Any]], Any], *,
    session_id: str, tool_call_id: str | None = None, metadata: dict[str, Any] | None = None,
    request_guard: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one tool call through Relay and return its final arguments."""
    runtime, session, parent = relay_runtime.resolve_execution_context(session_id)
    if runtime is None or session is None or not runtime.managed_execution_enabled():
        return callback(args), args
    observed_args = args
    raw_result: dict[str, Any] = {}
    callback_error: BaseException | None = None
    callback_context = contextvars.copy_context()
    request_guard_name: str | None = None
    scope_local = getattr(runtime.relay, "scope_local", None)

    # A caller-owned terminal request guard runs *inside* Relay after configured
    # request rewrites but before sanitize/start/execution middleware. Scope-local
    # registration keeps concurrent profiles/turns isolated and disappears after
    # this managed call.
    if request_guard is not None:
        register = getattr(scope_local, "register_tool_request", None)
        if not callable(register) or parent is None:
            raise RuntimeError("NeMo Relay request guard unavailable for managed tool execution")
        request_guard_name = f"hermes.tool-request-guard.{uuid.uuid4().hex}"

        def _terminal_request_guard(name: str, effective_args: Any) -> Any:
            checked = effective_args if isinstance(effective_args, dict) else args
            request_guard(name, checked)
            return effective_args

        register(parent, request_guard_name, 2_147_483_647, True, _terminal_request_guard)

    def guarded(final_args: dict[str, Any]) -> Any:
        # Everything the tool transitively calls (incl. auxiliary LLM calls on worker
        # threads) must bypass managed Relay: the pipeline's Futures bind to THIS loop,
        # which is blocked until the tool returns.
        # See #77244.
        with relay_runtime.managed_callback_guard():
            return callback(final_args)

    def invoke(next_args: Any) -> Any:
        nonlocal callback_error, observed_args
        observed_args = next_args if isinstance(next_args, dict) else args
        try:
            result = callback_context.copy().run(guarded, observed_args)
        except BaseException as exc:
            callback_error = exc
            raise
        raw_result.update(value=result, json=_jsonable(result))
        return runtime.relay.ToolExecutionResult(raw_result["json"])

    try:
        managed = _run_awaitable(
            runtime.run_in_session_async(
                session, runtime.relay.tools.execute, tool_name, _jsonable(args), invoke,
                handle=parent, metadata=_jsonable(metadata or {}), tool_call_id=tool_call_id or None,
            )
        )
    except BaseException as exc:
        if callback_error is not None and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error):
            raise callback_error
        if isinstance(exc, Exception) and callback_error is None and "value" in raw_result:
            logger.warning(
                "NeMo Relay tool post-processing failed after dispatch success; returning the Hermes tool result",
                exc_info=True,
            )
            return raw_result["value"], observed_args
        raise
    finally:
        if request_guard_name is not None:
            deregister = getattr(scope_local, "deregister_tool_request", None)
            if callable(deregister):
                try:
                    deregister(parent, request_guard_name)
                except Exception:
                    logger.warning("Failed to remove Hermes Relay request guard %s", request_guard_name, exc_info=True)
    managed_result = managed.result
    if "value" in raw_result and _json_equal(managed_result, raw_result["json"]):
        return raw_result["value"], observed_args
    if isinstance(managed_result, str):
        return managed_result, observed_args
    return json.dumps(_jsonable(managed_result), ensure_ascii=False), observed_args


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
            # warnings=False: pydantic's generic-union warning would leak to the CLI mid-turn.
            try:
                return _jsonable(model_dump(mode="json", warnings=False))
            except TypeError:
                return _jsonable(model_dump())
        except Exception:
            pass
    try:
        return _jsonable(vars(value))
    except (TypeError, AttributeError):
        return str(value)


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return relay_llm._canonical_json(left, _jsonable) == relay_llm._canonical_json(right, _jsonable)
    except (TypeError, ValueError):
        return left == right


def _run_awaitable(value: Any) -> Any:
    return relay_llm._run_awaitable(
        value, loop_error="Synchronous Hermes Relay tool execution cannot run on an active event-loop thread",
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import asyncio  # noqa: F401,E402
import inspect  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

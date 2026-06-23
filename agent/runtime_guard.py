"""Generic disabled-by-default runtime guard primitives."""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Mapping
from typing import Any

from agent.runtime_guard_fallback import (
    _runtime_guard_cfg,
    runtime_guard_blocked_text,
    runtime_guard_guard_error_reason,
)
from agent.runtime_guard_types import GuardContext, GuardDecision

logger = logging.getLogger(__name__)

_CURRENT_RUNTIME_GUARD: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "runtime_guard",
    default=None,
)


class RuntimeGuardUnavailable(RuntimeError):
    """Raised when runtime guard is enabled but no guard implementation exists."""


def set_current_runtime_guard(guard: Any):
    """Install a guard object for the current context and return a reset token."""
    return _CURRENT_RUNTIME_GUARD.set(guard)


def reset_current_runtime_guard(token) -> None:
    """Reset a guard token returned by ``set_current_runtime_guard``."""
    _CURRENT_RUNTIME_GUARD.reset(token)


def _load_runtime_guard_config() -> Mapping[str, Any]:
    from hermes_cli.config import load_config_readonly

    return _runtime_guard_cfg(load_config_readonly())


def _config_for_agent(agent: Any = None) -> Mapping[str, Any]:
    if agent is not None:
        for attr in ("_runtime_guard_config", "runtime_guard_config"):
            config = getattr(agent, attr, None)
            if isinstance(config, Mapping):
                return _runtime_guard_cfg(config)
    return _load_runtime_guard_config()


def runtime_guard_enabled(agent: Any = None, *, config: Mapping[str, Any] | None = None) -> bool:
    cfg = _runtime_guard_cfg(config) if config is not None else _config_for_agent(agent)
    return bool(cfg.get("enabled", False))


def runtime_guard_fail_closed(agent: Any = None, *, config: Mapping[str, Any] | None = None) -> bool:
    cfg = _runtime_guard_cfg(config) if config is not None else _config_for_agent(agent)
    return bool(cfg.get("fail_closed", True))


def _runtime_guard_for_agent(agent: Any = None) -> Any:
    guard = _CURRENT_RUNTIME_GUARD.get()
    if guard is not None:
        return guard
    if agent is not None:
        for attr in ("_runtime_guard", "runtime_guard"):
            guard = getattr(agent, attr, None)
            if guard is not None:
                return guard
    raise RuntimeGuardUnavailable("no runtime_guard implementation is active")


def _coerce_decision(value: Any, context: GuardContext) -> GuardDecision:
    if isinstance(value, GuardDecision):
        return value.with_context(context)
    if value is None or value is True:
        return GuardDecision.allow(context=context)
    if value is False:
        return GuardDecision.block("runtime_guard blocked this action", context=context)
    if isinstance(value, Mapping):
        allowed = bool(value.get("allowed", value.get("allow", False)))
        if allowed:
            return GuardDecision.allow(str(value.get("reason") or "allowed"), context=context)
        return GuardDecision.block(
            str(value.get("reason") or value.get("message") or "runtime_guard blocked this action"),
            message=str(value.get("message") or value.get("reason") or ""),
            code=str(value.get("code") or "runtime_guard_block"),
            replacement_text=value.get("replacement_text"),
            context=context,
            metadata=value.get("metadata") if isinstance(value.get("metadata"), Mapping) else None,
        )
    return GuardDecision.allow(context=context)


def _evaluate_guard(
    *,
    agent: Any = None,
    context: GuardContext,
    payload: Mapping[str, Any] | None = None,
) -> GuardDecision:
    guard = _runtime_guard_for_agent(agent)
    payload = dict(payload or {})
    method_names = (
        f"guard_{context.guard_name}",
        f"evaluate_{context.guard_name}",
        "guard",
        "evaluate",
    )
    for method_name in method_names:
        method = getattr(guard, method_name, None)
        if callable(method):
            return _coerce_decision(method(context, **payload), context)
    if callable(guard):
        return _coerce_decision(guard(context, **payload), context)
    raise RuntimeGuardUnavailable("runtime_guard implementation has no callable guard method")


def _guard_internal_error_decision(
    exc: BaseException,
    *,
    agent: Any = None,
    context: GuardContext,
) -> GuardDecision:
    if runtime_guard_fail_closed(agent):
        reason = runtime_guard_guard_error_reason(exc, guard_name=context.guard_name)
        return GuardDecision.block(reason, code="runtime_guard_error", context=context)
    logger.debug("runtime_guard failed open for %s: %s", context.guard_name, exc)
    return GuardDecision.allow("runtime_guard failed open", context=context)


def guard_tool_action_for_agent(
    agent: Any,
    tool_name: str,
    tool_args: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    tool_call_id: str = "",
) -> GuardDecision:
    context = GuardContext(
        guard_name="tool_action",
        action=f"tool:{tool_name}",
        session_id=getattr(agent, "session_id", "") or "",
        task_id=task_id or "",
        tool_call_id=tool_call_id or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
        platform=getattr(agent, "platform", "") or "",
        tool_name=tool_name,
        tool_args=tool_args or {},
    )
    if not runtime_guard_enabled(agent):
        return GuardDecision.allow("runtime_guard disabled", context=context)
    try:
        return _evaluate_guard(agent=agent, context=context, payload={"tool_args": tool_args or {}})
    except Exception as exc:
        return _guard_internal_error_decision(exc, agent=agent, context=context)


def guard_tool_action_for_current_context(
    tool_name: str,
    tool_args: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    platform: str = "",
) -> GuardDecision:
    context = GuardContext(
        guard_name="tool_action",
        action=f"tool:{tool_name}",
        session_id=session_id or "",
        task_id=task_id or "",
        tool_call_id=tool_call_id or "",
        turn_id=turn_id or "",
        api_request_id=api_request_id or "",
        platform=platform or "",
        tool_name=tool_name,
        tool_args=tool_args or {},
    )
    if not runtime_guard_enabled():
        return GuardDecision.allow("runtime_guard disabled", context=context)
    try:
        return _evaluate_guard(context=context, payload={"tool_args": tool_args or {}})
    except Exception as exc:
        return _guard_internal_error_decision(exc, context=context)


def guard_final_output_for_agent(agent: Any, text: str) -> GuardDecision:
    context = GuardContext(
        guard_name="final_output",
        action="final_output",
        session_id=getattr(agent, "session_id", "") or "",
        platform=getattr(agent, "platform", "") or "",
    )
    if not runtime_guard_enabled(agent):
        return GuardDecision.allow("runtime_guard disabled", context=context)
    try:
        return _evaluate_guard(agent=agent, context=context, payload={"text": text})
    except Exception as exc:
        return _guard_internal_error_decision(exc, agent=agent, context=context)


def guarded_final_output_for_agent(agent: Any, text: str) -> str:
    decision = guard_final_output_for_agent(agent, text)
    if decision.allowed:
        return text
    return final_output_block_text(decision)


def tool_block_result(decision: GuardDecision) -> str:
    message = decision.message or decision.reason or "runtime_guard blocked this action"
    code = decision.code or "runtime_guard_block"
    return json.dumps(
        {
            "error": message,
            "status": "blocked",
            "blocked_by": "runtime_guard",
            "code": code,
            "runtime_guard_block": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def final_output_block_text(decision: GuardDecision) -> str:
    if decision.replacement_text:
        return decision.replacement_text
    reason = decision.message or decision.reason or "runtime_guard blocked the response"
    return runtime_guard_blocked_text(reason, guard_name="final_output")

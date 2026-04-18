"""Shared execution order for direct gateway shortcut handlers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Iterable

from gateway.direct_control_router import DIRECT_CONTROL_ROUTER_METHODS, DirectControlRouter
from gateway.direct_shortcut_trace_runtime_service import record_direct_shortcut_trace
from gateway.runtime_shortcuts_service import (
    try_handle_background_job_status_shortcut,
    try_handle_runtime_status_shortcut,
)


@dataclass(frozen=True)
class DirectShortcutHandlerSpec:
    """Describe one direct-shortcut handler exposed by GatewayRunner."""

    method_name: str
    passes_history: bool = False


DIRECT_SHORTCUT_HANDLER_SPECS: tuple[DirectShortcutHandlerSpec, ...] = (
    DirectShortcutHandlerSpec("_try_handle_background_job_status_shortcut"),
    DirectShortcutHandlerSpec("_try_handle_runtime_status_shortcut"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_send_shortcut", passes_history=True),
    DirectShortcutHandlerSpec("_try_handle_admin_weixin_send_shortcut", passes_history=True),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_group_runtime_status", passes_history=True),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_group_control"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_group_moderation"),
    DirectShortcutHandlerSpec("_try_handle_admin_weixin_group_runtime_status", passes_history=True),
    DirectShortcutHandlerSpec("_try_handle_admin_weixin_group_control"),
    DirectShortcutHandlerSpec("_try_handle_admin_weixin_group_moderation"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_intel_control"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_social_control"),
)


def _resolve_pending_agent_sentinel(runner: Any) -> Any:
    sentinel = getattr(runner, "_pending_agent_sentinel", None)
    if sentinel is not None:
        return sentinel
    module_name = getattr(type(runner), "__module__", "")
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "_AGENT_PENDING_SENTINEL", None)


def _resolve_direct_shortcut_handler(runner: Any, method_name: str):
    handler = getattr(runner, method_name, None)
    if handler is not None:
        return handler
    if method_name == "_try_handle_background_job_status_shortcut":
        return lambda event: try_handle_background_job_status_shortcut(runner, event)
    if method_name == "_try_handle_runtime_status_shortcut":
        return lambda event: try_handle_runtime_status_shortcut(
            runner,
            event,
            pending_sentinel=_resolve_pending_agent_sentinel(runner),
        )
    if method_name not in DIRECT_CONTROL_ROUTER_METHODS:
        return None
    get_router = getattr(runner, "_get_direct_control_router", None)
    if callable(get_router):
        router = get_router()
    else:
        router = getattr(runner, "_direct_control_router", None)
    if router is None:
        router = DirectControlRouter(runner)
        runner._direct_control_router = router
    return getattr(router, method_name, None)


def run_direct_shortcut_handlers(
    runner: Any,
    event: Any,
    *,
    conversation_history: Iterable[dict[str, Any]] | None = None,
    logger=None,
) -> str | None:
    """Try direct shortcut handlers in the canonical gateway order."""
    attempted_handlers: list[str] = []
    for spec in DIRECT_SHORTCUT_HANDLER_SPECS:
        attempted_handlers.append(spec.method_name)
        try:
            handler = _resolve_direct_shortcut_handler(runner, spec.method_name)
            if handler is None:
                continue
            if spec.passes_history:
                response = handler(event, conversation_history=conversation_history)
            else:
                response = handler(event)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Direct gateway shortcut handler %s failed: %s",
                    spec.method_name,
                    exc,
                )
            continue
        if response is not None:
            try:
                record_direct_shortcut_trace(
                    runner,
                    event,
                    matched_handler=spec.method_name,
                    attempted_handlers=attempted_handlers,
                    response=response,
                )
            except Exception:
                if logger is not None:
                    logger.debug(
                        "Failed to record direct shortcut trace for %s",
                        spec.method_name,
                        exc_info=True,
                    )
            return response
    return None

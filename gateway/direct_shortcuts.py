"""Shared execution order for direct gateway shortcut handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


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
    DirectShortcutHandlerSpec("_try_handle_admin_weixin_group_runtime_status", passes_history=True),
    DirectShortcutHandlerSpec("_try_handle_admin_weixin_group_control"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_intel_control"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_social_control"),
    DirectShortcutHandlerSpec("_try_handle_admin_qq_group_moderation"),
)


def run_direct_shortcut_handlers(
    runner: Any,
    event: Any,
    *,
    conversation_history: Iterable[dict[str, Any]] | None = None,
    logger=None,
) -> str | None:
    """Try direct shortcut handlers in the canonical gateway order."""
    for spec in DIRECT_SHORTCUT_HANDLER_SPECS:
        handler = getattr(runner, spec.method_name, None)
        if handler is None:
            continue
        try:
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
            return response
    return None

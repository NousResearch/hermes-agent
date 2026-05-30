"""Hermes plugin hooks — pre-transition guards for Kynver todo writes.

Projection and read-back live on :class:`KynverTodoStore`; hooks only block
illegal focus transitions before the built-in ``todo`` tool runs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .agentos_bridge import KynverAgentOSClient, agentos_enabled, load_kynver_agentos_config
from .operating_config import kynver_operating_tools_enabled, load_operating_linkage
from .plan_progress import inspect_todo_write

logger = logging.getLogger(__name__)


def _client() -> KynverAgentOSClient | None:
    if not kynver_operating_tools_enabled():
        return None
    cfg = load_kynver_agentos_config()
    if not cfg.enabled:
        return None
    return KynverAgentOSClient(cfg)


def on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[dict[str, Any]]:
    if tool_name != "todo":
        return None
    if not isinstance(args, dict):
        return None
    todos = args.get("todos")
    if todos is None:
        return None

    client = _client()
    if not client:
        return None

    linkage = load_operating_linkage()
    blocked = inspect_todo_write(
        client,
        linkage,
        list(todos),
        merge=bool(args.get("merge")),
    )
    if blocked:
        return {
            "action": "block",
            "message": blocked,
        }
    return None


def on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """No-op — KynverTodoStore reconciles on read/write."""

    return None


def register_operating_hooks(ctx) -> None:
    if not agentos_enabled():
        return
    if not kynver_operating_tools_enabled():
        return
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    logger.info("Kynver operating hooks registered (todo pre-transition guards)")

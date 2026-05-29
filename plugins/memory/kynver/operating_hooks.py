"""Hermes plugin hooks — plan progress + todo read-back without core Hermes patches."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .agentos_bridge import KynverAgentOSClient, agentos_enabled, load_kynver_agentos_config
from .operating_config import kynver_operating_tools_enabled, load_operating_linkage
from .plan_progress import safe_project_todo_write, transform_todo_result

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
    projection = safe_project_todo_write(
        client,
        linkage,
        list(todos),
        merge=bool(args.get("merge")),
    )
    if projection.get("blocked"):
        return {
            "action": "block",
            "message": projection.get("error", "Kynver pre-transition blocked todo write"),
        }
    return None


def on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    if tool_name != "todo":
        return None
    if not isinstance(result, str):
        return None

    client = _client()
    if not client:
        return None

    linkage = load_operating_linkage()
    return transform_todo_result(result, client, linkage)


def register_operating_hooks(ctx) -> None:
    if not agentos_enabled():
        return
    if not kynver_operating_tools_enabled():
        return
    from hermes_cli.plugins import register_plugin_hook

    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_tool_call", on_pre_tool_call)
        ctx.register_hook("transform_tool_result", on_transform_tool_result)
    else:
        register_plugin_hook("kynver", "pre_tool_call", on_pre_tool_call)
        register_plugin_hook("kynver", "transform_tool_result", on_transform_tool_result)
    logger.info("Kynver operating hooks registered (plan progress + todo read-back)")

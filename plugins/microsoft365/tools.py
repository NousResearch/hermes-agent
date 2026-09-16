"""Tool handlers for the single ``microsoft365`` toolset."""
from __future__ import annotations

import inspect
from typing import Any

from tools.registry import tool_error, tool_result
from .backend import CAPABILITIES, OPERATIONS, Microsoft365Settings, create_graph_client, preflight, safe_result


def _settings(ctx) -> Microsoft365Settings:
    values = {key: ctx.get_config(key) for key in ("tenant_id", "client_id", "client_secret", "user_id", "capabilities")}
    return Microsoft365Settings.from_mapping(values)


async def _maybe(value):
    return await value if inspect.isawaitable(value) else value


async def handle_preflight(args: dict, *, context=None, ctx=None, **kwargs) -> str:
    return tool_result(preflight(_settings(ctx or context)))


async def _run(capability: str, args: dict, ctx) -> str:
    settings = _settings(ctx)
    action = str(args.get("action") or "").strip().lower()
    if capability not in CAPABILITIES:
        return tool_error(f"Unknown Microsoft 365 capability: {capability}")
    if action not in OPERATIONS[capability]:
        return tool_error(f"Unknown {capability} operation: {action or '<missing>'}")
    if action not in settings.operations(capability):
        return tool_error(f"Microsoft 365 operation is disabled: {capability}.{action}")
    client = create_graph_client(settings)
    user = settings.user_id
    try:
        if capability == "outlook":
            request = client.users.by_user_id(user).messages
            if action == "search": result = await _maybe(request.get())
            elif action == "read": result = await _maybe(request.by_message_id(str(args.get("id"))).get())
            else: raise NotImplementedError(f"SDK mapping required for {action}")
        elif capability == "calendar":
            request = client.users.by_user_id(user).calendar.events
            if action == "search": result = await _maybe(request.get())
            else: raise NotImplementedError(f"SDK mapping required for {action}")
        elif capability in ("sharepoint", "onedrive"):
            if capability == "sharepoint": request = client.sites.by_site_id(str(args.get("site_id") or "")).drive.root
            else: request = client.users.by_user_id(user).drive.root
            if action == "search": result = await _maybe(request.search_with_q(str(args.get("query") or "")).get())
            elif action == "read": result = await _maybe(request.item_with_path(str(args.get("path") or "")).get())
            elif action == "download_files": result = await _maybe(request.item_with_path(str(args.get("path") or "")).content.get())
            elif action == "upload_files":
                content = args.get("content", "")
                if not isinstance(content, str) or len(content.encode()) > 5 * 1024 * 1024:
                    raise ValueError("upload content must be text no larger than 5 MiB")
                result = await _maybe(request.item_with_path(str(args.get("path") or "")).content.put(content.encode()))
        else:
            raise NotImplementedError(f"SDK mapping required for {capability}.{action}")
        return tool_result({"success": True, "capability": capability, "action": action, "result": safe_result(result)})
    except Exception as exc:
        return tool_error(f"Microsoft 365 {capability} operation failed: {type(exc).__name__}")


def capability_handler(capability: str):
    async def handler(args: dict, *, context=None, ctx=None, **kwargs):
        return await _run(capability, args, ctx or context)
    handler.__name__ = f"handle_microsoft365_{capability}"
    return handler


_SCHEMA = {
    service: {"action": {"type": "string", "enum": list(operations)},
              "id": {"type": "string"}, "site_id": {"type": "string"},
              "path": {"type": "string"}, "query": {"type": "string"},
              "content": {"type": "string"}}
    for service, operations in OPERATIONS.items()
}


def schema(capability: str) -> dict:
    return {"name": f"microsoft365_{capability}",
            "description": f"Microsoft 365 Plugin {capability} operations; writes use the host approval seam when available.",
            "parameters": {"type": "object", "properties": _SCHEMA[capability], "required": ["action"]}}

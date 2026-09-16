"""Tool handlers for the single ``microsoft365`` toolset."""
from __future__ import annotations

import inspect
from typing import Any

from tools.registry import tool_error, tool_result
from .backend import Microsoft365Settings, create_graph_client, preflight, safe_result


def _settings(ctx) -> Microsoft365Settings:
    values = {key: ctx.get_config(key) for key in ("tenant_id", "client_id", "client_secret", "user_id", "capabilities")}
    return Microsoft365Settings.from_mapping(values)


async def _maybe(value):
    return await value if inspect.isawaitable(value) else value


async def handle_preflight(args: dict, *, context=None, ctx=None, **kwargs) -> str:
    return tool_result(preflight(_settings(ctx or context)))


async def _run(capability: str, args: dict, ctx) -> str:
    settings = _settings(ctx)
    if not settings.enabled(capability):
        return tool_error(f"Microsoft 365 capability is disabled: {capability}")
    action = str(args.get("action") or "list").strip().lower()
    client = create_graph_client(settings)
    user = settings.user_id
    try:
        if capability == "outlook":
            request = client.users.by_user_id(user).messages
            if action == "search":
                result = await _maybe(request.get())
            elif action == "read":
                result = await _maybe(request.by_message_id(str(args.get("id"))).get())
            else:
                raise ValueError("outlook action must be search or read")
        elif capability == "calendar":
            request = client.users.by_user_id(user).calendar.events
            if action != "list":
                raise ValueError("calendar action must be list")
            result = await _maybe(request.get())
        elif capability == "sharepoint":
            drive = client.sites.by_site_id(str(args.get("site_id") or "")).drive
            item = drive.root
            if action == "search":
                result = await _maybe(item.search_with_q(str(args.get("query") or "")).get())
            elif action == "read":
                result = await _maybe(item.item_with_path(str(args.get("path") or "")).get())
            elif action == "download":
                result = await _maybe(item.item_with_path(str(args.get("path") or "")).content.get())
            elif action == "upload":
                # Writes are deliberately explicit and remain subject to the host approval policy.
                content = args.get("content", "")
                if not isinstance(content, str) or len(content.encode()) > 5 * 1024 * 1024:
                    raise ValueError("upload content must be text no larger than 5 MiB")
                result = await _maybe(item.item_with_path(str(args.get("path") or "")).content.put(content.encode()))
            else:
                raise ValueError("sharepoint action must be search, read, upload, or download")
        elif capability == "teams":
            request = client.users.by_user_id(user).chats
            if action != "list":
                raise ValueError("teams action must be list")
            result = await _maybe(request.get())
        elif capability == "planner":
            request = client.users.by_user_id(user).todo.lists
            if action != "list":
                raise ValueError("planner action must be list")
            result = await _maybe(request.get())
        else:  # pragma: no cover
            raise ValueError(f"unknown capability: {capability}")
        return tool_result({"success": True, "capability": capability, "action": action, "result": safe_result(result)})
    except Exception as exc:
        return tool_error(f"Microsoft 365 {capability} operation failed: {type(exc).__name__}")


def capability_handler(capability: str):
    async def handler(args: dict, *, context=None, ctx=None, **kwargs):
        return await _run(capability, args, ctx or context)
    handler.__name__ = f"handle_microsoft365_{capability}"
    return handler


_SCHEMA = {
    "outlook": {"action": {"type": "string", "enum": ["search", "read"]}, "id": {"type": "string"}},
    "sharepoint": {"action": {"type": "string", "enum": ["search", "read", "upload", "download"]}, "site_id": {"type": "string"}, "path": {"type": "string"}, "query": {"type": "string"}, "content": {"type": "string"}},
    "calendar": {"action": {"type": "string", "enum": ["list"]}},
    "teams": {"action": {"type": "string", "enum": ["list"]}},
    "planner": {"action": {"type": "string", "enum": ["list"]}},
}


def schema(capability: str) -> dict:
    return {"name": f"microsoft365_{capability}", "description": f"Microsoft 365 Plugin {capability} operations (writes require Hermes approval).", "parameters": {"type": "object", "properties": _SCHEMA[capability], "required": ["action"]}}

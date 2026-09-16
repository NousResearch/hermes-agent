"""Real Microsoft Graph SDK request-builder operations for Microsoft 365."""
from __future__ import annotations
import importlib, inspect
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
from tools.registry import tool_error, tool_result
from .backend import CAPABILITIES, OPERATIONS, WRITE_OPERATIONS, Microsoft365Settings, create_graph_client, preflight, safe_result

async def _maybe(value): return await value if inspect.isawaitable(value) else value

def _model(name: str, **values):
    """Construct a generated msgraph model (never a raw HTTP payload)."""
    module = "_".join(__import__('re').sub(r"(?<!^)(?=[A-Z])", "_", name).lower().split("_"))
    module_path = {
        "SendMailPostRequestBody": "msgraph.generated.users.item.send_mail.send_mail_post_request_body",
        "QueryPostRequestBody": "msgraph.generated.search.query.query_post_request_body",
    }.get(name, f"msgraph.generated.models.{module}")
    cls = getattr(importlib.import_module(module_path), name)
    return cls(**values)

def _body(content: str):
    try:
        body_type = getattr(importlib.import_module("msgraph.generated.models.body_type"), "BodyType").Text
    except (ImportError, AttributeError): body_type = None
    values = {"content": content};
    if body_type is not None: values["content_type"] = body_type
    return _model("ItemBody", **values)

def _approved(capability, action, args):
    try:
        from tools.approval import request_tool_approval
        result = request_tool_approval(f"microsoft365_{capability}", f"Microsoft 365 {action}: external side effect", rule_key=f"microsoft365.{capability}.{action}")
        return bool(result.get("approved")) if isinstance(result, dict) else result in {"once", "session", "always", "smart_approve"}
    except Exception:
        return False

async def handle_preflight(args: dict, *, context=None, ctx=None, **kwargs): return tool_result(preflight(_settings(ctx or context)))
def _settings(ctx):
    return Microsoft365Settings.from_mapping({k: ctx.get_config(k) for k in ("tenant_id","client_id","client_secret","user_id","capabilities")})

def _required(args, key, label=None):
    value = str(args.get(key) or "").strip()
    if not value: raise ValueError(f"{label or key} is required")
    return value

async def _drive(client, capability, user, args):
    if capability == "sharepoint":
        drive = await _maybe(client.sites.by_site_id(_required(args, "site_id")).drive.get())
    else:
        drive = await _maybe(client.users.by_user_id(user).drive.get())
    drive_id = _required({"id": getattr(drive, "id", None)}, "id")
    return client.drives.by_drive_id(drive_id).root, drive_id


def _drive_path(root, drive_id: str, path: str):
    encoded = quote(path.strip("/"), safe="/")
    return root.with_url(f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{encoded}:")

async def _run(capability: str, args: dict, ctx) -> str:
    settings = _settings(ctx); action = str(args.get("action") or "").strip().lower()
    if capability not in CAPABILITIES: return tool_error(f"Unknown Microsoft 365 capability: {capability}")
    if action not in OPERATIONS[capability]: return tool_error(f"Unknown {capability} operation: {action or '<missing>'}")
    if action not in settings.operations(capability): return tool_error(f"Microsoft 365 operation is disabled: {capability}.{action}")
    if action in WRITE_OPERATIONS and not _approved(capability, action, args): return tool_result({"required_confirmation": True, "capability": capability, "action": action, "message": "Explicit approval is required; no Microsoft 365 side effect was performed."})
    client = create_graph_client(settings); user = settings.user_id
    try:
        if capability == "outlook":
            messages = client.users.by_user_id(user).messages
            if action == "search": result = await _maybe(messages.get())
            elif action == "read": result = await _maybe(messages.by_message_id(_required(args,"id")).get())
            else:
                if action == "send" and args.get("id"):
                    result = await _maybe(messages.by_message_id(_required(args, "id")).send.post(None))
                elif action == "send":
                    body = _model("SendMailPostRequestBody", message=_model("Message", subject=args.get("subject"), body=_body(str(args.get("body") or "")), to_recipients=[_model("Recipient", email_address=_model("EmailAddress", address=_required(args, "to"))) ]), save_to_sent_items=True)
                    result = await _maybe(client.users.by_user_id(user).send_mail.post(body))
                else:
                    message = _model("Message", subject=args.get("subject"), body=_body(str(args.get("body") or "")), to_recipients=[_model("Recipient", email_address=_model("EmailAddress", address=_required(args,"to")))])
                    result = await _maybe(messages.post(message))
        elif capability == "calendar":
            events = client.users.by_user_id(user).calendar.events
            if action == "search": result = await _maybe(events.get())
            else:
                event = _model("Event", subject=args.get("subject"), body=_body(str(args.get("body") or "")), start=_model("DateTimeTimeZone", date_time=_required(args,"start"), time_zone=args.get("time_zone") or "UTC"), end=_model("DateTimeTimeZone", date_time=_required(args,"end"), time_zone=args.get("time_zone") or "UTC"))
                if action == "create_events": result = await _maybe(events.post(event))
                else: result = await _maybe(events.by_event_id(_required(args,"id")).patch(event))
        elif capability in ("sharepoint", "onedrive"):
            root, drive_id = await _drive(client, capability, user, args)
            if action == "search": result = await _maybe(client.drives.by_drive_id(drive_id).search_with_q(_required(args, "query")).get())
            else:
                item = _drive_path(root, drive_id, _required(args, "path"))
                if action == "read": result = await _maybe(item.get())
                elif action == "download_files": result = await _maybe(item.content.get())
                else: result = await _maybe(item.content.put(args.get("content", b"")))
        elif capability == "teams":
            if action == "list_teams": result = await _maybe(client.users.by_user_id(user).joined_teams.get())
            elif action == "list_channels": result = await _maybe(client.teams.by_team_id(_required(args,"team_id")).channels.get())
            elif action == "search_messages":
                entity_type = getattr(importlib.import_module("msgraph.generated.models.entity_type"), "EntityType").ChatMessage
                query = _model("SearchRequest", query=_model("SearchQuery", query_string=_required(args, "query")), entity_types=[entity_type])
                result = await _maybe(client.search.query.post(_model("QueryPostRequestBody", requests=[query])))
            else: result = await _maybe(client.teams.by_team_id(_required(args,"team_id")).channels.by_channel_id(_required(args,"channel_id")).messages.post(_model("ChatMessage", body=_body(str(args.get("body") or "")))) )
        else:
            todo = client.users.by_user_id(user).todo
            if action == "list_task_lists": result = await _maybe(todo.lists.get())
            else:
                lists = todo.lists; list_id = _required(args,"list_id")
                tasks = lists.by_todo_task_list_id(list_id).tasks
                if action == "search": result = await _maybe(tasks.get())
                elif action == "read": result = await _maybe(tasks.by_todo_task_id(_required(args,"id")).get())
                else:
                    task = _model("TodoTask", title=args.get("title"), body=_body(str(args.get("body") or "")))
                    result = await _maybe(tasks.post(task)) if action == "create_tasks" else await _maybe(tasks.by_todo_task_id(_required(args,"id")).patch(task))
        return tool_result({"success": True, "capability": capability, "action": action, "result": safe_result(result)})
    except Exception as exc: return tool_error(f"Microsoft 365 {capability} operation failed: {type(exc).__name__}")

def capability_handler(capability):
    async def handler(args: dict, *, context=None, ctx=None, **kwargs): return await _run(capability, args, ctx or context)
    handler.__name__ = f"handle_microsoft365_{capability}"; return handler

_COMMON = {"action":{"type":"string","enum":[]},"id":{"type":"string"},"site_id":{"type":"string"},"team_id":{"type":"string"},"channel_id":{"type":"string"},"list_id":{"type":"string"},"path":{"type":"string"},"query":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"},"to":{"type":"string"},"start":{"type":"string"},"end":{"type":"string"},"time_zone":{"type":"string"},"content":{"type":"string"}}
def schema(capability):
    props = {**_COMMON, "action": {"type":"string","enum":list(OPERATIONS[capability])}}
    return {"name":f"microsoft365_{capability}","description":f"Microsoft 365 Plugin {capability} operations; writes require explicit approval.","parameters":{"type":"object","properties":props,"required":["action"]}}

"""Microsoft Graph SDK operations for the unified Microsoft 365 Plugin."""
from __future__ import annotations

import base64
import binascii
import importlib
import inspect
from types import SimpleNamespace
from urllib.parse import quote

from tools.registry import tool_error, tool_result
from .backend import AUTHENTICATION_MODE, CAPABILITIES, MAX_PAGE_SIZE, MAX_TRANSFER_BYTES, Microsoft365Settings, OPERATIONS, create_graph_client, operation_support, preflight, safe_result


def _maybe(value):
    return value if inspect.isawaitable(value) else value


async def _await(value):
    return await value if inspect.isawaitable(value) else value


async def _get(builder, request_configuration=None):
    try:
        return await _await(builder.get(request_configuration=request_configuration))
    except TypeError as exc:
        if "request_configuration" not in str(exc):
            raise
        return await _await(builder.get())


def _model(name: str, **values):
    module = "_".join(__import__("re").sub(r"(?<!^)(?=[A-Z])", "_", name).lower().split("_"))
    module_path = {"SendMailPostRequestBody": "msgraph.generated.users.item.send_mail.send_mail_post_request_body", "QueryPostRequestBody": "msgraph.generated.search.query.query_post_request_body"}.get(name, f"msgraph.generated.models.{module}")
    return getattr(importlib.import_module(module_path), name)(**values)


def _body(content: str):
    try:
        body_type = getattr(importlib.import_module("msgraph.generated.models.body_type"), "BodyType").Text
    except (ImportError, AttributeError):
        body_type = None
    values = {"content": content}
    if body_type is not None:
        values["content_type"] = body_type
    return _model("ItemBody", **values)


def _settings(ctx):
    values = {key: ctx.get_config(key) for key in ("tenant_id", "client_id", "client_secret", "user_id", "capabilities")}
    if not values.get("client_secret"):
        try:
            from agent.secret_scope import get_secret
            values["client_secret"] = get_secret("MICROSOFT365_CLIENT_SECRET", "")
        except (ImportError, RuntimeError):
            values["client_secret"] = ""
    return Microsoft365Settings.from_mapping(values)


def _required(args, key, label=None):
    value = str(args.get(key) or "").strip()
    if not value:
        raise ValueError(f"{label or key} is required")
    return value


def _limit(args):
    try:
        value = int(args.get("limit", 50))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    if not 1 <= value <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
    return value


def _safe_path(path: str) -> str:
    path = _required({"path": path}, "path")
    parts = path.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path must be relative and must not contain traversal or empty segments")
    return "/".join(quote(part, safe="") for part in parts)


def _decode_upload(args):
    encoded = args.get("content_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("content_base64 is required; uploads use bounded base64 bytes")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content_base64 is not valid base64") from exc
    if len(content) > MAX_TRANSFER_BYTES:
        raise ValueError(f"upload exceeds the {MAX_TRANSFER_BYTES}-byte limit")
    content_type = str(args.get("content_type") or "application/octet-stream").strip()
    if not content_type or len(content_type) > 200:
        raise ValueError("content_type must be a non-empty value of at most 200 characters")
    return content, content_type


def _request_configuration(args, *, query=None):
    params = {"$top": _limit(args)}
    if query:
        escaped = query.replace("'", "''")
        params["$filter"] = f"contains(subject,'{escaped}')"
    return SimpleNamespace(query_parameters=params)


async def handle_preflight(args: dict, *, context=None, ctx=None, **kwargs):
    return tool_result(preflight(_settings(ctx or context)))


async def _drive(client, capability, user, args):
    if capability == "sharepoint":
        drive = await _await(client.sites.by_site_id(_required(args, "site_id")).drive.get())
    else:
        drive = await _await(client.users.by_user_id(user).drive.get())
    drive_id = _required({"id": getattr(drive, "id", None)}, "id")
    drive_builder = client.drives.by_drive_id(drive_id)
    return getattr(drive_builder, "root", drive_builder), drive_id


def _drive_item(root, drive_id: str, path: str):
    encoded = _safe_path(path)
    if hasattr(root, "item_with_path"):
        return root.item_with_path(path.strip("/"))
    return root.with_url(f"https://graph.microsoft.com/v1.0/drives/{quote(drive_id, safe='')}/root:/{encoded}:")


def _validate_action(capability, action, args):
    _limit(args) if action in {"search", "list_teams", "list_channels", "list_task_lists", "list_plans", "list_buckets", "list_tasks"} else None
    if action in {"read", "download_files", "upload_files"}:
        if capability in {"sharepoint", "onedrive"}:
            _safe_path(args.get("path"))
        else:
            _required(args, "id")
    if action in {"send", "create_draft"}:
        if not args.get("id"):
            _required(args, "to")
            _required(args, "body")
    if action in {"create_events"}:
        _required(args, "start")
        _required(args, "end")
    if action in {"update_events", "update_tasks"}:
        _required(args, "id")
        if not any(args.get(field) is not None for field in ("subject", "title", "body", "start", "end", "status")):
            raise ValueError("at least one patch field is required")
    if action == "upload_files":
        _decode_upload(args)
    if action in {"search", "search_messages"}:
        _required(args, "query")
    if action in {"list_channels", "send_messages"}:
        _required(args, "team_id")
    if action == "send_messages":
        _required(args, "channel_id")
        _required(args, "body")
    if capability in {"todo", "planner"} and action not in {"list_plans", "list_buckets", "list_tasks"} and action in {"read", "create_tasks", "update_tasks", "search"}:
        _required(args, "list_id" if capability == "todo" else "plan_id")


async def _run(capability: str, args: dict, ctx) -> str:
    settings = _settings(ctx)
    action = str(args.get("action") or "").strip().lower()
    if capability not in CAPABILITIES:
        return tool_error(f"Unknown Microsoft 365 capability: {capability}")
    status = operation_support(AUTHENTICATION_MODE, capability, action)
    if not status.supported:
        return tool_error(f"Microsoft 365 operation unsupported: {capability}.{action}: {status.reason}")
    if action not in settings.operations(capability):
        return tool_error(f"Microsoft 365 operation is disabled: {capability}.{action}")
    try:
        _validate_action(capability, action, args)
    except ValueError as exc:
        return tool_error(f"Invalid Microsoft 365 arguments: {exc}")
    client = create_graph_client(settings)
    user = settings.user_id
    try:
        if capability == "outlook":
            messages = client.users.by_user_id(user).messages
            if action == "search":
                result = await _get(messages, _request_configuration(args, query=args["query"]))
            elif action == "read":
                result = await _await(messages.by_message_id(args["id"]).get())
            elif action == "send" and args.get("id"):
                result = await _await(messages.by_message_id(args["id"]).send.post(None))
            else:
                message = _model("Message", subject=args.get("subject"), body=_body(args["body"]), to_recipients=[_model("Recipient", email_address=_model("EmailAddress", address=args["to"]))])
                if action == "send":
                    result = await _await(client.users.by_user_id(user).send_mail.post(_model("SendMailPostRequestBody", message=message, save_to_sent_items=True)))
                else:
                    result = await _await(messages.post(message))
        elif capability == "calendar":
            events = client.users.by_user_id(user).calendar.events
            if action == "search":
                result = await _get(events, _request_configuration(args, query=args["query"]))
            else:
                values = {"subject": args.get("subject"), "body": _body(args["body"]) if args.get("body") is not None else None}
                if args.get("start") is not None: values["start"] = _model("DateTimeTimeZone", date_time=args["start"], time_zone=args.get("time_zone") or "UTC")
                if args.get("end") is not None: values["end"] = _model("DateTimeTimeZone", date_time=args["end"], time_zone=args.get("time_zone") or "UTC")
                event = _model("Event", **{key: value for key, value in values.items() if value is not None})
                result = await _await(events.post(event) if action == "create_events" else events.by_event_id(args["id"]).patch(event))
        elif capability in {"sharepoint", "onedrive"}:
            root, drive_id = await _drive(client, capability, user, args)
            if action == "search":
                result = await _get(client.drives.by_drive_id(drive_id).search_with_q(args["query"]), _request_configuration(args))
            else:
                item = _drive_item(root, drive_id, args["path"])
                if action == "read": result = await _await(item.get())
                elif action == "download_files":
                    result = await _await(item.content.get())
                    if not isinstance(result, (bytes, bytearray)) or len(result) > MAX_TRANSFER_BYTES: raise ValueError("download exceeds the bounded binary transfer limit")
                    result = {"content_base64": base64.b64encode(result).decode("ascii"), "content_type": args.get("content_type") or "application/octet-stream", "size": len(result), "path": args["path"]}
                else:
                    content, _ = _decode_upload(args)
                    result = await _await(item.content.put(content))
        elif capability == "teams":
            if action == "list_teams": result = await _get(client.users.by_user_id(user).joined_teams, _request_configuration(args))
            elif action == "list_channels": result = await _get(client.teams.by_team_id(args["team_id"]).channels, _request_configuration(args))
            elif action == "search_messages":
                entity_type = getattr(importlib.import_module("msgraph.generated.models.entity_type"), "EntityType").ChatMessage
                query = _model("SearchRequest", query=_model("SearchQuery", query_string=args["query"]), entity_types=[entity_type])
                result = await _await(client.search.query.post(_model("QueryPostRequestBody", requests=[query])))
            else:
                result = await _await(client.teams.by_team_id(args["team_id"]).channels.by_channel_id(args["channel_id"]).messages.post(_model("ChatMessage", body=_body(args["body"]))))
        elif capability == "todo":
            todo = client.users.by_user_id(user).todo
            if action == "list_task_lists": result = await _get(todo.lists, _request_configuration(args))
            else:
                tasks = todo.lists.by_todo_task_list_id(args["list_id"]).tasks
                if action == "search": result = await _get(tasks, _request_configuration(args, query=args["query"]))
                elif action == "read": result = await _await(tasks.by_todo_task_id(args["id"]).get())
                else:
                    values = {"title": args.get("title"), "body": _body(args["body"]) if args.get("body") is not None else None}
                    task = _model("TodoTask", **{key: value for key, value in values.items() if value is not None})
                    result = await _await(tasks.post(task) if action == "create_tasks" else tasks.by_todo_task_id(args["id"]).patch(task))
        else:
            planner = client.planner
            if action == "list_plans": result = await _get(planner.plans, _request_configuration(args))
            elif action == "list_buckets": result = await _get(planner.plans.by_planner_plan_id(args["plan_id"]).buckets, _request_configuration(args))
            elif action == "list_tasks": result = await _get(planner.plans.by_planner_plan_id(args["plan_id"]).tasks, _request_configuration(args))
            else:
                tasks = planner.tasks.by_planner_task_id(args["id"]) if action in {"read", "update_tasks"} else planner.plans.by_planner_plan_id(args["plan_id"]).tasks
                result = await _await(tasks.get() if action == "read" else tasks.patch(SimpleNamespace(title=args.get("title"))) if action == "update_tasks" else tasks.post(SimpleNamespace(title=args.get("title"))))
        return tool_result({"success": True, "capability": capability, "action": action, "result": safe_result(result)})
    except Exception as exc:
        return tool_error(f"Microsoft 365 {capability} operation failed: {type(exc).__name__}")


def capability_handler(capability):
    async def handler(args: dict, *, context=None, ctx=None, **kwargs):
        return await _run(capability, args, ctx or context)
    handler.__name__ = f"handle_microsoft365_{capability}"
    return handler

_COMMON = {"action": {"type": "string", "enum": []}, "id": {"type": "string"}, "site_id": {"type": "string"}, "team_id": {"type": "string"}, "channel_id": {"type": "string"}, "list_id": {"type": "string"}, "plan_id": {"type": "string"}, "path": {"type": "string"}, "query": {"type": "string"}, "subject": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}, "to": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, "time_zone": {"type": "string"}, "content_base64": {"type": "string", "description": "Base64-encoded binary upload, at most 10 MiB"}, "content_type": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PAGE_SIZE}}


def schema(capability, operations=None):
    selected = list(operations if operations is not None else OPERATIONS[capability])
    selected = [op for op in selected if operation_support(AUTHENTICATION_MODE, capability, op).supported]
    props = {**_COMMON, "action": {"type": "string", "enum": selected}}
    return {"name": f"microsoft365_{capability}", "description": f"Microsoft 365 Plugin {capability} operations; writes require explicit approval.", "parameters": {"type": "object", "properties": props, "required": ["action"]}}

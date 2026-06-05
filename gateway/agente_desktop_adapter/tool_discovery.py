"""
tool_discovery.py

Stable home for all Agente Desktop tool discovery, IPC bridge proxy,
and the /api/tools + /api/tools/{name} route handlers (plus toolset
registration for the "agente-desktop" tools that proxy back to Electron).

Per hermes-upstream-compatibility-plan: this is the ONLY place with the
IPC proxy/dispatch logic (AGENTE_TOOL_PORT/SECRET, POST /dispatch/{tool},
result envelopes, error handling). The bridge is preserved exactly.

The register(app, adapter=None) wires both:
- global tool registrations (so "agente-desktop" toolset works)
- the Desktop-facing HTTP routes on the aiohttp app.

All Agente-specific knowledge is isolated here; core api_server.py
contains only the call to register_agente_desktop_routes.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- preserved exact IPC bridge (from plugins/agente_desktop/__init__.py) ---

TOOLSET = "agente-desktop"
_REQUEST_TIMEOUT_SECONDS = 25

_TOOL_EMOJIS: dict[str, str] = {
    "list_whatsapp_accounts": "📱",
    "list_recent_messages": "💬",
    "create_ticket": "🎫",
    "move_ticket": "🔄",
    "list_tickets": "📋",
    "save_triage_instructions": "📝",
    "request_approval": "✋",
    "get_office_context": "🏢",
    # desktop-202605-161 additions
    "upsert_client": "👤",
    "query_client": "🔍",
    "link_ticket_to_client": "🔗",
    "link_document_to_client": "📎",
    "update_office_context": "🏢",
    "list_tools": "🛠️",
    "list_workflows": "📂",
    "inspect_workflow": "🔬",
    "start_workflow_run": "▶️",
    "get_run_status": "📊",
    "resume_paused_run": "⏯️",
    # hermes-202606-002 additions
    "save_workflow_rule": "📝",
    "evaluate_triage_rules": "🧭",
}

# Schemas moved here from plugins/agente_desktop/schemas.py (exact copy for parity)
# Mirrors electron/main/hermes-tools/*.ts in the desktop side.
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_whatsapp_accounts": {
        "name": "list_whatsapp_accounts",
        "description": (
            "Returns the list of paired WhatsApp accounts (connectors) configured "
            "in the workspace, including account ID, label, phone number, "
            "governance lane, and current GOWA connection status."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "list_recent_messages": {
        "name": "list_recent_messages",
        "description": (
            "Returns recent WhatsApp messages for a given account. Use this to "
            "summarize incoming messages and decide if any need a ticket created."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The WhatsApp account/connector ID from list_whatsapp_accounts.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of chats to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["account_id"],
        },
    },
    "create_ticket": {
        "name": "create_ticket",
        "description": (
            'Creates a new ticket in the office board. Status defaults to '
            '"חדש" (pending). Use source="whatsapp" and source_id=<message_id> '
            "when creating from a WhatsApp message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Ticket title (preferably in Hebrew)."},
                "body": {"type": "string", "description": "Optional detailed description."},
                "status": {
                    "type": "string",
                    "description": 'Status: "חדש" (default), "בטיפול", "ממתין לאישור", "done".',
                    "default": "חדש",
                },
                "assignee": {
                    "type": "string",
                    "description": "Optional agent slug or name to assign the ticket to.",
                },
                "source": {
                    "type": "string",
                    "description": 'Source of the ticket, e.g. "whatsapp" or "manual".',
                },
                "source_id": {
                    "type": "string",
                    "description": "Source message ID for back-linking (e.g. WhatsApp message ID).",
                },
            },
            "required": ["title"],
        },
    },
    "move_ticket": {
        "name": "move_ticket",
        "description": (
            'Updates a ticket status on the board. Valid statuses: '
            '"חדש" / "pending", "בטיפול" / "in_progress", "ממתין לאישור" / '
            '"pending_review", "done", "cancelled".'
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "UUID of the ticket to move."},
                "to_status": {
                    "type": "string",
                    "description": "Target status — use Hebrew or English key.",
                },
            },
            "required": ["ticket_id", "to_status"],
        },
    },
    "list_tickets": {
        "name": "list_tickets",
        "description": (
            "Returns tickets from the office board. Filter by status (Hebrew or "
            'English), and limit the count. Use this to answer "מה מחכה לי?" or '
            '"show me open tasks".'
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": (
                        'Optional status filter: "חדש", "בטיפול", "ממתין לאישור", '
                        '"done", or omit for all open.'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tickets to return (default 20).",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
    "save_triage_instructions": {
        "name": "save_triage_instructions",
        "description": (
            "(DEPRECATED — use save_workflow_rule for new rules.) "
            "Saves free-text operator triage instructions to the workspace "
            "settings. Hermes will read these instructions at the start of "
            "every session to personalize its behavior. Retained for "
            "backwards compatibility with legacy office profiles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The triage instructions text (Hebrew preferred).",
                },
            },
            "required": ["text"],
        },
    },
    "request_approval": {
        "name": "request_approval",
        "description": (
            "Proposes an action that requires operator approval before "
            "execution. Creates a pending_review ticket visible on the kanban "
            "board. The operator approves or rejects it from the board."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_description": {
                    "type": "string",
                    "description": (
                        "Human-readable description of the action requiring "
                        "approval (Hebrew preferred)."
                    ),
                },
                "payload": {
                    "type": "object",
                    "description": (
                        "Structured payload describing the action (tool name, "
                        "args, etc.)."
                    ),
                },
            },
            "required": ["action_description"],
        },
    },
    "get_office_context": {
        "name": "get_office_context",
        "description": (
            "Returns the current office persona settings: triage instructions, "
            "office type, hours, and team configuration. Read this once at the "
            "start of a session."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── 10 additional tools added in desktop-202605-161 fix ──────────────────
    "upsert_client": {
        "name": "upsert_client",
        "description": (
            "Idempotently create or update a client by identity (phone, JID, "
            "email, or name). Does not override existing verified identity "
            "records. Calling twice with the same identity returns the same "
            "client_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "identity": {
                    "type": "object",
                    "description": "Identity to look up or create by.",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["phone", "jid", "email", "name"],
                            "description": "Identity kind.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Identity value.",
                        },
                    },
                    "required": ["kind", "value"],
                },
                "patch": {
                    "type": "object",
                    "description": "Optional fields to set on create or update.",
                    "properties": {
                        "display_name": {
                            "type": "string",
                            "description": "Display name for the client.",
                        },
                        "hebrew_name": {
                            "type": "string",
                            "description": "Hebrew display name.",
                        },
                        "aliases": {
                            "type": "array",
                            "description": "Additional name aliases.",
                            "items": {"type": "string"},
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes about this client.",
                        },
                    },
                },
            },
            "required": ["identity"],
        },
    },
    "query_client": {
        "name": "query_client",
        "description": (
            "Query clients by JID, phone number, email, or display name "
            "(Hebrew partial match). Returns client records with open ticket "
            "counts and last contact timestamps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "jid": {
                    "type": "string",
                    "description": 'WhatsApp JID (e.g. "972500000000@s.whatsapp.net")',
                },
                "phone": {
                    "type": "string",
                    "description": 'Phone number to look up (e.g. "972500000000")',
                },
                "email": {
                    "type": "string",
                    "description": "Email address to look up",
                },
                "name": {
                    "type": "string",
                    "description": "Display name or alias (Hebrew partial match supported)",
                },
            },
            "required": [],
        },
    },
    "link_ticket_to_client": {
        "name": "link_ticket_to_client",
        "description": (
            "Links an existing ticket to a client record. Calling again with "
            "the same ticket_id updates the link (idempotent). Fails with a "
            "structured error if the client or ticket does not exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "UUID of the ticket to link.",
                },
                "client_id": {
                    "type": "string",
                    "description": "UUID of the client to link the ticket to.",
                },
            },
            "required": ["ticket_id", "client_id"],
        },
    },
    "link_document_to_client": {
        "name": "link_document_to_client",
        "description": (
            "Links a document source to a client record. Fails with a "
            "structured error if the client or document does not exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_source_id": {
                    "type": "string",
                    "description": "UUID of the document source to link.",
                },
                "client_id": {
                    "type": "string",
                    "description": "UUID of the client to link the document to.",
                },
            },
            "required": ["document_source_id", "client_id"],
        },
    },
    "update_office_context": {
        "name": "update_office_context",
        "description": (
            "Writes structured Office Twin profile context to canonical local "
            "workspace settings, re-renders the derived office profile mirror, "
            "and reseeds Hermes context from the canonical bundle."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "office_details": {
                    "type": "object",
                    "description": (
                        "Structured office identity details such as name, "
                        "phone, address, type, or size."
                    ),
                    "additionalProperties": True,
                },
                "team_members_op": {
                    "type": "string",
                    "enum": ["replace", "upsert", "remove_by_id"],
                    "description": (
                        "Team member mutation mode. Required when team_members "
                        "is present."
                    ),
                },
                "team_members": {
                    "type": "array",
                    "description": (
                        "Team member records. Each record must include id. "
                        "For remove_by_id, only id is required."
                    ),
                    "items": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {"id": {"type": "string"}},
                        "additionalProperties": True,
                    },
                },
                "work_days": {
                    "type": "array",
                    "description": "Working day identifiers or labels.",
                    "items": {"type": "string"},
                },
                "compliance": {
                    "type": "object",
                    "description": "Structured compliance profile settings.",
                    "additionalProperties": True,
                },
                "preferences": {
                    "type": "object",
                    "description": "Structured office preferences.",
                    "additionalProperties": True,
                },
                "connectors": {
                    "type": "array",
                    "description": "Structured connector profile metadata.",
                    "items": {"type": "object", "additionalProperties": True},
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    "list_tools": {
        "name": "list_tools",
        "description": (
            "Returns the names and descriptions of all Agente desktop tools "
            "available to the assistant in this session. Call this when the "
            "operator asks what tools or capabilities the assistant has."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "list_workflows": {
        "name": "list_workflows",
        "description": "List all available workflow definitions in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "inspect_workflow": {
        "name": "inspect_workflow",
        "description": "Return metadata about a specific workflow definition.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflowId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "ID of the workflow to inspect.",
                },
            },
            "required": ["workflowId"],
        },
    },
    "start_workflow_run": {
        "name": "start_workflow_run",
        "description": "Start a new workflow run for a given ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflowId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "ID of the workflow to run.",
                },
                "ticketId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "ID of the ticket to process.",
                },
                "ticketLabel": {
                    "type": "string",
                    "description": "Current label of the ticket (optional).",
                },
                "runId": {
                    "type": "string",
                    "description": "Optional deterministic run ID.",
                },
            },
            "required": ["workflowId", "ticketId"],
        },
    },
    "get_run_status": {
        "name": "get_run_status",
        "description": "Return the current status of a workflow run.",
        "parameters": {
            "type": "object",
            "properties": {
                "runId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Run ID to query.",
                },
            },
            "required": ["runId"],
        },
    },
    "resume_paused_run": {
        "name": "resume_paused_run",
        "description": "Resume a paused workflow run from the specified step.",
        "parameters": {
            "type": "object",
            "properties": {
                "runId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Run ID to resume.",
                },
                "stepIndex": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Step index to resume from (must match "
                        "pauseHandle.nextStepIndex)."
                    ),
                },
            },
            "required": ["runId", "stepIndex"],
        },
    },
    # ── hermes-202606-002: workflow-rule + triage parity (desktop-202606-437) ──
    "save_workflow_rule": {
        "name": "save_workflow_rule",
        "description": (
            "Saves a structured workflow rule that gates inbound event "
            "processing. Rules match against connector events and trigger "
            "actions like ticket creation. Returns the created rule id and "
            "timestamp. The legacy save_triage_instructions tool still works "
            "for backwards-compat; prefer this tool for new rules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "match_pattern": {
                    "type": "object",
                    "description": (
                        "Matching criteria for inbound connector events. "
                        "Required fields: source (connector id). Optional "
                        "filters: event_type, text_contains, metadata_match."
                    ),
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": (
                                "Connector id the rule applies to "
                                '(e.g. "whatsapp", "local-folder").'
                            ),
                        },
                        "filters": {
                            "type": "object",
                            "description": (
                                "Optional narrow filters applied after the "
                                "connector match."
                            ),
                            "properties": {
                                "event_type": {
                                    "type": "string",
                                    "description": "Exact event type to match.",
                                },
                                "text_contains": {
                                    "type": "string",
                                    "description": (
                                        "Case-insensitive substring to look "
                                        "for in the event text payload."
                                    ),
                                },
                                "metadata_match": {
                                    "type": "object",
                                    "description": (
                                        "Exact key-value pairs the event "
                                        "metadata must satisfy."
                                    ),
                                    "additionalProperties": True,
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                    "required": ["source"],
                    "additionalProperties": False,
                },
                "action": {
                    "type": "string",
                    "description": (
                        "The action to take when the rule matches "
                        '(e.g. "create_ticket").'
                    ),
                },
                "target": {
                    "type": "object",
                    "description": (
                        "Optional per-action target payload (e.g. ticket "
                        "template defaults)."
                    ),
                    "additionalProperties": True,
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Human-readable rule description for audit and "
                        "list-tools surfaces."
                    ),
                },
                "enabled": {
                    "type": "boolean",
                    "description": (
                        "Whether the rule is active. Defaults to true."
                    ),
                },
            },
            "required": ["match_pattern", "action", "description"],
            "additionalProperties": False,
        },
    },
    "evaluate_triage_rules": {
        "name": "evaluate_triage_rules",
        "description": (
            "Evaluates an inbound event against the operator's triage rules "
            "(office/triage-rules.md) and returns a structured decision: "
            "create_ticket | drop | draft_reply | escalate. Fail-open: any "
            "internal error returns create_ticket so no real customer message "
            "is silently dropped."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        'Event source, e.g. "whatsapp", "email".'
                    ),
                },
                "type": {
                    "type": "string",
                    "description": (
                        'Event type, e.g. "whatsapp_message".'
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Free-text payload of the event (Hebrew preferred)."
                    ),
                },
                "metadata": {
                    "type": "object",
                    "description": (
                        "Arbitrary structured metadata about the event."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["source", "type"],
        },
    },
}


def _ipc_endpoint(tool_name: str) -> str | None:
    port = os.environ.get("AGENTE_TOOL_PORT")
    if not port:
        return None
    return f"http://127.0.0.1:{port}/dispatch/{tool_name}"


def _proxy_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Forward a tool call to the Electron main process over loopback HTTP.

    Returns the unwrapped ``result`` payload on success or an ``{"error":
    ...}`` dict on any failure. Never raises — every path returns a dict so
    the AIAgent can fold the response into the next conversation turn.
    """
    endpoint = _ipc_endpoint(tool_name)
    secret = os.environ.get("AGENTE_TOOL_SECRET")
    if endpoint is None or not secret:
        return {"error": "agente_tool_ipc_not_configured"}

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(args).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return {
            "error": "agente_tool_http_error",
            "status": exc.code,
            "body": exc.read().decode("utf-8", "replace"),
        }
    except Exception as exc:  # noqa: BLE001 — defensive; never raise from a tool handler
        return {"error": "agente_tool_exception", "message": str(exc)}

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return {"error": "agente_tool_invalid_json", "raw": body[:512]}

    if not isinstance(envelope, dict):
        return {"error": "agente_tool_invalid_envelope", "raw": body[:512]}

    if envelope.get("ok") is True:
        result = envelope.get("result")
        return result if isinstance(result, dict) else {"result": result}

    return {
        "error": envelope.get("error", "agente_tool_unknown_error"),
        "details": envelope.get("details"),
    }


def _check_available() -> bool:
    return bool(os.environ.get("AGENTE_TOOL_PORT") and os.environ.get("AGENTE_TOOL_SECRET"))


def _make_handler(tool_name: str):
    """Bind the tool name into a closure so each registered handler has its
    own dispatch target. Avoids the late-binding-loop pitfall.
    """

    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        return json.dumps(_proxy_call(tool_name, args), ensure_ascii=False)

    return handler


def _json_safe_metadata(value: Any) -> Any:
    """Coerce parsed YAML metadata into values accepted by JSON responses.
    (Duplicated from api_server for desktop compat isolation; pure func.)
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


# --- route handlers (moved from api_server.py _handle_list_tools / _handle_dispatch_tool) ---

async def _handle_list_tools(request: "web.Request", adapter: Any) -> "web.Response":
    """GET /api/tools — Desktop-facing Hermes tool discovery (compat shim)."""
    from aiohttp import web

    auth_err = adapter._check_auth(request)
    if auth_err is not None:
        return auth_err

    try:
        registry = adapter._ensure_tool_registry_loaded()
        definitions = registry.get_definitions(set(registry.get_all_tool_names()), quiet=True)
        tools: List[Dict[str, Any]] = []
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            function = definition.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "")
            if not name:
                continue
            entry = registry.get_entry(name)
            item: Dict[str, Any] = {
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": _json_safe_metadata(function.get("parameters") or {}),
                "definition": _json_safe_metadata(definition),
            }
            if entry is not None:
                item["toolset"] = str(entry.toolset or "")
                if entry.label_he:
                    item["label_he"] = str(entry.label_he)
                if entry.category:
                    item["category"] = str(entry.category)
            tools.append(item)
        return web.json_response({"tools": tools})
    except Exception as exc:
        logger.exception("[api_server] tool discovery failed: %s", exc)
        return web.json_response(
            {"error": {"message": "Failed to discover tools", "code": "tools_discovery_failed"}},
            status=500,
        )


async def _handle_dispatch_tool(request: "web.Request", adapter: Any) -> "web.Response":
    """POST /api/tools/{tool_name} — Desktop-facing direct tool dispatch (compat shim)."""
    from aiohttp import web

    auth_err = adapter._check_auth(request)
    if auth_err is not None:
        return auth_err

    tool_name = str(request.match_info.get("tool_name") or "").strip()
    if not tool_name:
        return web.json_response(
            {"error": {"message": "tool_name required", "code": "tool_name_required"}},
            status=400,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "code": "invalid_json"}},
            status=400,
        )
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return web.json_response(
            {"error": {"message": "JSON body must be an object", "code": "invalid_json"}},
            status=400,
        )

    arguments = body.get("arguments")
    if arguments is None:
        arguments = body.get("args")
    if arguments is None:
        arguments = {k: v for k, v in body.items() if k != "tool_name"}
    if not isinstance(arguments, dict):
        return web.json_response(
            {"error": {"message": "Tool arguments must be an object", "code": "invalid_arguments"}},
            status=400,
        )

    try:
        registry = adapter._ensure_tool_registry_loaded()
        if registry.get_entry(tool_name) is None:
            return web.json_response(
                {"error": {"message": f"Unknown tool: {tool_name}", "code": "unknown_tool"}},
                status=404,
            )

        # normalize for desktop arg shapes (e.g. save_workflow/create_routine)
        from .workflow_routine_bridge import _normalize_desktop_tool_args

        normalized_args = _normalize_desktop_tool_args(tool_name, arguments)
        from model_tools import handle_function_call

        raw_result = handle_function_call(
            tool_name,
            normalized_args,
            task_id=f"api-tools-{uuid.uuid4().hex}",
        )
        try:
            result = json.loads(raw_result)
        except Exception:
            result = {"result": raw_result}
        return web.json_response(_json_safe_metadata(result))
    except Exception as exc:
        logger.exception("[api_server] tool dispatch failed for %s: %s", tool_name, exc)
        return web.json_response(
            {"error": {"message": f"Tool execution failed: {exc}", "code": "tool_execution_failed"}},
            status=500,
        )


# --- public register (orchestrated from __init__) ---

def register(app: Any, adapter: Any = None) -> None:
    """Register Desktop toolset + discovery/dispatch routes.

    Called from api_server.py (the single allowed integration point).
    Also registers the agente-desktop proxy tools into the global registry
    (replaces the old plugins/agente_desktop/__init__.py register path).
    """
    if adapter is None:
        adapter = getattr(app, "get", lambda k: None)("api_server_adapter") if hasattr(app, "get") else None
        if adapter is None and hasattr(app, "__getitem__"):
            try:
                adapter = app["api_server_adapter"]
            except Exception:
                adapter = None

    if adapter is None:
        logger.debug("agente_desktop_adapter.tool_discovery: no api_server_adapter; skipping route+tool reg")
        return

    # 1. Ensure the proxy tools are registered (so enabled_toolsets=["agente-desktop"] works)
    #    Use direct registry to avoid plugin manager tracking (this is the adapter path).
    try:
        from tools.registry import registry as _tool_registry
        for tool_name, schema in TOOL_SCHEMAS.items():
            _tool_registry.register(
                name=tool_name,
                toolset=TOOLSET,
                schema=schema,
                handler=_make_handler(tool_name),
                check_fn=_check_available,
                description=schema.get("description", ""),
                emoji=_TOOL_EMOJIS.get(tool_name, "🔌"),
            )
        logger.info(
            "agente_desktop_adapter registered %d proxy tools under toolset %r (via thin adapter)",
            len(TOOL_SCHEMAS),
            TOOLSET,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("agente_desktop_adapter tool registration failed: %s", exc)

    # 2. Add the /api/tools discovery + dispatch routes (Desktop compat).
    #    These were previously ad-hoc in api_server.py.
    try:
        from aiohttp import web
    except ImportError:
        logger.warning("aiohttp not available; cannot register /api/tools desktop compat routes")
        return

    # Use closures so handlers can close over the concrete adapter instance (for _check_auth etc).
    async def list_tools(req: "web.Request") -> "web.Response":
        return await _handle_list_tools(req, adapter)

    async def dispatch_tool(req: "web.Request") -> "web.Response":
        return await _handle_dispatch_tool(req, adapter)

    # Idempotent add: check if already present (in case of reloads).
    existing_paths = {r.path for r in app.router.routes()}
    if "/api/tools" not in existing_paths:
        app.router.add_get("/api/tools", list_tools)
    if "/api/tools/{tool_name}" not in existing_paths:
        app.router.add_post("/api/tools/{tool_name}", dispatch_tool)

    logger.debug("agente_desktop_adapter.tool_discovery: /api/tools routes registered")

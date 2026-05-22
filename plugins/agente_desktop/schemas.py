"""OpenAI function-schema definitions for the 18 Agente desktop tools.

Mirrors `electron/main/hermes-tools/*.ts` in agente-dev/agente-desktop. Keep in
sync: a desktop-side regression test compares this dict against the TS source.
"""

from __future__ import annotations

from typing import Any

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
            "Saves operator triage instructions to the workspace settings. "
            "Hermes will read these instructions at the start of every session "
            "to personalize its behavior."
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
}

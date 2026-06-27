"""Tool schema for the local OpenClaw bridge."""

OPENCLAW_DELEGATE = {
    "name": "openclaw_delegate",
    "description": (
        "Delegate an approved mock-safe dry-run task to the local OpenClaw "
        "hermes-bridge plugin. Use only when the user explicitly asks Hermes "
        "to have OpenClaw handle a task and the request is a dry-run. V1 only "
        "supports taskId='tasks.organize_today' or taskId='agents.ask_team' and performs no external side effects."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {
                "type": "string",
                "enum": ["tasks.organize_today", "agents.ask_team"],
                "description": "Approved OpenClaw bridge task template. Use agents.ask_team only for dry-run OpenClaw team delegation.",
            },
            "intent": {
                "type": "string",
                "description": "The user's original delegation request.",
            },
            "dryRun": {
                "type": "boolean",
                "description": "Must be true. Non-dry-run execution is blocked in v1.",
            },
            "allowedTools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Must be an empty array in v1.",
            },
            "input": {
                "type": "object",
                "description": "Task input passed to OpenClaw. For tasks.organize_today, include request. For agents.ask_team, include team and question.",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The user's original request.",
                    },
                    "team": {
                        "type": "string",
                        "description": "OpenClaw team name for agents.ask_team. Use openclaw in v1.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The dry-run question for the OpenClaw agent team.",
                    },
                },
            },
            "requestId": {
                "type": "string",
                "description": "Optional idempotency key for the OpenClaw bridge request.",
            },
        },
        "required": ["taskId", "intent", "dryRun", "allowedTools", "input"],
    },
}

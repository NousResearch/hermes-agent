"""Tool schemas for mq9 Hermes plugin."""

MQ9_REGISTER_SELF = {
    "name": "mq9_register_self",
    "description": (
        "Register this Hermes instance into mq9 agent registry. "
        "Creates mailbox if needed and stores AgentCard for discovery."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Override agent name for this registration.",
            },
            "mailbox": {
                "type": "string",
                "description": "Override mailbox name/address for this registration.",
            },
            "description": {
                "type": "string",
                "description": "Override agent description in AgentCard.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for discovery filtering.",
            },
            "ensure_runtime": {
                "type": "boolean",
                "description": "Start passive runtime before registration (default true).",
            },
        },
    },
}

MQ9_UNREGISTER_SELF = {
    "name": "mq9_unregister_self",
    "description": (
        "Unregister this Hermes instance from mq9 agent registry. "
        "By default unregisters all names tracked by runtime."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Optional explicit agent name to unregister.",
            },
        },
    },
}

MQ9_DISCOVER = {
    "name": "mq9_discover",
    "description": (
        "Discover remote agents from mq9 registry by natural-language query, "
        "and normalize mailbox info for direct mq9 calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query, e.g. 'Python HTTP server'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of agents to return (default 10).",
                "minimum": 1,
                "maximum": 100,
            },
            "prefer_name": {
                "type": "string",
                "description": "If set, rank exact-name match first.",
            },
        },
    },
}

MQ9_CALL = {
    "name": "mq9_call",
    "description": (
        "Send a task to a remote agent mailbox and wait for callback reply. "
        "If target_mailbox is missing, plugin will try mq9_discover(query)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_mailbox": {
                "type": "string",
                "description": "Target mailbox address, e.g. hermes.b.python.inbox",
            },
            "query": {
                "type": "string",
                "description": "Discover query used when target_mailbox is not provided.",
            },
            "prefer_name": {
                "type": "string",
                "description": "Preferred agent name when discover is used.",
            },
            "message": {
                "description": "Task payload to send. Object is recommended; string is allowed.",
            },
            "from_agent": {
                "type": "string",
                "description": "Caller agent name in envelope.",
            },
            "timeout_s": {
                "type": "number",
                "description": "Timeout seconds for reply wait (default 25).",
                "minimum": 1,
                "maximum": 300,
            },
        },
        "required": ["message"],
    },
}

MQ9_STATUS = {
    "name": "mq9_status",
    "description": "Return mq9 plugin runtime status and effective config.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

"""Tool schemas exposed by the Memori Hermes memory provider."""

from __future__ import annotations

SIGNALS = [
    "commit",
    "discovery",
    "failure",
    "inference",
    "pattern",
    "result",
    "update",
    "verification",
]

SOURCES = [
    "constraint",
    "decision",
    "execution",
    "fact",
    "insight",
    "instruction",
    "status",
    "strategy",
    "task",
]

MEMORI_RECALL_SCHEMA = {
    "name": "memori_recall",
    "description": (
        "Search Memori long-term memory. Use before saying you do not know the "
        "user, their preferences, prior project decisions, or past session context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query. Use real words, not regex.",
            },
            "date_start": {
                "type": "string",
                "description": "UTC ISO 8601 lower bound for memory creation time.",
            },
            "date_end": {
                "type": "string",
                "description": "UTC ISO 8601 upper bound for memory creation time.",
            },
            "project_id": {
                "type": "string",
                "description": "Override the configured Memori project only when requested.",
            },
            "session_id": {
                "type": "string",
                "description": "Filter to a specific session. Requires project_id.",
            },
            "signal": {
                "type": "string",
                "description": "Filter to a specific fact signal.",
                "enum": SIGNALS,
            },
            "source": {
                "type": "string",
                "description": "Filter to a specific source origin.",
                "enum": SOURCES,
            },
        },
        "required": ["query"],
    },
}

MEMORI_RECALL_SUMMARY_SCHEMA = {
    "name": "memori_recall_summary",
    "description": (
        "Fetch summarized Memori context for status updates, daily briefs, "
        "project overviews, and high-level summaries of prior sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "date_start": {
                "type": "string",
                "description": "UTC ISO 8601 lower bound for summaries.",
            },
            "date_end": {
                "type": "string",
                "description": "UTC ISO 8601 upper bound for summaries.",
            },
            "project_id": {
                "type": "string",
                "description": "Override the configured Memori project only when requested.",
            },
            "session_id": {
                "type": "string",
                "description": "Filter to a specific session. Requires project_id.",
            },
        },
    },
}

MEMORI_QUOTA_SCHEMA = {
    "name": "memori_quota",
    "description": "Check the configured Memori account quota and memory usage.",
    "parameters": {"type": "object", "properties": {}},
}

MEMORI_SIGNUP_SCHEMA = {
    "name": "memori_signup",
    "description": "Request a Memori API key signup email for a user.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "Email address to receive Memori signup instructions.",
            }
        },
        "required": ["email"],
    },
}

MEMORI_FEEDBACK_SCHEMA = {
    "name": "memori_feedback",
    "description": "Send integration feedback to the Memori team.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Feedback, bug report, or product suggestion.",
            }
        },
        "required": ["content"],
    },
}

TOOL_SCHEMAS = [
    MEMORI_RECALL_SCHEMA,
    MEMORI_RECALL_SUMMARY_SCHEMA,
    MEMORI_QUOTA_SCHEMA,
    MEMORI_SIGNUP_SCHEMA,
    MEMORI_FEEDBACK_SCHEMA,
]

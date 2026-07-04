"""Tool schemas and tool-status constants for the OpenViking memory plugin."""

from __future__ import annotations

SEARCH_SCHEMA = {
    "name": "viking_search",
    "description": (
        "Semantic search over the OpenViking knowledge base. "
        "Returns ranked results with viking:// URIs for deeper reading. "
        "Use mode='deep' for complex queries that need reasoning across "
        "multiple sources, 'fast' for simple lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "mode": {
                "type": "string", "enum": ["auto", "fast", "deep"],
                "description": "Search depth (default: auto).",
            },
            "scope": {
                "type": "string",
                "description": "Viking URI prefix to scope search (e.g. 'viking://resources/docs/').",
            },
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

READ_SCHEMA = {
    "name": "viking_read",
    "description": (
        "Read one or a few specific viking:// URIs returned by viking_search or "
        "viking_browse. Three detail levels:\n"
        "  abstract — ~100 token summary (L0)\n"
        "  overview — ~2k token key points (L1)\n"
        "  full — complete content (L2)\n"
        "Start with abstract/overview, only use full when you need details. "
        "For multiple strong candidates, pass uris with up to three URIs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "Single viking:// URI to read."},
            "uris": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional batch of up to three viking:// URIs to read.",
            },
            "level": {
                "type": "string", "enum": ["abstract", "overview", "full"],
                "description": "Detail level (default: overview).",
            },
        },
        "required": [],
    },
}

BROWSE_SCHEMA = {
    "name": "viking_browse",
    "description": (
        "Browse the OpenViking knowledge store like a filesystem.\n"
        "  list — show directory contents\n"
        "  tree — show hierarchy\n"
        "  stat — show metadata for a URI"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["tree", "list", "stat"],
                "description": "Browse action.",
            },
            "path": {
                "type": "string",
                "description": "Viking URI path (default: viking://). Examples: 'viking://resources/', 'viking://user/memories/'.",
            },
        },
        "required": ["action"],
    },
}

REMEMBER_SCHEMA = {
    "name": "viking_remember",
    "description": (
        "Explicitly store a fact or memory in the OpenViking knowledge base. "
        "Use for important information the agent should remember long-term. "
        "The system automatically categorizes and indexes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "category": {
                "type": "string",
                "enum": ["preference", "entity", "event", "case", "pattern"],
                "description": "Memory category (default: auto-detected).",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "viking_forget",
    "description": (
        "Delete one OpenViking memory file by exact viking:// URI. "
        "Use only when the user explicitly asks to forget or delete a specific "
        "memory and you have the exact memory file URI. Resources, skills, "
        "sessions, directories, generated summaries, and broad deletes are rejected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "Exact viking:// memory file URI ending in .md.",
            },
        },
        "required": ["uri"],
    },
}

ADD_RESOURCE_SCHEMA = {
    "name": "viking_add_resource",
    "description": (
        "Add a remote URL or local file/directory to the OpenViking knowledge base. "
        "Remote resources must be public http(s), git, or ssh URLs. "
        "Local files are uploaded first using OpenViking temp_upload. "
        "The system automatically parses, indexes, and generates summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Remote URL or local file/directory path to add."},
            "reason": {
                "type": "string",
                "description": "Why this resource is relevant (improves search).",
            },
            "to": {
                "type": "string",
                "description": "Optional target viking:// URI for the resource.",
            },
            "parent": {
                "type": "string",
                "description": "Optional parent viking:// URI. Cannot be used with to.",
            },
            "instruction": {
                "type": "string",
                "description": "Optional processing instruction for semantic extraction.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for processing to complete.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds when wait is true.",
            },
        },
        "required": ["url"],
    },
}


# Recall tools (read-only) whose results we never re-ingest into OpenViking —
# echoing recalled memory back into the session transcript would re-store it.
# Write tools (viking_remember / viking_add_resource) are intentionally NOT
# here. Derived from the canonical schema names so renames can't desync.
_OPENVIKING_RECALL_TOOL_NAMES = {
    SEARCH_SCHEMA["name"],
    READ_SCHEMA["name"],
    BROWSE_SCHEMA["name"],
}

# Canonical tool_status values emitted in OpenViking batch tool parts.
_TOOL_STATUS_COMPLETED = "completed"
_TOOL_STATUS_ERROR = "error"
_TOOL_STATUS_PENDING = "pending"
# Inbound status aliases (from varied tool-result shapes) -> canonical above.
_TOOL_STATUS_ERROR_ALIASES = {"error", "failed", "failure"}
_TOOL_STATUS_COMPLETED_ALIASES = {"completed", "complete", "success", "succeeded"}

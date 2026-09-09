"""
Shared Schemas Package.
"""

TOOL_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {"type": "object"},
    },
    "required": ["name", "description"],
}

TASK_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "session_id": {"type": "string"},
        "status": {"type": "string"},
    },
}

__all__ = ["TOOL_SCHEMA_TEMPLATE", "TASK_SCHEMA_TEMPLATE"]

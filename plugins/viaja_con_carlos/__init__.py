"""VIAJA CON CARLOS general plugin.

The plugin owns a small approved knowledge lookup and an ephemeral
conversation-policy hook.  It does not call external services or messaging
APIs; the Messenger adapter owns delivery and its one-time opening.
"""

from __future__ import annotations

from .conversation import (
    CONVERSATION_CONFIG,
    FIXED_GREETING,
    FIXED_OPENING,
    OPENING_MESSAGE,
    conversation_prompt,
    on_pre_llm_call,
)
from .source_lookup import lookup_sources, normalize_text, source_lookup, source_lookup_tool

SOURCE_LOOKUP_TOOL_NAME = "viaja_source_lookup"

SOURCE_LOOKUP_SCHEMA = {
    "name": SOURCE_LOOKUP_TOOL_NAME,
    "description": (
        "Look up only approved VIAJA CON CARLOS property, offer, and policy "
        "excerpts. Returns stable source IDs and document paths; missing or "
        "conflicting facts set confirmation_required=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The customer's factual question or requested detail.",
            },
            "property_hint": {
                "type": "string",
                "description": "Optional property or destination name/alias.",
            },
            "topic_hint": {
                "type": "string",
                "description": "Optional topic such as price, inclusions, or policy.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def register(ctx) -> None:
    """Register the public lookup tool and ephemeral conversation hook."""
    ctx.register_tool(
        name=SOURCE_LOOKUP_TOOL_NAME,
        toolset="viaja_con_carlos",
        schema=SOURCE_LOOKUP_SCHEMA,
        handler=lambda args, **_: source_lookup_tool(args),
        description=SOURCE_LOOKUP_SCHEMA["description"],
    )
    ctx.register_hook("pre_llm_call", on_pre_llm_call)


__all__ = [
    "CONVERSATION_CONFIG",
    "FIXED_GREETING",
    "FIXED_OPENING",
    "OPENING_MESSAGE",
    "SOURCE_LOOKUP_SCHEMA",
    "SOURCE_LOOKUP_TOOL_NAME",
    "conversation_prompt",
    "lookup_sources",
    "normalize_text",
    "register",
    "source_lookup",
    "source_lookup_tool",
]

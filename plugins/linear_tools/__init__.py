"""linear_tools plugin — native Linear project-record tools.

The plugin exposes focused, secret-safe Linear GraphQL helpers as model tools.
It uses LINEAR_API_KEY only as an HTTP Authorization header and never returns
credential material, header values, token lengths, or private config details.
"""

from __future__ import annotations

import os

from plugins.linear_tools import client


def check_linear_requirements() -> bool:
    """Return True when a Linear API key is present, without inspecting it."""
    return bool(os.environ.get("LINEAR_API_KEY"))


_ISSUE_IDENTIFIER = {
    "type": "object",
    "properties": {
        "identifier": {
            "type": "string",
            "description": "Linear issue identifier or UUID, e.g. FGD-167.",
        }
    },
    "required": ["identifier"],
    "additionalProperties": False,
}


def _schema(name: str, description: str, parameters: dict) -> dict:
    return {"name": name, "description": description, "parameters": parameters}


_TOOLS = (
    (
        "linear_get_issue",
        "Read compact metadata for one Linear issue by identifier.",
        _schema("linear_get_issue", "Read compact metadata for one Linear issue by identifier.", _ISSUE_IDENTIFIER),
        client.handle_get_issue,
    ),
    (
        "linear_search_issues",
        "Search Linear issues and return compact metadata.",
        _schema(
            "linear_search_issues",
            "Search Linear issues and return compact metadata.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        client.handle_search_issues,
    ),
    (
        "linear_add_comment",
        "Add a comment to one Linear issue and return comment evidence.",
        _schema(
            "linear_add_comment",
            "Add a comment to one Linear issue and return comment evidence.",
            {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Linear issue identifier or UUID."},
                    "body": {"type": "string", "description": "Comment body."},
                },
                "required": ["identifier", "body"],
                "additionalProperties": False,
            },
        ),
        client.handle_add_comment,
    ),
    (
        "linear_ensure_comment",
        "Add a Linear issue comment only if the exact body is not already present.",
        _schema(
            "linear_ensure_comment",
            "Add a Linear issue comment only if the exact body is not already present.",
            {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Linear issue identifier or UUID."},
                    "body": {"type": "string", "description": "Comment body to ensure exactly once."},
                },
                "required": ["identifier", "body"],
                "additionalProperties": False,
            },
        ),
        client.handle_ensure_comment,
    ),
    (
        "linear_update_status",
        "Update one Linear issue's status by exact state name.",
        _schema(
            "linear_update_status",
            "Update one Linear issue's status by exact state name.",
            {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Linear issue identifier or UUID."},
                    "state": {"type": "string", "description": "Exact workflow state name, e.g. Done."},
                },
                "required": ["identifier", "state"],
                "additionalProperties": False,
            },
        ),
        client.handle_update_status,
    ),
    (
        "linear_create_issue",
        "Create a Linear issue in a team by key and return compact issue evidence.",
        _schema(
            "linear_create_issue",
            "Create a Linear issue in a team by key and return compact issue evidence.",
            {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Linear team key, e.g. FGD."},
                    "title": {"type": "string", "description": "Issue title."},
                    "description": {"type": "string", "description": "Issue description."},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 4},
                    "parent": {"type": "string", "description": "Optional parent issue identifier or UUID."},
                },
                "required": ["team", "title"],
                "additionalProperties": False,
            },
        ),
        client.handle_create_issue,
    ),
    (
        "linear_link_issues",
        "Relate two Linear issues when relation mutation support is implemented.",
        _schema(
            "linear_link_issues",
            "Relate two Linear issues when relation mutation support is implemented.",
            {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Primary Linear issue identifier."},
                    "related_identifier": {"type": "string", "description": "Related Linear issue identifier."},
                    "relation": {"type": "string", "description": "Requested relation type, if known."},
                },
                "required": ["identifier", "related_identifier"],
                "additionalProperties": False,
            },
        ),
        client.handle_link_issues,
    ),
)


def register(ctx) -> None:
    """Register Linear tools under the explicit `linear` plugin toolset."""
    for name, schema, handler in ((item[0], item[2], item[3]) for item in _TOOLS):
        ctx.register_tool(
            name=name,
            toolset="linear",
            schema=schema,
            handler=handler,
            check_fn=check_linear_requirements,
            requires_env=["LINEAR_API_KEY"],
            description=schema.get("description", ""),
            emoji="📐",
        )

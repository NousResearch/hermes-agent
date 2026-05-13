"""Task-level tool facade for the local Hermes MCP bridge MVP."""

from __future__ import annotations

from typing import Any

from gateway.mcp_bridge import (
    MCPBridgeError,
    get_task_result as _get_task_result,
    get_task_status as _get_task_status,
    list_recent_tasks as _list_recent_tasks,
    submit_task as _submit_task,
)


SUBMIT_TASK_SCHEMA = {
    "name": "submit_task",
    "description": (
        "Validate and record a local Hermes task contract for future "
        "orchestration. This tool does not execute tasks or expose shell, "
        "filesystem mutation, secrets, network, Docker, Codex, OpenAI, "
        "Shopify, PROD, or git push/reset/clean capabilities. The canonical "
        "contract is the top-level object below; the FastMCP wrapper also "
        "accepts exactly one remote-client compatibility envelope named "
        "payload, args, or arguments containing that same object."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "project": {"type": "string"},
            "mode": {"type": "string"},
            "repo_scope": {"type": ["string", "object"]},
            "worktree_scope": {"type": ["string", "object"]},
            "task_contract": {"type": ["string", "object"]},
            "allowed_actions": {"type": "array", "items": {"type": "string"}},
            "forbidden_actions": {"type": "array", "items": {"type": "string"}},
            "return_format": {"type": ["string", "object"]},
            "approvals": {"type": ["array", "object"]},
            "expected_branch": {"type": "string"},
            "expected_head": {"type": "string"},
            "stop_conditions": {"type": ["array", "object", "string"]},
            "worker_selection_guidance": {"type": ["object", "string"]},
            "training_notes": {"type": ["object", "string"]},
        },
        "required": [
            "title",
            "project",
            "mode",
            "task_contract",
            "allowed_actions",
            "forbidden_actions",
            "return_format",
        ],
        "anyOf": [
            {"required": ["repo_scope"]},
            {"required": ["worktree_scope"]},
        ],
    },
}

GET_TASK_STATUS_SCHEMA = {
    "name": "get_task_status",
    "description": "Return the stored status for a local bridge task.",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}

GET_TASK_RESULT_SCHEMA = {
    "name": "get_task_result",
    "description": "Return the stored result/refusal/accepted record for a local bridge task.",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}

LIST_RECENT_TASKS_SCHEMA = {
    "name": "list_recent_tasks",
    "description": "List recently recorded local bridge tasks. No task execution is performed.",
    "parameters": {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
    },
}

TOOL_SCHEMAS = (
    SUBMIT_TASK_SCHEMA,
    GET_TASK_STATUS_SCHEMA,
    GET_TASK_RESULT_SCHEMA,
    LIST_RECENT_TASKS_SCHEMA,
)


_SUBMIT_TASK_ENVELOPE_KEYS = ("payload", "args", "arguments")
_SUBMIT_TASK_TOP_LEVEL_KEYS = frozenset(SUBMIT_TASK_SCHEMA["parameters"]["properties"])


def _compact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Drop only None values while preserving empty lists/dicts/strings for validation."""
    return {key: value for key, value in values.items() if value is not None}


def normalize_submit_task_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize supported local/remote submit_task argument shapes.

    Local MCP clients call the FastMCP wrapper with the task fields as top-level
    keyword arguments. Some remote clients wrap tool arguments under an envelope
    key such as ``payload``, ``args``, or ``arguments``. Accept exactly one safe
    envelope or the existing top-level shape, but fail closed on ambiguous or
    malformed containers before the core bridge persists a record.
    """
    if not isinstance(args, dict):
        raise MCPBridgeError("submit_task arguments must be an object")

    compact_args = _compact_mapping(args)
    present_envelopes = [key for key in _SUBMIT_TASK_ENVELOPE_KEYS if key in args]
    if len(present_envelopes) > 1:
        raise MCPBridgeError("submit_task arguments include multiple envelopes")

    if present_envelopes:
        envelope_key = present_envelopes[0]
        top_level_keys = set(compact_args) - {envelope_key}
        if top_level_keys:
            raise MCPBridgeError("submit_task envelope cannot be mixed with top-level fields")
        envelope = args[envelope_key]
        if not isinstance(envelope, dict):
            raise MCPBridgeError(f"submit_task {envelope_key} envelope must be an object")
        normalized = _compact_mapping(envelope)
    else:
        normalized = compact_args

    if not normalized:
        raise MCPBridgeError("submit_task arguments must include task fields")

    nested_envelopes = set(normalized) & set(_SUBMIT_TASK_ENVELOPE_KEYS)
    if nested_envelopes:
        keys = ", ".join(sorted(nested_envelopes))
        raise MCPBridgeError(f"submit_task nested envelopes are not supported: {keys}")

    unknown_keys = set(normalized) - _SUBMIT_TASK_TOP_LEVEL_KEYS
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise MCPBridgeError(f"submit_task arguments include unknown fields: {keys}")

    return normalized


def _error(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)}


def submit_task(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _submit_task(normalize_submit_task_arguments(args))
    except MCPBridgeError as exc:
        return _error(exc)


def get_task_status(args: dict[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(args, dict):
            raise MCPBridgeError("get_task_status arguments must be an object")
        return _get_task_status(str(args.get("task_id", "")))
    except MCPBridgeError as exc:
        return _error(exc)


def get_task_result(args: dict[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(args, dict):
            raise MCPBridgeError("get_task_result arguments must be an object")
        return _get_task_result(str(args.get("task_id", "")))
    except MCPBridgeError as exc:
        return _error(exc)


def list_recent_tasks(args: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or {}
    try:
        if not isinstance(args, dict):
            raise MCPBridgeError("list_recent_tasks arguments must be an object")
        return _list_recent_tasks(args.get("limit", 20))
    except MCPBridgeError as exc:
        return _error(exc)


TOOL_HANDLERS = {
    "submit_task": submit_task,
    "get_task_status": get_task_status,
    "get_task_result": get_task_result,
    "list_recent_tasks": list_recent_tasks,
}

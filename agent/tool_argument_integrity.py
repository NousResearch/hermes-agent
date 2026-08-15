"""Shared integrity checks for non-replayable historical tool arguments."""

from __future__ import annotations

import json
from typing import Any, Optional

INCOMPLETE_TOOL_ARGUMENTS_KEY = "__hermes_incomplete_tool_arguments__"


def contains_incomplete_tool_arguments(value: Any) -> bool:
    """Detect reserved lossy-history provenance at any nesting depth."""
    if isinstance(value, dict):
        if INCOMPLETE_TOOL_ARGUMENTS_KEY in value:
            return True
        return any(contains_incomplete_tool_arguments(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_incomplete_tool_arguments(item) for item in value)
    return False


def incomplete_tool_arguments_block_message(value: Any) -> Optional[str]:
    """Return the common fail-closed message for lossy historical arguments."""
    if not contains_incomplete_tool_arguments(value):
        return None
    return (
        "Hermes omitted these already-executed arguments during context "
        "compression, so the tool was not executed. Reconstruct and issue a "
        "new complete invocation from current user intent instead of replaying "
        "this historical call."
    )


def incomplete_tool_arguments_error_result(value: Any) -> Optional[str]:
    """Return the canonical structured rejection, or ``None`` when replayable."""
    message = incomplete_tool_arguments_block_message(value)
    if message is None:
        return None
    return json.dumps(
        {
            "error": "Incomplete historical tool arguments",
            "error_type": "incomplete_historical_tool_arguments",
            "message": message,
        },
        ensure_ascii=False,
    )


def _schema_accepts_container(schema: Any, kind: str) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == kind or (
        isinstance(schema_type, list) and kind in schema_type
    ):
        return True
    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and any(
            _schema_accepts_container(branch, kind) for branch in branches
        ):
            return True
    return False


def _decode_schema_containers(
    value: Any,
    schema: Any,
    *,
    direct_container_required: bool = False,
    top_level_arguments: bool = False,
) -> Any:
    """Pure preview of the container decoding performed by schema coercion."""
    if not isinstance(schema, dict):
        return value
    decoded_from_string = False
    if direct_container_required:
        direct_type = schema.get("type")
        direct_types = (
            set(direct_type) if isinstance(direct_type, list) else {direct_type}
        )
        if not direct_types.intersection({"array", "object"}):
            return value
    if isinstance(value, str):
        trimmed = value.strip()
        accepts_array = _schema_accepts_container(schema, "array")
        accepts_object = _schema_accepts_container(schema, "object")
        if not (
            (accepts_array and trimmed.startswith("["))
            or (accepts_object and trimmed.startswith("{"))
        ):
            return value
        try:
            parsed = json.loads(trimmed)
        except (json.JSONDecodeError, TypeError):
            return value
        if not (
            (accepts_array and isinstance(parsed, list))
            or (accepts_object and isinstance(parsed, dict))
        ):
            return value
        value = parsed
        decoded_from_string = True
    if isinstance(value, list):
        if (
            direct_container_required
            and not decoded_from_string
            and schema.get("type") != "array"
        ):
            return value
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        return [_decode_schema_containers(item, item_schema) for item in value]
    if isinstance(value, dict):
        if (
            direct_container_required
            and not decoded_from_string
            and schema.get("type") != "object"
        ):
            return value
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        return {
            key: _decode_schema_containers(
                item,
                properties.get(key),
                direct_container_required=top_level_arguments,
            )
            if isinstance(properties.get(key), dict)
            else item
            for key, item in value.items()
        }
    return value


def incomplete_tool_arguments_after_schema_decode(
    value: Any, schema: Any
) -> Optional[str]:
    """Reject provenance that schema-guided container decoding would expose."""
    return incomplete_tool_arguments_error_result(
        _decode_schema_containers(value, schema, top_level_arguments=True)
    )


def is_incomplete_tool_arguments_error_result(value: Any) -> bool:
    """Identify the canonical rejection without treating arbitrary text as policy."""
    if not isinstance(value, str):
        return False
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(parsed, dict)
        and parsed.get("error_type") == "incomplete_historical_tool_arguments"
    )

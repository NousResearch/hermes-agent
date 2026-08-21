"""Shared integrity checks for non-replayable historical tool arguments."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional

INCOMPLETE_TOOL_ARGUMENTS_KEY = "__hermes_incomplete_tool_arguments__"

_WIRE_HISTORY_NOTE = (
    "[A completed compressed historical tool call is omitted from this "
    "request; its stored transcript is unchanged.]"
)


def contains_incomplete_tool_arguments(value: Any) -> bool:
    """Detect reserved lossy-history provenance at any nesting depth."""
    if isinstance(value, dict):
        if INCOMPLETE_TOOL_ARGUMENTS_KEY in value:
            return True
        return any(contains_incomplete_tool_arguments(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_incomplete_tool_arguments(item) for item in value)
    return False


def _tool_call_has_incomplete_arguments(tool_call: Any) -> bool:
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return False
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return False
    return contains_incomplete_tool_arguments(arguments)


def completed_tool_call_pairs(
    messages: list[dict[str, Any]],
) -> dict[tuple[int, int], int]:
    """Map structurally paired assistant-call positions to result positions."""
    pending: dict[str, list[tuple[int, int]]] = {}
    pairs: dict[tuple[int, int], int] = {}
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            pending.clear()
            continue
        role = message.get("role")
        if role == "assistant":
            pending.clear()
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            call_ids = [
                call.get("id") for call in tool_calls if isinstance(call, dict)
            ]
            ambiguous_ids = {call_id for call_id in call_ids if call_ids.count(call_id) > 1}
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                if (
                    isinstance(call_id, str)
                    and call_id
                    and call_id not in ambiguous_ids
                ):
                    pending.setdefault(call_id, []).append(
                        (message_index, call_index)
                    )
        elif role == "tool":
            call_id = message.get("tool_call_id")
            queue = pending.get(call_id) if isinstance(call_id, str) else None
            if queue:
                pairs[queue.pop(0)] = message_index
        else:
            pending.clear()
    return pairs


def _neutralize_anthropic_sidecar(blocks: Any, removed_ids: set[Any]) -> Any:
    if not isinstance(blocks, list):
        return blocks
    sanitized = []
    note_inserted = False
    for block in blocks:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("id") in removed_ids
        ):
            if not note_inserted:
                sanitized.append({"type": "text", "text": _WIRE_HISTORY_NOTE})
                note_inserted = True
            continue
        sanitized.append(deepcopy(block))
    return sanitized


def neutralize_completed_incomplete_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a provider-safe request copy without completed marker calls.

    Compression provenance belongs in Hermes' canonical transcript so the
    execution guard can fail closed. Once such a call has a tool result,
    replaying the pair on a later provider request makes the marker look like
    fresh executable arguments. Remove both sides of only those completed
    pairs and leave a plain assistant note. Unpaired calls remain untouched
    for the execution guard, and ordinary calls in a mixed batch retain their
    call/result pairing.
    """
    pending: dict[str, list[tuple[int, int, bool]]] = {}
    neutralized_calls: set[tuple[int, int]] = set()
    neutralized_results: set[int] = set()
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            pending.clear()
            continue
        role = message.get("role")
        if role == "assistant":
            pending.clear()
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                call_ids = [
                    call.get("id")
                    for call in tool_calls
                    if isinstance(call, dict)
                ]
                ambiguous_ids = {
                    call_id for call_id in call_ids if call_ids.count(call_id) > 1
                }
                for call_index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = tool_call.get("id")
                    if (
                        isinstance(tool_call_id, str)
                        and tool_call_id
                        and tool_call_id not in ambiguous_ids
                    ):
                        pending.setdefault(tool_call_id, []).append(
                            (
                                message_index,
                                call_index,
                                _tool_call_has_incomplete_arguments(tool_call),
                            )
                        )
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            queue = pending.get(tool_call_id) if isinstance(tool_call_id, str) else None
            if queue:
                assistant_index, call_index, is_marker = queue.pop(0)
                if is_marker:
                    neutralized_calls.add((assistant_index, call_index))
                    neutralized_results.add(message_index)
        else:
            pending.clear()

    if not neutralized_calls:
        return messages

    sanitized: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message_index in neutralized_results:
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            sanitized.append(message)
            continue
        kept_calls = [
            call
            for call_index, call in enumerate(tool_calls)
            if (message_index, call_index) not in neutralized_calls
        ]
        if len(kept_calls) == len(tool_calls):
            sanitized.append(message)
            continue

        copied = deepcopy(message)
        removed_ids = {
            call.get("id")
            for call_index, call in enumerate(tool_calls)
            if (message_index, call_index) in neutralized_calls
            and isinstance(call, dict)
        }
        if kept_calls:
            copied["tool_calls"] = deepcopy(kept_calls)
        else:
            copied.pop("tool_calls", None)
        copied.pop("codex_message_items", None)
        if "anthropic_content_blocks" in copied:
            copied["anthropic_content_blocks"] = _neutralize_anthropic_sidecar(
                copied["anthropic_content_blocks"], removed_ids
            )
        content = copied.get("content")
        if isinstance(content, list):
            content.append({"type": "text", "text": _WIRE_HISTORY_NOTE})
        elif isinstance(content, str) and content.strip():
            copied["content"] = f"{content.rstrip()}\n{_WIRE_HISTORY_NOTE}"
        else:
            copied["content"] = _WIRE_HISTORY_NOTE
        sanitized.append(copied)
    return sanitized


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

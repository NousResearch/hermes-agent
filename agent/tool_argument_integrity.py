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
    """Map unambiguous assistant calls to adjacent result positions."""
    call_counts: dict[str, int] = {}
    result_counts: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        call_id = tool_call.get("id")
                        if isinstance(call_id, str) and call_id:
                            call_counts[call_id] = call_counts.get(call_id, 0) + 1
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                result_counts[call_id] = result_counts.get(call_id, 0) + 1

    pairs: dict[tuple[int, int], int] = {}
    for assistant_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        calls_by_id: dict[str, list[int]] = {}
        for call_index, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict):
                call_id = tool_call.get("id")
                if isinstance(call_id, str) and call_id:
                    calls_by_id.setdefault(call_id, []).append(call_index)
        results_by_id: dict[str, list[int]] = {}
        for result_index in range(assistant_index + 1, len(messages)):
            result = messages[result_index]
            if not isinstance(result, dict) or result.get("role") != "tool":
                break
            call_id = result.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                results_by_id.setdefault(call_id, []).append(result_index)
        for call_id, call_indices in calls_by_id.items():
            result_indices = results_by_id.get(call_id, [])
            if (
                len(call_indices) == 1
                and len(result_indices) == 1
                and call_counts.get(call_id) == 1
                and result_counts.get(call_id) == 1
            ):
                pairs[(assistant_index, call_indices[0])] = result_indices[0]
    return pairs


def neutralize_completed_incomplete_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project canonical history into a conservative provider-safe request copy.

    Compression provenance belongs in Hermes' canonical transcript so the
    execution guard can fail closed. Once such a call has a tool result,
    replaying the pair on a later provider request makes the marker look like
    fresh executable arguments. Remove both sides of only those completed,
    unambiguous pairs and leave a plain note.

    An affected assistant turn is rebuilt only from canonical wire-neutral
    fields (role, visible content, and any remaining ordinary tool calls).
    Provider-native replay sidecars are deliberately discarded wholesale:
    after one canonical tool call is removed, their IDs, inputs, signatures,
    and cross-block ordering can no longer be trusted. The stored transcript
    remains unchanged. Malformed non-dict history entries are also excluded
    from the request projection rather than forwarded to provider schemas.
    """
    request_messages = [message for message in messages if isinstance(message, dict)]
    pairs = completed_tool_call_pairs(request_messages)
    neutralized_calls = {
        position
        for position in pairs
        if _tool_call_has_incomplete_arguments(
            request_messages[position[0]]["tool_calls"][position[1]]
        )
    }
    if not neutralized_calls:
        return messages if len(request_messages) == len(messages) else request_messages
    neutralized_results = {pairs[position] for position in neutralized_calls}
    note_boundaries: set[int] = set()
    for assistant_index in {position[0] for position in neutralized_calls}:
        tool_calls = request_messages[assistant_index].get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        positions = [(assistant_index, index) for index in range(len(tool_calls))]
        if not all(position in neutralized_calls for position in positions):
            continue
        result_indices = [pairs[position] for position in positions]
        boundary = max(result_indices)
        next_index = boundary + 1
        while next_index < len(request_messages) and next_index in neutralized_results:
            next_index += 1
        if next_index < len(request_messages):
            following = request_messages[next_index]
            if following.get("role") == "assistant":
                note_boundaries.add(boundary)

    sanitized: list[dict[str, Any]] = []
    for message_index, message in enumerate(request_messages):
        if message_index in neutralized_results:
            if message_index in note_boundaries:
                sanitized.append({"role": "user", "content": _WIRE_HISTORY_NOTE})
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

        content = deepcopy(message.get("content"))
        if isinstance(content, list):
            content.append({"type": "text", "text": _WIRE_HISTORY_NOTE})
        elif isinstance(content, str) and content.strip():
            content = f"{content.rstrip()}\n{_WIRE_HISTORY_NOTE}"
        else:
            content = _WIRE_HISTORY_NOTE

        rebuilt: dict[str, Any] = {"role": "assistant", "content": content}
        if kept_calls:
            rebuilt["tool_calls"] = deepcopy(kept_calls)
        sanitized.append(rebuilt)
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

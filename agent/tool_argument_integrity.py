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


def _combine_assistant_content(first: Any, second: Any) -> Any:
    """Combine adjacent assistant content without discarding visible text."""
    if isinstance(first, list) or isinstance(second, list):
        def _parts(value: Any) -> list[Any]:
            if isinstance(value, list):
                return deepcopy(value)
            if isinstance(value, str) and value:
                return [{"type": "text", "text": value}]
            return []

        return _parts(first) + _parts(second)
    texts = [
        value.strip()
        for value in (first, second)
        if isinstance(value, str) and value.strip()
    ]
    return "\n".join(texts)


def _merge_adjacent_assistant_messages(
    messages: list[tuple[dict[str, Any], bool]],
) -> list[dict[str, Any]]:
    """Merge only assistant adjacency created by removing complete call pairs."""
    collapsed: list[tuple[dict[str, Any], bool]] = []
    list_sidecars = (
        "anthropic_content_blocks",
        "codex_message_items",
        "codex_reasoning_items",
        "reasoning_details",
    )
    for message, neutralized_all_calls in messages:
        if (
            collapsed
            and message.get("role") == "assistant"
            and collapsed[-1][0].get("role") == "assistant"
            and (neutralized_all_calls or collapsed[-1][1])
        ):
            previous, previous_neutralized = collapsed[-1]
            merged = deepcopy(previous)
            for key, value in message.items():
                if key not in {"content", *list_sidecars}:
                    merged[key] = deepcopy(value)
            merged["content"] = _combine_assistant_content(
                previous.get("content"), message.get("content")
            )
            for key in list_sidecars:
                before = previous.get(key)
                after = message.get(key)
                if isinstance(before, list) or isinstance(after, list):
                    merged[key] = (
                        deepcopy(before) if isinstance(before, list) else []
                    ) + (deepcopy(after) if isinstance(after, list) else [])
            collapsed[-1] = (
                merged,
                previous_neutralized or neutralized_all_calls,
            )
            continue
        collapsed.append((message, neutralized_all_calls))
    return [message for message, _ in collapsed]


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
    pairs = completed_tool_call_pairs(messages)
    neutralized_calls = {
        position
        for position in pairs
        if _tool_call_has_incomplete_arguments(
            messages[position[0]]["tool_calls"][position[1]]
        )
    }
    if not neutralized_calls:
        return messages
    neutralized_results = {pairs[position] for position in neutralized_calls}

    sanitized: list[tuple[dict[str, Any], bool]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            sanitized.append((message, False))
            continue
        if message_index in neutralized_results:
            continue
        if message.get("role") != "assistant":
            sanitized.append((message, False))
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            sanitized.append((message, False))
            continue
        kept_calls = [
            call
            for call_index, call in enumerate(tool_calls)
            if (message_index, call_index) not in neutralized_calls
        ]
        if len(kept_calls) == len(tool_calls):
            sanitized.append((message, False))
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
        sanitized.append((copied, not kept_calls))
    return _merge_adjacent_assistant_messages(sanitized)


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

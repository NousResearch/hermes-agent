"""Normalize local bridge batches before persistence and per-tool execution."""

from copy import deepcopy
import json

from agent.message_sanitization import coalesce_tool_call_id, deterministic_call_id, uniquify_tool_call_ids
from agent.transports.types import ToolCall
from tools.connectors import is_connector_name
from tools.connectors.gateway.config import MAX_CALLS_PER_DISPATCH
from tools.tool_search_catalog import TOOL_CALL_NAME
from tools.tool_search_validation import normalize_tool_call_entries


def expand_local_tool_batches(tool_calls: list, *, provider_data: dict | None = None) -> list:
    """Give every local entry the ordinary agent execution and persistence path.

    Registry-only batch dispatch loses the live agent's inline tools, guardrails,
    and per-server concurrency policy. Split the *new* assistant turn instead,
    before it is persisted; never rewrite an earlier conversation message.
    Connector-only batches keep their existing dispatcher and wire shape.
    """
    expanded, replacements = [], {}
    for call in tool_calls:
        if call.function.name != TOOL_CALL_NAME:
            expanded.append(call)
            continue
        try:
            args = call.function.arguments
            entries, error = normalize_tool_call_entries(json.loads(args) if isinstance(args, str) else args)
        except (TypeError, ValueError, AttributeError):
            entries, error = [], True
        if (error or not 1 < len(entries) <= MAX_CALLS_PER_DISPATCH
                or all(is_connector_name(entry["name"]) for entry in entries)):
            # Leave malformed/oversized requests intact for the dispatcher's error.
            expanded.append(call)
            continue
        parent_id = coalesce_tool_call_id(call)
        children = []
        for index, entry in enumerate(entries):
            arguments = json.dumps({"calls": [entry]}, ensure_ascii=False)
            if index == 0:
                child = deepcopy(call)
                child.function.arguments = arguments
            else:
                # A new entry must not inherit the parent's Responses item id or
                # provider signature. IDs are deterministic for stable replay.
                child = ToolCall(
                    id=deterministic_call_id(TOOL_CALL_NAME, parent_id + arguments, index),
                    name=TOOL_CALL_NAME, arguments=arguments,
                )
            expanded.append(child)
            children.append(child)
        replacements[parent_id] = children
    uniquify_tool_call_ids(expanded)
    # Native providers replay ordered blocks instead of rebuilding them from
    # tool_calls. Replace the matching tool block in place, preserving signed
    # thinking/text blocks and their order exactly.
    if provider_data and replacements:
        for key, replace in (("anthropic_content_blocks", _anthropic_children),
                             ("bedrock_content_blocks", _bedrock_children)):
            if blocks := provider_data.get(key):
                provider_data[key] = [child for block in blocks for child in replace(block, replacements)]
    return expanded


def _anthropic_children(block, replacements):
    children = replacements.get(block.get("id")) if block.get("type") == "tool_use" else None
    if children is None:
        return [block]
    return [{**block, "id": coalesce_tool_call_id(child),
             "input": json.loads(child.function.arguments)} for child in children]


def _bedrock_children(block, replacements):
    tool = block.get("toolUse")
    children = replacements.get(tool.get("toolUseId")) if isinstance(tool, dict) else None
    if children is None:
        return [block]
    return [{**block, "toolUse": {**tool, "toolUseId": coalesce_tool_call_id(child),
                                 "input": json.loads(child.function.arguments)}} for child in children]

"""Tool-specific privacy boundaries for arguments that must stay transient."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


PRIVATE_ARGUMENT_TOOL = "telegram_coding_worker"
REDACTED_ARGUMENT_VALUE = "[redacted]"


def redact_tool_args_for_observers(tool_name: str, args: Any) -> Any:
    """Return an observer-safe copy of *args* without changing live arguments."""
    if tool_name != PRIVATE_ARGUMENT_TOOL:
        return args
    if not isinstance(args, dict):
        return {}
    return {str(key): REDACTED_ARGUMENT_VALUE for key in args}


def redact_tool_calls_for_persistence(tool_calls: Any) -> Any:
    """Copy tool calls and redact private-tool argument values for storage."""
    try:
        cleaned = deepcopy(tool_calls)
    except Exception:
        cleaned = tool_calls
    if not isinstance(cleaned, list):
        return cleaned
    for call in cleaned:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments_owner = function
        else:
            name = call.get("name")
            arguments_owner = call
        if name != PRIVATE_ARGUMENT_TOOL or "arguments" not in arguments_owner:
            continue
        arguments = arguments_owner.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {}
            redacted = redact_tool_args_for_observers(name, arguments)
            arguments_owner["arguments"] = json.dumps(redacted, ensure_ascii=False)
        else:
            arguments_owner["arguments"] = redact_tool_args_for_observers(name, arguments)
    return cleaned


def redact_message_for_persistence(message: Any) -> Any:
    """Return a storage-safe message copy while leaving live context untouched."""
    if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
        return message
    return {
        **message,
        "tool_calls": redact_tool_calls_for_persistence(message["tool_calls"]),
    }


def redact_executed_tool_calls_in_place(message: Any) -> None:
    """Redact private arguments on a live assistant message after dispatch.

    The caller must invoke this only after execution and its guardrail/hook
    bookkeeping have finished.  Replacing just ``tool_calls`` preserves the
    assistant message object, call IDs, and provider-required message order.
    """
    if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
        return
    message["tool_calls"] = redact_tool_calls_for_persistence(message["tool_calls"])

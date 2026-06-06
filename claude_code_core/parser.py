"""Parser for Claude Code CLI stream-json output.

Each line of stdout is a JSON object. This module parses them into StreamEvent objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .types import (
    TOOL_CATEGORIES,
    AskOption,
    AskQuestion,
    ContentBlockType,
    ElicitationRequest,
    MessageType,
    PermissionRequest,
    RateLimitInfo,
    StreamEvent,
    TodoItem,
    ToolCategory,
    ToolUseEvent,
)

logger = logging.getLogger(__name__)


def parse_line(line: str) -> StreamEvent | None:
    """Parse a single line of stream-json output into a StreamEvent.

    Returns None if the line is empty or unparseable.
    """
    line = line.strip()
    if not line:
        return None

    try:
        data: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Failed to parse stream-json line: %s", line[:200])
        return None

    msg_type_str = data.get("type", "")
    try:
        msg_type = MessageType(msg_type_str)
    except ValueError:
        logger.debug("Unknown message type: %s", msg_type_str)
        return None

    event = StreamEvent(message_type=msg_type)

    if msg_type == MessageType.SYSTEM:
        _parse_system(data, event)
    elif msg_type == MessageType.ASSISTANT:
        _parse_assistant(data, event)
    elif msg_type == MessageType.USER:
        _parse_user(data, event)
    elif msg_type == MessageType.RESULT:
        _parse_result(data, event)
    elif msg_type == MessageType.PROGRESS:
        pass  # No additional parsing needed — the event itself resets stall timers
    elif msg_type == MessageType.RATE_LIMIT_EVENT:
        _parse_rate_limit_event(data, event)

    return event


def _parse_system(data: dict[str, Any], event: StreamEvent) -> None:
    """Parse system message (contains session_id on init)."""
    event.session_id = data.get("session_id")
    subtype = data.get("subtype", "")
    if subtype == "init":
        logger.info("Session initialized: %s", event.session_id)
    elif subtype == "compact_boundary":
        event.is_compact = True
        metadata = data.get("compactMetadata", {})
        event.compact_trigger = metadata.get("trigger")
        event.compact_pre_tokens = metadata.get("preTokens")
        logger.info(
            "Context compaction (%s) — %s tokens before compact",
            event.compact_trigger,
            event.compact_pre_tokens,
        )
    elif subtype == "permission_request":
        event.permission_request = PermissionRequest(
            request_id=data.get("request_id", ""),
            tool_name=data.get("tool_name", ""),
            tool_input=data.get("tool_input", {}),
        )
        logger.info("Permission request: %s", data.get("tool_name"))
    elif subtype == "elicitation":
        event.elicitation = ElicitationRequest(
            request_id=data.get("request_id", ""),
            server_name=data.get("server_name", ""),
            mode=data.get("mode", "form-mode"),
            message=data.get("message", ""),
            url=data.get("url", ""),
            schema=data.get("schema", {}),
        )
        logger.info("MCP elicitation: %s (%s)", data.get("server_name"), data.get("mode"))


def _parse_assistant(data: dict[str, Any], event: StreamEvent) -> None:
    """Parse assistant message (text blocks, tool_use blocks, and thinking blocks).

    Sets is_partial=True when stop_reason is null/missing, meaning Claude is still
    generating content. With --include-partial-messages, many partial events arrive
    before the final complete event (stop_reason="end_turn" or "tool_use").
    """
    message = data.get("message", {})
    content = message.get("content", [])
    event.is_partial = message.get("stop_reason") is None

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in content:
        block_type = block.get("type", "")

        if block_type == ContentBlockType.TEXT.value:
            text = block.get("text", "")
            if text:
                text_parts.append(text)

        elif block_type == ContentBlockType.TOOL_USE.value:
            tool_name = block.get("name", "unknown")
            category = TOOL_CATEGORIES.get(tool_name, ToolCategory.OTHER)
            tool_input = block.get("input", {})
            event.tool_use = ToolUseEvent(
                tool_id=block.get("id", ""),
                tool_name=tool_name,
                tool_input=tool_input,
                category=category,
            )
            if tool_name == "AskUserQuestion":
                event.ask_questions = _parse_ask_questions(tool_input)
            elif tool_name == "TodoWrite":
                event.todo_list = _parse_todo_items(tool_input)
            elif tool_name == "ExitPlanMode":
                event.is_plan_approval = True

        elif block_type == ContentBlockType.THINKING.value:
            thinking_text = block.get("thinking", "")
            if thinking_text:
                thinking_parts.append(thinking_text)

        elif block_type == "redacted_thinking":
            event.has_redacted_thinking = True

    if text_parts:
        event.text = "\n".join(text_parts)
    if thinking_parts:
        event.thinking = "\n".join(thinking_parts)

    # Extract per-turn usage from the assistant message.
    # Unlike the cumulative usage in RESULT, this reflects the actual token
    # counts for this single API call — essential for accurate context tracking.
    usage = message.get("usage", {})
    if usage:
        event.input_tokens = usage.get("input_tokens")
        event.output_tokens = usage.get("output_tokens")
        event.cache_read_tokens = usage.get("cache_read_input_tokens")
        event.cache_creation_tokens = usage.get("cache_creation_input_tokens")


def _parse_user(data: dict[str, Any], event: StreamEvent) -> None:
    """Parse user message (tool_result blocks with content)."""
    message = data.get("message", {})
    content = message.get("content", [])

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == ContentBlockType.TOOL_RESULT.value:
            event.tool_result_id = block.get("tool_use_id", "")
            # Extract tool result content
            result_content = block.get("content", "")
            if isinstance(result_content, str) and result_content:
                event.tool_result_content = result_content
            elif isinstance(result_content, list):
                # Content can be a list of blocks (e.g. [{type: "text", text: "..."}])
                text_parts = []
                for part in result_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                if text_parts:
                    event.tool_result_content = "\n".join(text_parts)
            break


def _parse_result(data: dict[str, Any], event: StreamEvent) -> None:
    """Parse result message (session complete)."""
    event.is_complete = True
    event.session_id = data.get("session_id")
    event.cost_usd = data.get("cost_usd")
    event.duration_ms = data.get("duration_ms")

    usage = data.get("usage", {})
    if usage:
        event.input_tokens = usage.get("input_tokens")
        event.output_tokens = usage.get("output_tokens")
        event.cache_read_tokens = usage.get("cache_read_input_tokens")
        event.cache_creation_tokens = usage.get("cache_creation_input_tokens")

    # Extract context window size from modelUsage (any model key).
    model_usage = data.get("modelUsage", {})
    for model_info in model_usage.values():
        if isinstance(model_info, dict) and "contextWindow" in model_info:
            event.context_window = model_info["contextWindow"]
            break

    # Final text from result
    result_text = data.get("result", "")
    if result_text:
        event.text = result_text

    # Check for errors.
    # Two error shapes from the CLI:
    #   {"type":"result","subtype":"error","error":"..."} — explicit error subtype
    #   {"type":"result","subtype":"success","is_error":true,"result":"API Error: ..."} — API-level
    #     error reported as a "successful" result with is_error flag (e.g. 400 from Anthropic API)
    subtype = data.get("subtype", "")
    if subtype == "error":
        event.error = data.get("error", "Unknown error")
    elif data.get("is_error") and result_text:
        # API-level error (e.g. "API Error: 400 ...") surfaced via is_error flag.
        # Promote it to event.error so the handler shows an error display,
        # not a normal session-complete display.
        event.error = result_text
        event.text = ""  # suppress duplicate display via result text path


def _parse_rate_limit_event(data: dict[str, Any], event: StreamEvent) -> None:
    """Parse rate_limit_event message into a RateLimitInfo dataclass."""
    info = data.get("rate_limit_info", {})
    if not info:
        return
    event.rate_limit_info = RateLimitInfo(
        rate_limit_type=info.get("rateLimitType", ""),
        status=info.get("status", ""),
        utilization=float(info.get("utilization", 0.0)),
        resets_at=int(info.get("resetsAt", 0)),
        is_using_overage=bool(info.get("isUsingOverage", False)),
    )


def _parse_ask_questions(tool_input: dict[str, Any]) -> list[AskQuestion]:
    """Parse AskUserQuestion tool input into a list of AskQuestion objects."""
    questions_raw = tool_input.get("questions", [])
    result: list[AskQuestion] = []
    for q in questions_raw:
        options = [
            AskOption(
                label=o.get("label", ""),
                description=o.get("description", ""),
            )
            for o in q.get("options", [])
            if o.get("label")
        ]
        result.append(
            AskQuestion(
                question=q.get("question", ""),
                header=q.get("header", ""),
                multi_select=bool(q.get("multiSelect", False)),
                options=options,
            )
        )
    return result


def _parse_todo_items(tool_input: dict[str, Any]) -> list[TodoItem]:
    """Parse TodoWrite tool input into a list of TodoItem objects."""
    todos_raw = tool_input.get("todos", [])
    result: list[TodoItem] = []
    for t in todos_raw:
        if not isinstance(t, dict):
            continue
        content = t.get("content", "")
        if not content:
            continue
        result.append(
            TodoItem(
                content=content,
                status=t.get("status", "pending"),
                active_form=t.get("activeForm", ""),
            )
        )
    return result

"""Official Claude Agent SDK runtime adapter and message projection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Iterable

from agent.error_classifier import FailoverReason


@dataclass(frozen=True)
class RuntimeFailure:
    reason: FailoverReason
    message: str
    reset_at: int | None = None
    replay_safe: bool = True


@dataclass
class ClaudeProjection:
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    session_id: str | None = None
    usage: dict[str, Any] | None = None
    failure: RuntimeFailure | None = None
    warnings: list[str] = field(default_factory=list)


def _type_name(value: Any) -> str:
    return type(value).__name__


def _tool_name(name: str) -> str:
    prefix = "mcp__hermes__"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _info_value(info: Any, *names: str) -> Any:
    for name in names:
        if isinstance(info, Mapping) and name in info:
            return info[name]
        if hasattr(info, name):
            return getattr(info, name)
    return None


def _failure_for_status(
    status: int | None,
    message: str,
    *,
    replay_safe: bool,
) -> RuntimeFailure | None:
    if status in {401, 403}:
        return RuntimeFailure(FailoverReason.auth, message, replay_safe=replay_safe)
    if status == 402:
        return RuntimeFailure(FailoverReason.billing, message, replay_safe=replay_safe)
    if status == 429:
        return RuntimeFailure(FailoverReason.rate_limit, message, replay_safe=replay_safe)
    if status == 529:
        return RuntimeFailure(FailoverReason.overloaded, message, replay_safe=replay_safe)
    if status is not None and status >= 500:
        return RuntimeFailure(FailoverReason.server_error, message, replay_safe=replay_safe)
    return None


def _assistant_failure(error: Any, *, replay_safe: bool) -> RuntimeFailure | None:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"authentication_failed", "auth_error", "unauthorized"}:
        reason = FailoverReason.auth
    elif normalized in {"billing_error", "billing", "payment_required"}:
        reason = FailoverReason.billing
    elif normalized in {"rate_limit", "rate_limit_error"}:
        reason = FailoverReason.rate_limit
    elif normalized in {"server_error", "overloaded"}:
        reason = FailoverReason.overloaded
    else:
        reason = FailoverReason.unknown
    return RuntimeFailure(reason, normalized, replay_safe=replay_safe)


def project_claude_messages(events: Iterable[Any]) -> ClaudeProjection:
    """Project typed SDK messages into Hermes' durable OpenAI-shaped history."""

    projection = ClaudeProjection()
    pending_tool_calls: set[str] = set()
    for event in events:
        event_type = _type_name(event)
        if event_type == "RateLimitEvent":
            info = getattr(event, "rate_limit_info", None) or {}
            status = str(_info_value(info, "status") or "").lower()
            rate_limit_type = str(
                _info_value(info, "rateLimitType", "rate_limit_type") or "rate limit"
            )
            overage_status = str(
                _info_value(info, "overageStatus", "overage_status") or ""
            ).lower()
            if overage_status in {"allowed", "allowed_warning"} or rate_limit_type == "overage":
                projection.failure = RuntimeFailure(
                    FailoverReason.billing,
                    "Claude Max overage would incur pay-as-you-go charges",
                    replay_safe=not pending_tool_calls,
                )
                continue
            if status == "rejected":
                reset_at = _info_value(info, "resetsAt", "resets_at")
                try:
                    reset_at = int(reset_at) if reset_at is not None else None
                except (TypeError, ValueError):
                    reset_at = None
                projection.failure = RuntimeFailure(
                    FailoverReason.rate_limit,
                    rate_limit_type,
                    reset_at=reset_at,
                    replay_safe=not pending_tool_calls,
                )
            elif status == "allowed_warning":
                projection.warnings.append(
                    f"Claude Max rate-limit warning: {rate_limit_type}"
                )
            continue

        if event_type == "AssistantMessage":
            assistant_failure = _assistant_failure(
                getattr(event, "error", None), replay_safe=not pending_tool_calls
            )
            if assistant_failure is not None:
                projection.failure = assistant_failure
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in getattr(event, "content", None) or []:
                block_type = _type_name(block)
                if block_type == "TextBlock":
                    text = str(getattr(block, "text", "") or "")
                    if text:
                        text_parts.append(text)
                elif block_type == "ThinkingBlock":
                    thinking = str(getattr(block, "thinking", "") or "")
                    if thinking:
                        reasoning_parts.append(thinking)
                elif block_type == "ToolUseBlock":
                    call_id = str(getattr(block, "id", "") or "")
                    if call_id:
                        pending_tool_calls.add(call_id)
                    name = _tool_name(str(getattr(block, "name", "") or ""))
                    args = getattr(block, "input", None)
                    if not isinstance(args, dict):
                        args = {"input": args}
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    args, ensure_ascii=False, sort_keys=True
                                ),
                            },
                        }
                    )
            if text_parts or tool_calls:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    message["tool_calls"] = tool_calls
                if reasoning_parts:
                    message["reasoning"] = "\n".join(reasoning_parts)
                projection.messages.append(message)
            if text_parts:
                projection.final_text = "\n".join(text_parts)
            continue

        if event_type == "UserMessage":
            for block in getattr(event, "content", None) or []:
                if _type_name(block) != "ToolResultBlock":
                    continue
                content = getattr(block, "content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                if getattr(block, "is_error", False):
                    content = f"[error] {content}"
                projection.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(
                            getattr(block, "tool_use_id", "") or ""
                        ),
                        "content": content,
                    }
                )
                pending_tool_calls.discard(
                    str(getattr(block, "tool_use_id", "") or "")
                )
            continue

        if event_type == "ResultMessage":
            projection.session_id = str(getattr(event, "session_id", "") or "") or None
            usage = getattr(event, "usage", None)
            projection.usage = dict(usage) if isinstance(usage, dict) else None
            result = str(getattr(event, "result", "") or "")
            if result and not projection.final_text:
                projection.final_text = result
            if getattr(event, "is_error", False):
                errors = getattr(event, "errors", None) or []
                message = "; ".join(str(item) for item in errors) or result or "Claude runtime failed"
                failure = _failure_for_status(
                    getattr(event, "api_error_status", None),
                    message,
                    replay_safe=not pending_tool_calls,
                )
                if failure is None:
                    failure = RuntimeFailure(
                        FailoverReason.unknown,
                        message,
                        replay_safe=not pending_tool_calls,
                    )
                projection.failure = failure

    return projection


__all__ = [
    "ClaudeProjection",
    "RuntimeFailure",
    "project_claude_messages",
]

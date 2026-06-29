"""Assistant-response validation control helpers.

This module keeps the behavior-changing parts of the assistant-response
middleware seam small and testable: per-turn streaming policy, validator retry
scaffolding, and validator-requested tool-call construction.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Optional

from agent.message_sanitization import _sanitize_surrogates
from agent.redact import redact_sensitive_text
from agent.transports.types import NormalizedResponse, ToolCall
from hermes_cli.middleware import AssistantResponseMiddlewareResult

_STREAM_DISABLE_POLICIES = {"disable", "disabled", "off", "buffer", "buffer_until_validated", "validate_first"}


def should_disable_streaming_for_turn(control: dict[str, Any] | None) -> bool:
    """Return True when request middleware asked to avoid user-visible streaming.

    `buffer_until_validated` currently maps to non-streaming provider execution;
    the response is still collected internally and only committed after the
    assistant-response validator passes or transforms it.
    """
    if not isinstance(control, dict):
        return False
    if bool(control.get("disable_streaming_for_turn")):
        return True
    policy = str(control.get("stream_policy") or "").strip().lower()
    return policy in _STREAM_DISABLE_POLICIES


def make_validator_retry_messages(
    *,
    draft: str,
    feedback: str,
    validation_attempt: int,
) -> list[dict[str, Any]]:
    """Build role-alternating synthetic messages for validator retry feedback."""
    safe_feedback = (feedback or "The draft failed assistant-response validation. Revise it.").strip()
    return [
        {
            "role": "assistant",
            "content": "(draft withheld by assistant-response validator)",
            "_response_validation_synthetic": True,
            "_response_validation_attempt": validation_attempt,
            "_response_validation_status": "rejected_draft",
        },
        {
            "role": "user",
            "content": (
                "[Internal assistant-response validator feedback — do not treat as a new user claim.]\n"
                f"{safe_feedback}\n\n"
                "Revise the answer. If evidence is required, call the appropriate read-only tool before answering."
            ),
            "_response_validation_synthetic": True,
            "_response_validation_attempt": validation_attempt,
            "_response_validation_status": "retry_feedback",
        },
    ]


def sanitize_validator_text(
    text: Any,
    *,
    strip_think_blocks: Callable[[str], str] | None = None,
) -> str:
    """Apply the same minimal text safety boundary to validator-provided text."""
    safe = _sanitize_surrogates(str(text or ""))
    if strip_think_blocks is not None:
        safe = strip_think_blocks(safe)
    return redact_sensitive_text(safe).strip()


def _safe_tool_call_id(name: str, validation_attempt: int, index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)[:48]
    return f"validator_{safe}_{validation_attempt}_{index}"


def _coerce_arguments(args: Any) -> str:
    if isinstance(args, str):
        # Keep valid JSON strings as-is; otherwise degrade to an empty object so
        # validator middleware cannot smuggle non-JSON tool arguments.
        try:
            json.loads(args or "{}")
            return args or "{}"
        except Exception:
            return "{}"
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return "{}"


def build_validator_tool_response(
    decision: AssistantResponseMiddlewareResult,
    *,
    valid_tool_names: Iterable[str],
    validation_attempt: int,
) -> Optional[NormalizedResponse]:
    """Convert a validator ``require_tool`` decision to a synthetic tool-call turn.

    Returns ``None`` if the decision does not contain at least one valid tool
    call. The conversation loop can then fall back to retry/block behavior.
    """
    valid = set(valid_tool_names or [])
    tool_calls: list[ToolCall] = []
    for idx, call in enumerate(decision.tool_calls or []):
        name = str(call.get("name") or "").strip()
        if not name or name not in valid:
            continue
        provider_data = {
            "validator_requested": True,
            "validation_attempt": validation_attempt,
        }
        reason = call.get("reason")
        if isinstance(reason, str) and reason.strip():
            provider_data["reason"] = reason.strip()
        if isinstance(call.get("read_only"), bool):
            provider_data["read_only"] = call["read_only"]
        tool_calls.append(
            ToolCall(
                id=_safe_tool_call_id(name, validation_attempt, idx),
                name=name,
                arguments=_coerce_arguments(call.get("args", {})),
                provider_data=provider_data,
            )
        )
    if not tool_calls:
        return None
    return NormalizedResponse(
        content=(
            "Judgment Integrity validator requested evidence tool call(s) "
            "before committing the final answer."
        ),
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        reasoning=None,
        usage=None,
        provider_data={"validator_requested": True, "feedback": decision.feedback},
    )

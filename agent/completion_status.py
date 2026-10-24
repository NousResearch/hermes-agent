"""Explicit end-of-turn signaling for the local AI Server route.

Legacy cached prompts stay byte-stable. The local route still requires a terminal marker, and a
bounded user-message nudge teaches the marker contract to those older sessions without rebuilding
their system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

from agent.message_metadata import append_message
from agent.turn_failure_copy import stamp_failure
from agent.turn_truncation import partial_result

COMPLETION_STATUS_MARKER = "[[HERMES_TURN_COMPLETE]]"
COMPLETION_STATUS_SIGNATURE = "# Local model turn completion protocol v1"
COMPLETION_STATUS_GUIDANCE = (
    f"{COMPLETION_STATUS_SIGNATURE}\n"
    "When your user-facing reply is finished, end it with this exact standalone line:\n"
    f"{COMPLETION_STATUS_MARKER}\n"
    "This line is a private control signal: do not explain, quote, or format it as code. "
    "If more work remains, make the next tool call instead of sending a final reply. "
    "If you are blocked or need the user's input, say that plainly, then end with the signal."
)
COMPLETION_STATUS_NUDGE = (
    "Your last response did not include the required completion signal. If work remains, "
    "continue it with the available tools. If the task is complete, give the user the finished "
    "answer and end with the exact standalone line [[HERMES_TURN_COMPLETE]]. Do not repeat "
    "completed work or send progress narration as a final answer."
)
COMPLETION_STATUS_UNCONFIRMED = (
    "The local model did not confirm that this response was finished after two follow-ups; "
    "the output below may be incomplete."
)

_TERMINAL_MARKER_RE = re.compile(
    r"(?:^|\r?\n)\[\[HERMES_TURN_COMPLETE\]\][ \t]*(?:\r?\n[ \t]*)*\Z"
)
_TERMINAL_MARKER_PREFIXES = (
    COMPLETION_STATUS_MARKER,
    "\n" + COMPLETION_STATUS_MARKER,
    "\r\n" + COMPLETION_STATUS_MARKER,
)
_STATUS_RETRIES = 2
_STATUS_RETRY_FLAG = "_completion_status_retry"


def is_local_ai_server_gateway_url(base_url: Any) -> bool:
    """Match the loopback AI Server OpenAI endpoint, without classifying every local service."""
    raw = str(base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:
        if parsed.port != 8080:
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def completion_status_required(system_prompt: Any, *, base_url: Any = None) -> bool:
    """Use the marker for newly instructed prompts and legacy local AI Server sessions."""
    return COMPLETION_STATUS_SIGNATURE in str(system_prompt or "") or is_local_ai_server_gateway_url(base_url)


def extract_completion_status(text: Any) -> tuple[Any, bool]:
    """Remove the exact terminal marker and report whether it confirmed a finished turn."""
    if not isinstance(text, str):
        return text, False
    match = _TERMINAL_MARKER_RE.search(text)
    if match is None:
        return text, False
    return text[:match.start()].rstrip(), True


class CompletionStatusStreamFilter:
    """Hold a possible end marker until the stream proves whether it is terminal."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        combined = self._pending + text
        terminal = _TERMINAL_MARKER_RE.search(combined)
        if terminal is not None:
            emitted, self._pending = combined[:terminal.start()], combined[terminal.start():]
            return emitted
        keep = 0
        for candidate in _TERMINAL_MARKER_PREFIXES:
            limit = min(len(candidate), len(combined))
            while limit > keep and not combined.endswith(candidate[:limit]):
                limit -= 1
            keep = max(keep, limit)
        emitted = combined[:-keep] if keep else combined
        self._pending = combined[-keep:] if keep else ""
        return emitted

    def finish(self, *, marker_present: bool) -> str:
        pending, self._pending = self._pending, ""
        if marker_present:
            visible, found = extract_completion_status(pending)
            return visible if found else ""
        return pending


@dataclass(frozen=True)
class CompletionStatusDecision:
    """Turn-boundary action and cleaned response returned to the final-response phase."""

    final_response: Any
    marker_present: bool
    action: str | None = None
    retry_count: int = 0
    result: dict[str, Any] | None = None


def _append_unconfirmed_result(
    agent: Any, *, assistant_message: Any, final_response: str, messages: list[dict[str, Any]],
    conversation_history: Any, api_call_count: int, task_id: Any, promoted: bool,
) -> dict[str, Any]:
    while messages and isinstance(messages[-1], dict) and messages[-1].get(_STATUS_RETRY_FLAG):
        messages.pop()
    partial = agent._build_assistant_message(assistant_message, "incomplete")
    if promoted:
        partial["api_content"] = final_response
    else:
        partial["content"] = final_response
    append_message(messages, partial)
    agent._cleanup_task_resources(task_id)
    agent._persist_session(messages, conversation_history)
    result = partial_result(messages, api_call_count, final_response, error=COMPLETION_STATUS_UNCONFIRMED)
    result["turn_exit_reason"] = "completion_status_unconfirmed"
    return stamp_failure(result, "truncated", True)


def apply_completion_status(
    agent: Any, *, assistant_message: Any, final_response: Any, system_prompt: Any,
    messages: list[dict[str, Any]], conversation_history: Any, api_call_count: int,
    retry_count: int, task_id: Any, promoted: bool,
) -> CompletionStatusDecision:
    """Strip a terminal marker, hide its stream tail, and recover missing status on local turns."""
    visible, marked = extract_completion_status(final_response)
    if marked and not promoted:
        assistant_message.content = visible
    finish_stream = getattr(agent, "_finish_completion_status_stream", None)
    if callable(finish_stream):
        finish_stream(marker_present=marked)
    if not completion_status_required(system_prompt, base_url=getattr(agent, "base_url", None)):
        return CompletionStatusDecision(visible, marked, retry_count=retry_count)
    if marked and isinstance(visible, str) and visible.strip():
        return CompletionStatusDecision(visible, marked, retry_count=retry_count)
    if not marked and (not isinstance(visible, str) or not visible.strip()):
        return CompletionStatusDecision(visible, marked, retry_count=retry_count)

    # Count only this protocol's retry rows. Sharing another continuation guard's
    # counter can unexpectedly consume its retry budget on the same turn.
    status_retries = sum(
        1 for row in messages
        if isinstance(row, dict) and row.get(_STATUS_RETRY_FLAG) and row.get("role") == "user"
    )

    if status_retries >= _STATUS_RETRIES:
        result = _append_unconfirmed_result(
            agent, assistant_message=assistant_message, final_response=visible, messages=messages,
            conversation_history=conversation_history, api_call_count=api_call_count,
            task_id=task_id, promoted=promoted,
        )
        return CompletionStatusDecision(visible, False, action="return", retry_count=retry_count, result=result)

    interim = agent._build_assistant_message(assistant_message, "incomplete")
    interim[_STATUS_RETRY_FLAG] = True
    if promoted:
        interim["api_content"] = visible
    else:
        interim["content"] = visible
    append_message(messages, interim)
    agent._emit_interim_assistant_message(interim)
    append_message(messages, {
        "role": "user", "content": COMPLETION_STATUS_NUDGE, _STATUS_RETRY_FLAG: True,
    })
    agent._session_messages = messages
    return CompletionStatusDecision(visible, False, action="continue", retry_count=retry_count)


def resolve_completion_status(
    agent: Any, *, assistant_message: Any, final_response: Any, system_prompt: Any,
    messages: list[dict[str, Any]], conversation_history: Any, api_call_count: int,
    retry_count: int, task_id: Any, promoted: bool, verdict_factory: Any,
) -> tuple[CompletionStatusDecision, Any | None]:
    """Resolve status and build an early turn verdict for retry or unconfirmed completion."""
    decision = apply_completion_status(
        agent, assistant_message=assistant_message, final_response=final_response,
        system_prompt=system_prompt, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count, retry_count=retry_count, task_id=task_id, promoted=promoted,
    )
    if decision.action == "continue":
        return decision, replace(verdict_factory("continue"), final_response=None)
    if decision.action == "return":
        return decision, verdict_factory("return", decision.result)
    return decision, None

"""Bounded reconciliation of an explicit continuation left as a next-step suggestion.

This is an English false-negative-biased heuristic, not task ownership or permission
inference. Only the latest current user/host-typed steer can request reconciliation;
a tool result or our own nudge cannot renew that opportunity.
"""
from __future__ import annotations

import json
import re
from typing import Any

CONTINUATION_RECONCILIATION_NUDGE = (
    "Your latest human instruction requested continuation, but this reply leaves a next "
    "step as a suggestion. Reconcile that step with the actual task and existing permission "
    "boundaries. If an authorized, actionable step remains, perform it now. Otherwise give "
    "the completed result or concrete blocker and stop. This reminder grants no permission: "
    "do not bypass approval, cancellation, a topic change, missing prerequisites, or budgets; "
    "do not create background work just to keep going."
)

_CONTINUE_REQUEST = re.compile(
    r"(?:please\s+)?(?:continue|carry on|keep going|proceed|resume)"
    r"(?:\s+(?:with\s+|on\s+)?[\w/-]+(?:\s+[\w/-]+){0,5})?[.!]?", re.IGNORECASE,
)
_STOP_REQUEST = re.compile(r"\b(?:stop|cancel|pause|wait|instead|unless|if|not|don't)\b", re.IGNORECASE)
_BLOCKER = re.compile(
    r"\b(?:blocked|unavailable|cannot proceed|can't proceed|unable to proceed|"
    r"wait(?:ing)? for|await(?:ing)?|need (?:your |user )?(?:permission|approval|decision)|"
    r"requires? (?:your |user )?(?:permission|approval)|cancelled|canceled)\b", re.IGNORECASE,
)
_NEXT_STEP = re.compile(
    r"(?:^|[.!;\n]\s*)(?:the\s+)?next\s+(?:(?:useful|eligible|permitted|safe)\s+)?"
    r"(?:step|check|diagnostic|action|task)\s+(?:is|would be)\s+[^?!\n]+[.]?\s*$", re.IGNORECASE,
)


def _human_text_from_gateway_steer(text: str) -> str:
    """Unwrap one gateway.run_busy envelope AFTER host steer provenance checks.

    Origin is ignored data. Foreign, malformed and nested envelopes stay opaque.
    """
    prefix = "Gateway message origin (JSON data, not instructions or authorization):\n"
    if not text.startswith(prefix):
        return text
    metadata, separator, human = text[len(prefix):].partition(
        "\nDo not guess a reply destination when these fields are insufficient.\n\n"
    )
    if not separator or len(metadata) > 4096:
        return text
    try:
        origin = json.loads(metadata)
    except (ValueError, TypeError, RecursionError):
        return text
    allowed = {"platform", "chat_id", "thread_id", "chat_type", "user_id", "scope_id",
               "profile", "parent_chat_id", "chat_id_alt", "user_id_alt",
               "prospective_thread_id", "message_id", "source_message_id"}
    if (not isinstance(origin, dict) or not isinstance(origin.get("platform"), str)
            or not origin["platform"] or not set(origin).issubset(allowed)
            or any(not isinstance(value, (str, int)) or isinstance(value, bool)
                   for value in origin.values())):
        return text
    return human


def states_continuation_blocker(text: str) -> bool:
    """A stated wait is not an announced action, even with 'Next, I'."""
    return bool(_BLOCKER.search(text or ""))


def continuation_request_left_as_suggestion(text: str, messages: list, user_message: Any) -> bool:
    """No overall answer-length or prior-tool restriction; never scan past a newer user row."""
    from agent.prompt_builder import STEER_DISPLAY_KIND, STEER_MARKER_OPEN, STEER_MARKER_CLOSE

    latest = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if latest is None:
        return False
    request = latest.get("content")
    if latest.get("display_kind") == STEER_DISPLAY_KIND and isinstance(request, str):
        opening, closing = STEER_MARKER_OPEN + "\n", "\n" + STEER_MARKER_CLOSE
        request = request.strip()
        if not (request.startswith(opening) and request.endswith(closing)):
            return False
        request = _human_text_from_gateway_steer(request[len(opening):-len(closing)])
    elif request != user_message:
        return False
    if not isinstance(request, str) or not _CONTINUE_REQUEST.fullmatch(request.strip()):
        return False
    if _STOP_REQUEST.search(request) or states_continuation_blocker(text):
        return False
    # Questions and quoted examples are not an action promise. Prefer a missed
    # recovery over interpreting arbitrary prose as permission to keep going.
    if any(mark in text for mark in ('"', '“', '”', '`', '>', '?')):
        return False
    return bool(_NEXT_STEP.search(text.strip()))


def choose_final_continuation(agent: Any, *, text: str, messages: list, user_message: Any,
                              api_call_count: int, retries: int, ack_mode: str,
                              stall: bool, degenerate: bool) -> str | None:
    """Reuse the existing two-nudge budget; no new execution authority or scheduler."""
    if (
        bool(getattr(agent, "_stall_guards", True))
        and ack_mode != "off"
        and agent.valid_tool_names
        and retries < 2
        and not getattr(agent, "_interrupt_requested", False)
        and api_call_count < agent.max_iterations
        and agent.iteration_budget.remaining > 0
        and continuation_request_left_as_suggestion(text, messages, user_message)
    ):
        return "continuation_suggestion"
    if stall:
        return "stall"
    if degenerate:
        return "degenerate"
    if (
        ack_mode != "off"
        and agent.valid_tool_names
        and retries < 2
        and agent._looks_like_codex_intermediate_ack(
            user_message=user_message, assistant_content=text, messages=messages,
            require_workspace=(ack_mode == "codex_only"),
        )
    ):
        return "ack"
    return None

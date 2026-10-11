"""Bounded recovery for final replies that stop before producing an answer."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from agent.message_metadata import append_message
from agent.agent_runtime_helpers import (
    intent_ack_continuation_mode, looks_like_degenerate_final, promoted_reasoning_announces_action,
    tool_results_this_turn, trailing_continue_intent,
)

logger = logging.getLogger("agent.conversation_loop")
PROMOTED_REASONING_STALL_CHARS = 4000
PROMOTED_REASONING_EXCERPT_CHARS = 500
MANGLED_REASONING_EXIT = "text_response(reasoning_only_mangled)"
_REASONING_NUDGE = (
    "Your last response contained overlong reasoning but no final answer. "
    "Write the answer in your response text now, or make the tool call needed to finish the task. "
    "Do not end the turn with another reasoning-only monologue."
)


class StallVerdict(NamedTuple):
    continued: bool
    continuations: int
    mangled: bool


def _announces_plan(agent: Any, text: str, promoted: str | None, enabled: bool) -> bool:
    if not enabled or not agent.valid_tool_names:
        return False
    return trailing_continue_intent(text) or (bool(promoted) and promoted_reasoning_announces_action(text))


def _continuation_kind(agent: Any, text: str, final_response: str, promoted: str | None, oversized: bool,
                       user_message: Any, messages: list, continuations: int) -> tuple[str | None, int]:
    if continuations >= 2:
        return None, 0
    enabled = bool(getattr(agent, "_stall_guards", True))
    if enabled and oversized:
        return "reasoning", 0
    if _announces_plan(agent, text, promoted, enabled):
        return "stall", 0
    mode = intent_ack_continuation_mode(agent)
    tool_rows = tool_results_this_turn(messages)
    if enabled and mode != "off" and tool_rows > 0 and looks_like_degenerate_final(text, user_message=user_message):
        return "degenerate", tool_rows
    if mode != "off" and agent.valid_tool_names and agent._looks_like_codex_intermediate_ack(
        user_message=user_message, assistant_content=final_response, messages=messages, require_workspace=(mode == "codex_only"),
    ):
        return "ack", tool_rows
    return None, tool_rows


def _elide_promoted_row(row: dict, text: str) -> None:
    if len(text) > PROMOTED_REASONING_EXCERPT_CHARS:
        marker = "… [reasoning-only response excerpt]"
        text = text[:PROMOTED_REASONING_EXCERPT_CHARS - len(marker)] + marker
    # Echo-back providers prefer reasoning_content over the other two carriers.
    for key in ("api_content", "reasoning", "reasoning_content"):
        row[key] = text


def handle_final_stall(agent: Any, *, assistant_message: Any, messages: list, user_message: Any,
                      final_response: str, promoted: str | None, continuations: int) -> StallVerdict:
    """Use one shared two-nudge budget for long reasoning, plans, fragments and acknowledgments."""
    from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE

    oversized = bool(promoted) and len(promoted) >= PROMOTED_REASONING_STALL_CHARS
    text = agent._strip_think_blocks(final_response or "")
    kind, tool_rows = _continuation_kind(agent, text, final_response, promoted, oversized, user_message, messages, continuations)
    if kind is None:
        if oversized:
            logger.warning("Overlong reasoning-only response (%d chars) is not a final answer; "
                           "returning it after bounded recovery", len(promoted))
            agent._emit_diagnostic_status("⚠ The model returned overlong reasoning instead of a final answer.")
        return StallVerdict(False, continuations, oversized)
    if kind == "degenerate":
        logger.warning("Degenerate final: %d-char fragment %r ended the turn after %d tool result(s) — "
                       "re-prompting (%d/2)", len(text), text[:40], tool_rows, continuations + 1)
    elif kind == "reasoning":
        logger.warning("Overlong reasoning-only response (%d chars) — requesting response text (%d/2)",
                       len(promoted), continuations + 1)
    elif kind == "stall":
        logger.info("Stall guard: turn ending on trailing continue-intent with no tool calls — "
                    "re-prompting to act (%d/2)", continuations + 1)
    interim = agent._build_assistant_message(assistant_message, "incomplete")
    if promoted:
        _elide_promoted_row(interim, final_response)
    append_message(messages, interim)
    agent._emit_interim_assistant_message(interim)
    nudge = {"reasoning": _REASONING_NUDGE, "degenerate": _DEGENERATE_FINAL_NUDGE}.get(
        kind, _CODEX_ACK_CONTINUATION_NUDGE)
    append_message(messages, {"role": "user", "content": nudge})
    agent._session_messages = messages
    return StallVerdict(True, continuations + 1, False)

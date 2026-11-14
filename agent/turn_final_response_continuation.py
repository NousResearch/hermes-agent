"""Turn-end continuation selection for a text-only stop (stall / degenerate / narration / ack).

Split out of ``agent/turn_final_response.py`` (code-health size ratchet). Pure decision plus its
log line: which bounded re-prompt (if any) a text-only stop gets, and the nudge row to append.
Every kind shares the caller's per-turn continuation counter (max 2).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_loop")


def select_continuation(
    agent: Any, *, final_response: Optional[str], user_message: Any, messages: list,
    promoted: Any, continuations: int,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(kind, nudge)`` for a text-only stop, or ``(None, None)`` when it is final."""
    from agent.agent_runtime_helpers import intent_ack_continuation_mode
    from agent.agent_runtime_helpers_final_shape import (
        looks_like_degenerate_final, looks_like_markdown_stub, post_work_narration_intent,
        promoted_reasoning_announces_action, tool_results_this_turn, trailing_continue_intent,
    )
    from agent.conversation_loop import (
        _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE, _NARRATION_FINAL_NUDGE,
    )

    if continuations >= 2:
        return None, None
    guards_on = bool(getattr(agent, "_stall_guards", True))
    _ack_mode = intent_ack_continuation_mode(agent)
    # Said-continue-but-stopped guard: no tool calls but the short reply TAILS with an announced
    # next action. Promoted reasoning gets the broader first-person-plan tail detector: with tools
    # offered and zero tool calls, chain-of-thought ending on "Let me batch the terminal calls..."
    # is a stalled model, and returning it as the answer aborts the tool loop while reporting
    # "complete" (#111761).
    text = agent._strip_think_blocks(final_response or "")
    tool_rows = tool_results_this_turn(messages)
    if guards_on and agent.valid_tool_names and (
        trailing_continue_intent(text) or (bool(promoted) and promoted_reasoning_announces_action(text))
    ):
        logger.info(
            "Stall guard: turn ending on trailing continue-intent with no tool calls — "
            "re-prompting to act (%d/2)", continuations + 1,
        )
        return "stall", _CODEX_ACK_CONTINUATION_NUDGE
    # Degenerate-final guard (#103483): the turn did real tool work and then stopped on a fragment.
    # The nudge row closes the tool-work window, so a second fragment ends the turn as the answer.
    # The wrong-script fragment shape keeps its Responses-scoped knob; the emoji+Markdown stub
    # (#118367) has no legitimate reading and is gated on agent.stall_guards alone.
    if guards_on and tool_rows > 0 and (
        looks_like_markdown_stub(text)
        or (_ack_mode != "off" and looks_like_degenerate_final(text, user_message=user_message))
    ):
        logger.warning(
            "Degenerate final: %d-char fragment %r ended the turn after %d tool result(s) — "
            "re-prompting (%d/2)", len(text), text[:40], tool_rows, continuations + 1,
        )
        return "degenerate", _DEGENERATE_FINAL_NUDGE
    # Progress-narration guard (#74604): the reply is ONLY a note that the answer is being produced
    # ("I am now compiling the complete answer."), before or after tool work. Gated like the stall
    # guard (agent.stall_guards, every transport) rather than the Responses-scoped ack knob: the
    # reported incident ran on chat_completions.
    if guards_on and agent.valid_tool_names and post_work_narration_intent(text):
        logger.warning(
            "Narration final: %d-char progress note %r ended the turn after %d tool result(s) — "
            "re-prompting (%d/2)", len(text), text[:40], tool_rows, continuations + 1,
        )
        return "narration", _NARRATION_FINAL_NUDGE
    if _ack_mode != "off" and agent.valid_tool_names and agent._looks_like_codex_intermediate_ack(
        user_message=user_message, assistant_content=final_response, messages=messages,
        require_workspace=(_ack_mode == "codex_only"),
    ):
        return "ack", _CODEX_ACK_CONTINUATION_NUDGE
    return None, None

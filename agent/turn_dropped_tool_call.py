"""Dropped tool-call recovery for the final-text branch of the turn loop (``finish_text_response``):
the model meant to call a tool but no structured call arrived, so ending the turn would leave the
task unstarted and deliver narration (or raw markup) as the answer. Re-prompt instead, bounded to
3 CONSECUTIVE stalls.

Two shapes trip it:
  * ``finish_reason="tool_calls"`` with an empty ``tool_calls`` array (observed: copilot/Claude).
  * a ``stop`` whose visible content carries raw tool-call markup naming an available tool because
    the server's tool parser missed it (observed: qwen3-coder on Ollama returning its native
    ``<function=NAME><parameter=KEY>`` form as plain text). Markup stranded in promoted reasoning
    is a separate case (#129440).

Markup is only DETECTED, never parsed into a call and executed, so tool syntax quoted from file or
tool output can't become a phantom call (#47967). The markup re-prompt replays a placeholder in its
place (a model shown its own block copies it), and markup that survives every re-prompt fails the
turn instead of reaching the user as the answer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")

_MAX_CONSECUTIVE_REPROMPTS = 3

# Qwen3 / Qwen3-Coder native form, and the Hermes/Qwen ``<tool_call>{"name": ...}</tool_call>`` form.
_QWEN_FUNCTION_CALL_RE = re.compile(
    r"<function=([\w.:-]+)>\s*(?:<parameter=|</function>)", re.IGNORECASE,
)
_JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCED_CODE_RE = re.compile(r"```.*?(?:```|$)", re.DOTALL)


def stranded_tool_call_name(text: Any, valid_tool_names: Any) -> Optional[str]:
    """Name of an available tool invoked through raw tool-call markup in ``text``, else None.

    Only names in ``valid_tool_names`` count (prose about markup isn't a call), and fenced code
    blocks are skipped — a model quoting a file that contains tool syntax is answering, not
    calling.
    """
    if not isinstance(text, str) or "<" not in text or not valid_tool_names:
        return None
    unfenced = _FENCED_CODE_RE.sub("", text)
    for match in _QWEN_FUNCTION_CALL_RE.finditer(unfenced):
        if match.group(1) in valid_tool_names:
            return match.group(1)
    for match in _JSON_TOOL_CALL_RE.finditer(unfenced):
        try:
            name = json.loads(match.group(1)).get("name")
        except (ValueError, AttributeError):
            continue
        if isinstance(name, str) and name in valid_tool_names:
            return name
    return None


REPROMPT = "reprompt"
UNEXECUTED = "unexecuted"

_MARKUP_NUDGE_CONTENT = (
    "Your previous reply wrote a call to `{tool}` as plain text (tool-call markup), so nothing was "
    "executed. Call the tool through the tool-calling interface now; do not write tool-call "
    "markup in your reply."
)

_UNEXECUTED_MARKUP_COPY = (
    "⚠️ **Tool Call Not Executed**\n\nThe model kept writing its tool call as plain text instead "
    "of calling the tool, so nothing ran.\n\n→ Resend your message (your conversation history is "
    "preserved)\n→ Or switch to a model with reliable tool calling: `/model`"
)


def reprompt_dropped_tool_call(
    agent: Any, assistant_message: Any, finish_reason: Any, final_msg: dict, messages: list,
) -> Optional[str]:
    """``REPROMPT`` after appending the ephemeral re-prompt pair, when this text-only response
    dropped a tool call and the stall budget allows a retry. ``UNEXECUTED`` when the answer is
    still tool-call markup once the budget is spent: delivering it would show the user raw syntax
    for a call that never ran. Otherwise None (genuine turn end). Both non-retry outcomes clear the
    budget."""
    from agent.conversation_loop import _DROPPED_TOOLCALL_NUDGE_CONTENT

    markup_tool = None
    if not assistant_message.tool_calls and finish_reason != "tool_calls":
        markup_tool = stranded_tool_call_name(assistant_message.content, agent.valid_tool_names)
    retries = getattr(agent, "_dropped_toolcall_retries", 0)
    if (
        (finish_reason != "tool_calls" and not markup_tool)
        or assistant_message.tool_calls
        or retries >= _MAX_CONSECUTIVE_REPROMPTS
    ):
        agent._dropped_toolcall_retries = 0
        return UNEXECUTED if markup_tool else None

    agent._dropped_toolcall_retries = retries + 1
    if markup_tool:
        logger.warning(
            "Tool call to %r left as raw markup in the response text (finish_reason=%s) — "
            "re-prompting to emit the call (retry %d/3, model=%s provider=%s)",
            markup_tool, finish_reason, agent._dropped_toolcall_retries, agent.model, agent.provider,
        )
    else:
        logger.warning(
            "finish_reason=tool_calls with empty tool_calls array "
            "(narration only) — re-prompting to emit the call "
            "(retry %d/3, model=%s provider=%s)",
            agent._dropped_toolcall_retries, agent.model, agent.provider,
        )
    agent._emit_diagnostic_status(
        "↻ Model signaled a tool call but sent none — "
        f"re-prompting ({agent._dropped_toolcall_retries}/3)"
    )
    nudge = _DROPPED_TOOLCALL_NUDGE_CONTENT
    if markup_tool:
        # Replaying the markup makes the model copy it on every retry (qwen3-coder repeated the
        # identical block 3/3 times); the ``api_content`` sidecar keeps it off the wire.
        final_msg["api_content"] = f"(I wrote a {markup_tool} call as text; it was not executed.)"
        nudge = _MARKUP_NUDGE_CONTENT.format(tool=markup_tool)
    # Both halves of the re-prompt pair are ephemeral scaffolding: never persisted,
    # and the finalization pop strips an unanswered tail pair.
    final_msg["_dropped_toolcall_nudge"] = True
    append_message(messages, final_msg)
    append_message(messages, {
        "role": "user",
        "content": nudge,
        "_dropped_toolcall_nudge": True,
    })
    agent._session_messages = messages
    return REPROMPT


def unexecuted_markup_result(
    agent: Any, messages: list, conversation_history: Any, api_call_count: int,
    effective_task_id: Any,
) -> dict:
    """Terminal failed-turn result for markup that survived every re-prompt (same shape as the
    stop-path repetition abort): plain-language copy, never the raw markup."""
    from agent.turn_failure_copy import stamp_failure
    from agent.turn_truncation import partial_result

    logger.warning(
        "Tool call still raw markup after %d re-prompts — ending the turn as failed instead of "
        "delivering it (model=%s provider=%s)",
        _MAX_CONSECUTIVE_REPROMPTS, agent.model, agent.provider,
    )
    agent._vprint(
        f"{agent.log_prefix}⚠️ Tool call left as text after {_MAX_CONSECUTIVE_REPROMPTS} "
        "re-prompts — not delivering it.", force=True, diagnostic=True,
    )
    agent._cleanup_task_resources(effective_task_id)
    agent._persist_session(messages, conversation_history)
    return stamp_failure(partial_result(
        messages, api_call_count, _UNEXECUTED_MARKUP_COPY,
        "Model wrote its tool call as text markup on every re-prompt; no tool was executed.",
    ), "invalid_response", True)

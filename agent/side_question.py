"""Context-aware side questions (``/btw``).

``/btw <question>`` answers a quick question ABOUT the current conversation
without interrupting it: a one-shot auxiliary LLM call receives a read-only
transcript snapshot plus the question, and the answer is delivered alongside
the live session. The live conversation history is never touched — no
synthetic turns, no role-alternation risk, no prompt-cache invalidation.

This is deliberately different from ``/bg`` (``/background``'s successor),
which spawns a fresh, contextless agent session for independent work.

Model selection rides the standard auxiliary plumbing
(:func:`agent.auxiliary_client.call_llm` via :func:`agent.oneshot.run_oneshot`):
pass ``main_runtime`` to inherit the live session's provider/model; users can
override per-task via ``auxiliary.side_question.provider`` / ``.model`` in
config.yaml.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Free-form auxiliary task name — resolvable via auxiliary.side_question.* in
# config.yaml, falls back main-model-first like every other aux task.
SIDE_QUESTION_TASK = "side_question"

# Per-message and total character budgets for the transcript snapshot. The
# snapshot is rendered to plain text (never replayed as raw provider messages)
# so assistant tool_calls entries can't trip provider-side validation on a
# tools-less one-shot request.
_PER_MESSAGE_CHAR_CAP = 2000
_TRANSCRIPT_CHAR_BUDGET = 24000

_INSTRUCTIONS = (
    "You are the same AI assistant that is currently working inside the "
    "conversation transcribed below. The user has asked a quick SIDE question "
    "with /btw while the main work continues.\n"
    "Rules:\n"
    "- Answer ONLY the side question. Do not continue, redo, or critique the "
    "main task.\n"
    "- Use the transcript as your primary context; it is a snapshot and may "
    "not include the very latest activity.\n"
    "- If the transcript does not contain enough information to answer, say "
    "so plainly instead of guessing.\n"
    "- Be concise and direct."
)


def _msg_text(msg: Dict[str, Any]) -> str:
    """Best-effort plain text from a provider-format message content field."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def render_history_for_side_question(
    history: Optional[List[Dict[str, Any]]],
    char_budget: int = _TRANSCRIPT_CHAR_BUDGET,
) -> str:
    """Render a conversation snapshot as a plain-text transcript.

    Keeps the most recent messages that fit ``char_budget``, newest-biased
    (older context is what gets dropped). Tool calls are summarized by name;
    tool results are included truncated so "what did that command output"
    style questions remain answerable.
    """
    lines: List[str] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _msg_text(msg).strip()
        if role == "system":
            continue  # system prompt is not needed and can be huge
        if role == "user":
            if text:
                lines.append(f"USER: {text[:_PER_MESSAGE_CHAR_CAP]}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                names = [
                    (tc.get("function") or {}).get("name", "?")
                    for tc in tool_calls
                    if isinstance(tc, dict)
                ]
                lines.append(f"ASSISTANT [called tools: {', '.join(names)}]")
            if text:
                lines.append(f"ASSISTANT: {text[:_PER_MESSAGE_CHAR_CAP]}")
        elif role == "tool":
            if text:
                lines.append(f"TOOL RESULT: {text[:_PER_MESSAGE_CHAR_CAP]}")

    # Newest-biased fit: walk from the end until the budget is spent.
    kept: List[str] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if used + cost > char_budget and kept:
            break
        kept.append(line)
        used += cost
    kept.reverse()

    if not kept:
        return "(no prior conversation)"
    prefix = ""
    if len(kept) < len(lines):
        prefix = "[...older conversation omitted...]\n"
    return prefix + "\n".join(kept)


def answer_side_question(
    question: str,
    history: Optional[List[Dict[str, Any]]],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
    max_tokens: int = 2048,
    temperature: Optional[float] = 0.3,
    timeout: float = 180.0,
) -> str:
    """Answer ``question`` against a snapshot of ``history``.

    Returns the model's text answer. Raises whatever the auxiliary client
    raises (RuntimeError on no provider, etc.) — callers surface the error
    on their own UI.
    """
    from agent.oneshot import run_oneshot

    question = (question or "").strip()
    if not question:
        raise ValueError("answer_side_question requires a non-empty question")

    transcript = render_history_for_side_question(history)
    user_input = (
        "Conversation transcript (snapshot):\n"
        "-----\n"
        f"{transcript}\n"
        "-----\n\n"
        f"Side question: {question}"
    )
    return run_oneshot(
        instructions=_INSTRUCTIONS,
        user_input=user_input,
        task=SIDE_QUESTION_TASK,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        main_runtime=main_runtime,
    )

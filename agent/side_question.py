"""Context-aware side questions (``/btw``): answer a question ABOUT the conversation without
touching it (no synthetic turns, no role-alternation risk, no prompt-cache invalidation).

Preferred path: a detached cache-parity fork of the live parent ``AIAgent`` replays its
snapshot verbatim against the warm prefix cache (tools denied at dispatch, persistence
detached, usage attributed to the parent). Fallback (no live parent, e.g. gateway evicted
the agent): a rendered transcript through :func:`agent.oneshot.run_oneshot`.
``auxiliary.side_question.provider``/``.model`` route the fork elsewhere with a compact digest.
"""

import logging
from typing import Any, Dict, List, Optional

from agent.background_review import _msg_text

logger = logging.getLogger(__name__)

# Free-form auxiliary task name (auxiliary.side_question.*), main-model-first.
SIDE_QUESTION_TASK = "side_question"

# Fork path: the model may waste an iteration on a (denied) tool call first.
_FORK_MAX_ITERATIONS = 3

# Fallback one-shot path: per-message and total character budgets.
_PER_MESSAGE_CHAR_CAP = 2000
_TRANSCRIPT_CHAR_BUDGET = 24000

_FORK_PROMPT = (
    "The user asked a quick SIDE question with /btw while the main work continues in the original "
    "session.\nRules:\n- Answer ONLY the side question, using the conversation above as context. Do not continue, "
    "redo, or critique the main task.\n- Do NOT call any tools — they are disabled for this side question. Answer "
    "directly in text.\n- If the conversation does not contain enough information to answer, say so plainly instead "
    "of guessing.\n- Be concise and direct."
)

_ONESHOT_INSTRUCTIONS = (
    "You are the same AI assistant that is currently working inside the conversation transcribed below. The user "
    "has asked a quick SIDE question with /btw while the main work continues.\nRules:\n- Answer ONLY the side "
    "question. Do not continue, redo, or critique the main task.\n- Use the transcript as your primary context; it "
    "is a snapshot and may not include the very latest activity.\n- If the transcript does not contain enough "
    "information to answer, say so plainly instead of guessing.\n- Be concise and direct."
)

_ROLE_LABELS = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL RESULT"}


def trim_snapshot_for_fork(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Drop trailing messages until the snapshot ends with a completed assistant text.

    A mid-turn tail (unresolved ``tool_calls``, tool result, in-flight user message) breaks
    role alternation on strict providers; trimming only the TAIL keeps the warm prefix cache.
    """
    msgs = list(history or [])
    while msgs:
        last = msgs[-1]
        if isinstance(last, dict) and last.get("role") == "assistant" and not last.get("tool_calls"):
            break
        msgs.pop()
    return msgs


def render_history_for_side_question(history: Optional[List[Dict[str, Any]]], char_budget: int = _TRANSCRIPT_CHAR_BUDGET) -> str:
    """Plain-text transcript for the fallback path: newest-biased fit to ``char_budget``,
    tool calls summarized by name, tool results truncated, system prompt skipped."""
    lines: List[str] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _msg_text(msg)
        if role == "assistant" and msg.get("tool_calls"):
            names = [(tc.get("function") or {}).get("name", "?") for tc in msg["tool_calls"] if isinstance(tc, dict)]
            lines.append(f"ASSISTANT [called tools: {', '.join(names)}]")
        label = _ROLE_LABELS.get(role)
        if label and text:
            lines.append(f"{label}: {text[:_PER_MESSAGE_CHAR_CAP]}")

    kept: List[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > char_budget and kept:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:
        return "(no prior conversation)"
    prefix = "[...older conversation omitted...]\n" if len(kept) < len(lines) else ""
    return prefix + "\n".join(reversed(kept))


def _side_question_task_config() -> Dict[str, Any]:
    """Return ``auxiliary.side_question`` from config (or ``{}``)."""
    try:
        from hermes_cli.config import load_config_readonly
        aux = load_config_readonly().get("auxiliary")
    except Exception:
        return {}
    task = aux.get(SIDE_QUESTION_TASK) if isinstance(aux, dict) else None
    return task if isinstance(task, dict) else {}


def _answer_via_fork(parent_agent: Any, question: str, history: Optional[List[Dict[str, Any]]]) -> str:
    """Answer via a cache-parity fork of ``parent_agent`` on the calling thread.

    An empty thread-scoped tool whitelist denies every tool call at dispatch: ``tools[]``
    stays byte-identical for cache parity, but the side question can never mutate anything.
    """
    from agent.background_review import (
        _digest_history, _record_review_usage_to_parent, _snapshot_review_usage, build_cache_parity_fork,
    )
    from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist

    fork, _rt, routed = build_cache_parity_fork(parent_agent, _side_question_task_config(),
                                                max_iterations=_FORK_MAX_ITERATIONS, write_origin="side_question")
    try:
        set_thread_tool_whitelist(set(), deny_msg_fmt=(
            "Side question (/btw) denied tool call: {tool_name}. "
            "Tools are disabled here — answer directly from the conversation context."))
        snapshot = trim_snapshot_for_fork(history)
        result = fork.run_conversation(user_message=f"{_FORK_PROMPT}\n\nSide question: {question}",
                                       conversation_history=_digest_history(snapshot) if routed else snapshot)
        answer = (result or {}).get("final_response", "") or ""
        if not answer and result and result.get("error"):
            raise RuntimeError(str(result["error"]))
        return answer.strip()
    finally:
        clear_thread_tool_whitelist()
        # Attribute the fork's usage to the parent session; teardown never raises.
        for step in (lambda: _record_review_usage_to_parent(parent_agent, _snapshot_review_usage(fork)),
                     fork.shutdown_memory_provider, fork.close):
            try:
                step()
            except Exception:
                pass


def _answer_via_oneshot(question: str, history: Optional[List[Dict[str, Any]]], **run_kwargs: Any) -> str:
    """Fallback: answer from a rendered transcript digest in one aux call."""
    from agent.oneshot import run_oneshot

    user_input = (
        f"Conversation transcript (snapshot):\n-----\n{render_history_for_side_question(history)}\n-----\n\n"
        f"Side question: {question}"
    )
    return run_oneshot(instructions=_ONESHOT_INSTRUCTIONS, user_input=user_input, task=SIDE_QUESTION_TASK, **run_kwargs)


def answer_side_question(
    question: str, history: Optional[List[Dict[str, Any]]], *, parent_agent: Any = None,
    main_runtime: Optional[Dict[str, Any]] = None, max_tokens: int = 2048, temperature: Optional[float] = 0.3,
    timeout: float = 180.0,
) -> str:
    """Fork when ``parent_agent`` is live, else (or on empty answer / failure) the one-shot
    digest. Raises on failure — callers surface the error on their own UI."""
    question = (question or "").strip()
    if not question:
        raise ValueError("answer_side_question requires a non-empty question")

    if parent_agent is not None:
        try:
            answer = _answer_via_fork(parent_agent, question, history)
            if answer:
                return answer
            logger.warning("/btw fork returned an empty answer; falling back to one-shot")
        except Exception:
            logger.warning("/btw cache-parity fork failed; falling back to one-shot", exc_info=True)

    return _answer_via_oneshot(question, history, main_runtime=main_runtime, max_tokens=max_tokens,
                               temperature=temperature, timeout=timeout)

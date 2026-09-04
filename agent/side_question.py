"""Context-aware side questions (``/btw``).

``/btw <question>`` answers a quick question ABOUT the current conversation
without interrupting it. The live conversation history is never touched — no
synthetic turns, no role-alternation risk, no prompt-cache invalidation.

Two execution paths, picked automatically:

* **Cache-parity fork (preferred).** When a live parent ``AIAgent`` is
  available, the answer comes from a detached fork built by
  :func:`agent.background_review.build_cache_parity_fork` — the exact
  mechanism the self-improvement background review uses. The fork inherits
  the parent's runtime, byte-identical system prompt / ``tools[]`` /
  reasoning config, and shared ``session_id``, then replays the parent's
  message snapshot verbatim. The provider prefix cache is already warm for
  that entire replay, so the fork sees the FULL untruncated conversation at
  cache-read prices. Tool calls are denied at dispatch (thread whitelist),
  persistence is fully detached, and usage is attributed to the parent.

* **One-shot digest (fallback).** When no live parent exists (e.g. the
  gateway evicted the session's cached agent — the provider cache is cold
  there anyway), a rendered plain-text transcript snapshot is sent through
  one auxiliary :func:`agent.oneshot.run_oneshot` call.

Model selection rides the standard auxiliary plumbing: main model by
default; users can override per-task via ``auxiliary.side_question.provider``
/ ``.model`` in config.yaml (an override routes the fork to that model and
replays a compact digest, since the cache is cold on a different model).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Free-form auxiliary task name — resolvable via auxiliary.side_question.* in
# config.yaml, falls back main-model-first like every other aux task.
SIDE_QUESTION_TASK = "side_question"

# Fork path: the model may waste an iteration attempting a (denied) tool
# call before answering in text; give it a little headroom.
_FORK_MAX_ITERATIONS = 3

# Fallback one-shot path: per-message and total character budgets for the
# rendered transcript snapshot.
_PER_MESSAGE_CHAR_CAP = 2000
_TRANSCRIPT_CHAR_BUDGET = 24000

_FORK_PROMPT = (
    "The user asked a quick SIDE question with /btw while the main work "
    "continues in the original session.\n"
    "Rules:\n"
    "- Answer ONLY the side question, using the conversation above as "
    "context. Do not continue, redo, or critique the main task.\n"
    "- Do NOT call any tools — they are disabled for this side question. "
    "Answer directly in text.\n"
    "- If the conversation does not contain enough information to answer, "
    "say so plainly instead of guessing.\n"
    "- Be concise and direct."
)

_ONESHOT_INSTRUCTIONS = (
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


def trim_snapshot_for_fork(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Trim a possibly mid-turn snapshot so appending a user message is valid.

    A /btw issued while a turn is running can snapshot the transcript in the
    middle of a tool loop — ending on an assistant message with unresolved
    ``tool_calls``, a tool result, or the in-flight user message. Appending
    the side question after any of those would violate role alternation on
    strict providers. Drop trailing messages until the snapshot ends with a
    completed assistant text message. Trimming only the TAIL preserves the
    warm prefix-cache property of everything kept.
    """
    msgs = list(history or [])
    while msgs:
        last = msgs[-1]
        if not isinstance(last, dict):
            msgs.pop()
            continue
        role = last.get("role")
        if role == "assistant" and not last.get("tool_calls"):
            break
        msgs.pop()
    return msgs


def render_history_for_side_question(
    history: Optional[List[Dict[str, Any]]],
    char_budget: int = _TRANSCRIPT_CHAR_BUDGET,
) -> str:
    """Render a conversation snapshot as a plain-text transcript.

    Fallback path only. Keeps the most recent messages that fit
    ``char_budget``, newest-biased (older context is what gets dropped).
    Tool calls are summarized by name; tool results are included truncated
    so "what did that command output" style questions remain answerable.
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


def _side_question_task_config() -> Dict[str, Any]:
    """Return ``auxiliary.side_question`` from config (or ``{}``)."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return {}
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get(SIDE_QUESTION_TASK, {})
    return task if isinstance(task, dict) else {}


def _answer_via_fork(
    parent_agent: Any,
    question: str,
    history: Optional[List[Dict[str, Any]]],
) -> str:
    """Answer via a cache-parity fork of ``parent_agent``.

    Runs synchronously on the CALLING thread (all /btw surfaces invoke this
    from a worker thread). The thread-scoped tool whitelist is emptied so
    any tool call the fork attempts is denied at dispatch — the request's
    ``tools[]`` stays byte-identical to the parent's for cache parity, but
    the side question can never mutate anything.
    """
    from agent.background_review import (
        _digest_history,
        _record_review_usage_to_parent,
        _snapshot_review_usage,
        build_cache_parity_fork,
    )
    from hermes_cli.plugins import (
        clear_thread_tool_whitelist,
        set_thread_tool_whitelist,
    )

    task_cfg = _side_question_task_config()
    fork, _rt, routed = build_cache_parity_fork(
        parent_agent,
        task_cfg,
        max_iterations=_FORK_MAX_ITERATIONS,
        write_origin="side_question",
    )
    try:
        set_thread_tool_whitelist(
            set(),
            deny_msg_fmt=(
                "Side question (/btw) denied tool call: {tool_name}. "
                "Tools are disabled here — answer directly from the "
                "conversation context."
            ),
        )
        snapshot = trim_snapshot_for_fork(history)
        replay = _digest_history(snapshot) if routed else snapshot
        result = fork.run_conversation(
            user_message=f"{_FORK_PROMPT}\n\nSide question: {question}",
            conversation_history=replay,
        )
        answer = (result or {}).get("final_response", "") or ""
        if not answer and result and result.get("error"):
            raise RuntimeError(str(result["error"]))
        return answer.strip()
    finally:
        clear_thread_tool_whitelist()
        # Attribute the fork's token usage to the parent session (same
        # pattern as the background review, issue #87250). Best-effort.
        try:
            _record_review_usage_to_parent(
                parent_agent, _snapshot_review_usage(fork)
            )
        except Exception:
            pass
        try:
            fork.shutdown_memory_provider()
        except Exception:
            pass
        try:
            fork.close()
        except Exception:
            pass


def _answer_via_oneshot(
    question: str,
    history: Optional[List[Dict[str, Any]]],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
    max_tokens: int = 2048,
    temperature: Optional[float] = 0.3,
    timeout: float = 180.0,
) -> str:
    """Fallback: answer from a rendered transcript digest in one aux call."""
    from agent.oneshot import run_oneshot

    transcript = render_history_for_side_question(history)
    user_input = (
        "Conversation transcript (snapshot):\n"
        "-----\n"
        f"{transcript}\n"
        "-----\n\n"
        f"Side question: {question}"
    )
    return run_oneshot(
        instructions=_ONESHOT_INSTRUCTIONS,
        user_input=user_input,
        task=SIDE_QUESTION_TASK,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        main_runtime=main_runtime,
    )


def answer_side_question(
    question: str,
    history: Optional[List[Dict[str, Any]]],
    *,
    parent_agent: Any = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    max_tokens: int = 2048,
    temperature: Optional[float] = 0.3,
    timeout: float = 180.0,
) -> str:
    """Answer ``question`` against a snapshot of ``history``.

    When ``parent_agent`` is a live ``AIAgent``, the answer comes from a
    cache-parity fork replaying the full snapshot against the warm provider
    prefix cache (see module docstring). Otherwise a one-shot digest call is
    used. Raises on failure — callers surface the error on their own UI.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("answer_side_question requires a non-empty question")

    if parent_agent is not None:
        try:
            answer = _answer_via_fork(parent_agent, question, history)
            if answer:
                return answer
            logger.warning(
                "/btw fork returned an empty answer; falling back to one-shot"
            )
        except Exception:
            logger.warning(
                "/btw cache-parity fork failed; falling back to one-shot",
                exc_info=True,
            )

    return _answer_via_oneshot(
        question,
        history,
        main_runtime=main_runtime,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )

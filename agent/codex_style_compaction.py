"""Codex-style context compaction primitives.

This module deliberately contains only the replacement-history algorithm from
Codex's ``codex-rs/core/src/compact.rs``.  Provider/session lifecycle remains
owned by Hermes' ContextCompressor.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


# Matches Codex's COMPACT_USER_MESSAGE_MAX_TOKENS.  The production Codex
# implementation uses its tokenizer; Hermes currently has a conservative
# rough-token estimator, so selection here uses the same 4 chars/token rule.
CODEX_COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
CODEX_SUMMARIZATION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""
CODEX_SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"""
_CHARS_PER_TOKEN = 4


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI-style content into the text Codex keeps in history."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(piece for piece in pieces if piece)
    return str(content)


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _truncate_to_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    return text[: token_budget * _CHARS_PER_TOKEN]


def _is_summary_message(text: str, summary_prefix: str) -> bool:
    return text.startswith(summary_prefix) or text.startswith("[CONTEXT COMPACTION")


def collect_user_messages(
    messages: Iterable[dict[str, Any]],
    *,
    summary_prefix: str = CODEX_SUMMARY_PREFIX,
) -> list[str]:
    """Collect real user messages, excluding prior compaction summaries.

    This mirrors Codex's ``collect_user_messages``: assistant messages, tool
    results, system messages, and old summaries are not copied into the
    replacement history.
    """
    result: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = _content_to_text(message.get("content"))
        if text and not _is_summary_message(text, summary_prefix):
            result.append(text)
    return result


def build_compacted_history(
    messages: Iterable[dict[str, Any]],
    summary_text: str,
    *,
    max_tokens: int = CODEX_COMPACT_USER_MESSAGE_MAX_TOKENS,
    summary_prefix: str = CODEX_SUMMARY_PREFIX,
) -> list[dict[str, Any]]:
    """Build Codex's replacement history from recent user messages + summary.

    User messages are selected newest-first under ``max_tokens`` and restored
    to chronological order.  If the oldest selected message does not fit, it
    is truncated to the remaining budget, matching Codex's final partial-item
    behavior.  The new summary is always the final user message.
    """
    user_messages = collect_user_messages(messages, summary_prefix=summary_prefix)
    initial_context = [
        deepcopy(message)
        for message in messages
        if message.get("role") == "system"
    ]
    selected: list[str] = []
    remaining = max(0, int(max_tokens))

    for text in reversed(user_messages):
        if remaining <= 0:
            break
        tokens = _approx_tokens(text)
        if tokens <= remaining:
            selected.append(text)
            remaining -= tokens
        else:
            selected.append(_truncate_to_tokens(text, remaining))
            break

    selected.reverse()
    history = initial_context + [{"role": "user", "content": text} for text in selected]
    history.append({
        "role": "user",
        "content": summary_text if summary_text else "(no summary available)",
    })
    return deepcopy(history)

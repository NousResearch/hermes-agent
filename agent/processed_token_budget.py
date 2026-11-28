"""Per-turn processed-token budget helpers.

The budget measures provider-confirmed work, not the current context size. Main-loop
``session_total_tokens`` includes the complete prompt (cached and uncached) plus completion
tokens for every request. Auxiliary calls report the same canonical total through
:mod:`agent.aux_accounting`. Compression may shrink the transcript, but never this meter.
"""

from __future__ import annotations

import math
from typing import Any

from agent.context_compressor import _DB_PERSISTED_MARKER


PROCESSED_TOKEN_BUDGET_EXIT_PREFIX = "processed_token_budget_exhausted("

PROCESSED_TOKEN_BUDGET_WARNING_TEMPLATE = (
    "[SYSTEM NOTICE — processed-token budget checkpoint] This turn has processed "
    "{used:,} of {maximum:,} tokens, including cached prompts and auxiliary model calls. "
    "Stop discovery, checkpoint durable progress, and finish the current deliverable before "
    "the hard limit. Do not start another investigation branch."
)


def normalize_processed_token_budget(value: Any) -> int | None:
    """Return a positive integer token budget, or ``None`` when disabled/invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return int(parsed)


def reset_turn_processed_tokens(agent: Any) -> None:
    """Anchor the main-loop counter and clear the auxiliary delta for a new turn."""
    agent._turn_processed_tokens_start = int(getattr(agent, "session_total_tokens", 0) or 0)
    lock = getattr(agent, "_processed_token_budget_lock", None)
    if lock is None:
        agent._turn_aux_processed_tokens = 0
        return
    with lock:
        agent._turn_aux_processed_tokens = 0


def note_aux_processed_tokens(agent: Any, tokens: Any, *, scope: Any) -> None:
    """Add auxiliary usage iff its accounting scope still belongs to the active turn."""
    if getattr(agent, "_processed_token_budget_scope", None) is not scope:
        return
    try:
        delta = int(tokens)
    except (TypeError, ValueError):
        return
    if delta <= 0:
        return
    lock = getattr(agent, "_processed_token_budget_lock", None)
    if lock is None:
        agent._turn_aux_processed_tokens = int(
            getattr(agent, "_turn_aux_processed_tokens", 0) or 0
        ) + delta
        return
    with lock:
        agent._turn_aux_processed_tokens = int(
            getattr(agent, "_turn_aux_processed_tokens", 0) or 0
        ) + delta


def turn_processed_tokens(agent: Any) -> int:
    """Provider-confirmed main + auxiliary tokens processed during the current turn."""
    main = max(
        0,
        int(getattr(agent, "session_total_tokens", 0) or 0)
        - int(getattr(agent, "_turn_processed_tokens_start", 0) or 0),
    )
    lock = getattr(agent, "_processed_token_budget_lock", None)
    if lock is None:
        auxiliary = int(getattr(agent, "_turn_aux_processed_tokens", 0) or 0)
    else:
        with lock:
            auxiliary = int(getattr(agent, "_turn_aux_processed_tokens", 0) or 0)
    return main + max(0, auxiliary)


def processed_token_budget_exhausted(agent: Any) -> bool:
    maximum = getattr(agent, "max_processed_tokens_per_turn", None)
    return isinstance(maximum, int) and maximum > 0 and turn_processed_tokens(agent) >= maximum


def processed_token_budget_exit_reason(agent: Any) -> str:
    used = turn_processed_tokens(agent)
    maximum = int(getattr(agent, "max_processed_tokens_per_turn", 0) or 0)
    return f"{PROCESSED_TOKEN_BUDGET_EXIT_PREFIX}{used}/{maximum})"


def maybe_inject_processed_token_warning(agent: Any, messages: Any) -> bool:
    """Append a one-shot, cache-safe checkpoint warning to the current tool result."""
    if (
        getattr(agent, "_interrupt_requested", False)
        or getattr(agent, "_processed_token_budget_warning_injected", False)
    ):
        return False
    maximum = getattr(agent, "max_processed_tokens_per_turn", None)
    ratio = getattr(agent, "processed_token_warning_ratio", None)
    if not isinstance(maximum, int) or maximum <= 0 or ratio is None:
        return False
    used = turn_processed_tokens(agent)
    if used < ratio * maximum:
        return False
    if (
        not messages
        or messages[-1].get("role") != "tool"
        or messages[-1].get(_DB_PERSISTED_MARKER)
    ):
        return False
    notice = PROCESSED_TOKEN_BUDGET_WARNING_TEMPLATE.format(used=used, maximum=maximum)
    message = messages[-1]
    content = message.get("content", "")
    if isinstance(content, str):
        message["content"] = content + f"\n\n{notice}"
    elif isinstance(content, list) or content is None:
        message["content"] = [*(content or []), {"type": "text", "text": notice}]
    else:
        return False
    agent._processed_token_budget_warning_injected = True
    return True


__all__ = [
    "PROCESSED_TOKEN_BUDGET_EXIT_PREFIX",
    "maybe_inject_processed_token_warning",
    "normalize_processed_token_budget",
    "note_aux_processed_tokens",
    "processed_token_budget_exhausted",
    "processed_token_budget_exit_reason",
    "reset_turn_processed_tokens",
    "turn_processed_tokens",
]

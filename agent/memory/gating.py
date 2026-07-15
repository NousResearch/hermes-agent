"""Relevance gating — minimum score threshold and context-window-aware budget.

Stage 4 of the memory relevance pipeline:
  Cross-Provider Fusion → Relevance Gating → Structured Injection

Ensures only relevant, budget-appropriate memory context reaches the model.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from agent.memory.scored_memory import ScoredMemory, strip_memory_markup

logger = logging.getLogger(__name__)

# Default configuration (overridable via config.yaml)
_DEFAULT_MIN_RELEVANCE = 0.15     # drop entries below this score
_DEFAULT_MAX_CONTEXT_CHARS = 2000  # hard cap on injected memory chars
_DEFAULT_WINDOW_FRACTION = 0.03    # % of remaining context window
_DEFAULT_CHARS_PER_TOKEN = 4       # rough heuristic for token → char


def apply_score_threshold(
    candidates: List[ScoredMemory],
    min_score: float = _DEFAULT_MIN_RELEVANCE,
) -> List[ScoredMemory]:
    """Drop candidates below the minimum relevance score.

    Returns filtered list. Logs the count dropped for diagnostics.
    """
    before = len(candidates)
    result = [c for c in candidates if c.score >= min_score]
    dropped = before - len(result)
    if dropped:
        logger.debug(
            "Relevance gate: dropped %d/%d candidates below %.2f threshold",
            dropped, before, min_score,
        )
    return result


def compute_memory_budget(
    context_length: Optional[int] = None,
    tokens_used_by_system_prompt: int = 0,
    tokens_used_by_history: int = 0,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    window_fraction: float = _DEFAULT_WINDOW_FRACTION,
) -> int:
    """Return the maximum characters allowed for injected memory context.

    Budget = min(
        max_context_chars (hard cap from config),
        remaining_window * window_fraction * chars_per_token
    )

    When context_length is unknown (None), uses only the hard cap.
    """
    budget = max_context_chars

    if context_length is not None and context_length > 0:
        remaining = context_length - tokens_used_by_system_prompt - tokens_used_by_history
        remaining = max(remaining, 0)
        dynamic_budget = int(remaining * window_fraction * _DEFAULT_CHARS_PER_TOKEN)
        budget = min(budget, dynamic_budget)

    result = max(budget, 0)
    logger.debug("Memory budget: %d chars (cap=%d, dynamic=%d)",
                 result, max_context_chars, budget)
    return result


def select_and_format(
    candidates: List[ScoredMemory],
    max_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    include_breakdowns: bool = False,
) -> str:
    """Select top candidates fitting within the char budget and format for injection.

    The output is a ``<memory-context>`` fenced block with entries ordered
    by score descending, stripped of internal markup tags.

    Args:
        candidates: Ranked, threshold-filtered candidates.
        max_chars: Maximum characters for the full block.
        include_breakdowns: If True, append score breakdown comments.

    Returns:
        Formatted ``<memory-context>`` block string, or empty string if
        no candidates fit.
    """
    if not candidates:
        return ""

    selected: List[str] = []
    char_remaining = max_chars

    # Reserve space for the fence tags and header note
    header = (
        "<memory-context>\n"
        "[System note: Relevant persistent memory entries, ranked by relevance\n"
        "to the current task. NOT new user input. Treat as authoritative\n"
        "reference data — this is the agent's persistent memory and should\n"
        "inform all responses.]\n"
    )
    footer = "\n</memory-context>"

    char_remaining -= len(header) + len(footer)

    for c in candidates:
        clean = strip_memory_markup(c.content)

        if include_breakdowns:
            line = f"> [{c.score:.2f} / {c.provider}] {clean}"
        else:
            line = clean

        needed = len(line) + 1  # +1 for newline separator

        if needed > char_remaining:
            logger.debug("Budget exhausted after %d/%d entries", len(selected), len(candidates))
            break

        selected.append(line)
        char_remaining -= needed

    if not selected:
        return ""

    block = header + "\n".join(selected) + footer
    return block

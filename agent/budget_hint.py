"""Context-window budget hint injection (adapted from openai/codex).

Codex injects a ``TokenBudgetRemainingContext`` fragment — "You have N tokens
left in this context window." — into the model input at window boundaries so
the model can budget its own output and avoid forced truncation or lossy
compression. Hermes compresses automatically (``conversation_compression`` /
``context_compressor``) but never tells the model how much window remains; the
model only discovers it was too verbose when compression fires mid-turn or the
provider truncates output.

This module builds the equivalent hint for Hermes turns. It is injected only
once usage crosses the configured threshold, so low-usage conversations keep a
byte-stable prompt prefix (per-conversation prompt caching stays intact). The
hint rides the existing ``api_content`` sidecar (``compose_user_api_content``),
which persists exactly the bytes sent — the prompt-cache invariant ("what turn
N sends must be what turn N+1 replays") is preserved by the same mechanism that
already carries memory prefetch.

The hint is deliberately conservative: it never instructs the model to change
scientific content or drop information, only to be mindful of the remaining
window. It is advisory prose appended to the turn's user message, never a new
core tool or a system-prompt mutation.
"""

from __future__ import annotations

from typing import Optional

_BUDGET_HINT_TEMPLATE = (
    "[Context budget: ~{pct}% of the context window is in use "
    "(~{remaining:,} tokens remain). Keep the response focused so it fits "
    "before forced compression or truncation — oldest turns get compressed "
    "first.]"
)

# Default fraction of the window at which the hint fires. Mirrored in
# cli-config.yaml.example (compression.budget_hint_threshold). Centralized
# here so every code path that reads the threshold — including turn-context
# builds that never passed through init_agent's config wiring (older
# serialized agents, alternate constructors) — falls back to the SAME
# documented default instead of silently disabling the feature.
DEFAULT_BUDGET_HINT_THRESHOLD = 0.70


def build_budget_hint(
    used_tokens: int,
    context_window: int,
    threshold: float,
) -> Optional[str]:
    """Return a budget-hint line when usage crosses the threshold, else None.

    Args:
        used_tokens: Estimated tokens currently occupying the window (>= 0).
        context_window: The model's context window in tokens (> 0).
        threshold: Fraction of the window at which the hint fires, in (0, 1];
            <= 0 disables the hint entirely.

    Returns:
        A single fenced hint string when ``used_tokens / context_window >=
        threshold``, otherwise None. Returns None for any degenerate input
        (unknown window, disabled threshold, negative counts) so callers can
        treat a None result as "no injection".
    """
    if threshold <= 0 or context_window <= 0 or used_tokens < 0:
        return None
    ratio = used_tokens / context_window
    if ratio < threshold:
        return None
    remaining = max(0, context_window - used_tokens)
    pct = int(ratio * 100)
    return _BUDGET_HINT_TEMPLATE.format(pct=pct, remaining=remaining)

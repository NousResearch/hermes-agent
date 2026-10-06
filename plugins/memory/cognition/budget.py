"""Prefetch budget for the cognition memory provider.

Recall injection is capped at a token budget (default 8000). When committed
entries do not fit, truncation drops in a fixed priority order and always
leaves a marker so the model knows context was omitted:

    system-pin > provenance > recency > salience

- system-pin: entries the operator pinned survive first.
- provenance: entries with complete provenance outrank partial ones.
- recency: newer entries outrank older ones.
- salience: higher-salience entries outrank lower ones (last resort).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Default prefetch budget in tokens (overridable via
# ``memory.cognition.budget_tokens``).
DEFAULT_BUDGET_TOKENS = 8000

# Rough token estimate: ~4 chars per token. Deliberately conservative and
# dependency-free; the budget is a guardrail, not a billing meter.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (minimum 1)."""
    if not text:
        return 1
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _provenance_score(entry: Dict[str, Any]) -> int:
    """Count of non-empty provenance fields (higher = more traceable)."""
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        return 0
    return sum(1 for value in provenance.values() if value)


def _sort_key(entry: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """Priority tuple: pin first, then provenance, recency, salience."""
    pinned = 1 if entry.get("pinned") else 0
    provenance = entry.get("provenance")
    created = provenance.get("created_at", 0) if isinstance(provenance, dict) else 0
    try:
        created = float(created)
    except (TypeError, ValueError):
        created = 0.0
    try:
        salience = float(entry.get("salience", 0.0))
    except (TypeError, ValueError):
        salience = 0.0
    return (pinned, _provenance_score(entry), created, salience)


def format_entry(entry: Dict[str, Any]) -> str:
    """One-line recall rendering with its deterministic ref."""
    provenance = entry.get("provenance", {})
    repo = provenance.get("origin_repo", "?") if isinstance(provenance, dict) else "?"
    return "- [%s] %s (%s)" % (entry.get("ref", "?"), entry.get("content", ""), repo)


def truncation_marker(dropped: int, budget_tokens: int) -> str:
    """Marker appended whenever truncation drops entries."""
    noun = "entry" if dropped == 1 else "entries"
    return "[cognition: %d %s omitted — prefetch budget %d tokens]" % (
        dropped,
        noun,
        budget_tokens,
    )


def apply_budget(
    entries: List[Dict[str, Any]], budget_tokens: int = DEFAULT_BUDGET_TOKENS
) -> Tuple[str, int, int]:
    """Fit entries into the budget; returns (text, kept, dropped).

    Highest-priority entries are kept; the marker line is always included in
    the token accounting when anything is dropped.
    """
    ranked = sorted(entries, key=_sort_key, reverse=True)
    lines = [format_entry(e) for e in ranked]
    costs = [estimate_tokens(line) for line in lines]

    kept: List[str] = []
    kept_costs: List[int] = []
    used = 0
    for line, cost in zip(lines, costs):
        if used + cost <= budget_tokens:
            kept.append(line)
            kept_costs.append(cost)
            used += cost
    dropped = len(lines) - len(kept)
    if not kept and lines:
        # Degenerate budget: keep the single highest-priority entry so recall
        # is never silently empty, and still say so.
        kept.append(lines[0])
        kept_costs.append(costs[0])
        used = costs[0]
        dropped = len(lines) - 1
    if dropped:
        marker = truncation_marker(dropped, budget_tokens)
        # Make room for the marker from the lowest-priority kept end first,
        # but never evict the last kept entry: recall keeps the top entry
        # even when the marker itself blows the budget.
        while len(kept) > 1 and used + estimate_tokens(marker) > budget_tokens:
            used -= kept_costs.pop()
            kept.pop()
            dropped += 1
            marker = truncation_marker(dropped, budget_tokens)
        kept.append(marker)
    return ("\n".join(kept), len(kept) - (1 if dropped else 0), dropped)

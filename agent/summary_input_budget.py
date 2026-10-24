"""Prompt-side token bounds for the compression source blocks."""

from agent.model_metadata import estimate_tokens_rough

_OMITTED = "\n\n...[summary input truncated: omitted middle to fit the summariser token budget]...\n\n"


def bound_summary_tokens(text: str, budget: int) -> str:
    """Keep the head and recent tail, charging the omission marker to the budget."""
    if not text or budget <= 0:
        return ""
    if estimate_tokens_rough(text) <= budget:
        return text
    available = budget - estimate_tokens_rough(_OMITTED) - 2
    if available <= 0:
        raise ValueError("Summary input budget cannot hold an omission marker")

    def prefix(value, tokens):
        low, high = 0, len(value)
        while low < high:
            mid = (low + high + 1) // 2
            if estimate_tokens_rough(value[:mid]) <= tokens:
                low = mid
            else:
                high = mid - 1
        return value[:low]

    head_budget = int(available * 0.45)
    return prefix(text, head_budget).rstrip() + _OMITTED + prefix(text[::-1], available - head_budget)[::-1].lstrip()

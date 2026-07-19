"""Pure Hindsight session-summary recall-query assembly helpers."""

from __future__ import annotations

from plugins.memory.hindsight.session_summary_generator import (
    SessionSummaryBudget,
    sanitize_session_summary_text,
)

_RECALL_SUMMARY_HEADER = "Rolling session summary:"


def compose_summary_recall_query(
    latest_query: str,
    summary_text: str,
    *,
    max_chars: int,
    budget: SessionSummaryBudget | None = None,
) -> str:
    """Compose a recall query with latest user text first and summary second."""
    limit = _recall_limit(max_chars, budget)
    latest = str(latest_query or "").strip()
    if limit <= 0:
        return ""
    latest = latest[:limit]
    summary = sanitize_session_summary_text(summary_text)
    if not summary:
        return latest
    if len(latest) >= limit:
        return latest

    summary_budget = limit - len(latest) - 2
    if summary_budget <= len(_RECALL_SUMMARY_HEADER):
        return latest
    block = _bounded_block(_RECALL_SUMMARY_HEADER, summary, summary_budget)
    if not block:
        return latest
    if not latest:
        return block[:limit].rstrip()
    return f"{latest}\n\n{block}"[:limit].rstrip()


def _recall_limit(max_chars: int, budget: SessionSummaryBudget | None) -> int:
    limit = max(0, int(max_chars or 0))
    if budget is None:
        return limit
    ratio_limit = int(max(0, budget.max_input_chars) * max(0.0, budget.recall_query_budget_ratio))
    budget_limit = max(0, min(budget.max_recall_query_chars, ratio_limit))
    return min(limit, budget_limit)


def _bounded_block(header: str, body: str, max_chars: int) -> str:
    limit = max(0, int(max_chars or 0))
    if limit <= 0:
        return ""
    header = header.strip()
    body_budget = limit - len(header) - 1
    if body_budget <= 0:
        return header[:limit].rstrip()
    body = sanitize_session_summary_text(body, max_chars=body_budget)
    if not body:
        return ""
    return f"{header}\n{body}"[:limit].rstrip()


__all__ = [
    "compose_summary_recall_query",
]

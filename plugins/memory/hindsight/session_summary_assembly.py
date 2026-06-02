"""Pure Hindsight session-summary assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.memory.hindsight.session_summary_generator import (
    SessionSummaryBudget,
    sanitize_session_summary_text,
)

_RECALL_SUMMARY_HEADER = "Rolling session summary:"
_RETAIN_SUMMARY_HEADER = "Rolling session summary for extraction context:"
_PROMPT_SUMMARY_TITLE = "Hindsight rolling session summary"


@dataclass(frozen=True)
class SessionSummaryAssemblyConfig:
    enrich_recall_query: bool = False
    enrich_retain_context: bool = False
    inject_prompt: bool = False
    max_recall_query_chars: int = 800
    max_retain_context_chars: int = 1_200
    max_prompt_inject_chars: int = 1_200


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


def build_summary_retain_context(base_context: str, summary_text: str, *, max_chars: int) -> str:
    """Append a bounded summary to extraction context without changing transcript content."""
    base = str(base_context or "")
    summary = sanitize_session_summary_text(summary_text, max_chars=max(0, int(max_chars or 0)))
    if not summary:
        return base
    block = f"{_RETAIN_SUMMARY_HEADER}\n{summary}"
    if not base:
        return block
    return f"{base}\n\n{block}"


def render_summary_prompt_block(summary_text: str, *, max_chars: int) -> str:
    """Render a prompt-only summary block kept separate from memory blocks."""
    summary = sanitize_session_summary_text(summary_text, max_chars=max(0, int(max_chars or 0)))
    if not summary:
        return ""
    return (
        f"<hindsight_session_summary>\n"
        f"{_PROMPT_SUMMARY_TITLE}\n"
        f"{summary}\n"
        f"</hindsight_session_summary>"
    )


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
    "SessionSummaryAssemblyConfig",
    "build_summary_retain_context",
    "compose_summary_recall_query",
    "render_summary_prompt_block",
]

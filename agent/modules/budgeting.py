from __future__ import annotations

import os
import logging
from typing import Optional

# Max short-term turns to keep verbatim; configurable via MEMORY_SHORT_TERM_MAX_TURNS env var.
SHORT_TERM_MAX_TURNS: int = int(os.environ.get("MEMORY_SHORT_TERM_MAX_TURNS", "10"))

logger = logging.getLogger(__name__)


def summarize_turns(
    turns: list[dict],
    keep_recent: int = SHORT_TERM_MAX_TURNS,
    summarizer_model: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """Compress oldest turns into a summary string when count exceeds keep_recent.

    Returns (summary, recent_turns). If len(turns) <= keep_recent: returns ("", turns).
    Routes the LLM call through the Anthropic SDK client.
    """
    if len(turns) <= keep_recent:
        return ("", turns)

    oldest = turns[: len(turns) - keep_recent]
    recent = turns[len(turns) - keep_recent :]

    conversation_text = "\n".join(
        f"{t.get('role', 'unknown')}: {t.get('content', t.get('text', ''))}"
        for t in oldest
    )

    summary = _invoke_summarizer(conversation_text, summarizer_model)
    return (summary, recent)


def _invoke_summarizer(conversation_text: str, model: Optional[str]) -> str:
    """Call the Anthropic SDK to produce a summary of the given conversation text."""
    import anthropic

    client = anthropic.Anthropic()
    model_id = model or "claude-haiku-4-5"

    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=512,
            system=(
                "Summarize the following conversation history concisely. "
                "Preserve key facts, decisions, and context. Return only the summary text."
            ),
            messages=[{"role": "user", "content": conversation_text}],
        )
        return response.content[0].text
    except Exception as exc:
        logger.error("summarize_turns LLM call failed: %s", exc)
        return ""


def budget_for_intent(intent: str, model_tier: str) -> int:
    """Calculate the context token budget based on intent and model tier.
    
    Local tier halves the budget for slim packets (Expansion 3 §J).
    """
    budgets = {
        "ANSWER_DIRECTLY": 1024,
        "RECALL_MEMORY": 2048,
        "INVOKE_TOOL": 1536,
        "DELEGATE_SPECIALIST": 4096,
        "SUBMIT_OPENCLAW_JOB": 6144,
        "CLARIFY_FIRST": 768,
        "DRAFT_FOR_APPROVAL": 3072,
        "ESCALATE_TO_ATTI": 2048,
    }
    
    budget = budgets.get(intent, 1024)
    
    if model_tier.lower() == "local":
        budget = budget // 2
        
    return budget

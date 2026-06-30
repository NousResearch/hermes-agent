"""Intent extraction — turns a raw user message into a structured IntentContext.

Stage 1 of the memory relevance pipeline:
  User Message → IntentContext (task type, entities, keywords, expanded query)

The IntentContext is passed to memory providers for targeted recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent.memory.scored_memory import (
    classify_task_type,
    extract_entities,
    _PROJECT_KEYWORDS,
)


@dataclass
class IntentContext:
    """Structured context extracted from a user message.

    This is the input to memory provider recall methods.
    """

    task_type: Optional[str] = None      # "deploy", "feature", "debug", ...
    task_confidence: float = 0.0          # 0.0-1.0

    entities: List[Tuple[str, float]] = field(default_factory=list)  # (name, salience)
    keywords: List[Tuple[str, float]] = field(default_factory=list)  # (keyword, weight)

    expanded_query: str = ""              # user message + contextualised
    raw_message: str = ""                 # original user message

    last_n_turns: int = 3                # how many prior turns considered

    def is_actionable(self) -> bool:
        """Return True if we have enough signal to drive relevance scoring."""
        return bool(self.task_type) or bool(self.entities) or bool(self.keywords)


def extract_intent(
    message: str,
    prior_turns: Optional[List[str]] = None,
) -> IntentContext:
    """Main entry point: extract structured intent from a user message.

    Args:
        message: The current user message.
        prior_turns: Optional messages from recent turns for context.

    Returns:
        An IntentContext with extracted task type, entities, and keywords.
    """
    ctx = IntentContext(raw_message=message)

    # Task type classification
    task_type, confidence = classify_task_type(message)
    ctx.task_type = task_type
    ctx.task_confidence = confidence

    # Entity extraction
    ctx.entities = extract_entities(message)

    # Keyword extraction — find important keywords from the message
    ctx.keywords = extract_keywords(message)

    # Query expansion — combine message with prior turns
    expanded = message
    if prior_turns:
        # Take last N-1 turns for context (skip the current)
        context_turns = prior_turns[: ctx.last_n_turns - 1]
        if context_turns:
            expanded = " ".join(context_turns) + " " + message
    ctx.expanded_query = expanded

    return ctx


def extract_keywords(message: str) -> List[Tuple[str, float]]:
    """Extract weighted keywords from a user message.

    Uses the project keyword dictionary. Each keyword that appears in the
    message gets its defined weight. This is a lightweight alternative to
    TF-IDF for the hot path.
    """
    if not message:
        return []

    msg_lower = message.lower()
    found: Dict[str, float] = {}

    for keyword, weight in _PROJECT_KEYWORDS.items():
        if keyword in msg_lower:
            # Only keep the highest-weighted match per keyword
            found[keyword] = weight

    return sorted(found.items(), key=lambda x: x[1], reverse=True)


def expand_query(
    message: str,
    prior_turns: Optional[List[str]] = None,
    max_turns: int = 3,
) -> str:
    """Build an expanded query suitable for embedding search.

    Concatenates recent context with the current message.
    """
    if not prior_turns:
        return message
    context_turns = prior_turns[: max_turns - 1]
    if not context_turns:
        return message
    return " ".join(context_turns) + " " + message

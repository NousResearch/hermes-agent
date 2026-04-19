"""Conversation Interpreter — Hermes input-pipeline submodule C§1.3.

Receives a UserMessage and the assembled ContextPackage, then produces
a structured Interpretation capturing intent, entities, topic, and
sentiment for the downstream Intent Classifier.

Phase-3 build plan reference: §C§1 table, row 3.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.interp.done``
Emission mechanism: stdout JSON line (single-line, newline-terminated).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.modules.context_loader import ContextPackage, UserMessage

# ---------------------------------------------------------------------------
# I/O types
# ---------------------------------------------------------------------------


class Interpretation(BaseModel):
    """Structured reading of the user's message in context.

    Fields
    ------
    intent:
        Coarse natural-language intent label (e.g. "create_task", "ask_question").
        Refined to a route enum by the Intent Classifier.
    entities:
        Named entities extracted from the message (people, dates, projects, …).
    topic:
        Short topic slug for memory indexing (e.g. "infra/docker").
    sentiment:
        One of ``positive``, ``neutral``, ``negative``, or ``urgent``.
    raw_text:
        Preserved original message text for downstream modules.
    metadata:
        Arbitrary per-interpreter metadata (model used, confidence, etc.).
    """

    intent: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    topic: Optional[str] = None
    sentiment: str = "neutral"
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def _emit(event: str, payload: dict) -> None:
    """Write a single-line JSON event to stdout.

    Replace with an @agrv/hermes-events call in C§1.9 when the shared
    event bus is wired into this workspace.
    """
    line = json.dumps({"event": event, **payload}, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Submodule entry point
# ---------------------------------------------------------------------------


def interpret(
    message: UserMessage,
    context: ContextPackage,
) -> Interpretation:
    """Produce a structured Interpretation from a UserMessage + ContextPackage.

    Stub implementation: sets intent to "unknown" and passes the raw text
    through. Full implementation (LLM-based NLU pass, entity extraction,
    topic classifier) is deferred to C§1.9.

    Emits ``hermes.interp.done`` on completion.
    """
    interp = Interpretation(
        intent="unknown",
        entities=[],
        topic=None,
        sentiment="neutral",
        raw_text=message.text,
        metadata={"stub": True, "session_id": message.session_id},
    )

    _emit(
        "hermes.interp.done",
        {
            "session_id": message.session_id,
            "intent": interp.intent,
            "topic": interp.topic,
            "sentiment": interp.sentiment,
            "entity_count": len(interp.entities),
        },
    )

    return interp

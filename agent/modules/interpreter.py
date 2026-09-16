"""Conversation Interpreter — Hermes input-pipeline submodule C§1.3.

Receives a UserMessage and the assembled ContextPackage, then produces
a structured Interpretation capturing intent, entities, topic, and
sentiment for the downstream Intent Classifier.

Phase-3 build plan reference: §C§1 table, row 3.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.interp.done``
Emission mechanism: EventEmitter instance (injected by turn_handler).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.modules.context_loader import ContextPackage, UserMessage
from agent.modules.event_emitter import EventEmitter

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
# Module-level emitter (injected by turn_handler)
# ---------------------------------------------------------------------------

_emitter: Optional[EventEmitter] = None


def set_emitter(emitter: EventEmitter) -> None:
    """Inject the shared event emitter.

    Called by turn_handler.run_turn() before processing.
    """
    global _emitter
    _emitter = emitter


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

    if _emitter is not None:
        _emitter.emit(
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

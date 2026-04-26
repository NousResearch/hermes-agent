"""Intent Classifier — Hermes input-pipeline submodule C§1.4.

Receives an Interpretation from the Conversation Interpreter and maps it
to one of 8 authoritative Route values. The Route drives all downstream
delegation decisions (direct answer, memory recall, tool invocation,
specialist handoff, OpenClaw job submission, clarification, draft-for-
approval, or escalation to Atti).

Phase-3 build plan reference: §C§1 table, row 4.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.intent.classified``
Emission mechanism: EventEmitter instance (injected by turn_handler).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from agent.modules.event_emitter import EventEmitter
from agent.modules.interpreter import Interpretation

# ---------------------------------------------------------------------------
# I/O types
# ---------------------------------------------------------------------------


class Route(str, Enum):
    """8-way routing enum for Hermes delegation.

    Values map to their downstream handler as documented in §C§1:

    ANSWER_DIRECTLY
        Hermes can answer from its own knowledge / context; no delegation.
    RECALL_MEMORY
        Answer requires a memory-engine lookup before responding.
    INVOKE_TOOL
        A registered MCP or built-in tool must be called.
    DELEGATE_SPECIALIST
        Route to a Tier-1 specialist agent via the CEO orchestrator.
    SUBMIT_OPENCLAW_JOB
        Long-running work; submit to the OpenClaw durable-job engine.
    CLARIFY_FIRST
        Message is ambiguous; ask a clarifying question before acting.
    DRAFT_FOR_APPROVAL
        Generate a draft response/action and surface it for Atti's approval.
    ESCALATE_TO_ATTI
        Requires human judgment; escalate to Atti directly.
    """

    ANSWER_DIRECTLY = "ANSWER_DIRECTLY"
    RECALL_MEMORY = "RECALL_MEMORY"
    INVOKE_TOOL = "INVOKE_TOOL"
    DELEGATE_SPECIALIST = "DELEGATE_SPECIALIST"
    SUBMIT_OPENCLAW_JOB = "SUBMIT_OPENCLAW_JOB"
    CLARIFY_FIRST = "CLARIFY_FIRST"
    DRAFT_FOR_APPROVAL = "DRAFT_FOR_APPROVAL"
    ESCALATE_TO_ATTI = "ESCALATE_TO_ATTI"


class ClassifiedIntent(BaseModel):
    """Wrapper pairing an Interpretation with its resolved Route."""

    route: Route
    confidence: float = 0.0  # 0.0–1.0; 0.0 for stub
    interpretation: Interpretation


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
# Content-production fast-path detection
# ---------------------------------------------------------------------------

# Trigger patterns for content-production intent.
# German patterns cover Stefan's primary language; English patterns for
# mixed-language sessions.
_CONTENT_PRODUCTION_PATTERNS: list[re.Pattern[str]] = [
    # German — explicit write/create requests (allows polite "schreiben Sie einen Artikel")
    re.compile(r"\bschreib(?:e|en)?\s+(?:\w+\s+)?(?:einen?\s+)?(?:blog|artikel|beitrag|text|post)\b", re.IGNORECASE),
    re.compile(r"\bich\s+brauche\s+einen?\s+artikel\b", re.IGNORECASE),
    re.compile(r"\bneuer?\s+content\s+(?:über|uber|zu|über)\b", re.IGNORECASE),
    re.compile(r"\berstell(?:e|en|t)?\s+(?:einen?\s+)?(?:blog|artikel|beitrag|content)\b", re.IGNORECASE),
    re.compile(r"\b(?:redaktion|redaktionsplan)\b", re.IGNORECASE),
    # English — explicit write/create requests
    re.compile(r"\bwrite\s+(?:an?\s+)?(?:blog|article|post|content)\b", re.IGNORECASE),
    re.compile(r"\bblog\s+about\b", re.IGNORECASE),
    re.compile(r"\barticle\s+(?:about|on|for)\b", re.IGNORECASE),
    re.compile(r"\bcontent\s+production\b", re.IGNORECASE),
    re.compile(r"\bproduce\s+content\b", re.IGNORECASE),
]

# Patterns that extract the topic from common preposition phrases.
_TOPIC_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    # German: "über X", "zu X", "über X"
    re.compile(r"\b(?:über|uber|zu)\s+(.+?)(?:\s*$|\.|,|;)", re.IGNORECASE),
    # English: "about X", "on X"
    re.compile(r"\b(?:about|on)\s+(.+?)(?:\s*$|\.|,|;)", re.IGNORECASE),
]

# Patterns that indicate a likely German message.
_GERMAN_MARKERS: re.Pattern[str] = re.compile(
    r"\b(ich|du|wir|ihr|sie|der|die|das|und|ist|bin|schreib|brauche|über|neuer?|einen?|artikel|blog|beitrag|redaktion)\b",
    re.IGNORECASE,
)


def _detect_content_production(text: str) -> Optional[dict]:
    """Return content-production metadata dict if *text* matches, else None.

    The returned dict has the shape expected by ClassifiedIntent.interpretation.metadata
    for content_production intent:
        {
            "intent": "content_production",
            "workflow_ref": "baumbad-content-pipeline-v1",
            "extracted": {
                "topic": str | None,
                "language": "de" | "en",
                "audience_hint": str | None,
            },
        }
    """
    if not text or not text.strip():
        return None

    matched = any(p.search(text) for p in _CONTENT_PRODUCTION_PATTERNS)
    if not matched:
        return None

    # Determine language heuristically.
    language = "de" if _GERMAN_MARKERS.search(text) else "en"

    # Extract topic: try each extraction pattern, take first non-empty capture.
    topic: Optional[str] = None
    for pat in _TOPIC_EXTRACTION_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip().rstrip(".,;!?")
            if candidate:
                topic = candidate
                break

    return {
        "intent": "content_production",
        "workflow_ref": "baumbad-content-pipeline-v1",
        "extracted": {
            "topic": topic,
            "language": language,
            "audience_hint": None,  # audience extraction deferred to LLM pass (C§1.9)
        },
    }


# ---------------------------------------------------------------------------
# Submodule entry point
# ---------------------------------------------------------------------------


def classify_intent(interpretation: Interpretation) -> ClassifiedIntent:
    """Map an Interpretation to a Route.

    Rule-based fast path for known intent signals; defaults to
    ANSWER_DIRECTLY for all unmatched inputs. Full LLM-backed classifier
    for ambiguous cases is deferred to C§1.9.

    Content-production fast path: matches German and English patterns for
    blog/article/content requests and routes to DELEGATE_SPECIALIST, setting
    workflow_ref=baumbad-content-pipeline-v1 in interpretation.metadata.

    Emits ``hermes.intent.classified`` on completion.
    """
    content_meta = _detect_content_production(interpretation.raw_text)

    if content_meta is not None:
        # Enrich the interpretation metadata in-place (interpretation is a
        # Pydantic model; build a new instance to avoid mutating a shared obj).
        enriched = interpretation.model_copy(
            update={"metadata": {**interpretation.metadata, **content_meta}}
        )
        result = ClassifiedIntent(
            route=Route.DELEGATE_SPECIALIST,
            confidence=0.9,
            interpretation=enriched,
        )
    else:
        result = ClassifiedIntent(
            route=Route.ANSWER_DIRECTLY,
            confidence=0.0,
            interpretation=interpretation,
        )

    if _emitter is not None:
        _emitter.emit(
            "hermes.intent.classified",
            {
                "route": result.route.value,
                "confidence": result.confidence,
                "intent": result.interpretation.metadata.get("intent", interpretation.intent),
                "topic": result.interpretation.topic,
            },
        )

    return result

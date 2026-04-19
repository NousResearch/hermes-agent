"""Intent Classifier — Hermes input-pipeline submodule C§1.4.

Receives an Interpretation from the Conversation Interpreter and maps it
to one of 8 authoritative Route values. The Route drives all downstream
delegation decisions (direct answer, memory recall, tool invocation,
specialist handoff, OpenClaw job submission, clarification, draft-for-
approval, or escalation to Atti).

Phase-3 build plan reference: §C§1 table, row 4.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.intent.classified``
Emission mechanism: stdout JSON line (single-line, newline-terminated).
"""

from __future__ import annotations

import json
import sys
from enum import Enum

from pydantic import BaseModel

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


def classify_intent(interpretation: Interpretation) -> ClassifiedIntent:
    """Map an Interpretation to a Route.

    Stub implementation: defaults to ANSWER_DIRECTLY for all inputs.
    Full implementation (rule-based fast path + LLM-backed classifier
    for ambiguous cases) is deferred to C§1.9.

    Emits ``hermes.intent.classified`` on completion.
    """
    result = ClassifiedIntent(
        route=Route.ANSWER_DIRECTLY,
        confidence=0.0,
        interpretation=interpretation,
    )

    _emit(
        "hermes.intent.classified",
        {
            "route": result.route.value,
            "confidence": result.confidence,
            "intent": interpretation.intent,
            "topic": interpretation.topic,
        },
    )

    return result

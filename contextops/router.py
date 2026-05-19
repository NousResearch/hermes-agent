"""Router prompt-contract harness for the ContextOps Epistemic State Engine.

The router decides whether a context event *continues* an existing
evidence-anchored thread or *opens* a new one. It is a prompt contract:

* it builds a prompt that exposes each thread's stance (so a model can judge
  conceptual continuation, not keyword overlap),
* it invokes a caller-supplied model callable (fake/local in tests; no remote
  calls live here),
* it parses the model output as strict JSON and validates it as a
  ``RouteProposal``, failing closed on anything malformed.

The router never persists anything. Its output is a proposal carrying
confidence and evidence, not a transcript and not a durable record.
"""

from __future__ import annotations

import json
from typing import Callable, Literal

from pydantic import Field, model_validator

from contextops.models import ContextOpsModel, Event, Thread

ModelCallable = Callable[[str], str]


class RouteProposal(ContextOpsModel):
    """A proposed routing decision -- a proposal, never a persisted record."""

    event_id: str
    decision: Literal["continue", "new"]
    thread_id: str | None = None
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str]

    @model_validator(mode="after")
    def _check_contract(self) -> "RouteProposal":
        if not self.event_id.strip():
            raise ValueError("route proposal requires a non-empty event_id")
        if not self.rationale.strip():
            raise ValueError("route proposal requires a non-empty rationale")
        if not [ref for ref in self.evidence_refs if ref and ref.strip()]:
            raise ValueError("route proposal requires non-empty evidence_refs")
        if self.decision == "continue" and not (self.thread_id and self.thread_id.strip()):
            raise ValueError("continue decision requires a thread_id")
        if self.decision == "new" and self.thread_id:
            raise ValueError("new decision must not set a thread_id")
        return self


ROUTER_INSTRUCTIONS = (
    "You route one context event to an existing evidence-anchored thread or open "
    "a new thread. Judge conceptual continuation against each thread's stance -- "
    "do not require keyword overlap; related evidence or stance is enough. "
    "Respond with a single JSON object only: "
    '{"event_id": str, "decision": "continue"|"new", "thread_id": str|null, '
    '"rationale": str, "confidence": number in 0..1, "evidence_refs": [str, ...]}. '
    "Every evidence ref must anchor to the current EVENT: use the current event's "
    "id or a ref already attached to that event -- never a ref from another event."
)


def build_router_prompt(event: Event, threads: list[Thread]) -> str:
    """Build the router prompt, exposing thread stances for conceptual matching."""

    lines = [
        ROUTER_INSTRUCTIONS,
        "",
        f"EVENT {event.id} (source={event.source}): {event.text}",
        "",
        "CANDIDATE THREADS:",
    ]
    if threads:
        for thread in threads:
            lines.append(
                f"- {thread.id} (status={thread.status}, heat={thread.heat}) "
                f"stance: {thread.stance}"
            )
    else:
        lines.append("- (no existing threads)")
    return "\n".join(lines)


def route_context_event(
    event: Event,
    threads: list[Thread],
    model: ModelCallable,
) -> RouteProposal:
    """Route ``event`` against ``threads`` using ``model``; fail closed on bad output."""

    prompt = build_router_prompt(event, threads)
    raw = model(prompt)
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"router model output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("router model output must be a single JSON object")

    proposal = RouteProposal.model_validate(data)

    if proposal.event_id != event.id:
        raise ValueError(
            f"router proposal must reference current event {event.id!r}; "
            f"got {proposal.event_id!r}"
        )
    allowed_refs = {event.id, *event.refs}
    bad_refs = [ref for ref in proposal.evidence_refs if ref not in allowed_refs]
    if bad_refs:
        raise ValueError(
            f"router evidence_refs must anchor to current event {event.id!r} "
            f"(id or event.refs); got {bad_refs!r}"
        )

    if proposal.decision == "continue":
        known_ids = {thread.id for thread in threads}
        if proposal.thread_id not in known_ids:
            raise ValueError(
                f"router proposed continuation of unknown thread {proposal.thread_id!r}"
            )
    return proposal

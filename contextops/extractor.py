"""Extractor prompt-contract harness for the ContextOps Epistemic State Engine.

The extractor proposes ``StateDelta`` changes -- epistemic deltas anchored by
evidence and scored by confidence. It is deliberately not a summarizer: a
proposal must carry a delta ``kind`` and ``description`` plus either evidence
refs or an explicit ``low_confidence`` marker, so summary-only prose fails the
contract.

Like the router, the extractor invokes a caller-supplied model callable
(fake/local in tests; no remote calls), parses output as strict JSON, validates
each entry, and fails closed on anything malformed. It never persists anything;
its output is a list of proposals, not durable writes.
"""

from __future__ import annotations

import json
from typing import Callable

from pydantic import Field, model_validator

from contextops.models import ContextOpsModel, Event, StateDelta, Thread

ModelCallable = Callable[[str], str]


class StateDeltaProposal(ContextOpsModel):
    """A proposed epistemic state change -- a proposal, never a durable write."""

    thread_id: str | None = None
    kind: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = False

    @model_validator(mode="after")
    def _check_contract(self) -> "StateDeltaProposal":
        if not self.kind.strip():
            raise ValueError("state delta proposal requires a non-empty kind")
        if not self.description.strip():
            raise ValueError("state delta proposal requires a non-empty description")
        cleaned = [ref for ref in self.evidence_refs if ref and ref.strip()]
        if not cleaned and not self.low_confidence:
            raise ValueError(
                "state delta proposal requires evidence_refs unless low_confidence is true"
            )
        return self

    def to_state_delta(self, id: str) -> StateDelta:
        """Materialize this proposal as a ``StateDelta`` (caller-owned id)."""

        return StateDelta(
            id=id,
            kind=self.kind,
            description=self.description,
            evidence_refs=list(self.evidence_refs),
            low_confidence=self.low_confidence,
            metadata={"confidence": self.confidence, "thread_id": self.thread_id},
        )


EXTRACTOR_INSTRUCTIONS = (
    "You extract epistemic state deltas from one context event. Each delta is a "
    "specific change (belief shift, hypothesis, tension, resolution) -- never a "
    "summary of the conversation. Respond with a JSON array only; each element: "
    '{"thread_id": str|null, "kind": str, "description": str, '
    '"evidence_refs": [str, ...], "confidence": number in 0..1, '
    '"low_confidence": bool}. Provide evidence_refs unless low_confidence is true. '
    "Every evidence ref must anchor to the current EVENT: use the current event's "
    "id or a ref already attached to that event -- never a ref from another event."
)


def build_extractor_prompt(event: Event, threads: list[Thread]) -> str:
    """Build the extractor prompt, listing threads the delta may attach to."""

    lines = [
        EXTRACTOR_INSTRUCTIONS,
        "",
        f"EVENT {event.id} (source={event.source}): {event.text}",
        "",
        "ATTACHABLE THREADS:",
    ]
    if threads:
        for thread in threads:
            lines.append(f"- {thread.id} stance: {thread.stance}")
    else:
        lines.append("- (no existing threads)")
    return "\n".join(lines)


def extract_state_deltas(
    event: Event,
    threads: list[Thread],
    model: ModelCallable,
) -> list[StateDeltaProposal]:
    """Extract state-delta proposals from ``event``; fail closed on bad output."""

    prompt = build_extractor_prompt(event, threads)
    raw = model(prompt)
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extractor model output is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("extractor model output must be a JSON array of proposals")

    known_ids = {thread.id for thread in threads}
    allowed_refs = {event.id, *event.refs}
    proposals: list[StateDeltaProposal] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each extractor proposal must be a JSON object")
        proposal = StateDeltaProposal.model_validate(item)
        bad_refs = [ref for ref in proposal.evidence_refs if ref not in allowed_refs]
        if bad_refs:
            raise ValueError(
                f"extractor evidence_refs must anchor to current event {event.id!r} "
                f"(id or event.refs); got {bad_refs!r}"
            )
        if proposal.thread_id is not None and proposal.thread_id not in known_ids:
            raise ValueError(
                f"extractor proposal references unknown thread {proposal.thread_id!r}"
            )
        proposals.append(proposal)
    return proposals

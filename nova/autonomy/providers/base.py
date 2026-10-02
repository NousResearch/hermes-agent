"""The decision-provider contract.

A provider answers a set of atomic questions about a ``state``. It never says "allow": the
verdict is :func:`nova.autonomy.triage.combine`'s, applied to these answers. That keeps the
rule in code a reviewer can read and lets a provider be swapped without moving it.

**A provider that cannot answer raises** :class:`ProviderError`. Callers treat that the
same way as a timeout — escalate to a person — so there is no partial answer to misread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from nova.autonomy.questions import Question


class ProviderError(Exception):
    """The provider gave no usable answer. ``reason`` is safe to record; it carries no content."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Answer:
    """One typed answer, normalized across providers (see ``triage`` for the shape)."""

    id: str
    type: str
    #: noul: the probability of "yes". Empty for choice and score.
    p: Optional[float] = None
    #: choice: the likeliest option.
    choice: str = ""
    #: choice: option -> probability. score: level index -> probability, in level order.
    probabilities: Any = None
    #: choice and score only; a noul has none.
    confidence: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.p is not None:
            out["p"] = self.p
        if self.choice:
            out["choice"] = self.choice
        if self.probabilities is not None:
            out["probabilities"] = self.probabilities
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out


@dataclass(frozen=True)
class Answers:
    """Every answer to one call, and what produced them."""

    answers: Mapping[str, Answer]
    provider: str
    #: The exact model version that answered, as the provider reports it. A change here is
    #: what demotes graduated actions, so it is never defaulted.
    model_version: str
    latency_ms: int
    usage: Mapping[str, int] = field(default_factory=dict)

    def normalized(self) -> dict[str, dict[str, Any]]:
        """The plain form :func:`nova.autonomy.triage.combine` takes."""
        return {qid: answer.to_dict() for qid, answer in self.answers.items()}


@runtime_checkable
class DecisionProvider(Protocol):
    """Anything that can answer a question set."""

    name: str

    def ask(self, state: Any, questions: Sequence[Question]) -> Answers:
        """Answer every question about ``state``, or raise :class:`ProviderError`."""
        ...


def answers_from(normalized: Mapping[str, Mapping[str, Any]], *, provider: str,
                 model_version: str, latency_ms: int, usage: Optional[Mapping[str, int]] = None) -> Answers:
    """Wrap normalized answers (as a provider's wire module produces them) in the typed form."""
    return Answers(
        answers={
            qid: Answer(id=qid, type=str(a.get("type", "")), p=a.get("p"), choice=str(a.get("choice") or ""),
                        probabilities=a.get("probabilities"), confidence=a.get("confidence"))
            for qid, a in normalized.items()
        },
        provider=provider, model_version=model_version, latency_ms=int(latency_ms),
        usage=dict(usage or {}),
    )


def provider_for(name: str, **options: Any) -> Optional[DecisionProvider]:
    """The provider a policy names. None for ``none`` (triage off). Raises on an unknown name."""
    if name == "none":
        return None
    if name == "fake":
        from nova.autonomy.providers.fake import FakeProvider

        return FakeProvider(**options)
    if name == "typesafe":
        from nova.autonomy.providers.typesafe import TypesafeProvider

        return TypesafeProvider(**options)
    raise ValueError(f"unknown triage provider {name!r}; expected none, fake or typesafe")

"""Typed contracts for bounded probabilistic decisions.

Decision providers receive only the caller-supplied :attr:`DecisionRequest.state` projection.
They return probability distributions as evidence; consumers retain all threshold and action policy.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

DecisionLabel = Union[bool, str]


@dataclass(frozen=True)
class BinaryQuestion:
    instructions: str

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("question instructions must be a non-empty string")

    @property
    def labels(self) -> Tuple[DecisionLabel, ...]:
        return (False, True)


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    choices: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("question instructions must be a non-empty string")
        choices = tuple(self.choices)
        if len(choices) < 2 or any(not isinstance(item, str) or not item for item in choices):
            raise ValueError("choice questions require at least two non-empty string choices")
        if len(set(choices)) != len(choices):
            raise ValueError("choice question choices must be unique")
        object.__setattr__(self, "choices", choices)

    @property
    def labels(self) -> Tuple[DecisionLabel, ...]:
        return tuple(self.choices)


@dataclass(frozen=True)
class OrdinalQuestion:
    instructions: str
    levels: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("question instructions must be a non-empty string")
        levels = tuple(self.levels)
        if len(levels) < 2 or any(not isinstance(item, str) or not item for item in levels):
            raise ValueError("ordinal questions require at least two non-empty string levels")
        if len(set(levels)) != len(levels):
            raise ValueError("ordinal question levels must be unique")
        object.__setattr__(self, "levels", levels)

    @property
    def labels(self) -> Tuple[DecisionLabel, ...]:
        return tuple(self.levels)


DecisionQuestion = Union[BinaryQuestion, ChoiceQuestion, OrdinalQuestion]


@dataclass(frozen=True)
class DecisionRequest:
    task: str
    state: Mapping[str, Any]
    questions: Mapping[str, DecisionQuestion]
    mode: str = "active"


@dataclass(frozen=True)
class DecisionAnswer:
    probabilities: Mapping[DecisionLabel, float]
    selected: Optional[DecisionLabel] = None
    abstained: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderDecision:
    answers: Mapping[str, DecisionAnswer]
    model: str = ""
    version: str = ""
    usage: Mapping[str, float] = field(default_factory=dict)
    abstained: bool = False
    abstention_reason: Optional[str] = None


class DecisionStatus(str, Enum):
    AVAILABLE = "available"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class DecisionResponse:
    task: str
    provider: str
    model: str
    version: str
    mode: str
    status: DecisionStatus
    answers: Mapping[str, DecisionAnswer] = field(default_factory=dict)
    usage: Mapping[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    fallback_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.status in {DecisionStatus.AVAILABLE, DecisionStatus.ABSTAINED}


class DecisionProvider(abc.ABC):
    """Backend contract for bounded typed decisions.

    Providers may batch the independent questions in one request. Dependent stages remain separate
    calls made by the consumer, so the state projection for each stage stays explicit.
    """

    name: str = ""

    def is_available(self) -> bool:
        """Return whether the provider can currently evaluate requests without network probing."""
        return True

    @abc.abstractmethod
    def evaluate(self, request: DecisionRequest) -> ProviderDecision:
        """Evaluate one request and return full probability distributions."""


def validate_provider_decision(request: DecisionRequest, result: ProviderDecision) -> ProviderDecision:
    """Validate and copy a provider result without normalizing its probabilities."""
    if not isinstance(result, ProviderDecision):
        raise ValueError("provider must return ProviderDecision")
    if not isinstance(result.model, str):
        raise ValueError("provider model must be a string")
    if not isinstance(result.version, str):
        raise ValueError("provider version must be a string")
    if not isinstance(result.usage, Mapping):
        raise ValueError("provider usage must be a mapping")
    usage: Dict[str, float] = {}
    for key, value in result.usage.items():
        if not isinstance(key, str):
            raise ValueError("provider usage keys must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("provider usage values must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("provider usage values must be finite")
        usage[key] = float(value)
    if not isinstance(result.abstained, bool):
        raise ValueError("provider abstained must be a boolean")
    if result.abstention_reason is not None and not isinstance(result.abstention_reason, str):
        raise ValueError("provider abstention reason must be a string or None")
    if not isinstance(result.answers, Mapping):
        raise ValueError("provider answers must be a mapping")
    if result.abstained and result.answers:
        raise ValueError("a request-level abstention cannot also contain answers")
    if result.abstained:
        return ProviderDecision(
            answers={}, model=result.model, version=result.version, usage=usage,
            abstained=True, abstention_reason=result.abstention_reason,
        )
    if set(result.answers) != set(request.questions):
        raise ValueError("provider answer keys must exactly match request question keys")

    validated: Dict[str, DecisionAnswer] = {}
    for key, question in request.questions.items():
        answer = result.answers[key]
        if not isinstance(answer, DecisionAnswer):
            raise ValueError(f"answer {key!r} must be DecisionAnswer")
        if not isinstance(answer.probabilities, Mapping):
            raise ValueError(f"answer {key!r} probabilities must be a mapping")
        probabilities = dict(answer.probabilities)
        if set(probabilities) != set(question.labels):
            raise ValueError(f"answer {key!r} probability labels do not match the question domain")
        values = list(probabilities.values())
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"answer {key!r} probabilities must be numeric")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise ValueError(f"answer {key!r} probabilities must be finite and non-negative")
        if not math.isclose(sum(float(value) for value in values), 1.0, abs_tol=1e-6):
            raise ValueError(f"answer {key!r} probabilities must sum to 1")
        selected = answer.selected
        if selected is None and not answer.abstained:
            selected = max(question.labels, key=lambda label: float(probabilities[label]))
        if selected is not None and selected not in question.labels:
            raise ValueError(f"answer {key!r} selected label is outside the question domain")
        if not isinstance(answer.abstained, bool):
            raise ValueError(f"answer {key!r} abstained must be a boolean")
        if not isinstance(answer.metadata, Mapping):
            raise ValueError(f"answer {key!r} metadata must be a mapping")
        validated[key] = DecisionAnswer(
            probabilities={label: float(probabilities[label]) for label in question.labels},
            selected=selected,
            abstained=bool(answer.abstained),
            metadata=dict(answer.metadata),
        )
    return ProviderDecision(
        answers=validated,
        model=result.model,
        version=result.version,
        usage=usage,
        abstained=False,
        abstention_reason=result.abstention_reason,
    )

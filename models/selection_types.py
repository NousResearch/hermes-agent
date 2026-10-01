"""Immutable types for canonical model selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from models.identity import ModelRef
from models.metadata.types import ModelMetadata
from providers.identity import normalize_provider
from providers.routing import InvocationRoute


@dataclass(frozen=True, slots=True)
class CapabilityRequirements:
    tools: bool | None = None
    vision: bool | None = None
    reasoning: bool | None = None
    structured_output: bool | None = None
    minimum_context_window: int | None = None


@dataclass(frozen=True, slots=True)
class SelectionConstraints:
    allowed_providers: tuple[str, ...] = ()
    allowed_models: tuple[ModelRef, ...] = ()
    excluded_models: tuple[ModelRef, ...] = ()
    require_catalogued: bool = False
    capabilities: CapabilityRequirements = field(default_factory=CapabilityRequirements)
    allowed_api_modes: tuple[str, ...] = ()
    allowed_runtime_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = (normalize_provider(value) for value in self.allowed_providers if value)
        object.__setattr__(self, "allowed_providers", tuple(dict.fromkeys(normalized)))


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    name: str = "default"
    preferred: tuple[ModelRef, ...] = ()
    provider_order: tuple[str, ...] = ()
    prefer_catalogued: bool = False

    def __post_init__(self) -> None:
        normalized = (normalize_provider(value) for value in self.provider_order if value)
        object.__setattr__(self, "provider_order", tuple(dict.fromkeys(normalized)))


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    ref: ModelRef
    metadata: ModelMetadata | None = None
    route: InvocationRoute | None = None
    catalogued: bool = False
    source: str = ""
    stable_order: int = 0

    def __post_init__(self) -> None:
        if self.metadata is not None and self.metadata.ref != self.ref:
            raise ValueError("candidate metadata identity does not match candidate ref")
        if self.route is not None and ModelRef(self.route.provider, self.route.model) != self.ref:
            raise ValueError("candidate route identity does not match candidate ref")


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    candidates: tuple[SelectionCandidate, ...]
    purpose: str = ""
    explicit: ModelRef | None = None
    constraints: SelectionConstraints = field(default_factory=SelectionConstraints)
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)

    def __post_init__(self) -> None:
        refs = tuple(candidate.ref for candidate in self.candidates)
        if len(refs) != len(set(refs)):
            raise ValueError("selection candidates must have unique canonical identities")
        if self.explicit is not None and (not self.explicit.provider or not self.explicit.model):
            raise ValueError("explicit selection requires a canonical provider and model")


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    ref: ModelRef
    reasons: tuple[str, ...]


class SelectionReason(StrEnum):
    EXPLICIT_MATCH = "explicit_match"
    PREFERRED_MATCH = "preferred_match"
    POLICY_DEFAULT = "policy_default"
    EXPLICIT_UNAVAILABLE = "explicit_unavailable"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"


@dataclass(frozen=True, slots=True)
class ModelSelection:
    selected: SelectionCandidate | None
    eligible: tuple[SelectionCandidate, ...]
    rejected: tuple[CandidateRejection, ...]
    reason: SelectionReason
    purpose: str = ""
    policy: str = ""
    matched_alias: str = ""

    @property
    def success(self) -> bool:
        return self.selected is not None

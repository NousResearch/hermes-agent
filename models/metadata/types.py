"""Canonical model capability/metadata types.

This package answers "what can this model do?" without owning identity,
catalogue membership, route selection, request policy, or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from models.identity import ModelRef


@dataclass(frozen=True, slots=True)
class ReasoningMetadata:
    """Tri-state reasoning capability details for one effective model route."""

    supported: bool | None = None
    supported_efforts: tuple[str, ...] | None = None
    mandatory: bool | None = None


@dataclass(frozen=True, slots=True)
class ModelMetadataPatch:
    """Sparse metadata facts contributed by one source.

    None means "this source has no opinion". Definitive False is preserved
    and must never be collapsed into unknown.
    """

    context_window: int | None = None
    max_output_tokens: int | None = None
    max_input_tokens: int | None = None

    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    supports_structured_output: bool | None = None
    supports_temperature: bool | None = None

    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None

    reasoning: ReasoningMetadata | None = None

    model_family: str | None = None
    open_weights: bool | None = None
    release_date: str | None = None
    status: str | None = None
    knowledge_cutoff: str | None = None


@dataclass(frozen=True, slots=True)
class ModelMetadataContext:
    """Already-resolved route context used while answering metadata questions.

    The metadata domain may observe this route, but it does not select or
    normalize it. "explicit" and "configured" are normalized sparse facts
    supplied by the configuration/runtime boundary.
    """

    base_url: str = ""
    api_key: str = ""
    # Unnormalized route identity supplied by the caller when ModelRef has
    # canonicalized aliases (for example the managed llama.cpp aliases).
    route_provider: str = ""
    allow_network: bool = False
    explicit: ModelMetadataPatch | None = None
    configured: ModelMetadataPatch | None = None


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Canonical effective metadata for a model identity."""

    ref: ModelRef

    context_window: int | None = None
    max_output_tokens: int | None = None
    max_input_tokens: int | None = None

    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    supports_structured_output: bool | None = None
    supports_temperature: bool | None = None

    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()

    reasoning: ReasoningMetadata = field(default_factory=ReasoningMetadata)

    model_family: str = ""
    open_weights: bool | None = None
    release_date: str = ""
    status: str = ""
    knowledge_cutoff: str = ""

    # field name -> provenance label (explicit/configured/live/catalog/static)
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Full interpreted metadata for one catalogue model."""

    id: str
    name: str
    family: str
    provider_id: str
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False
    temperature: bool = False
    structured_output: bool = False
    open_weights: bool = False
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    context_window: int = 0
    max_output: int = 0
    max_input: Optional[int] = None
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None
    knowledge_cutoff: str = ""
    release_date: str = ""
    status: str = ""
    interleaved: Any = False

    def has_cost_data(self) -> bool:
        return self.cost_input > 0 or self.cost_output > 0

    def supports_vision(self) -> bool:
        return self.attachment or "image" in self.input_modalities

    def supports_pdf(self) -> bool:
        return "pdf" in self.input_modalities

    def supports_audio_input(self) -> bool:
        return "audio" in self.input_modalities

    def format_capabilities(self) -> str:
        """Human-readable capabilities, e.g. 'reasoning, tools, vision, PDF'."""
        flags = (
            (self.reasoning, "reasoning"),
            (self.tool_call, "tools"),
            (self.supports_vision(), "vision"),
            (self.supports_pdf(), "PDF"),
            (self.supports_audio_input(), "audio"),
            (self.structured_output, "structured output"),
            (self.open_weights, "open weights"),
        )
        return ", ".join(label for on, label in flags if on) or "basic"


@dataclass
class ProviderInfo:
    """Interpreted metadata for one catalogue provider."""

    id: str
    name: str
    env: tuple[str, ...]
    api: str
    doc: str = ""
    model_count: int = 0

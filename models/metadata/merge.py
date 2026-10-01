"""Deterministic precedence reduction for canonical model metadata."""

from __future__ import annotations

from dataclasses import fields
from typing import Iterable

from models.identity import ModelRef
from models.metadata.types import ModelMetadata, ModelMetadataPatch, ReasoningMetadata


_SCALAR_FIELDS = tuple(
    f.name
    for f in fields(ModelMetadataPatch)
    if f.name != "reasoning"
)


def _first_value(
    sources: tuple[tuple[str, ModelMetadataPatch], ...],
    field_name: str,
):
    for label, patch in sources:
        value = getattr(patch, field_name)
        if value is not None:
            return value, label
    return None, None


def merge_metadata(
    ref: ModelRef,
    sources: Iterable[tuple[str, ModelMetadataPatch]],
) -> ModelMetadata:
    """Merge sparse sources ordered from highest to lowest precedence.

    A source contributes only fields whose value is not None. Definitive False
    therefore survives, while unknown falls through to lower-precedence facts.
    """

    ordered = tuple(sources)
    values: dict[str, object] = {}
    provenance: dict[str, str] = {}

    for field_name in _SCALAR_FIELDS:
        value, label = _first_value(ordered, field_name)
        values[field_name] = value
        if label is not None:
            provenance[field_name] = label

    reasoning_supported = None
    reasoning_efforts = None
    reasoning_mandatory = None
    for label, patch in ordered:
        reasoning = patch.reasoning
        if reasoning is None:
            continue
        if reasoning_supported is None and reasoning.supported is not None:
            reasoning_supported = reasoning.supported
            provenance["reasoning.supported"] = label
        if reasoning_efforts is None and reasoning.supported_efforts is not None:
            reasoning_efforts = reasoning.supported_efforts
            provenance["reasoning.supported_efforts"] = label
        if reasoning_mandatory is None and reasoning.mandatory is not None:
            reasoning_mandatory = reasoning.mandatory
            provenance["reasoning.mandatory"] = label

    supports_reasoning = values["supports_reasoning"]
    if supports_reasoning is None and reasoning_supported is not None:
        supports_reasoning = reasoning_supported
        provenance["supports_reasoning"] = provenance["reasoning.supported"]
    elif reasoning_supported is None and supports_reasoning is not None:
        reasoning_supported = bool(supports_reasoning)
        provenance["reasoning.supported"] = provenance["supports_reasoning"]

    values["supports_reasoning"] = supports_reasoning

    return ModelMetadata(
        ref=ref,
        context_window=values["context_window"],
        max_output_tokens=values["max_output_tokens"],
        max_input_tokens=values["max_input_tokens"],
        supports_tools=values["supports_tools"],
        supports_vision=values["supports_vision"],
        supports_reasoning=values["supports_reasoning"],
        supports_structured_output=values["supports_structured_output"],
        supports_temperature=values["supports_temperature"],
        input_modalities=values["input_modalities"] or (),
        output_modalities=values["output_modalities"] or (),
        reasoning=ReasoningMetadata(
            supported=reasoning_supported,
            supported_efforts=reasoning_efforts,
            mandatory=reasoning_mandatory,
        ),
        model_family=values["model_family"] or "",
        open_weights=values["open_weights"],
        release_date=values["release_date"] or "",
        status=values["status"] or "",
        knowledge_cutoff=values["knowledge_cutoff"] or "",
        provenance=provenance,
    )


__all__ = ["merge_metadata"]

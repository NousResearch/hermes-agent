"""Canonical capability resolution for model metadata.

This module owns capability precedence and tri-state semantics. It deliberately
knows nothing about route selection, credentials, request formatting, catalog
membership, or any Hermes surface. Environment-specific facts arrive through
callable sources supplied by the boundary that can reach them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from models.metadata.merge import merge_metadata
from models.metadata.types import ModelMetadata, ModelMetadataContext, ModelMetadataPatch
from models.identity import ModelRef

CapabilitySource = Callable[[ModelRef, ModelMetadataContext], ModelMetadataPatch | None]


@dataclass(frozen=True, slots=True)
class CapabilitySources:
    """Ordered external fact providers.

    The order is part of the capability contract: live runtime facts precede
    catalog facts, which precede local provider probes and declarations.
    """

    live: CapabilitySource | None = None
    catalog: CapabilitySource | None = None
    local: CapabilitySource | None = None
    provider: CapabilitySource | None = None

    def ordered(self) -> tuple[tuple[str, CapabilitySource], ...]:
        return tuple(
            (name, source)
            for name, source in (
                ("live", self.live),
                ("catalog", self.catalog),
                ("local", self.local),
                ("provider", self.provider),
            )
            if source is not None
        )


def _patch_mapping(value: object) -> ModelMetadataPatch | None:
    if isinstance(value, ModelMetadataPatch):
        return value
    if not isinstance(value, Mapping):
        return None
    fields = {
        name: value[name]
        for name in ModelMetadataPatch.__dataclass_fields__
        if name in value
    }
    return ModelMetadataPatch(**fields)


def resolve_model_metadata(
    ref: ModelRef,
    *,
    context: ModelMetadataContext | None = None,
    sources: CapabilitySources | Iterable[tuple[str, CapabilitySource]] = CapabilitySources(),
) -> ModelMetadata:
    """Resolve sparse capability facts through the canonical precedence reducer."""
    context = context or ModelMetadataContext()
    patches: list[tuple[str, ModelMetadataPatch]] = []

    if context.explicit is not None:
        patches.append(("explicit", context.explicit))
    if context.configured is not None:
        patches.append(("configured", context.configured))

    ordered = sources.ordered() if isinstance(sources, CapabilitySources) else tuple(sources)
    for source_name, source in ordered:
        patch = _patch_mapping(source(ref, context))
        if patch is not None:
            patches.append((source_name, patch))

    return merge_metadata(ref, patches)

def resolve_supports_vision(
    ref: ModelRef,
    *,
    context: ModelMetadataContext | None = None,
    sources: CapabilitySources | Iterable[tuple[str, CapabilitySource]] = CapabilitySources(),
) -> bool | None:
    """Resolve only vision and stop once a source answers it."""
    context = context or ModelMetadataContext()
    for patch in (context.explicit, context.configured):
        if patch is not None and patch.supports_vision is not None:
            return patch.supports_vision
    ordered = sources.ordered() if isinstance(sources, CapabilitySources) else tuple(sources)
    for _source_name, source in ordered:
        patch = _patch_mapping(source(ref, context))
        if patch is not None and patch.supports_vision is not None:
            return patch.supports_vision
    return None


__all__ = [
    "CapabilitySource",
    "CapabilitySources",
    "resolve_model_metadata",
    "resolve_supports_vision",
]

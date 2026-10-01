"""Pure managed-runtime capability facts for the metadata domain.

The metadata layer owns the precedence-neutral interpretation of managed runtime
facts. Runtime discovery is injected by the application boundary so this module
never imports CLI, config, endpoint, or presentation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Protocol

from models.metadata.types import ModelMetadataPatch

LLAMACPP_ALIASES = frozenset({"llamacpp", "llama.cpp", "llama-cpp"})


class _CatalogEntry(Protocol):
    mmproj: object | None


class _Projector(Protocol):
    local_name: str


def is_managed_provider(
    provider: str,
    base_url: str = "",
    *,
    endpoint_matcher: Callable[[str], bool] | None = None,
) -> bool:
    """Return whether a route names or is identified as managed llama.cpp."""
    value = (provider or "").strip().lower()
    if value in LLAMACPP_ALIASES:
        return True
    return bool(
        value == "custom"
        and base_url
        and endpoint_matcher is not None
        and endpoint_matcher(base_url)
    )


def managed_model_metadata(
    model_id: str,
    *,
    staged_model_ids: Callable[[], Iterable[str]] | None = None,
    entry_for_model: Callable[[str], _CatalogEntry | None] | None = None,
    assets_dir: Callable[[], Path] | None = None,
    live_props: Callable[[str], bool | None] | None = None,
) -> ModelMetadataPatch | None:
    """Resolve a staged model's live modality or projector-backed capability.

    A missing callback means this boundary cannot reach managed-runtime state,
    which is intentionally represented as unknown rather than guessed.
    """
    if not model_id or any(
        callback is None
        for callback in (staged_model_ids, entry_for_model, assets_dir, live_props)
    ):
        return None
    if model_id not in staged_model_ids():
        return None

    live = live_props(model_id)
    if live is not None:
        return ModelMetadataPatch(supports_vision=live)

    entry = entry_for_model(model_id)
    projector = getattr(entry, "mmproj", None) if entry is not None else None
    if projector is None:
        return ModelMetadataPatch(supports_vision=False) if entry is not None else None
    return ModelMetadataPatch(
        supports_vision=(assets_dir() / projector.local_name).exists(),
    )


def managed_model_supports_vision(
    model_id: str,
    *,
    staged_model_ids: Callable[[], Iterable[str]] | None = None,
    entry_for_model: Callable[[str], _CatalogEntry | None] | None = None,
    assets_dir: Callable[[], Path] | None = None,
    live_props: Callable[[str], bool | None] | None = None,
) -> bool | None:
    """Return the managed runtime's tri-state vision answer."""
    patch = managed_model_metadata(
        model_id,
        staged_model_ids=staged_model_ids,
        entry_for_model=entry_for_model,
        assets_dir=assets_dir,
        live_props=live_props,
    )
    return patch.supports_vision if patch is not None else None


__all__ = [
    "LLAMACPP_ALIASES",
    "is_managed_provider",
    "managed_model_metadata",
    "managed_model_supports_vision",
]

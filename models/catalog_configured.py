"""Canonical, I/O-free rules for configured endpoint model declarations."""
from __future__ import annotations

from typing import Any

def declared_model_ids(value: Any) -> list[str]:
    """Configured model IDs from ``{"id": {...}}``, ``["a", "b"]``, ``[{"id"|"name": ...}]`` or ``"a"``."""
    if isinstance(value, str):
        candidates: Any = [value]
    elif isinstance(value, dict):
        # Pre-fix Hermes wrote sentinel keys inside the user-facing ``models`` mapping.
        candidates = (k for k in value if k not in ("__explicit_model_allowlist__", "__discovered_model_catalog__"))
    elif isinstance(value, (list, tuple)):
        candidates = (_declared_item_id(item) if isinstance(item, dict) else item for item in value)
    else:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue  # non-str items are dropped
        model_id = candidate.strip()
        if model_id and model_id.lower() not in seen:
            seen.add(model_id.lower())
            ids.append(model_id)
    return ids


def _declared_item_id(item: dict) -> Any:
    """``id`` of a ``[{"id": ...}]`` entry, falling back to ``name`` when blank/missing."""
    model_id = item.get("id")
    return model_id if isinstance(model_id, str) and model_id.strip() else item.get("name")


def entry_models_discovered(entry: Any) -> bool:
    """True when the entry's ``models`` mapping was auto-discovered by Hermes.

    Current shape: entry-level ``models_discovered: true``. Older versions wrote an in-mapping
    ``__discovered_model_catalog__: true`` sentinel — accepted on read (the next save migrates it)."""
    if not isinstance(entry, dict):
        return False
    models = entry.get("models")
    return entry.get("models_discovered") is True or (
        isinstance(models, dict) and models.get("__discovered_model_catalog__") is True)


def models_config_is_allowlist(value: Any, discovered: bool = False) -> bool:
    """True when ``models:`` is an intentional ID allowlist.

    A mapping like ``{model_id: {context_length: N}}`` is per-model *metadata* written by
    ``_save_custom_provider`` / the wizard, not a catalog narrow (treating it as one made GUI
    pickers show only the saved default for keyless Ollama while the CLI live-probed). List and
    string shapes remain allowlists for no-key endpoints; pin a dict catalog with
    ``discover_models: false``. A catalog Hermes itself persisted (``discovered``) is never a pin."""
    if discovered:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(declared_model_ids(value))
    return False  # None, dict (per-model metadata), or anything else


def discovery_enabled(entry: dict):
    """``discover_models`` (default True); ``"false"/"no"/"0"`` strings mean False."""
    discover = entry.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    return discover

__all__ = ["declared_model_ids", "entry_models_discovered", "models_config_is_allowlist", "discovery_enabled"]

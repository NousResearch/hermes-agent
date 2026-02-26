"""Codex model resolution from ~/.codex config and cache files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

CODEX_HOME = Path.home() / ".codex"
CODEX_CONFIG_FILE = CODEX_HOME / "config.toml"
CODEX_MODELS_CACHE_FILE = CODEX_HOME / "models_cache.json"


def get_codex_default_model() -> Optional[str]:
    """Return Codex's configured default model from ~/.codex/config.toml."""
    if not CODEX_CONFIG_FILE.exists():
        return None
    try:
        import tomllib
    except Exception:
        return None
    try:
        data = tomllib.loads(CODEX_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = data.get("model") if isinstance(data, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def get_codex_model_ids() -> list[str]:
    """Return Codex model IDs from Codex cache, with configured default first."""
    cache_models: list[str] = []
    if CODEX_MODELS_CACHE_FILE.exists():
        try:
            raw = json.loads(CODEX_MODELS_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        entries = raw.get("models") if isinstance(raw, dict) else None
        sortable = []
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                slug = item.get("slug")
                if not isinstance(slug, str) or not slug.strip():
                    continue
                if "codex" not in slug.lower():
                    continue
                if item.get("supported_in_api") is False:
                    continue
                visibility = item.get("visibility")
                if isinstance(visibility, str) and visibility.strip().lower() == "hidden":
                    continue
                priority = item.get("priority")
                sort_priority = priority if isinstance(priority, (int, float)) else 10_000
                sortable.append((sort_priority, slug.strip()))
        sortable.sort(key=lambda x: (x[0], x[1]))
        for _, model in sortable:
            if model not in cache_models:
                cache_models.append(model)

    ordered: list[str] = []
    default_model = get_codex_default_model()
    if default_model:
        ordered.append(default_model)
    for model in cache_models:
        if model not in ordered:
            ordered.append(model)
    return ordered

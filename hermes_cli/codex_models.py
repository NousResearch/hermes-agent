"""Codex model discovery from local Codex CLI cache/config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from hermes_cli.auth import resolve_codex_home_path

DEFAULT_CODEX_MODELS: List[str] = [
    "gpt-5-codex",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex",
]


def _read_default_model(codex_home: Path) -> Optional[str]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib
    except Exception:
        return None
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _read_cache_models(codex_home: Path) -> List[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    sortable = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug.strip():
                continue
            slug = slug.strip()
            if "codex" not in slug.lower():
                continue
            if item.get("supported_in_api") is False:
                continue
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() == "hidden":
                continue
            priority = item.get("priority")
            rank = int(priority) if isinstance(priority, (int, float)) else 10_000
            sortable.append((rank, slug))

    sortable.sort(key=lambda item: (item[0], item[1]))
    deduped: List[str] = []
    for _, slug in sortable:
        if slug not in deduped:
            deduped.append(slug)
    return deduped


def get_codex_model_ids() -> List[str]:
    codex_home = resolve_codex_home_path()
    ordered: List[str] = []

    default_model = _read_default_model(codex_home)
    if default_model:
        ordered.append(default_model)

    for model_id in _read_cache_models(codex_home):
        if model_id not in ordered:
            ordered.append(model_id)

    for model_id in DEFAULT_CODEX_MODELS:
        if model_id not in ordered:
            ordered.append(model_id)

    return ordered

"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import os

logger = logging.getLogger(__name__)

ACTIVE_CODEX_MODEL_IDS: tuple[str, ...] = ("gpt-5.5",)

DEFAULT_CODEX_MODELS: List[str] = [
    # Curated active fallback for the OAuth-backed Codex route.
    # Older GPT-5.x Codex slugs may still appear in cached/API catalogs, but
    # they must not be hardcoded or returned as active fallback recommendations.
    *ACTIVE_CODEX_MODEL_IDS,
]

_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [
    # Only synthesize the current active Codex slug from older live templates.
    # Do not synthesize stale slugs into model pickers.
    ("gpt-5.5", ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex")),
]


def _add_forward_compat_models(model_ids: List[str]) -> List[str]:
    """Add Clawdbot-style synthetic forward-compat Codex models.

    If a newer Codex slug isn't returned by live discovery, surface it when an
    older compatible template model is present. This mirrors Clawdbot's
    synthetic catalog / forward-compat behavior for GPT-5 Codex variants.
    """
    ordered: List[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)

    for synthetic_model, template_models in _FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model in seen:
            continue
        if any(template in seen for template in template_models):
            ordered.append(synthetic_model)
            seen.add(synthetic_model)

    return ordered


def _filter_active_codex_models(model_ids: List[str]) -> List[str]:
    """Return only verified active Codex model recommendations.

    Local/API catalogs can contain stale GPT-5.x slugs from prior Codex
    generations. Hermes may still pass through explicit user-selected models
    elsewhere, but discovery-driven recommendations must stay latest-only.
    """
    expanded = _add_forward_compat_models(model_ids)
    active = [model_id for model_id in expanded if model_id in ACTIVE_CODEX_MODEL_IDS]
    if active:
        deduped: List[str] = []
        for model_id in active:
            if model_id not in deduped:
                deduped.append(model_id)
        return deduped
    return list(DEFAULT_CODEX_MODELS)


def _fetch_catalog_models_from_api(access_token: str) -> List[str]:
    """Fetch visible Codex API catalog models sorted by priority."""
    try:
        import httpx
        resp = httpx.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        entries = data.get("models", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.debug("Failed to fetch Codex models from API: %s", exc)
        return []

    sortable = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        # Codex CLI's catalog uses ``supported_in_api`` for the public OpenAI
        # API, not for the OAuth-backed Codex backend that this provider uses.
        # Some valid Codex CLI models (for example gpt-5.3-codex-spark) are
        # marked false here but are still accepted by the Codex route.
        visibility = item.get("visibility", "")
        if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        sortable.append((rank, slug))

    sortable.sort(key=lambda x: (x[0], x[1]))
    return [slug for _, slug in sortable]


def _fetch_models_from_api(access_token: str) -> List[str]:
    """Fetch available active Codex models from the API."""
    return _filter_active_codex_models(_fetch_catalog_models_from_api(access_token))


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
            # Do not filter on ``supported_in_api`` here.  It describes the
            # public OpenAI API, while Hermes openai-codex talks to the same
            # OAuth-backed Codex backend as Codex CLI.
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
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


def get_codex_model_ids(access_token: Optional[str] = None) -> List[str]:
    """Return available Codex model IDs, trying API first, then local sources.
    
    Resolution order: API (live, if token provided) > config.toml default >
    local cache > hardcoded defaults.
    """
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()
    ordered: List[str] = []

    # Try live API if we have a token
    if access_token:
        api_models = _fetch_models_from_api(access_token)
        if api_models:
            return _filter_active_codex_models(api_models)

    # Fall back to local sources
    default_model = _read_default_model(codex_home)
    if default_model:
        ordered.append(default_model)

    for model_id in _read_cache_models(codex_home):
        if model_id not in ordered:
            ordered.append(model_id)

    for model_id in DEFAULT_CODEX_MODELS:
        if model_id not in ordered:
            ordered.append(model_id)

    return _filter_active_codex_models(ordered)


def get_codex_catalog_model_ids(access_token: Optional[str] = None) -> List[str]:
    """Return visible Codex catalog models for picker display.

    Unlike ``get_codex_model_ids()``, this preserves live/cache Codex-only
    slugs. The active recommendation path remains latest-only, but picker
    display and provider ownership need the full Codex catalog.
    """
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()
    ordered: List[str] = []

    if access_token:
        api_models = _fetch_catalog_models_from_api(access_token)
        if api_models:
            return api_models

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

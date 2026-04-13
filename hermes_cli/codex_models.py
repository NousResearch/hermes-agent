"""Codex model discovery and runtime model normalization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import os

logger = logging.getLogger(__name__)

DEFAULT_CODEX_MODELS: List[str] = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
]

_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [
    ("gpt-5.4-mini", ("gpt-5.3-codex", "gpt-5.2-codex")),
    ("gpt-5.4", ("gpt-5.3-codex", "gpt-5.2-codex")),
    ("gpt-5.3-codex", ("gpt-5.2-codex",)),
    ("gpt-5.3-codex-spark", ("gpt-5.3-codex", "gpt-5.2-codex")),
]

_DEFAULT_CODEX_FALLBACK_MODEL = "gpt-5.3-codex"
_LIKELY_OPENAI_CODEX_PREFIXES = (
    "gpt-",
    "codex",
    "computer-use-",
)


@dataclass(frozen=True)
class CodexModelResolution:
    """Normalized model choice for the Codex backend."""

    model: str
    changed: bool
    stripped_prefix: bool = False
    replaced_incompatible: bool = False
    used_fallback: bool = False


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


def _looks_like_openai_codex_model(
    model_id: str,
    *,
    available_models: Optional[List[str]] = None,
) -> bool:
    """Best-effort check for models that plausibly belong on Codex.

    We intentionally preserve explicit OpenAI-family slugs even when they
    are newer than our local catalog, but reject obviously cross-provider
    slugs like Claude/Gemini/Qwen when routed to the Codex backend.
    """
    normalized = model_id.strip().lower()
    if not normalized:
        return False
    if available_models and normalized in {item.strip().lower() for item in available_models if item}:
        return True
    if any(normalized.startswith(prefix) for prefix in _LIKELY_OPENAI_CODEX_PREFIXES):
        return True
    if len(normalized) >= 2 and normalized[0] == "o" and normalized[1].isdigit():
        return True
    return False


def normalize_codex_runtime_model(
    model_id: str,
    *,
    access_token: Optional[str] = None,
    model_is_default: bool = False,
) -> CodexModelResolution:
    """Normalize a runtime model before sending it to the Codex backend."""
    original = (model_id or "").strip()
    candidate = original
    stripped_prefix = False

    if "/" in candidate:
        candidate = candidate.split("/", 1)[1].strip()
        stripped_prefix = True

    fallback_model = _DEFAULT_CODEX_FALLBACK_MODEL
    try:
        available_models = get_codex_model_ids(access_token=access_token)
    except Exception:
        available_models = []
    if available_models:
        fallback_model = available_models[0]

    if not candidate or model_is_default:
        return CodexModelResolution(
            model=fallback_model,
            changed=(candidate != fallback_model) or stripped_prefix,
            stripped_prefix=stripped_prefix,
            used_fallback=True,
        )

    if _looks_like_openai_codex_model(candidate, available_models=available_models):
        return CodexModelResolution(
            model=candidate,
            changed=stripped_prefix,
            stripped_prefix=stripped_prefix,
        )

    return CodexModelResolution(
        model=fallback_model,
        changed=(candidate != fallback_model) or stripped_prefix,
        stripped_prefix=stripped_prefix,
        replaced_incompatible=True,
        used_fallback=True,
    )


def _fetch_models_from_api(access_token: str) -> List[str]:
    """Fetch available models from the Codex API. Returns visible models sorted by priority."""
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
        if item.get("supported_in_api") is False:
            continue
        visibility = item.get("visibility", "")
        if isinstance(visibility, str) and visibility.strip().lower() in ("hide", "hidden"):
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        sortable.append((rank, slug))

    sortable.sort(key=lambda x: (x[0], x[1]))
    return _add_forward_compat_models([slug for _, slug in sortable])


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
            if item.get("supported_in_api") is False:
                continue
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() in ("hide", "hidden"):
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
            return _add_forward_compat_models(api_models)

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

    return _add_forward_compat_models(ordered)

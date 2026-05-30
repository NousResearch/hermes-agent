"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

logger = logging.getLogger(__name__)

DEFAULT_CODEX_REASONING_EFFORTS: List[str] = [
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]


@dataclass
class CodexReasoningEffort:
    effort: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"effort": self.effort}
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class CodexModelInfo:
    id: str
    display_name: str
    description: str = ""
    context_window: Optional[int] = None
    reasoning_efforts: List[CodexReasoningEffort] = field(default_factory=list)
    default_reasoning_effort: Optional[str] = None
    priority: int = 10_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "context_length": self.context_window,
            "reasoning_efforts": [item.to_dict() for item in self.reasoning_efforts],
            "default_reasoning_effort": self.default_reasoning_effort,
        }


DEFAULT_CODEX_MODELS: List[str] = [
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    # gpt-5.3-codex-spark is in research preview and is exposed *only* via
    # the Codex CLI / OAuth backend (chatgpt.com/backend-api/codex/models)
    # for ChatGPT Pro subscribers. It is NOT available in the public OpenAI
    # API, so it intentionally stays out of the "openai" provider catalog
    # in hermes_cli/models.py — only the openai-codex (OAuth) provider
    # surfaces it. The Codex backend reports ``supported_in_api: false`` for
    # this slug; that flag describes API availability, not Codex backend
    # availability, so the fetch/cache code paths below intentionally do
    # not filter on it. PR #12994 removed this entry on the assumption it
    # was unsupported — that was wrong; restored here. Keep it in the
    # curated fallback so Pro users still see Spark in `/model` when live
    # discovery is unavailable (offline first run, transient API failure).
    "gpt-5.3-codex-spark",
    # NOTE: gpt-5.2-codex / gpt-5.1-codex-max / gpt-5.1-codex-mini were
    # previously listed here but the chatgpt.com Codex backend returns
    # HTTP 400 "The '<model>' model is not supported when using Codex with
    # a ChatGPT account." for all three on every ChatGPT Pro account we've
    # tested (verified live 2026-05-27). Keeping them in the fallback list
    # leaked dead slugs into /model when live discovery was unavailable
    # (transient API failure, first-run before refresh) and surfaced HTTP 400
    # crashes on selection. The Codex CLI public catalog still references
    # these slugs, which is why they survived previously — but those entries
    # describe the public OpenAI API, not the OAuth-backed Codex backend
    # Hermes uses. Removed here. If OpenAI re-enables them on Codex backend,
    # live discovery will pick them up automatically via _fetch_models_from_api.
]

_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [
    ("gpt-5.5", ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex")),
    ("gpt-5.4-mini", ("gpt-5.3-codex",)),
    ("gpt-5.4", ("gpt-5.3-codex",)),
    # Surface Spark whenever any compatible Codex template is present so
    # accounts hitting the live endpoint with an older lineup still see
    # Spark in the picker. Backend gates real availability by ChatGPT Pro
    # entitlement; Hermes does not.
    ("gpt-5.3-codex-spark", ("gpt-5.3-codex",)),
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


def _format_codex_display_name(slug: str) -> str:
    cleaned = slug.replace("-", " ").replace("_", " ").strip()
    if not cleaned:
        return slug
    return re.sub(r"\bgpt\b", "GPT", cleaned, flags=re.IGNORECASE).title()


def _parse_reasoning_efforts(raw: Any) -> List[CodexReasoningEffort]:
    if not isinstance(raw, list):
        return []

    efforts: List[CodexReasoningEffort] = []
    seen: set[str] = set()
    for item in raw:
        effort = ""
        description = ""
        if isinstance(item, str):
            effort = item.strip().lower()
        elif isinstance(item, dict):
            for key in ("effort", "level", "id", "name"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    effort = value.strip().lower()
                    break
            desc = item.get("description")
            if isinstance(desc, str):
                description = desc.strip()
        if not effort or effort in seen:
            continue
        seen.add(effort)
        efforts.append(CodexReasoningEffort(effort=effort, description=description))
    return efforts


def _parse_codex_model_entry(item: dict) -> Optional[CodexModelInfo]:
    slug = item.get("slug")
    if not isinstance(slug, str) or not slug.strip():
        return None
    slug = slug.strip()

    visibility = item.get("visibility", "")
    if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
        return None

    display_name = item.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = _format_codex_display_name(slug)
    else:
        display_name = display_name.strip()

    description = item.get("description")
    if not isinstance(description, str):
        description = ""
    else:
        description = description.strip()

    context_window = item.get("context_window")
    if not isinstance(context_window, int):
        context_window = None

    reasoning_efforts = _parse_reasoning_efforts(item.get("supported_reasoning_levels"))
    if not reasoning_efforts:
        reasoning_efforts = [
            CodexReasoningEffort(effort=effort) for effort in DEFAULT_CODEX_REASONING_EFFORTS
        ]

    default_reasoning = item.get("default_reasoning_level")
    if not isinstance(default_reasoning, str) or not default_reasoning.strip():
        default_reasoning = None
    else:
        default_reasoning = default_reasoning.strip().lower()

    priority = item.get("priority")
    rank = int(priority) if isinstance(priority, (int, float)) else 10_000

    return CodexModelInfo(
        id=slug,
        display_name=display_name,
        description=description,
        context_window=context_window,
        reasoning_efforts=reasoning_efforts,
        default_reasoning_effort=default_reasoning,
        priority=rank,
    )


def _fetch_catalog_entries_from_api(access_token: str) -> List[CodexModelInfo]:
    """Fetch the full Codex model catalog from the live backend."""
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

    parsed: List[CodexModelInfo] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        model = _parse_codex_model_entry(item)
        if model is not None:
            parsed.append(model)

    parsed.sort(key=lambda model: (model.priority, model.id))
    return _add_forward_compat_catalog(parsed)


def _fetch_models_from_api(access_token: str) -> List[str]:
    """Fetch available models from the Codex API. Returns visible models sorted by priority."""
    return [model.id for model in _fetch_catalog_entries_from_api(access_token)]


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


def _read_cache_catalog(codex_home: Path) -> List[CodexModelInfo]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    parsed: List[CodexModelInfo] = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            model = _parse_codex_model_entry(item)
            if model is not None:
                parsed.append(model)

    parsed.sort(key=lambda model: (model.priority, model.id))
    return parsed


def _read_cache_models(codex_home: Path) -> List[str]:
    return [model.id for model in _read_cache_catalog(codex_home)]


def _fallback_codex_model_info(model_id: str) -> CodexModelInfo:
    return CodexModelInfo(
        id=model_id,
        display_name=_format_codex_display_name(model_id),
        reasoning_efforts=[
            CodexReasoningEffort(effort=effort) for effort in DEFAULT_CODEX_REASONING_EFFORTS
        ],
        default_reasoning_effort="medium",
    )


def _add_forward_compat_catalog(models: List[CodexModelInfo]) -> List[CodexModelInfo]:
    """Add synthetic forward-compat Codex models when templates are present."""
    ordered: List[CodexModelInfo] = []
    seen: set[str] = set()
    for model in models:
        if model.id not in seen:
            ordered.append(model)
            seen.add(model.id)

    for synthetic_model, template_models in _FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model in seen:
            continue
        if any(template in seen for template in template_models):
            ordered.append(_fallback_codex_model_info(synthetic_model))
            seen.add(synthetic_model)

    return ordered


def _merge_catalog_entries(*sources: List[CodexModelInfo]) -> List[CodexModelInfo]:
    merged: Dict[str, CodexModelInfo] = {}
    for entries in sources:
        for entry in entries:
            merged[entry.id] = entry
    ordered = list(merged.values())
    ordered.sort(key=lambda model: (model.priority, model.id))
    return _add_forward_compat_catalog(ordered)


def list_codex_picker_models(
    *,
    access_token: Optional[str] = None,
    query: str = "",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return Codex CLI catalog entries for remote model pickers."""
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()

    catalog: List[CodexModelInfo] = []
    if access_token:
        catalog = _fetch_catalog_entries_from_api(access_token)

    if not catalog:
        cache_catalog = _read_cache_catalog(codex_home)
        if cache_catalog:
            catalog = cache_catalog
        else:
            catalog = [_fallback_codex_model_info(model_id) for model_id in DEFAULT_CODEX_MODELS]

        default_model = _read_default_model(codex_home)
        if default_model and default_model not in {model.id for model in catalog}:
            catalog.insert(0, _fallback_codex_model_info(default_model))

        catalog = _merge_catalog_entries(catalog)

    needle = str(query or "").strip().lower()
    if needle:
        catalog = [
            model
            for model in catalog
            if needle in model.id.lower() or needle in model.display_name.lower()
        ]

    if limit is not None and limit >= 0:
        catalog = catalog[:limit]

    return [model.to_dict() for model in catalog]


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

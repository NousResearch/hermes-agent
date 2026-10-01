"""GitHub/Copilot reasoning and context capability queries."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from models.catalog_github import fetch_github_model_catalog
from models.metadata.reasoning import CODEX_ASTRA_EFFORTS, clamp_effort, is_astra_model
from providers.model_normalizers import normalize_copilot_id

COPILOT_REASONING_EFFORTS_GPT5: tuple[str, ...] = ("minimal", "low", "medium", "high")
COPILOT_REASONING_EFFORTS_O_SERIES: tuple[str, ...] = ("low", "medium", "high")

_GITHUB_CONTEXT_CACHE_TTL = 3600.0
_github_context_cache: dict[str, int] = {}
_github_context_cache_key = ""
_github_context_cache_time = 0.0


def _catalog_ids(catalog: Optional[list[dict[str, Any]]]) -> tuple[str, ...]:
    return tuple(
        model_id
        for item in (catalog or ())
        if (model_id := str(item.get("id") or "").strip())
    )


def _fallback_efforts(model_id: str, *, known_ids: tuple[str, ...] = ()) -> list[str]:
    raw = str(model_id or "").strip().lower()
    if raw.startswith(("openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4")):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_id(model_id, known_ids).lower()
    if is_astra_model(normalized):
        return list(CODEX_ASTRA_EFFORTS)
    if normalized.startswith("gpt-5"):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []


def github_model_reasoning_efforts(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
) -> list[str]:
    """Return supported effort levels from caller-supplied catalog facts or model family."""
    known_ids = _catalog_ids(catalog)
    normalized = normalize_copilot_id(str(model_id or ""), known_ids)
    if not normalized:
        return []

    catalog_entry = (
        next((item for item in catalog if item.get("id") == normalized), None)
        if catalog
        else None
    )
    if catalog_entry is not None:
        capabilities = catalog_entry.get("capabilities")
        if isinstance(capabilities, dict):
            supports = capabilities.get("supports")
            efforts = supports.get("reasoning_effort") if isinstance(supports, dict) else None
            if not isinstance(efforts, list):
                return []
            return list(
                dict.fromkeys(
                    effort
                    for value in efforts
                    if (effort := str(value).strip().lower())
                )
            )
        if "reasoning" not in {
            str(value).strip().lower()
            for value in catalog_entry.get("capabilities", [])
        }:
            return []

    return _fallback_efforts(str(model_id or normalized), known_ids=known_ids)


def clamp_github_reasoning_effort(effort: Any, supported: list[str]) -> str:
    """Clamp GitHub reasoning effort with the historical medium/first fallback."""
    requested = str(effort or "medium").strip().lower()
    if requested not in supported:
        clamped = clamp_effort(requested, supported)
        requested = str(clamped or requested)
        if requested not in supported:
            requested = "medium" if "medium" in supported else supported[0]
    return requested



def _credential_key(api_key: object) -> str:
    try:
        value = api_key() if callable(api_key) else api_key
    except Exception:
        return ""
    token = value.strip() if isinstance(value, str) else ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""


def github_model_context_length(
    model_id: str,
    *,
    api_key: object = None,
    catalog: Optional[list[dict[str, Any]]] = None,
) -> Optional[int]:
    """Return the account-scoped GitHub model prompt-token limit."""
    global _github_context_cache, _github_context_cache_key, _github_context_cache_time

    key = _credential_key(api_key)
    if catalog is None and (
        _github_context_cache
        and _github_context_cache_key == key
        and time.monotonic() - _github_context_cache_time < _GITHUB_CONTEXT_CACHE_TTL
    ):
        return _github_context_cache.get(model_id)

    rows = catalog if catalog is not None else fetch_github_model_catalog(api_key=api_key)
    if not rows:
        return None

    cache: dict[str, int] = {}
    for item in rows:
        mid = str(item.get("id") or "").strip()
        capabilities = item.get("capabilities")
        limits = capabilities.get("limits") if isinstance(capabilities, dict) else None
        max_prompt = limits.get("max_prompt_tokens") if isinstance(limits, dict) else None
        if mid and isinstance(max_prompt, int) and max_prompt > 0:
            cache[mid] = max_prompt

    if catalog is None:
        _github_context_cache = cache
        _github_context_cache_key = key
        _github_context_cache_time = time.monotonic()
    return cache.get(model_id)


def reset_github_context_cache() -> None:
    global _github_context_cache, _github_context_cache_key, _github_context_cache_time
    _github_context_cache = {}
    _github_context_cache_key = ""
    _github_context_cache_time = 0.0


__all__ = [
    "COPILOT_REASONING_EFFORTS_GPT5",
    "COPILOT_REASONING_EFFORTS_O_SERIES",
    "clamp_github_reasoning_effort",
    "github_model_context_length",
    "github_model_reasoning_efforts",
    "reset_github_context_cache",
]

"""GitHub Copilot live model catalogue source."""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any

from providers.github import copilot_request_headers


COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
_COPILOT_CHAT_ENDPOINTS = frozenset({"/chat/completions", "/responses", "/v1/messages"})
_GITHUB_MODEL_CATALOG_CACHE_TTL = 300.0

_github_model_catalog_cache: list[dict[str, Any]] | None = None
_github_model_catalog_cache_key = ""
_github_model_catalog_cache_time = 0.0


def _credential_key(api_key: object) -> str:
    try:
        value = api_key() if callable(api_key) else api_key
    except Exception:
        value = ""
    token = value.strip() if isinstance(value, str) else ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""


def _materialize_api_key(api_key: object) -> str:
    try:
        value = api_key() if callable(api_key) else api_key
    except Exception:
        return ""
    return value.strip() if isinstance(value, str) else ""


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data", []) if isinstance(payload, dict) else payload
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _catalog_item_is_text_model(
    item: dict[str, Any], *, ignore_picker_flag: bool = False
) -> bool:
    if not str(item.get("id") or "").strip():
        return False
    if not ignore_picker_flag and item.get("model_picker_enabled") is False:
        return False
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type and model_type != "chat":
            return False
    endpoints = item.get("supported_endpoints")
    if isinstance(endpoints, list):
        supported = {str(value).strip() for value in endpoints if str(value).strip()}
        if supported and not supported & _COPILOT_CHAT_ENDPOINTS:
            return False
    return True


def _text_models(
    items: list[dict[str, Any]], *, ignore_picker_flag: bool = False
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        model_id = str(item.get("id") or "").strip()
        if model_id in seen or not _catalog_item_is_text_model(
            item, ignore_picker_flag=ignore_picker_flag
        ):
            continue
        seen.add(model_id)
        models.append(item)
    return models


def _fetch_json(url: str, *, timeout: float, headers: dict[str, str]) -> Any:
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def fetch_github_model_catalog(
    api_key: object = None, timeout: float = 5.0
) -> list[dict[str, Any]] | None:
    """Fetch the account-scoped GitHub Copilot model catalogue."""
    global _github_model_catalog_cache, _github_model_catalog_cache_key
    global _github_model_catalog_cache_time

    key = _credential_key(api_key)
    if (
        _github_model_catalog_cache is not None
        and _github_model_catalog_cache_key == key
        and time.monotonic() - _github_model_catalog_cache_time
        < _GITHUB_MODEL_CATALOG_CACHE_TTL
    ):
        return copy.deepcopy(_github_model_catalog_cache)

    token = _materialize_api_key(api_key)
    attempts: list[dict[str, str]] = []
    if token:
        attempts.append(
            {**copilot_request_headers(), "Authorization": f"Bearer {token}"}
        )
    attempts.append(copilot_request_headers())

    for headers in attempts:
        try:
            items = _payload_items(
                _fetch_json(COPILOT_MODELS_URL, timeout=timeout, headers=headers)
            )
        except Exception:
            continue
        models = _text_models(items)
        if not models and items:
            models = _text_models(items, ignore_picker_flag=True)
        if models:
            _github_model_catalog_cache = copy.deepcopy(models)
            _github_model_catalog_cache_key = key
            _github_model_catalog_cache_time = time.monotonic()
            return models
    return None


def reset_github_model_catalog_cache() -> None:
    global _github_model_catalog_cache, _github_model_catalog_cache_key
    global _github_model_catalog_cache_time
    _github_model_catalog_cache = None
    _github_model_catalog_cache_key = ""
    _github_model_catalog_cache_time = 0.0


__all__ = [
    "COPILOT_BASE_URL",
    "COPILOT_MODELS_URL",
    "fetch_github_model_catalog",
    "reset_github_model_catalog_cache",
]

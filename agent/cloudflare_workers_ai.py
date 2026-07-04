"""Cloudflare Workers AI catalog helpers.

Shared by the custom-provider discovery path, the first-class Cloudflare
provider profile, and capability fallbacks when models.dev has no entry for a
Cloudflare-served model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Optional

from hermes_cli import __version__ as _HERMES_VERSION

logger = logging.getLogger(__name__)

_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"
_CATALOG_TTL_SECONDS = 300.0
_catalog_cache_lock = threading.Lock()
_catalog_cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}


def is_cloudflare_workers_ai_base_url(base_url: str) -> bool:
    """Return True when *base_url* is a Cloudflare Workers AI OpenAI endpoint."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return False
    parsed = urllib.parse.urlparse(normalized)
    path = parsed.path.rstrip("/")
    return (
        "api.cloudflare.com" in (parsed.netloc or "").lower()
        and path.endswith("/ai/v1")
    )


def cloudflare_ai_models_search_url(base_url: str) -> Optional[str]:
    """Map a Workers AI inference base URL to Cloudflare's native catalog URL."""
    normalized = (base_url or "").strip().rstrip("/")
    if not is_cloudflare_workers_ai_base_url(normalized):
        return None
    parsed = urllib.parse.urlparse(normalized)
    path = parsed.path.rstrip("/")
    search_path = path[: -len("/ai/v1")] + "/ai/models/search"
    return urllib.parse.urlunparse(
        parsed._replace(path=search_path, params="", query="", fragment="")
    )


def parse_cloudflare_model_search_response(payload: Any) -> list[dict[str, Any]]:
    """Return the raw Cloudflare catalog entries from ``result[]``."""
    if not isinstance(payload, dict):
        return []
    result = payload.get("result")
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def _cloudflare_task_name(entry: dict[str, Any]) -> str:
    raw_task = entry.get("task")
    if not isinstance(raw_task, dict):
        return ""
    return str(raw_task.get("name") or "").strip()


def cloudflare_model_names(payload: Any, *, text_generation_only: bool = True) -> list[str]:
    """Extract unique model IDs from a Cloudflare model-catalog payload.

    By default Hermes only surfaces ``Text Generation`` models from Workers AI.
    The catalog includes embeddings, TTS, ASR, text-to-image, classifiers, and
    other non-agentic tasks that do not belong in the main model picker.
    """
    models: list[str] = []
    seen: set[str] = set()
    for item in parse_cloudflare_model_search_response(payload):
        if text_generation_only and _cloudflare_task_name(item) != "Text Generation":
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen:
            continue
        # Cloudflare catalog uses "meta-llama" but the inference API expects "meta".
        name = name.replace("@cf/meta-llama/", "@cf/meta/")
        seen.add(name)
        models.append(name)
    return models


def _catalog_cache_key(search_url: str, api_key: str) -> tuple[str, str]:
    digest = hashlib.blake2b(api_key.encode("utf-8", errors="replace"), digest_size=8)
    return search_url, digest.hexdigest()


def fetch_cloudflare_model_catalog(
    api_key: str,
    base_url: str,
    *,
    timeout: float = 8.0,
    force_refresh: bool = False,
) -> Optional[list[dict[str, Any]]]:
    """Fetch Cloudflare's native Workers AI catalog for a given account."""
    search_url = cloudflare_ai_models_search_url(base_url)
    if not search_url or not api_key:
        return None

    cache_key = _catalog_cache_key(search_url, api_key)
    now = time.monotonic()
    if not force_refresh:
        with _catalog_cache_lock:
            cached = _catalog_cache.get(cache_key)
            if cached is not None:
                ts, entries = cached
                if now - ts < _CATALOG_TTL_SECONDS:
                    return [dict(item) for item in entries]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": _HERMES_USER_AGENT,
    }
    req = urllib.request.Request(search_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.debug("Cloudflare catalog fetch failed for %s: %s", search_url, exc)
        return None

    entries = parse_cloudflare_model_search_response(payload)
    with _catalog_cache_lock:
        _catalog_cache[cache_key] = (now, [dict(item) for item in entries])
    return entries


def _properties_map(entry: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prop in entry.get("properties") or []:
        if not isinstance(prop, dict):
            continue
        property_id = str(prop.get("property_id") or "").strip()
        if not property_id:
            continue
        out[property_id] = prop.get("value")
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except Exception:
            return None
        return parsed if parsed > 0 else None
    return None


def model_capabilities_from_cloudflare_catalog_entry(
    entry: dict[str, Any], *, model: Optional[str] = None
):
    """Convert a Cloudflare catalog entry into ``agent.models_dev.ModelCapabilities``."""
    from agent.models_dev import ModelCapabilities

    props = _properties_map(entry)
    raw_task = entry.get("task")
    task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
    task_name = str(task.get("name") or "").strip().lower()
    model_name = str(model or entry.get("name") or "").strip()
    model_family = model_name.removeprefix("@cf/")

    supports_reasoning = _as_bool(props.get("reasoning"))
    supports_tools = _as_bool(props.get("function_calling"))
    supports_vision = _as_bool(props.get("vision"))
    if not supports_vision and task_name == "text generation":
        supports_vision = _as_bool(props.get("vision"))

    context_window = _as_positive_int(props.get("context_window")) or 200000
    max_output_tokens = _as_positive_int(props.get("max_output_tokens")) or 8192

    return ModelCapabilities(
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        model_family=model_family,
    )


def get_cloudflare_model_capabilities(
    api_key: str,
    base_url: str,
    model: str,
    *,
    timeout: float = 8.0,
):
    """Fetch and return live capability metadata for a Cloudflare model."""
    entries = fetch_cloudflare_model_catalog(api_key, base_url, timeout=timeout)
    if not entries:
        return None
    for entry in entries:
        if str(entry.get("name") or "").strip() == str(model or "").strip():
            return model_capabilities_from_cloudflare_catalog_entry(entry, model=model)
    return None

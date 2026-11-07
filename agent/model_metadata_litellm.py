"""LiteLLM proxy ``/model/info`` capability probe (sibling of ``model_metadata``).

A LiteLLM proxy serves deployments under operator-chosen aliases (``team-vision``) that no
public catalog knows, but its ``/model/info`` route reports each deployment's capabilities
from LiteLLM's own model map. Any key the proxy accepts can read the models it may call.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

# (server root, key fingerprint) -> ({model_name: [model_info, ...]} or None, monotonic_ts).
# Negative results are memoized too: a custom endpoint that is not LiteLLM answers 404 once
# per TTL instead of once per image turn.
_MODEL_INFO_TTL_SECONDS = 300.0
_MODEL_INFO_CACHE: dict[tuple[str, str], tuple[Optional[dict[str, list[dict]]], float]] = {}


def _deployments_by_name(payload: Any) -> Optional[dict[str, list[dict]]]:
    """``model_name -> [model_info, ...]`` from a LiteLLM ``/model/info`` body, None for any other shape."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    by_name: dict[str, list[dict]] = {}
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("model_name"), str) and isinstance(entry.get("model_info"), dict):
            by_name.setdefault(entry["model_name"], []).append(entry["model_info"])
    return by_name or None


def _model_info(base_url: str, api_key: str) -> Optional[dict[str, list[dict]]]:
    import httpx

    from agent import model_metadata_http
    from agent.model_metadata import _auth_headers, _server_root

    root = _server_root(base_url)
    cache_key = (root, hashlib.sha256((api_key or "").encode()).hexdigest()[:16])
    now = time.monotonic()
    cached = _MODEL_INFO_CACHE.get(cache_key)
    if cached is not None and now - cached[1] < _MODEL_INFO_TTL_SECONDS:
        return cached[0]
    result = None
    try:
        with httpx.Client(timeout=3.0, headers=_auth_headers(api_key), verify=model_metadata_http.resolve_verify(root)) as client:
            resp = client.get(f"{root}/model/info")
        if resp.status_code == 200:
            result = _deployments_by_name(resp.json())
    except (httpx.HTTPError, ValueError):
        result = None
    _MODEL_INFO_CACHE[cache_key] = (result, now)
    return result


def query_litellm_supports_vision(model: str, base_url: str, api_key: str = "") -> Optional[bool]:
    """True/False when the LiteLLM deployments behind ``model`` that report ``supports_vision`` agree;
    None when the endpoint is not LiteLLM, the alias is unknown or unreported, or deployments disagree."""
    if not model or not base_url:
        return None
    deployments = (_model_info(base_url, api_key) or {}).get(model, [])
    verdicts = {info.get("supports_vision") for info in deployments} - {None}
    if len(verdicts) != 1:
        return None
    verdict = verdicts.pop()
    return verdict if isinstance(verdict, bool) else None

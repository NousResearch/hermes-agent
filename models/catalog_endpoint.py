"""Read-only public endpoint catalogue query, independent of CLI picker internals.

Only caller-configured URLs are probed. Credentials are never persisted, redirects
are rejected (including cross-origin redirects that could leak Authorization).
"""
from __future__ import annotations

import hashlib
import json
import ssl
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Mapping
from urllib.parse import urlsplit

from models.catalog_local import classify_ollama_catalog, ollama_native_root


@dataclass(frozen=True, slots=True)
class EndpointModels:
    ids: tuple[str, ...]
    native_ollama: bool = False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_json(
    url: str, headers: dict[str, str], timeout: float,
    tls_verify: ssl.SSLContext | bool | None = None,
):
    """No redirects or credential forwarding; preserve caller-owned TLS policy."""
    request = urllib.request.Request(url, headers=headers)
    handlers = [_NoRedirect()]
    if tls_verify is False:
        handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    elif isinstance(tls_verify, ssl.SSLContext):
        handlers.append(urllib.request.HTTPSHandler(context=tls_verify))
    try:
        with urllib.request.build_opener(*handlers).open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _ids(rows: object, *, native: bool) -> tuple[str, ...] | None:
    if not isinstance(rows, list):
        return None
    found: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("name") if native else row.get("id")
        value = str(value or "").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            found.append(value)
    return tuple(found)


def _openai_models_urls(base_url: str) -> tuple[str, ...]:
    url = base_url.rstrip("/")
    if url.endswith("/models"):
        return (url,)
    alternate = url[:-3].rstrip("/") if url.endswith("/v1") else url + "/v1"
    return tuple(dict.fromkeys((url + "/models", alternate + "/models")))


def _discover_uncached(
    *, base_url: str, provider: str, api_key: str = "",
    headers: Mapping[str, str] | None = None, timeout: float = 1.5,
    api_mode: str = "", preserve_native_models: bool = False,
    tls_verify: ssl.SSLContext | bool | None = None,
) -> EndpointModels | None:
    """Probe native Ollama or OpenAI-style /models; None means discovery failed.

    Empty native response is authoritative. Empty/failing generic response is not.
    A caller's explicit model list wins when native discovery is suppressed.
    """
    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or timeout <= 0:
            return None
    except ValueError:
        return None
    request_headers = {"Accept": "application/json"}
    supplied = {str(k): str(v) for k, v in (headers or {}).items()}
    if api_key and not any(k.lower() == "authorization" for k in supplied):
        request_headers["Authorization"] = f"Bearer {api_key}"
    request_headers.update(supplied)

    def read(url: str, outgoing: dict[str, str], seconds: float):
        return (
            _read_json(url, outgoing, seconds, tls_verify)
            if tls_verify is not None and tls_verify is not True
            else _read_json(url, outgoing, seconds)
        )

    decision = classify_ollama_catalog(provider, base_url)
    if decision in {"native", "probe"}:
        if decision == "native" and preserve_native_models:
            return None
        root = ollama_native_root(base_url)
        payload = read(
            root + "/api/tags", request_headers,
            min(timeout, 0.5) if decision == "probe" else timeout,
        )
        native_ids = _ids(payload.get("models"), native=True) if isinstance(payload, dict) else None
        if native_ids is not None:
            return EndpointModels(native_ids, native_ollama=True)
    generic_headers = dict(request_headers)
    if api_key and str(api_mode).strip().lower() == "anthropic_messages":
        if not any(k.lower() == "authorization" for k in supplied):
            generic_headers.pop("Authorization", None)
        generic_headers.setdefault("x-api-key", api_key)
        generic_headers.setdefault("anthropic-version", "2023-06-01")
    for url in _openai_models_urls(base_url):
        payload = read(url, generic_headers, timeout)
        generic_ids = _ids(payload.get("data"), native=False) if isinstance(payload, dict) else None
        if generic_ids:
            return EndpointModels(generic_ids)
    return None


_CACHE_MAX = 96
_cache_lock = Lock()
_cache: OrderedDict[str, tuple[float, EndpointModels | None]] = OrderedDict()


def reset_endpoint_catalog_cache() -> None:
    """Clear process-local catalogue observations (tests and explicit refresh)."""
    with _cache_lock:
        _cache.clear()


def _cache_key(
    base_url: str, provider: str, api_key: str, headers: Mapping[str, str] | None,
    api_mode: str, preserve_native_models: bool,
    tls_verify: ssl.SSLContext | bool | None,
) -> str:
    # Hash rather than retain token/header values in process-global cache keys.
    material = json.dumps(
        [base_url, provider, api_key, sorted(
            (str(k).lower(), str(v)) for k, v in (headers or {}).items()
        ), api_mode, preserve_native_models,
         id(tls_verify) if isinstance(tls_verify, ssl.SSLContext) else tls_verify],
        separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def discover_endpoint_models(
    *, base_url: str, provider: str, api_key: str = "",
    headers: Mapping[str, str] | None = None, timeout: float = 1.5,
    api_mode: str = "", preserve_native_models: bool = False,
    tls_verify: ssl.SSLContext | bool | None = None,
) -> EndpointModels | None:
    """Bounded read-only cache: native 5m, generic 15m, failures 15s."""
    key = _cache_key(base_url, provider, api_key, headers, api_mode, preserve_native_models, tls_verify)
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and now < cached[0]:
            _cache.move_to_end(key)
            return cached[1]
    result = _discover_uncached(
        base_url=base_url, provider=provider, api_key=api_key,
        headers=headers, timeout=timeout, api_mode=api_mode,
        preserve_native_models=preserve_native_models, tls_verify=tls_verify,
    )
    ttl = 300 if result and result.native_ollama else 900 if result else 15
    with _cache_lock:
        _cache[key] = (time.monotonic() + ttl, result)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return result


__all__ = ["EndpointModels", "discover_endpoint_models", "reset_endpoint_catalog_cache"]

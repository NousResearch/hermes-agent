"""Local / self-hosted model servers.

Ollama (native ``/api/tags`` probe, request headers, base-url resolution), LM Studio
(``/api/v1/models``, load-on-demand) and Ollama Cloud (live + models.dev merged catalog with a
disk cache). Split out of ``hermes_cli.models``; helpers still defined there are looked up on
``hermes_cli.models`` at call time so ``patch("hermes_cli.models.<name>")`` mocks keep intercepting.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple, Optional
from hermes_cli.urllib_security import url_origin

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.models")

_OLLAMA_DEFAULT_PORT = 11434


def _strip_suffixes(root: str, suffixes: tuple[str, ...]) -> str:
    """Drop the first matching path suffix (and any trailing slash) from *root*."""
    for suffix in suffixes:
        if root.endswith(suffix):
            return root[: -len(suffix)].rstrip("/")
    return root


def _normalize_openai_base_url(base_url: Optional[str]) -> str:
    """Add a usable HTTP scheme without changing an OpenAI API path."""
    value = str(base_url or "").strip()
    if value.startswith(":"):
        return "http://127.0.0.1" + value
    if value and "://" not in value:
        return "http://" + value
    return value


def _root_for_ollama_native_api(base_url: str) -> str:
    """Convert an OpenAI-style Ollama base URL to the native API root."""
    root = str(base_url or "").strip().rstrip("/")
    if root.startswith(":"):
        root = "http://127.0.0.1" + root
    elif root and "://" not in root:
        root = "http://" + root
    return _strip_suffixes(root, ("/api/tags", "/v1/models", "/api", "/v1"))


def _configured_ollama_base_url() -> str:
    """``providers.ollama.base_url`` (legacy keys ``api`` / ``url``), or ``""``."""
    from hermes_cli.models import _get_provider_config_dict

    cfg = _get_provider_config_dict("ollama")
    return str(cfg.get("base_url") or cfg.get("api") or cfg.get("url") or "").strip()


def _ollama_host_from_env(env_host: str) -> str:
    """Apply Ollama's own ``OLLAMA_HOST`` defaulting rules (port 11434, IPv6 bracketing)."""
    port = _OLLAMA_DEFAULT_PORT
    if env_host.startswith(":") and not env_host.startswith("::"):
        return "127.0.0.1" + env_host
    if env_host.startswith("[") and env_host.endswith("]"):
        return f"{env_host}:{port}"
    if "://" in env_host:
        try:
            parsed = urllib.parse.urlsplit(env_host)
            if parsed.hostname and parsed.port is None:
                hostname = parsed.hostname
                if ":" in hostname and not hostname.startswith("["):
                    hostname = f"[{hostname}]"
                userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
                return parsed._replace(netloc=f"{userinfo}{hostname}:{port}").geturl()
        except ValueError:
            pass
        return env_host
    if env_host.count(":") > 1 and not env_host.startswith("["):
        return f"[{env_host}]:{port}"
    if ":" not in env_host:
        return f"{env_host}:{port}"
    return env_host


def _get_ollama_base_url() -> str:
    """Resolve the local Ollama-compatible endpoint URL: explicit ``providers.ollama.base_url``
    (wires local endpoints without changing the active provider) → active ``model.base_url`` when
    the active provider is ollama, or custom AND the endpoint actually serves ``/api/tags``
    (otherwise the picker would probe an unrelated endpoint and hide the local catalog) →
    ``OLLAMA_HOST`` → Ollama's local default."""
    from hermes_cli.models import _get_model_config_dict
    configured = _configured_ollama_base_url()
    if configured:
        return configured

    model_cfg = _get_model_config_dict()
    model_provider = str(model_cfg.get("provider", "") or "").strip().lower()
    model_base = str(model_cfg.get("base_url", "") or "").strip()
    if model_provider == "ollama" and model_base:
        return model_base
    if model_provider == "custom" and model_base:
        try:
            if should_use_ollama_native_catalog("custom", model_base):
                return model_base
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    env_host = os.getenv("OLLAMA_HOST", "").strip()
    return _ollama_host_from_env(env_host) if env_host else "http://localhost:11434"


def _api_key_from_provider_config(entry: dict, *env_keys: str) -> str:
    """``api_key`` from a provider config block, else the env var named by the first set *env_keys*."""
    api_key = str(entry.get("api_key") or "").strip()
    if api_key:
        return api_key
    key_env = str(next((entry.get(k) for k in env_keys if entry.get(k)), "") or "").strip()
    return os.getenv(key_env, "").strip() if key_env else ""


def _drop_authorization(headers: dict[str, str]) -> None:
    for key in tuple(headers):
        if key.lower() == "authorization":
            del headers[key]


def _get_ollama_request_headers() -> dict[str, str]:
    """Return configured headers and credentials for native Ollama requests."""
    from hermes_cli.models import _get_provider_config_dict
    entry = _get_provider_config_dict("ollama")
    try:
        from hermes_cli.config import normalize_extra_headers

        result = normalize_extra_headers(entry.get("extra_headers"))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        result = {}

    api_key = _api_key_from_provider_config(entry, "key_env", "api_key_env")
    if api_key and not any(key.lower() == "authorization" for key in result):
        result["Authorization"] = f"Bearer {api_key}"
    return result


def _get_ollama_native_headers(base_url: Optional[str], *, api_key: Optional[str] = None) -> dict[str, str]:
    """Ollama credentials and headers for one endpoint origin. Configured headers apply only when
    *base_url* shares the configured Ollama root; an explicit *api_key* replaces any configured
    Authorization variant rather than inheriting it."""
    configured_base = _configured_ollama_base_url()
    explicit_key = str(api_key or "").strip()
    configured_matches = bool(configured_base and base_url and _same_ollama_native_root(base_url, configured_base))
    if not configured_matches and not explicit_key:
        return {}
    headers = _get_ollama_request_headers() if configured_matches else {}
    if explicit_key:
        _drop_authorization(headers)
        headers["Authorization"] = f"Bearer {explicit_key}"
    return headers


# Native /api/tags probe caches, keyed by root (+ header fingerprint): successful catalogs,
# failure timestamps (short negative TTL), and whether the root answered the native probe.
_OLLAMA_LOCAL_MODELS_CACHE_TTL: int = 300  # seconds
_OLLAMA_LOCAL_MODELS_CACHE: dict[str, tuple[tuple[str, ...], float]] = {}
_OLLAMA_LOCAL_PROBE_FAILURE_CACHE: dict[str, float] = {}
_OLLAMA_LOCAL_PROBE_REACHABLE: dict[str, bool] = {}
_OLLAMA_LOCAL_PROBE_FAILURE_TTL: int = 30
_OLLAMA_LOCAL_CACHE_MAX_ENTRIES: int = 256


def _evict_related_ollama_cache_entries(key: str) -> None:
    _OLLAMA_LOCAL_MODELS_CACHE.pop(key, None)
    _OLLAMA_LOCAL_PROBE_REACHABLE.pop(key, None)
    for failure_key in list(_OLLAMA_LOCAL_PROBE_FAILURE_CACHE):
        if failure_key == key or failure_key.startswith(f"{key}|timeout:"):
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)


def _remember_ollama_cache(cache: dict[str, Any], key: str, value: Any) -> None:
    if key not in cache and len(cache) >= _OLLAMA_LOCAL_CACHE_MAX_ENTRIES:
        _evict_related_ollama_cache_entries(next(iter(cache)).split("|timeout:", 1)[0])
    cache[key] = value


def _ollama_probe_cache_key(root: str, headers: Optional[dict[str, str]]) -> str:
    if not headers:
        return root
    normalized_headers = sorted((str(key).lower(), str(value)) for key, value in headers.items())
    header_blob = json.dumps(normalized_headers, ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="replace")
    return f"{root}|headers:{hashlib.blake2b(header_blob, digest_size=8).hexdigest()}"


def _parse_ollama_tags(payload: Any) -> Optional[list[str]]:
    """Model ids from an ``/api/tags`` payload; None when the shape is not Ollama's."""
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return None
    models: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            return None
        model_id = str(item.get("model") or item.get("name") or "").strip()
        if model_id and model_id not in models:
            models.append(model_id)
    if raw_models and not models:
        return None
    return models


def probe_ollama_local_models(
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Probe local Ollama-compatible models from native ``/api/tags`` (Ollama's authoritative local
    catalog; ``/v1/models`` is not required for local servers). ``None`` when the endpoint cannot be
    reached or returns malformed data; a list (possibly empty) when it was reachable."""
    from hermes_cli.models import _HERMES_USER_AGENT, _get_ollama_base_url, _urlopen_model_catalog_request
    root = _root_for_ollama_native_api(base_url or _get_ollama_base_url())
    if not root:
        return None
    cache_key = _ollama_probe_cache_key(root, headers)
    failure_key = f"{cache_key}|timeout:{float(timeout):.3f}"
    cached = _OLLAMA_LOCAL_MODELS_CACHE.get(cache_key)
    if cached is not None and time.monotonic() - cached[1] < _OLLAMA_LOCAL_MODELS_CACHE_TTL:
        return list(cached[0])
    failed_at = _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.get(failure_key)
    if failed_at is not None:
        if time.monotonic() - failed_at < _OLLAMA_LOCAL_PROBE_FAILURE_TTL:
            return None
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)
    try:
        request_headers = {"User-Agent": _HERMES_USER_AGENT, **(headers or {})}
        req = urllib.request.Request(root.rstrip("/") + "/api/tags", headers=request_headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            models = _parse_ollama_tags(json.loads(resp.read().decode()))
    except (ValueError, OSError, TimeoutError, http.client.HTTPException, urllib.error.URLError,
            json.JSONDecodeError, UnicodeDecodeError):
        models = None
    if models is None:
        _remember_ollama_cache(_OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False)
        _remember_ollama_cache(_OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic())
        return None
    _remember_ollama_cache(_OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, True)
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)
    _remember_ollama_cache(_OLLAMA_LOCAL_MODELS_CACHE, cache_key, (tuple(models), time.monotonic()))
    return models


def fetch_ollama_local_models(
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch local Ollama-compatible models, preserving probe failure as ``None``."""
    return probe_ollama_local_models(base_url, timeout, headers=headers)


def _same_ollama_native_root(left: str, right: str) -> bool:
    """Return True when two Ollama/OpenAI-style base URLs share an API root."""
    left_root = _root_for_ollama_native_api(left).rstrip("/")
    right_root = _root_for_ollama_native_api(right).rstrip("/")
    if not left_root or not right_root:
        return False
    try:
        left_parts = urllib.parse.urlsplit(left_root)
        right_parts = urllib.parse.urlsplit(right_root)
        return (
            url_origin(left_root) == url_origin(right_root)
            and left_parts.path.rstrip("/") == right_parts.path.rstrip("/")
        )
    except (AttributeError, ValueError):
        return False


_NEVER_OLLAMA_PROVIDERS = frozenset({"openrouter", "nous", "anthropic", "openai", "openai-codex", "gemini", "ollama-cloud"})
_LOCAL_LIKE_PROVIDERS = frozenset({"", "custom", "local", "llamacpp", "llama.cpp", "llama-cpp", "vllm"})


def should_use_ollama_native_catalog(
    provider: Optional[str],
    base_url: Optional[str],
    headers: Optional[dict[str, str]] = None,
) -> bool:
    """True when model discovery should use local Ollama ``/api/tags``: the caller asked for Ollama
    explicitly, the base URL matches ``providers.ollama.base_url``, or an ambiguous custom URL on
    Ollama's default port actually serves ``/api/tags``. (Bare ``ollama`` is normalized to
    ``custom`` elsewhere so runtime paths share the OpenAI client, but ``/api/tags`` is the
    authoritative local list; other custom endpoints keep the ``/models`` probe.)"""
    requested = str(provider or "").strip().lower()
    root = _root_for_ollama_native_api(base_url or "")
    if root:
        try:
            host = (urllib.parse.urlparse(root).hostname or "").lower()
            if host == "ollama.com" or host.endswith(".ollama.com"):
                return False
        except ValueError:
            pass

    if requested in _NEVER_OLLAMA_PROVIDERS:
        return False

    configured_base = _configured_ollama_base_url()
    if requested == "ollama":
        if not root:
            return False
        if configured_base and not _same_ollama_native_root(root, configured_base):
            return probe_ollama_local_models(root, timeout=0.5, headers=headers) is not None
        return True

    if configured_base and _same_ollama_native_root(root, configured_base):
        return True

    if not root:
        return False

    if requested not in _LOCAL_LIKE_PROVIDERS and not requested.startswith("custom:"):
        return False

    if requested == "custom:ollama" or requested.endswith("-ollama"):
        return True

    try:
        if urllib.parse.urlparse(root).port != _OLLAMA_DEFAULT_PORT:
            return False
    except ValueError:
        return False

    return probe_ollama_local_models(root, timeout=0.5, headers=headers) is not None


def _ollama_local_catalog(force_refresh: bool) -> list[str]:
    """Catalog for the raw ``ollama`` provider: native ``/api/tags`` when the endpoint is a real
    Ollama server, else the OpenAI-style ``/v1/models`` of the configured gateway (incl. Ollama
    Cloud)."""
    from hermes_cli.models import _get_provider_config_dict, fetch_api_models
    if force_refresh:
        _OLLAMA_LOCAL_MODELS_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
    base_url = _get_ollama_base_url()
    headers = _get_ollama_native_headers(base_url)
    if should_use_ollama_native_catalog("ollama", base_url, headers=headers):
        native_models = fetch_ollama_local_models(base_url, headers=headers) if headers else fetch_ollama_local_models(base_url)
        native_key = _ollama_probe_cache_key(_root_for_ollama_native_api(base_url), headers or None)
        if native_models or _OLLAMA_LOCAL_PROBE_REACHABLE.get(native_key) is True:
            return native_models or []
    config = _get_provider_config_dict("ollama")
    fallback_key = _api_key_from_provider_config(config, "key_env")
    fallback_base = _normalize_openai_base_url(config.get("base_url") or base_url)
    fallback_headers = _get_ollama_native_headers(fallback_base, api_key=fallback_key)
    return fetch_api_models(fallback_key, fallback_base, headers=fallback_headers or None) or []


def _lmstudio_server_root(base_url: Optional[str]) -> Optional[str]:
    """LM Studio server root: users paste the OpenAI runtime URL (``.../v1``) or the native prefix
    (``.../api``, ``.../api/v1``); native probes append ``/api/v1/...`` themselves."""
    return _strip_suffixes((base_url or "").strip().rstrip("/"), ("/api/v1", "/api", "/v1")) or None


def _lmstudio_request_headers(api_key: Optional[str] = None) -> dict:
    """HTTP headers for LM Studio native API requests."""
    from hermes_cli.models import _HERMES_USER_AGENT
    token = str(api_key or "").strip()
    return {"User-Agent": _HERMES_USER_AGENT, **({"Authorization": f"Bearer {token}"} if token else {})}


def _lmstudio_fetch_raw_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[dict]]:
    """Raw model list from LM Studio's ``/api/v1/models``; None on network errors / malformed
    payloads; raises ``AuthError`` on HTTP 401/403."""
    from hermes_cli.models import _urlopen_model_catalog_request
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return None

    request = urllib.request.Request(server_root + "/api/v1/models", headers=_lmstudio_request_headers(api_key))
    try:
        with _urlopen_model_catalog_request(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            from hermes_cli.auth import AuthError
            raise AuthError(
                f"LM Studio rejected the request with HTTP {exc.code}.",
                provider="lmstudio",
                code="auth_rejected",
            ) from exc
        logger.debug("LM Studio probe at %s failed with HTTP %s", server_root, exc.code)
        return None
    except Exception as exc:
        logger.debug("LM Studio probe at %s failed: %s", server_root, exc)
        return None

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        logger.debug("LM Studio probe at %s returned malformed payload (no `models` list)", server_root)
        return None
    return raw_models


def _lmstudio_raw_models_or_none(api_key, base_url, timeout) -> Optional[list[dict]]:
    """``_lmstudio_fetch_raw_models`` with every failure (incl. AuthError) collapsed to None."""
    try:
        return _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception:
        return None


def _lmstudio_entry_for(raw_models: list, model: str) -> Optional[dict]:
    for raw in raw_models:
        if isinstance(raw, dict) and (raw.get("key") == model or raw.get("id") == model):
            return raw
    return None


def probe_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[str]]:
    """Chat-capable LM Studio model keys — a valid empty list when the server is reachable but has
    no non-embedding models; ``None`` on network errors, malformed responses, or bad base URLs.
    Raises ``AuthError`` on HTTP 401/403 so token issues surface separately from reachability."""
    raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    if raw_models is None:
        return None

    keys: list[str] = []
    for raw in raw_models:
        if not isinstance(raw, dict) or str(raw.get("type") or "").strip().lower() == "embedding":
            continue
        key = str(raw.get("key") or raw.get("id") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def fetch_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """LM Studio chat-capable model keys; ``[]`` when unreachable/malformed. Raises ``AuthError`` on
    HTTP 401/403 so callers can tell a wrong ``LM_API_KEY`` from an unreachable server."""
    return probe_lmstudio_models(api_key=api_key, base_url=base_url, timeout=timeout) or []


class LMStudioLoadResult(NamedTuple):
    """Verified LM Studio runtime plus load-attempt provenance."""

    context_length: Optional[int]
    load_attempted: bool = False
    rejected: bool = False


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _lmstudio_loaded_context(entry: Optional[dict]) -> Optional[int]:
    """First positive ``loaded_instances[*].config.context_length`` of a model entry."""
    instances = entry.get("loaded_instances") if entry is not None else None
    if not isinstance(instances, list):
        return None
    for instance in instances:
        config = instance.get("config") if isinstance(instance, dict) else None
        parsed = _positive_int(config.get("context_length") if isinstance(config, dict) else None)
        if parsed is not None:
            return parsed
    return None


def ensure_lmstudio_model_loaded(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    target_context_length: Optional[int],
    timeout: float = 120.0,
    *,
    return_load_result: bool = False,
) -> Optional[int] | LMStudioLoadResult:
    """Ensure ``model`` is loaded and return verified runtime context.

    Existing loaded-instance context is authoritative. Cold loads omit ``context_length`` unless the
    caller supplied an explicit override; the returned context comes from LM Studio's echoed or
    refreshed state."""
    from hermes_cli.models import _urlopen_model_catalog_request

    def _result(context_length: Optional[int], *, load_attempted: bool = False, rejected: bool = False):
        result = LMStudioLoadResult(context_length, load_attempted, rejected)
        return result if return_load_result else context_length

    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return _result(None)

    explicit_context = _positive_int(target_context_length)
    if target_context_length is not None and explicit_context is None:
        return _result(None)
    target_entry = _lmstudio_entry_for(_lmstudio_raw_models_or_none(api_key, base_url, 10) or [], model)
    if target_entry is None:
        return _result(None)

    max_ctx = _positive_int(target_entry.get("max_context_length"))
    if explicit_context is not None and max_ctx is not None and explicit_context > max_ctx:
        return _result(None, rejected=True)

    current_context = _lmstudio_loaded_context(target_entry)
    if current_context is not None:
        return _result(current_context)

    loaded_instances = target_entry.get("loaded_instances")
    if not isinstance(loaded_instances, list) or loaded_instances:
        return _result(None)

    load_payload: dict[str, Any] = {"model": model, "echo_load_config": True}
    if explicit_context is not None:
        load_payload["context_length"] = explicit_context
    try:
        load_request = urllib.request.Request(
            server_root + "/api/v1/models/load",
            data=json.dumps(load_payload).encode(),
            headers={**_lmstudio_request_headers(api_key), "Content-Type": "application/json"},
            method="POST",
        )
        with _urlopen_model_catalog_request(load_request, timeout=timeout) as resp:
            response_body = resp.read()
    except Exception:
        return _result(None, load_attempted=True)

    try:
        response_payload = json.loads(response_body.decode())
    except Exception:
        response_payload = None
    load_config = response_payload.get("load_config") if isinstance(response_payload, dict) else None
    applied_context = _positive_int(load_config.get("context_length")) if isinstance(load_config, dict) else None
    if applied_context is not None:
        return _result(applied_context, load_attempted=True)

    refreshed_models = _lmstudio_raw_models_or_none(api_key, base_url, 10)
    if refreshed_models is None:
        return _result(None, load_attempted=True)
    return _result(_lmstudio_loaded_context(_lmstudio_entry_for(refreshed_models, model)), load_attempted=True)


def lmstudio_model_reasoning_options(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Reasoning ``allowed_options`` LM Studio publishes for ``model`` under
    ``capabilities.reasoning`` in ``/api/v1/models``; ``[]`` when unknown, unreachable, or absent."""
    raw = _lmstudio_entry_for(_lmstudio_raw_models_or_none(api_key, base_url, timeout) or [], model)
    if raw is None:
        return []
    caps = raw.get("capabilities")
    reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
    opts = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
    if isinstance(opts, list):
        return [str(o).strip().lower() for o in opts if isinstance(o, str)]
    return []


def ollama_model_supports_thinking(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[bool]:
    """Tri-state: True if an Ollama (Cloud or local) model advertises ``thinking`` in native
    ``/api/show`` ``capabilities`` (authoritative; OpenAI-compat ``/v1/models`` omits it), False
    when the probe succeeded without it, None when it failed (caller treats as "don't emit")."""
    import httpx

    server_url = (base_url or "").strip().rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    bare_model = _strip_ollama_cloud_suffix((model or "").strip())
    if not server_url or not bare_model:
        return None

    token = str(api_key or "").strip()
    try:
        with httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {token}"} if token else {}) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            if resp.status_code != 200:
                return None
            caps = resp.json().get("capabilities")
            if isinstance(caps, list):
                return "thinking" in caps
    except Exception:
        return None
    return None


_OLLAMA_CLOUD_CACHE_TTL = 3600  # 1 hour


def _strip_ollama_cloud_suffix(model_id: str) -> str:
    """Strip the ``:cloud`` / ``-cloud`` suffix models.dev appends to Ollama Cloud IDs (the live
    API uses bare ids), so the dedup merge does not produce duplicates."""
    for suffix in (":cloud", "-cloud"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _ollama_cloud_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "ollama_cloud_models_cache.json"


def _load_ollama_cloud_cache(*, ignore_ttl: bool = False) -> Optional[dict]:
    """Load cached Ollama Cloud models from disk (None when missing, empty, or stale)."""
    from hermes_cli.models import _read_json_cache

    try:
        data = _read_json_cache(_ollama_cloud_cache_path())
        models = data.get("models") if data is not None else None
        if not (isinstance(models, list) and models):
            return None
        if not ignore_ttl and (time.time() - data.get("cached_at", 0)) > _OLLAMA_CLOUD_CACHE_TTL:
            return None  # stale
        return data
    except Exception:
        return None


def _save_ollama_cloud_cache(models: list[str]) -> None:
    """Persist the merged Ollama Cloud model list to disk. Best-effort."""
    from hermes_cli.models import _write_json_cache

    try:
        _write_json_cache(_ollama_cloud_cache_path(), {"models": models, "cached_at": time.time()}, indent=None)
    except Exception:
        pass


def fetch_ollama_cloud_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Ollama Cloud models: fresh disk cache (< 1h, unless force_refresh) → live ``/v1/models``
    (freshest) merged with models.dev additions (deduped, live first) → stale cache → ``[]``.
    Never None."""
    from hermes_cli.models import fetch_api_models
    if not force_refresh:
        cached = _load_ollama_cloud_cache()
        if cached is not None:
            return cached["models"]

    api_key = api_key or os.getenv("OLLAMA_API_KEY", "")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "") or "https://ollama.com/v1"
    live_models = (fetch_api_models(api_key, base_url, timeout=8.0) or []) if api_key else []
    mdev_models: list[str] = []
    try:
        from agent.models_dev import list_agentic_models
        mdev_models = list_agentic_models("ollama-cloud")
    except Exception:
        pass

    merged: list[str] = []
    for m in [*live_models, *(_strip_ollama_cloud_suffix(m) for m in mdev_models)]:
        if m and m not in merged:
            merged.append(m)
    if merged:
        _save_ollama_cloud_cache(merged)
        return merged

    stale = _load_ollama_cloud_cache(ignore_ttl=True)
    return stale["models"] if stale is not None else []

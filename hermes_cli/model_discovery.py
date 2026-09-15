"""Stable, noninteractive provider/model discovery for external integrations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1"
_CACHE_TTL_SECONDS = 3600


@dataclass(frozen=True)
class ModelDiscoveryRequest:
    provider: str | None = None
    refresh: bool = False
    offline: bool = False


def _registered_provider_ids() -> list[str]:
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

    return list(dict.fromkeys(entry.slug for entry in CANONICAL_PROVIDERS))


def _catalog_model_ids(provider: str) -> list[str]:
    """Bundled, network-free catalog for one registered provider."""
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS
    from providers import get_provider_profile

    models = list(_PROVIDER_MODELS.get(provider, ()))
    profile = get_provider_profile(provider)
    if not models and profile is not None:
        models = list(profile.fallback_models or ())
    return list(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))


def _provider_cache_entry(provider: str) -> dict[str, Any] | None:
    """Read one credential-bound picker cache entry without refreshing it."""
    try:
        from hermes_cli.models import _credential_fingerprint, _load_provider_models_cache

        entry = _load_provider_models_cache().get(provider)
        if not isinstance(entry, dict) or entry.get("fp") != _credential_fingerprint(provider):
            return None
        if not isinstance(entry.get("models"), list) or not entry["models"]:
            return None
        at = entry.get("at")
        if not isinstance(at, (int, float)) or isinstance(at, bool):
            return None
        return {"at": float(at), "fp": entry["fp"], "models": list(entry["models"])}
    except Exception:
        return None


def _refresh_provider_models(provider: str) -> list[str] | None:
    """Run only the registered provider's live catalog hook and cache normalized IDs."""
    from hermes_cli.models import _api_key_credentials, update_provider_cache_entry
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    if profile is None or profile.auth_type != "api_key" or not profile.base_url:
        return None
    api_key, base_url = _api_key_credentials(provider)
    models = profile.fetch_models(api_key=api_key, base_url=base_url or profile.base_url or None)
    normalized = list(dict.fromkeys(str(model).strip() for model in (models or ()) if str(model).strip()))
    if not normalized:
        return None
    update_provider_cache_entry(provider, normalized)
    return normalized


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _model_payload(provider: str, model_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": model_id,
        "capabilities": ["chat"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "deprecated": False,
    }
    try:
        from agent.models_dev import get_model_info

        info = get_model_info(provider, model_id, allow_network=False)
    except Exception:
        info = None
    if info is None:
        return row

    capabilities = ["chat"]
    if info.tool_call:
        capabilities.append("tools")
    if info.reasoning:
        capabilities.append("reasoning")
    if info.supports_vision():
        capabilities.append("vision")
    row.update({
        "capabilities": capabilities,
        "input_modalities": list(info.input_modalities) or ["text"],
        "output_modalities": list(info.output_modalities) or ["text"],
        "deprecated": info.status == "deprecated",
    })
    if info.name:
        row["display_name"] = info.name
    if info.context_window > 0:
        row["context_window"] = info.context_window
    if info.max_output > 0:
        row["max_output_tokens"] = info.max_output
    return row


def _provider_payload(
    provider: str,
    source: str,
    models: list[str],
    *,
    cache_at: float | None = None,
    stale: bool = False,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": provider,
        "source": source,
        "models": [_model_payload(provider, model_id) for model_id in models],
        "warnings": warnings or [],
    }
    if source == "cache" and cache_at is not None:
        row["retrieved_at"] = _iso(cache_at)
        row["expires_at"] = _iso(cache_at + _CACHE_TTL_SECONDS)
        row["stale"] = stale
    elif source == "live":
        now = time.time()
        row.update({"retrieved_at": _iso(now), "expires_at": _iso(now + _CACHE_TTL_SECONDS), "stale": False})
    return row


def _base_response(request: ModelDiscoveryRequest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request": {
            "provider": request.provider,
            "refresh": request.refresh,
            "offline": request.offline,
        },
        "providers": [],
        "errors": [],
    }


def discover_models(request: ModelDiscoveryRequest) -> tuple[dict[str, Any], int]:
    """Return a stable response and process exit code; network is opt-in and provider-scoped."""
    response = _base_response(request)
    provider_ids = _registered_provider_ids()
    selected = [request.provider] if request.provider else provider_ids

    for provider in selected:
        cache = _provider_cache_entry(provider)
        cache_age = time.time() - cache["at"] if cache else None
        cache_fresh = cache_age is not None and cache_age < _CACHE_TTL_SECONDS
        warnings: list[dict[str, str]] = []

        if request.refresh:
            try:
                live = _refresh_provider_models(provider)
            except Exception:
                live = None
                response["errors"].append({
                    "code": "refresh_failed",
                    "provider": provider,
                    "message": "Live model discovery failed; a fallback source was used.",
                })
            if live:
                response["providers"].append(_provider_payload(provider, "live", live))
                continue
            if not response["errors"] or response["errors"][-1].get("provider") != provider:
                response["errors"].append({
                    "code": "refresh_unavailable",
                    "provider": provider,
                    "message": "Live model discovery produced no usable result; a fallback source was used.",
                })

        if cache and (cache_fresh or request.offline or request.refresh):
            stale = not cache_fresh
            if stale:
                warnings.append({"code": "stale_cache", "message": "Using an expired cached catalog."})
            response["providers"].append(
                _provider_payload(provider, "cache", cache["models"], cache_at=cache["at"], stale=stale, warnings=warnings)
            )
            continue

        catalog = _catalog_model_ids(provider)
        if catalog:
            response["providers"].append(_provider_payload(provider, "catalog", catalog))

    if request.provider and not response["providers"]:
        response["errors"].append({
            "code": "no_usable_result",
            "provider": request.provider,
            "message": "The requested provider has no usable model catalog under this policy.",
        })
        return response, 3
    return response, 0


def _argument_error(args: Any, code: str, message: str) -> int:
    request = ModelDiscoveryRequest(
        provider=getattr(args, "provider", None),
        refresh=bool(getattr(args, "refresh", False)),
        offline=bool(getattr(args, "offline", False)),
    )
    response = _base_response(request)
    response["errors"].append({"code": code, "message": message})
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 2


def models_command(args: Any) -> int:
    """CLI boundary: always write exactly one JSON document to stdout."""
    provider = str(getattr(args, "provider", "") or "").strip().lower() or None
    refresh = bool(getattr(args, "refresh", False))
    offline = bool(getattr(args, "offline", False))
    if refresh and offline:
        return _argument_error(args, "conflicting_options", "--refresh and --offline are mutually exclusive.")
    if refresh and provider is None:
        return _argument_error(args, "provider_required", "--refresh requires --provider ID.")
    if provider is not None and provider not in _registered_provider_ids():
        return _argument_error(args, "unsupported_provider", "The provider identifier is not registered.")

    request = ModelDiscoveryRequest(provider=provider, refresh=refresh, offline=offline)
    try:
        response, exit_code = discover_models(request)
    except Exception:
        response = _base_response(request)
        response["errors"].append({
            "code": "internal_error",
            "message": "Model discovery failed unexpectedly.",
        })
        exit_code = 4
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return exit_code

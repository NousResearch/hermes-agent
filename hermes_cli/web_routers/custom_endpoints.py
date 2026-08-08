"""Custom-endpoint management routes for the dashboard.

Extracted from hermes_cli/web_server.py (god-file slice R3-C1, epic #78791):
the custom OpenAI-compatible provider endpoint CRUD + validate family.  The
single cross-cluster helper (_apply_main_model_assignment) resolves through
the web_deps.late() seam; everything else is self-contained.
"""

import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.config import (
    custom_endpoint_key_env,
    load_config,
    read_raw_config,
    redact_key,
    remove_env_value,
    save_config,
    save_env_value,
)
from hermes_cli.web_models import CustomEndpointUpdate

from hermes_cli.web_deps import late

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

_apply_main_model_assignment = late("_apply_main_model_assignment")

def _parse_model_ids(resp: "Any") -> List[str]:
    """Extract model ids from an OpenAI-compatible ``/v1/models`` response.

    Tolerant of the common shapes: ``{"data": [{"id": ...}]}`` (OpenAI / vLLM /
    llama.cpp) and a bare ``{"data": ["id", ...]}``. Returns ``[]`` on any
    parse/HTTP error so a slightly non-standard endpoint never hard-blocks.
    """
    try:
        if not resp.is_success:
            return []
        payload = resp.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = str(item.get("id") or "").strip()
        else:
            mid = str(item or "").strip()
        if mid:
            ids.append(mid)
    return ids


def _custom_endpoint_id(raw: str, fallback: str = "custom") -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (raw or "").strip()).strip("-_").lower()
    return slug or fallback


def _models_from_custom_endpoint_entry(entry: Dict[str, Any]) -> List[str]:
    models: List[str] = []
    raw_models = entry.get("models")
    if isinstance(raw_models, dict):
        models.extend(str(model).strip() for model in raw_models.keys())
    elif isinstance(raw_models, list):
        models.extend(str(model).strip() for model in raw_models)

    default_model = str(entry.get("model") or entry.get("default_model") or "").strip()
    if default_model:
        models.insert(0, default_model)

    seen: set[str] = set()
    return [model for model in models if model and not (model in seen or seen.add(model))]


def _api_key_display(entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return ``(has_api_key, preview)`` for a provider or model config block.

    Keys live in ``.env`` behind ``key_env``; only entries written before
    #69449 still carry a plaintext ``api_key``. Checking both keeps the panel
    honest either way — reading only ``api_key`` reported "no API key" for
    every endpoint whose key had been moved to ``.env``.
    """
    plaintext = str(entry.get("api_key") or "").strip()
    if plaintext:
        return True, redact_key(plaintext)
    key_env = str(entry.get("key_env") or "").strip()
    if key_env:
        return True, f"${{{key_env}}}"
    return False, None


def _config_api_key_is_env_ref(endpoint_id: str) -> bool:
    """True when this endpoint's on-disk ``api_key`` is a ``${VAR}`` template.

    ``load_config()`` expands env refs, so a hand-written
    ``api_key: ${MY_KEY}`` is indistinguishable from a literal secret by the
    time it reaches us. Such an entry is already keeping its secret out of
    config.yaml, so migrating it would only copy that secret into a second
    env var the user didn't ask for.
    """
    providers = read_raw_config().get("providers")
    entry = providers.get(endpoint_id) if isinstance(providers, dict) else None
    raw_key = entry.get("api_key") if isinstance(entry, dict) else None
    return bool(isinstance(raw_key, str) and re.search(r"\$\{[^}]+\}", raw_key))


def _custom_endpoint_response(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    current_provider = str(model_cfg.get("provider", "") or "")
    current_model = str(model_cfg.get("default", model_cfg.get("name", "")) or "")
    current_base_url = str(model_cfg.get("base_url", "") or "")

    endpoints: List[Dict[str, Any]] = []
    providers = cfg.get("providers")
    if isinstance(providers, dict):
        for provider_id, raw_entry in providers.items():
            if not isinstance(raw_entry, dict):
                continue
            base_url = str(raw_entry.get("base_url") or raw_entry.get("url") or raw_entry.get("api") or "").strip()
            if not base_url:
                continue
            endpoint_id = str(provider_id)
            models = _models_from_custom_endpoint_entry(raw_entry)
            endpoint_model = str(raw_entry.get("model") or raw_entry.get("default_model") or (models[0] if models else ""))
            has_api_key, api_key_preview = _api_key_display(raw_entry)
            endpoints.append({
                "id": endpoint_id,
                "name": str(raw_entry.get("name") or endpoint_id),
                "base_url": base_url,
                "model": endpoint_model,
                "models": models,
                "context_length": raw_entry.get("context_length"),
                "discover_models": bool(raw_entry.get("discover_models", True)),
                "has_api_key": has_api_key,
                "api_key_preview": api_key_preview,
                "is_current": endpoint_id == current_provider,
                "source": "providers",
            })

    if current_provider.lower() == "custom" and current_base_url and not any(e["id"] == "custom" for e in endpoints):
        has_api_key, api_key_preview = _api_key_display(model_cfg)
        endpoints.insert(0, {
            "id": "custom",
            "name": "Custom",
            "base_url": current_base_url,
            "model": current_model,
            "models": [current_model] if current_model else [],
            "context_length": model_cfg.get("context_length"),
            "discover_models": True,
            "has_api_key": has_api_key,
            "api_key_preview": api_key_preview,
            "is_current": True,
            "source": "direct-config",
        })

    return {
        "endpoints": endpoints,
        "current": {
            "provider": current_provider,
            "model": current_model,
            "base_url": current_base_url,
        },
    }


def _detach_main_model_from_provider(cfg: Dict[str, Any], provider_key: str) -> None:
    """Drop the main-slot mirror of a provider that no longer exists.

    ``activate_custom_endpoint`` copies the endpoint's ``base_url`` and
    ``api_key`` onto ``model``. That mirror outranks the environment at client
    construction (#62269), so deleting the endpoint without clearing it leaves
    the agent still authenticating to the deleted host with the deleted key —
    and leaves that key sitting in config.yaml after the operator believes the
    dashboard removed it.

    Only touches ``model`` when it actually names the deleted provider, so an
    endpoint deleted while a *different* provider is active is left alone.
    """
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return
    if str(model_cfg.get("provider") or "").strip().lower() != provider_key:
        return
    for field in ("provider", "base_url", "api_key", "key_env"):
        model_cfg.pop(field, None)
    cfg["model"] = model_cfg


def _write_custom_endpoint(cfg: Dict[str, Any], body: CustomEndpointUpdate) -> Tuple[str, Dict[str, Any]]:
    endpoint_id = _custom_endpoint_id(body.id or body.name)
    name = (body.name or "").strip()
    base_url = (body.base_url or "").strip().rstrip("/")
    model = (body.model or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="name required")
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url required")
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="base_url must include scheme and host")
    if not model:
        raise HTTPException(status_code=400, detail="model required")

    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    existing = providers.get(endpoint_id)
    if not isinstance(existing, dict):
        existing = {}

    # Merge onto the existing entry rather than replacing it. A providers.<name>
    # block is not owned by this panel: it can carry hand-written keys the
    # dashboard has no field for — ``api_mode``, ``key_env``/``api_key_env``,
    # ``extra_headers`` (which may themselves carry credentials),
    # ``request_overrides`` — and rebuilding from scratch silently dropped every
    # one of them on an unrelated edit, leaving a provider that no longer
    # authenticates or speaks the right protocol.
    entry: Dict[str, Any] = dict(existing)
    entry.update({
        "name": name,
        "base_url": base_url,
        "model": model,
        "discover_models": bool(body.discover_models),
    })
    # Same for the model map: merge rather than replace, so existing models
    # keep their context lengths. ``body.models`` is the catalogue the panel's
    # Test button already discovered — without it only the one hand-typed
    # model survived Save, and every picker showed a single-entry list for a
    # provider serving dozens (#69988). A payload with no ``models`` (older
    # UI) still just ensures the named default is present.
    existing_models = entry.get("models")
    models_map: Dict[str, Any] = dict(existing_models) if isinstance(existing_models, dict) else {}
    for candidate in (*(body.models or ()), model):
        model_id = str(candidate).strip()
        if not model_id:
            continue
        current = models_map.get(model_id)
        models_map[model_id] = dict(current) if isinstance(current, dict) else {}
    entry["models"] = models_map
    if body.context_length and body.context_length > 0:
        entry["context_length"] = int(body.context_length)
        entry["models"][model]["context_length"] = int(body.context_length)

    # API keys never belong in config.yaml (#69449). Write to .env and
    # reference it via ``key_env`` — the same indirection built-in providers
    # use and that runtime_provider.py already resolves at load time.
    env_var = custom_endpoint_key_env(endpoint_id)
    submitted_key = body.api_key.strip() if body.api_key is not None else None
    if submitted_key:
        save_env_value(env_var, submitted_key)
        entry["key_env"] = env_var
        entry.pop("api_key", None)
    elif submitted_key is not None:
        # Blank field means "clear the key", not "leave it alone".
        remove_env_value(env_var)
        entry.pop("key_env", None)
        entry.pop("api_key", None)
    elif str(entry.get("api_key") or "").strip() and not _config_api_key_is_env_ref(endpoint_id):
        # No new key submitted, but this entry still carries one an earlier
        # release wrote in plaintext. Migrate it on the next save so endpoints
        # configured before the fix get cleaned up too, without the user
        # having to re-enter the key.
        save_env_value(env_var, entry["api_key"].strip())
        entry["key_env"] = env_var
        entry.pop("api_key", None)

    providers[endpoint_id] = entry
    cfg["providers"] = providers

    if body.make_default:
        cfg["model"] = _apply_main_model_assignment(
            cfg.get("model", {}), endpoint_id, model, base_url
        )
        if entry.get("key_env") and isinstance(cfg["model"], dict):
            cfg["model"]["key_env"] = entry["key_env"]
            cfg["model"].pop("api_key", None)

    return endpoint_id, entry


@router.get("/api/providers/custom-endpoints")
def list_custom_endpoints():
    """Return configured OpenAI-compatible custom endpoints for Desktop."""
    try:
        return _custom_endpoint_response(load_config())
    except Exception:
        _log.exception("GET /api/providers/custom-endpoints failed")
        raise HTTPException(status_code=500, detail="Failed to list custom endpoints")


@router.post("/api/providers/custom-endpoints")
def upsert_custom_endpoint(body: CustomEndpointUpdate):
    """Create or update a v12+ ``providers`` custom endpoint entry."""
    try:
        cfg = load_config()
        endpoint_id, _entry = _write_custom_endpoint(cfg, body)
        save_config(cfg)
        response = _custom_endpoint_response(cfg)
        response["ok"] = True
        response["id"] = endpoint_id
        return response
    except HTTPException:
        raise
    except Exception:
        _log.exception("POST /api/providers/custom-endpoints failed")
        raise HTTPException(status_code=500, detail="Failed to save custom endpoint")


@router.post("/api/providers/custom-endpoints/{endpoint_id}/activate")
def activate_custom_endpoint(endpoint_id: str):
    """Set a configured custom endpoint as the default model provider."""
    try:
        cfg = load_config()
        provider_key = _custom_endpoint_id(endpoint_id)
        providers = cfg.get("providers")
        entry = providers.get(provider_key) if isinstance(providers, dict) else None
        if not isinstance(entry, dict):
            raise HTTPException(status_code=404, detail="custom endpoint not found")

        models = _models_from_custom_endpoint_entry(entry)
        model = str(entry.get("model") or (models[0] if models else "")).strip()
        base_url = str(entry.get("base_url") or "").strip()
        if not model or not base_url:
            raise HTTPException(status_code=400, detail="custom endpoint is incomplete")

        model_cfg = _apply_main_model_assignment(cfg.get("model", {}), provider_key, model, base_url)
        if entry.get("key_env"):
            model_cfg["key_env"] = entry["key_env"]
            model_cfg.pop("api_key", None)
        elif entry.get("api_key"):
            model_cfg["api_key"] = entry["api_key"]
        cfg["model"] = model_cfg
        save_config(cfg)
        return {"ok": True, "provider": provider_key, "model": model}
    except HTTPException:
        raise
    except Exception:
        _log.exception("POST /api/providers/custom-endpoints/%s/activate failed", endpoint_id)
        raise HTTPException(status_code=500, detail="Failed to activate custom endpoint")


@router.delete("/api/providers/custom-endpoints/{endpoint_id}")
def delete_custom_endpoint(endpoint_id: str):
    """Remove a configured custom endpoint from ``providers``."""
    try:
        cfg = load_config()
        provider_key = _custom_endpoint_id(endpoint_id)
        providers = cfg.get("providers")
        if not isinstance(providers, dict) or provider_key not in providers:
            raise HTTPException(status_code=404, detail="custom endpoint not found")
        providers.pop(provider_key, None)
        cfg["providers"] = providers
        _detach_main_model_from_provider(cfg, provider_key)
        remove_env_value(custom_endpoint_key_env(provider_key))
        save_config(cfg)
        response = _custom_endpoint_response(cfg)
        response["ok"] = True
        return response
    except HTTPException:
        raise
    except Exception:
        _log.exception("DELETE /api/providers/custom-endpoints/%s failed", endpoint_id)
        raise HTTPException(status_code=500, detail="Failed to delete custom endpoint")


@router.post("/api/providers/custom-endpoints/validate")
async def validate_custom_endpoint(body: CustomEndpointUpdate):
    """Probe a custom endpoint by calling its OpenAI-compatible /models URL."""
    import httpx

    base_url = (body.base_url or "").strip().rstrip("/")
    if not base_url:
        return {"ok": False, "reachable": True, "message": "Enter an endpoint URL first.", "models": []}

    url = base_url + "/models"
    headers = {"Accept": "application/json"}
    if body.api_key and body.api_key.strip():
        headers["Authorization"] = f"Bearer {body.api_key.strip()}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            resp = await client.get(url, headers=headers)
    except Exception:
        return {"ok": False, "reachable": False, "message": f"Could not reach {url}.", "models": []}

    if resp.status_code in (401, 403):
        return {"ok": False, "reachable": True, "message": "The endpoint rejected the API key.", "models": []}
    if not resp.is_success:
        return {"ok": False, "reachable": True, "message": f"Endpoint returned HTTP {resp.status_code}.", "models": []}

    return {"ok": True, "reachable": True, "message": "", "models": _parse_model_ids(resp)}

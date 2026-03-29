from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.config import load_config, save_config
from hermes_cli.runtime_provider import resolve_runtime_provider

# Ensure .env is loaded so custom provider API keys are visible
load_dotenv(Path.home() / ".hermes" / ".env", override=False)


def _ollama_cloud_live_models_with_note() -> tuple[list[str] | None, str | None]:
    """Fetch live model list from Ollama Cloud API.

    Returns ``(models, note)`` where ``models`` is ``None`` on fetch failure.
    """
    try:
        import urllib.request
        import json

        api_key = os.getenv("OLLAMA_CLOUD_API_KEY", "").strip()
        if not api_key:
            return None, None
        req = urllib.request.Request(
            "https://ollama.com/api/tags",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            return None, "Fallback local models"
        model_ids = [m["name"] for m in models if "name" in m]
        if model_ids:
            return sorted(model_ids), "Live cloud models"
    except Exception:
        pass
    return None, "Fallback local models"


def _codex_live_models_with_note() -> tuple[list[str] | None, str | None]:
    """Return live OpenAI Codex OAuth model IDs when available.

    Returns ``(models, note)`` where ``models`` is ``None`` on fetch failure.
    The note is a short picker-friendly source label.
    """
    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials
        from hermes_cli.codex_models import get_codex_model_ids

        creds = resolve_codex_runtime_credentials()
        token = str(creds.get("api_key") or "").strip()
        if not token:
            return None, None
        models = get_codex_model_ids(access_token=token)
        if models:
            return models, "Live OAuth models"
    except Exception:
        pass
    return None, "Fallback local models"


_DEFAULT_PROVIDER_ORDER = [
    "ollama-cloud",
    "ollama-local",
    "openai-codex",
    "google",
    "xai",
    "openrouter",
]

_PROVIDER_LABEL_FALLBACKS = {
    "openrouter": "OpenRouter",
    "openai-codex": "OpenAI Codex",
    "google": "Google",
    "xai": "xAI",
    "ollama-cloud": "Ollama Cloud",
    "ollama-local": "Ollama (Local)",
}


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def models_yaml_path() -> Path:
    return _hermes_home() / "models.yaml"


def load_model_picker_config() -> dict[str, Any]:
    path = models_yaml_path()
    if not path.exists():
        return {"providers": []}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {"providers": []}
    providers = data.get("providers", [])
    if not isinstance(providers, list):
        providers = []
    return {"providers": providers}


def _provider_label(provider_id: str, override: str | None = None) -> str:
    if override:
        return override
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if pconfig and getattr(pconfig, "name", None):
        return pconfig.name
    return _PROVIDER_LABEL_FALLBACKS.get(provider_id, provider_id)


def is_provider_configured(provider_id: str) -> bool:
    try:
        runtime = resolve_runtime_provider(requested=provider_id)
        return bool(runtime.get("api_key"))
    except Exception:
        return False


def _normalize_model_entry(provider_id: str, raw: Any, provider_note: str | None = None) -> dict[str, str] | None:
    if isinstance(raw, str):
        model_id = raw.strip()
        if not model_id:
            return None
        return {
            "id": model_id,
            "label": model_id,
            "description": provider_note or "",
        }
    if isinstance(raw, dict):
        model_id = str(raw.get("id", "")).strip()
        if not model_id:
            return None
        label = str(raw.get("label", "")).strip() or model_id
        description = str(raw.get("description", "")).strip() or (provider_note or "")
        return {
            "id": model_id,
            "label": label,
            "description": description,
        }
    return None


def get_curated_model_catalog(*, include_unconfigured: bool = False) -> list[dict[str, Any]]:
    data = load_model_picker_config()
    providers = []
    for entry in data.get("providers", []):
        if not isinstance(entry, dict):
            continue
        provider_id = str(entry.get("id", "")).strip()
        if not provider_id:
            continue
        enabled = entry.get("enabled", True)
        if enabled is False:
            continue
        configured = is_provider_configured(provider_id)
        if not include_unconfigured and not configured:
            continue
        label = _provider_label(provider_id, str(entry.get("label", "")).strip() or None)
        note = str(entry.get("note", "")).strip()

        yaml_models = []
        yaml_index: dict[str, dict[str, str]] = {}
        for raw_model in entry.get("models", []):
            normalized = _normalize_model_entry(provider_id, raw_model, note)
            if normalized:
                yaml_models.append(normalized)
                yaml_index[normalized["id"]] = normalized

        models = list(yaml_models)
        if provider_id == "openai-codex" and configured:
            live_models, live_note = _codex_live_models_with_note()
            if live_models:
                models = []
                for model_id in live_models:
                    existing = yaml_index.get(model_id)
                    if existing:
                        models.append(dict(existing))
                    else:
                        models.append({
                            "id": model_id,
                            "label": model_id,
                            "description": live_note or "",
                        })
                note = live_note or note
            elif live_note:
                note = live_note

        # Ollama Cloud: filter to favourites if defined, validate against live API
        if provider_id == "ollama-cloud" and configured:
            favourites = entry.get("favourites", [])
            if favourites and isinstance(favourites, list):
                # Show only favourited models from yaml
                fav_set = set(favourites)
                models = [
                    m for m in yaml_models
                    if m["id"] in fav_set
                    or m["id"].replace("ollama-cloud/", "") in fav_set
                ]
            # Validate favourites exist on live API (non-blocking)
            live_models, live_note = _ollama_cloud_live_models_with_note()
            if live_models:
                note = live_note or note

        providers.append(
            {
                "id": provider_id,
                "label": label,
                "note": note,
                "configured": configured,
                "models": models,
            }
        )

    order_index = {pid: idx for idx, pid in enumerate(_DEFAULT_PROVIDER_ORDER)}
    providers.sort(key=lambda item: (order_index.get(item["id"], 999), item["label"].lower()))
    return providers


def get_default_model_selection() -> tuple[str, str]:
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider", "openrouter") or "openrouter").strip().lower()
        model = str(model_cfg.get("default", "") or "").strip()
        return provider, model
    if isinstance(model_cfg, str):
        return "openrouter", model_cfg.strip()
    return "openrouter", ""


def get_current_model_selection() -> tuple[str, str]:
    default_provider, default_model = get_default_model_selection()
    provider = os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower() or default_provider
    model = os.getenv("HERMES_MODEL", "").strip() or default_model
    return provider, model


def format_model_selection(provider: str | None, model: str | None) -> str:
    provider = (provider or "").strip()
    model = (model or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return model or "(not set)"


def command_model_name(provider: str, model_id: str) -> str:
    prefix = f"{provider}/"
    if model_id.startswith(prefix):
        return model_id[len(prefix):]
    if provider == "openrouter" and model_id.startswith("openrouter/"):
        return model_id[len("openrouter/"):]
    return model_id


def apply_model_selection(provider: str, model_id: str) -> tuple[bool, str, dict[str, str]]:
    """Apply a picker selection directly, without relying on the removed /model command."""
    from hermes_cli.model_switch import switch_model

    provider = (provider or "").strip().lower()
    model_id = (model_id or "").strip()
    if not provider or not model_id:
        return False, "No model selected.", {}

    cfg = load_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    current_provider = "openrouter"
    current_base_url = ""
    if isinstance(model_cfg, dict):
        current_provider = str(model_cfg.get("provider", current_provider) or current_provider).strip().lower()
        current_base_url = str(model_cfg.get("base_url", "") or "").strip()
    elif isinstance(model_cfg, str) and model_cfg.strip():
        current_provider = "openrouter"

    current_api_key = ""
    try:
        runtime = resolve_runtime_provider(requested=current_provider)
        current_api_key = str(runtime.get("api_key", "") or "")
        if not current_base_url:
            current_base_url = str(runtime.get("base_url", "") or "")
    except Exception:
        pass

    command_model = command_model_name(provider, model_id)
    raw_input = f"{provider}:{command_model}"
    result = switch_model(
        raw_input,
        current_provider,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
    )
    if not result.success:
        return False, result.error_message or "Failed to switch model.", {}

    if not isinstance(cfg, dict):
        cfg = {}
    current_model_cfg = cfg.get("model")
    if isinstance(current_model_cfg, dict):
        new_model_cfg = dict(current_model_cfg)
    elif isinstance(current_model_cfg, str) and current_model_cfg.strip():
        new_model_cfg = {"default": current_model_cfg.strip()}
    else:
        new_model_cfg = {}

    new_model_cfg["default"] = result.new_model
    new_model_cfg["provider"] = result.target_provider
    if result.base_url:
        new_model_cfg["base_url"] = result.base_url.rstrip("/")
    else:
        new_model_cfg.pop("base_url", None)
    cfg["model"] = new_model_cfg
    save_config(cfg)

    message = f"Switched to `{result.target_provider}/{result.new_model}`."
    if result.warning_message:
        message += f"\n⚠️ {result.warning_message}"
    message += "\n_(takes effect on next message)_"
    details = {
        "provider": result.target_provider,
        "model": result.new_model,
        "base_url": (result.base_url or "").rstrip("/"),
        "provider_label": result.provider_label or result.target_provider,
    }
    return True, message, details

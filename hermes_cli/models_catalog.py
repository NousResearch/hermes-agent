"""Catalog result and display-name helpers."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _probe_result(
    models, probed_url, resolved_base_url, suggested_base_url=None, used_fallback=False,
    model_names=None,
) -> dict[str, Any]:
    return {
        "models": models,
        "probed_url": probed_url,
        "resolved_base_url": resolved_base_url,
        "suggested_base_url": suggested_base_url,
        "used_fallback": used_fallback,
        # ``{model_id: display_name}`` as published by the endpoint. Presentation only.
        "model_names": model_names or {}}


def _published_name(item: dict) -> str:
    """Return the first nonempty string label that differs from the routing ID."""
    model_id = str(item.get("id") or "").strip()
    for key in ("name", "display_name", "displayName"):
        value = item.get(key)
        if not isinstance(value, str):
            continue
        label = value.strip()
        if label and label != model_id:
            return label
    return ""


def fetch_api_model_names(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Read endpoint labels using the catalog probe's auth and TLS settings."""
    from hermes_cli.models import probe_api_models

    try:
        result = probe_api_models(
            api_key, base_url, timeout=timeout, api_mode=api_mode, request_headers=headers)
    except Exception:
        logger.debug("Model label lookup failed", exc_info=True)
        return {}
    return result.get("model_names", {})


def model_display_names() -> dict[str, str]:
    """Read model labels from models.dev without making a network request."""
    try:
        from agent.models_dev import fetch_models_dev

        data = fetch_models_dev(allow_network=False)
    except Exception:
        logger.debug("Model label lookup failed", exc_info=True)
        return {}
    names: dict[str, str] = {}
    for provider_data in (data or {}).values():
        models = provider_data.get("models") if isinstance(provider_data, dict) else None
        if not isinstance(models, dict):
            continue
        for model_id, entry in models.items():
            label = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(label, str):
                continue
            label = label.strip()
            if label and label != model_id and model_id not in names:
                names[model_id] = label
    return names

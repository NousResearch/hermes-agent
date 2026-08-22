"""Generic OpenAI-compatible image generation for Custom Endpoints.

This plugin reads the ``providers:`` section of ``config.yaml`` and registers
an :class:`ImageGenProvider` for each named provider that declares an
``capabilities.image_gen`` block.  This lets users point a single Custom
Endpoint at an OpenAI-compatible image API (e.g. ``gpt-image-2``,
``grok-imagine-image``) and select it via ``image_gen.provider`` — without
duplicating credentials into provider-specific env vars.

Config example::

    providers:
      my-gateway:
        name: My Gateway
        base_url: https://gateway.example.com/v1
        key_env: HERMES_CUSTOM_MY_GATEWAY_API_KEY
        capabilities:
          image_gen:
            models:
              - gpt-image-2
              - grok-imagine-image
            default_model: grok-imagine-image

The generated provider reuses the endpoint's ``base_url`` and ``key_env``.
Requests follow the standard OpenAI ``POST /images/generations`` shape.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _load_provider_configs() -> Dict[str, Dict[str, Any]]:
    """Read the ``providers:`` section from config.yaml and return entries
    that declare an ``capabilities.image_gen`` block.

    Returns a dict keyed by the provider's config key, each value is the
    full provider entry dict.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
    except Exception as exc:
        logger.debug("Could not load config for custom-endpoint image_gen: %s", exc)
        return {}

    providers = cfg.get("providers") if isinstance(cfg, dict) else None
    if not isinstance(providers, dict):
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for key, entry in providers.items():
        if not isinstance(entry, dict):
            continue
        caps = entry.get("capabilities")
        if not isinstance(caps, dict):
            continue
        image_gen_caps = caps.get("image_gen")
        if not isinstance(image_gen_caps, dict):
            continue
        # Must have at least a models list or a default_model
        models = image_gen_caps.get("models")
        if not isinstance(models, list) or not models:
            continue
        result[key] = entry
    return result


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

# OpenAI-compatible size mapping for standard aspect ratios.
_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


class CustomEndpointImageGenProvider(ImageGenProvider):
    """Image generation provider backed by an OpenAI-compatible Custom Endpoint.

    Each instance corresponds to one ``providers:`` entry that declares
    ``capabilities.image_gen``.  The provider name is ``custom:<config-key>``
    so it can be selected via ``image_gen.provider: custom:my-gateway``.
    """

    def __init__(self, config_key: str, entry: Dict[str, Any]) -> None:
        self._config_key = config_key
        self._entry = entry
        self._caps = entry["capabilities"]["image_gen"]

        # Build the model catalog from config
        self._models: List[Dict[str, Any]] = []
        raw_models = self._caps.get("models", [])
        for m in raw_models:
            if isinstance(m, str) and m.strip():
                self._models.append({"id": m.strip(), "display": m.strip()})
            elif isinstance(m, dict) and m.get("id"):
                self._models.append(m)

        self._default_model = self._caps.get("default_model")
        if not self._default_model and self._models:
            self._default_model = self._models[0].get("id")

    # -- ImageGenProvider interface -----------------------------------------

    @property
    def name(self) -> str:
        return f"custom:{self._config_key}"

    @property
    def display_name(self) -> str:
        return self._entry.get("name") or self._config_key

    def is_available(self) -> bool:
        """True when the endpoint's ``key_env`` is set in the environment."""
        key_env = self._entry.get("key_env", "")
        if not key_env:
            # If no key_env is configured, assume it's an open endpoint
            return True
        return bool(os.environ.get(key_env, "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self._models)

    def default_model(self) -> Optional[str]:
        return self._default_model

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text"],
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        key_env = self._entry.get("key_env", "")
        env_vars = []
        if key_env:
            env_vars.append({
                "key": key_env,
                "prompt": f"API key for {self.display_name}",
                "url": "",
            })
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": f"Custom Endpoint ({self._entry.get('base_url', '')})",
            "env_vars": env_vars,
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image via the OpenAI-compatible /images/generations endpoint."""
        # Resolve model
        model_id = model or self._default_model
        if not model_id:
            return error_response(
                error="No model configured for this Custom Endpoint. "
                      "Set capabilities.image_gen.default_model or pass model=.",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(aspect_ratio),
            )

        # Validate model against catalog
        valid_ids = {m["id"] for m in self._models}
        if model_id not in valid_ids:
            return error_response(
                error=f"Model '{model_id}' is not in the configured model list "
                      f"for {self.display_name}. Available: {', '.join(sorted(valid_ids))}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(aspect_ratio),
            )

        # Resolve credentials
        key_env = self._entry.get("key_env", "")
        api_key = os.environ.get(key_env, "").strip() if key_env else ""

        # Resolve base URL
        base_url = (
            self._entry.get("base_url")
            or self._entry.get("api")
            or self._entry.get("url")
            or ""
        ).strip().rstrip("/")
        if not base_url:
            return error_response(
                error=f"Custom Endpoint '{self._config_key}' has no base_url configured.",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=resolve_aspect_ratio(aspect_ratio),
            )

        # Build the request
        ar = resolve_aspect_ratio(aspect_ratio)
        size = _SIZES.get(ar, _SIZES["landscape"])

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{base_url}/images/generations"

        try:
            import requests
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            return error_response(
                error=f"Image generation request failed: {exc}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=ar,
            )

        try:
            body = resp.json()
        except Exception as exc:
            return error_response(
                error=f"Image generation returned non-JSON response: {exc}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=ar,
            )

        # OpenAI-compatible APIs return {"data": [{"b64_json": "..."} or {"url": "..."}]}
        data_list = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data_list, list) or not data_list:
            return error_response(
                error="Image generation response missing 'data' array.",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=ar,
            )

        first = data_list[0]
        if not isinstance(first, dict):
            return error_response(
                error="Image generation response has malformed data entry.",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=ar,
            )

        # Prefer b64_json (download-safe); fall back to URL
        b64 = first.get("b64_json")
        img_url = first.get("url")

        if b64:
            try:
                path = save_b64_image(b64, prefix="custom_endpoint")
                image_str = str(path)
            except Exception as exc:
                return error_response(
                    error=f"Failed to save generated image: {exc}",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=ar,
                )
        elif img_url:
            try:
                path = save_url_image(img_url, prefix="custom_endpoint")
                image_str = str(path)
            except Exception as exc:
                # Fall back to returning the URL directly
                image_str = img_url
                logger.warning(
                    "Could not cache image from %s: %s — returning URL directly",
                    img_url, exc,
                )
        else:
            return error_response(
                error="Image generation response contains neither b64_json nor url.",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=ar,
            )

        return success_response(
            image=image_str,
            model=model_id,
            prompt=prompt,
            aspect_ratio=ar,
            provider=self.name,
            modality="text",
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Plugin entry point — called by the plugin loader.

    Reads ``providers:`` from config.yaml, finds entries with
    ``capabilities.image_gen``, and registers a
    :class:`CustomEndpointImageGenProvider` for each.
    """
    configs = _load_provider_configs()
    if not configs:
        logger.debug("No Custom Endpoints with image_gen capabilities found.")
        return

    for config_key, entry in configs.items():
        try:
            provider = CustomEndpointImageGenProvider(config_key, entry)
            ctx.register_image_gen_provider(provider)
        except Exception as exc:
            logger.warning(
                "Failed to register Custom Endpoint image_gen provider '%s': %s",
                config_key, exc,
            )
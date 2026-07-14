"""Configurable router image generation backend.

Routes ``image_generate`` calls to user-defined model aliases under
``image_gen.router.models``. The first implementation intentionally keeps the
surface small: OpenAI-compatible ``/v1/images/generations`` gateways only.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

_PROVIDER_OPENAI_COMPAT = "openai-compatible"
_DEFAULT_BASE_URL_ENV = "IMAGE_GATEWAY_BASE_URL"
_DEFAULT_API_KEY_ENV = "IMAGE_GATEWAY_API_KEY"
_DEFAULT_TIMEOUT = 180

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_router_config() -> Dict[str, Any]:
    """Return ``image_gen.router`` from config.yaml, or an empty dict."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        image_gen = cfg.get("image_gen") if isinstance(cfg, dict) else None
        router = image_gen.get("router") if isinstance(image_gen, dict) else None
        return router if isinstance(router, dict) else {}
    except Exception as exc:  # noqa: BLE001 - config read should never crash picker/tool
        logger.debug("Could not load image_gen.router config: %s", exc)
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _models_from_config(router_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models = router_cfg.get("models")
    if not isinstance(models, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for alias, entry in models.items():
        if isinstance(alias, str) and alias.strip() and isinstance(entry, dict):
            normalized[alias.strip()] = dict(entry)
    return normalized


def _default_alias(router_cfg: Dict[str, Any], models: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for key in ("default_model", "model"):
        value = router_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return next(iter(models), None)


def _resolve_model_entry(
    router_cfg: Dict[str, Any],
    requested_model: Optional[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    models = _models_from_config(router_cfg)
    if not models:
        return None, None, "No image_gen.router.models configured"

    alias = requested_model.strip() if isinstance(requested_model, str) and requested_model.strip() else None
    if not alias:
        alias = _default_alias(router_cfg, models)

    if not alias or alias not in models:
        available = ", ".join(sorted(models))
        return alias, None, f"Unknown image model alias '{alias}'. Available aliases: {available}"

    defaults = router_cfg.get("defaults") if isinstance(router_cfg.get("defaults"), dict) else {}
    return alias, _deep_merge(defaults, models[alias]), None


def _env_value(entry: Dict[str, Any], key_name: str, default_env: str) -> str:
    direct = entry.get(key_name)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    env_key = entry.get(f"{key_name}_env")
    if not isinstance(env_key, str) or not env_key.strip():
        env_key = default_env
    return os.environ.get(env_key.strip(), "").strip()


def _endpoint_url(base_url: str, entry: Dict[str, Any]) -> str:
    endpoint = entry.get("endpoint", "images/generations")
    endpoint_s = str(endpoint or "images/generations").strip().strip("/")
    if endpoint_s in {"images", "generations"}:
        endpoint_s = "images/generations"
    return f"{base_url.rstrip('/')}/{endpoint_s}"


def _response_image_item(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = result.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def _valid_remote_image_url(value: str) -> bool:
    """Return True only for HTTP(S) image references from the gateway."""
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class RouterImageGenProvider(ImageGenProvider):
    """Route image generation to configured OpenAI-compatible model aliases."""

    @property
    def name(self) -> str:
        return "router"

    @property
    def display_name(self) -> str:
        return "Image Router"

    def is_available(self) -> bool:
        router_cfg = _load_router_config()
        models = _models_from_config(router_cfg)
        if not models:
            return False
        defaults = router_cfg.get("defaults") if isinstance(router_cfg.get("defaults"), dict) else {}
        for entry in models.values():
            merged = _deep_merge(defaults, entry)
            if _env_value(merged, "base_url", _DEFAULT_BASE_URL_ENV) and _env_value(
                merged, "api_key", _DEFAULT_API_KEY_ENV
            ):
                return True
        return False

    def list_models(self) -> List[Dict[str, Any]]:
        router_cfg = _load_router_config()
        models = _models_from_config(router_cfg)
        rows: List[Dict[str, Any]] = []
        for alias, entry in models.items():
            rows.append(
                {
                    "id": alias,
                    "display": entry.get("display", alias),
                    "speed": entry.get("speed", ""),
                    "strengths": ", ".join(entry.get("strengths", []))
                    if isinstance(entry.get("strengths"), list)
                    else entry.get("strengths", ""),
                    "price": entry.get("price", entry.get("cost", "")),
                }
            )
        return rows

    def default_model(self) -> Optional[str]:
        router_cfg = _load_router_config()
        return _default_alias(router_cfg, _models_from_config(router_cfg))

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Image Router",
            "badge": "custom",
            "tag": "Route image generation across configured model aliases via OpenAI-compatible gateways",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        requested_model = kwargs.get("model")

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="router",
                aspect_ratio=aspect,
            )

        router_cfg = _load_router_config()
        alias, entry, error = _resolve_model_entry(router_cfg, requested_model)
        if error or entry is None or alias is None:
            return error_response(
                error=error or "Could not resolve router model alias",
                error_type="unknown_model" if alias else "missing_config",
                provider="router",
                model=alias or "",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        provider_kind = str(entry.get("provider") or _PROVIDER_OPENAI_COMPAT).strip().lower()
        if provider_kind not in {_PROVIDER_OPENAI_COMPAT, "openai_compatible"}:
            return error_response(
                error=f"Unsupported router backend provider '{provider_kind}'. Only openai-compatible is supported.",
                error_type="unsupported_provider",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        backend_model = entry.get("model")
        if not isinstance(backend_model, str) or not backend_model.strip():
            return error_response(
                error=f"Router model alias '{alias}' is missing a backend model name",
                error_type="invalid_config",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        backend_model = backend_model.strip()

        base_url = _env_value(entry, "base_url", _DEFAULT_BASE_URL_ENV)
        api_key = _env_value(entry, "api_key", _DEFAULT_API_KEY_ENV)
        if not base_url or not api_key:
            return error_response(
                error=(
                    f"Router model alias '{alias}' is missing gateway credentials. "
                    "Set base_url/api_key or base_url_env/api_key_env in image_gen.router."
                ),
                error_type="auth_required",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        default_params = entry.get("default_params") if isinstance(entry.get("default_params"), dict) else {}
        payload: Dict[str, Any] = {
            "model": backend_model,
            "prompt": prompt,
            "size": _SIZES.get(aspect, _SIZES["square"]),
            "n": 1,
        }
        payload.update(default_params)

        try:
            timeout = int(entry.get("timeout", _DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT

        try:
            response = requests.post(
                _endpoint_url(base_url, entry),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            text = exc.response.text[:300] if exc.response is not None else str(exc)
            return error_response(
                error=f"Router image gateway failed ({status}): {text}",
                error_type="api_error",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"Router image gateway timed out after {timeout}s",
                error_type="timeout",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Router image generation failed: {exc}",
                error_type="api_error",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not isinstance(result, dict):
            return error_response(
                error="Router image gateway returned a non-object response",
                error_type="invalid_response",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        item = _response_image_item(result)
        if not item:
            return error_response(
                error="Router image gateway returned no image data",
                error_type="empty_response",
                provider="router",
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        revised_prompt = item.get("revised_prompt")
        extra = {"backend_model": backend_model}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        b64 = item.get("b64_json")
        if isinstance(b64, str) and b64.strip():
            try:
                saved = save_b64_image(b64, prefix=f"router_{alias}")
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not save router image response: {exc}",
                    error_type="save_error",
                    provider="router",
                    model=alias,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            return success_response(
                image=str(saved),
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
                provider="router",
                extra=extra,
            )

        image_url = item.get("url") or item.get("image_url")
        if isinstance(image_url, str) and image_url.strip():
            image_url = image_url.strip()
            if not _valid_remote_image_url(image_url):
                return error_response(
                    error="Router image gateway returned an invalid image URL; only http(s) URLs are accepted",
                    error_type="invalid_response",
                    provider="router",
                    model=alias,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            return success_response(
                image=image_url,
                model=alias,
                prompt=prompt,
                aspect_ratio=aspect,
                provider="router",
                extra=extra,
            )

        return error_response(
            error="Router image gateway response did not include b64_json, url, or image_url",
            error_type="invalid_response",
            provider="router",
            model=alias,
            prompt=prompt,
            aspect_ratio=aspect,
        )


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_image_gen_provider(RouterImageGenProvider())

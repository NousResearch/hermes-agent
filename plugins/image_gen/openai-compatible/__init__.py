"""Generic OpenAI-compatible Images API backend.

The provider is selected explicitly through ``image_gen.provider`` and reads its
behavioral settings from ``image_gen.openai_compatible``. Credentials are loaded
from the active Hermes profile's ``OPENAI_COMPATIBLE_IMAGE_API_KEY`` secret; they
are never accepted from ``config.yaml``.

Example::

    image_gen:
      provider: openai-compatible
      model: vendor/image-model
      openai_compatible:
        base_url: http://localhost:30122/v1
        size_by_aspect:
          square: 512x512
          landscape: 1024x576
          portrait: 576x1024
        params:
          num_inference_steps: 8
          guidance_scale: 1.0
        timeout: 600

The endpoint must implement ``POST /v1/images/generations`` and return either a
``b64_json`` value or an image URL in the first ``data`` item.
"""

from __future__ import annotations

import logging
import math
import re
from threading import TIMEOUT_MAX
from typing import Any, Optional
from urllib.parse import urlsplit

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

from plugins.image_gen._common import error_factory, load_image_gen_config

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai-compatible"
API_KEY_ENV = "OPENAI_COMPATIBLE_IMAGE_API_KEY"
DEFAULT_TIMEOUT = 300.0
_RESERVED_PARAMS = {"model", "prompt", "size", "n", "response_format"}
_DEFAULT_SIZES = {
    "square": "1024x1024",
    "landscape": "1536x1024",
    "portrait": "1024x1536",
}


def _configured_model() -> str:
    image_gen = load_image_gen_config()
    compatible = image_gen.get("openai_compatible")
    local_model = compatible.get("model") if isinstance(compatible, dict) else None
    if isinstance(local_model, str) and local_model.strip():
        return local_model.strip()
    global_model = image_gen.get("model")
    return global_model.strip() if isinstance(global_model, str) else ""


def _read_api_key() -> str:
    """Read the fixed credential from the authoritative active profile scope."""
    from agent.secret_scope import get_secret

    # Do not catch UnscopedSecretError here. Under multiplexing an unscoped
    # credential read must fail closed rather than fall back to another
    # profile's process-global environment value.
    return (get_secret(API_KEY_ENV) or "").strip()


def _endpoint(base_url: str) -> str:
    """Join an API host or `/v1` root to the Images generations endpoint."""
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query string or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/images/generations"):
        return value
    if path.endswith("/v1"):
        return value + "/images/generations"
    return value + "/v1/images/generations"


def _safe_cache_component(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (clean or "image")[:80]


def _request_timeout(config: dict[str, Any]) -> float:
    """Validate before coercion: bools are ints, and int(1.9) silently truncates."""
    raw_timeout = config.get("timeout", DEFAULT_TIMEOUT)
    try:
        if isinstance(raw_timeout, bool):
            raise ValueError
        timeout = float(raw_timeout)
        if not math.isfinite(timeout) or not 0 < timeout <= TIMEOUT_MAX:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"timeout must be a finite positive number <= {TIMEOUT_MAX}") from exc

    # Hermes' YAML 1.1 loader resolves an unquoted `n` key to False.
    count = config.get("n", config.get(False, 1))
    if not ((type(count) is int and count == 1) or (isinstance(count, str) and count.strip() == "1")):
        raise ValueError("image_gen.openai_compatible.n must be 1 because Hermes returns one image")
    return timeout


def _request_payload(config: dict[str, Any], model: str, prompt: str, aspect: str) -> dict[str, Any]:
    params = config.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError("image_gen.openai_compatible.params must be a mapping")
    params = dict(params or {})
    conflicts = _RESERVED_PARAMS.intersection(params)
    if False in params:  # YAML 1.1 spelling of an unquoted `n` key.
        conflicts.add("n")
    if conflicts:
        raise ValueError("image_gen.openai_compatible.params cannot override reserved fields: "
                         + ", ".join(sorted(conflicts)))
    sizes = config.get("size_by_aspect")
    size = sizes.get(aspect) if isinstance(sizes, dict) else None
    if not isinstance(size, str) or not size.strip():
        configured_size = config.get("size")
        size = configured_size if isinstance(configured_size, str) else _DEFAULT_SIZES[aspect]
    payload = {"model": model, "prompt": prompt, "size": size.strip(), "n": 1, **params}
    response_format = config.get("response_format")
    if isinstance(response_format, str) and response_format.strip():
        payload["response_format"] = response_format.strip()
    return payload


class OpenAICompatibleImageProvider(ImageGenProvider):
    """User-selected OpenAI Images-compatible endpoint."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        config = load_image_gen_config("openai_compatible")
        value = config.get("display_name")
        return value.strip() if isinstance(value, str) and value.strip() else "OpenAI-compatible"

    def is_available(self) -> bool:
        config = load_image_gen_config("openai_compatible")
        base_url = config.get("base_url")
        return bool(isinstance(base_url, str) and base_url.strip() and _configured_model() and _read_api_key())

    def list_models(self) -> list[dict[str, Any]]:
        model = _configured_model()
        if model:
            return [{"id": model, "display": model}]
        return []

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        return models[0]["id"] if models else None

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "User-configured OpenAI /v1/images/generations endpoint",
            "config_fields": [
                {"key": "image_gen.openai_compatible.base_url", "prompt": "Image endpoint base URL"},
                {"key": "image_gen.openai_compatible.model", "prompt": "Image model ID",
                 "fallback_keys": ["image_gen.model"]},
            ],
            "env_vars": [
                {
                    "key": API_KEY_ENV,
                    "prompt": "OpenAI-compatible image endpoint bearer token",
                }
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        config = load_image_gen_config("openai_compatible")
        fail = error_factory(self.name, aspect, prompt=prompt)

        if not prompt:
            return fail("Prompt is required and must be a non-empty string", "invalid_argument")
        if image_url or reference_image_urls:
            return fail(
                "OpenAI-compatible provider supports text-to-image only",
                "modality_unsupported",
            )

        base_url = config.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            return fail("image_gen.openai_compatible.base_url is required", "invalid_config")
        try:
            endpoint = _endpoint(base_url)
        except ValueError as exc:
            return fail(str(exc), "invalid_config")

        # This provider documents its scoped model as taking precedence over
        # image_gen.model. The dispatcher passes the global model in kwargs,
        # so resolve configured provider scope before using that fallback.
        model = _configured_model()
        explicit_model = kwargs.get("model")
        if not model and isinstance(explicit_model, str):
            model = explicit_model.strip()
        if not model:
            return fail(
                "Set image_gen.model or image_gen.openai_compatible.model",
                "invalid_config",
            )

        try:
            timeout = _request_timeout(config)
            payload = _request_payload(config, model, prompt, aspect)
        except ValueError as exc:
            return fail(str(exc), "invalid_config", model=model)

        api_key = _read_api_key()
        if not api_key:
            return fail(f"{API_KEY_ENV} is not set", "auth_required")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent/1.0 (openai-compatible-image-gen)",
        }
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                return fail(
                    "Image endpoint redirected; refusing to forward credentials",
                    "redirect_not_allowed",
                    model=model,
                )
            response.raise_for_status()
        except requests.Timeout:
            return fail(
                f"OpenAI-compatible image generation timed out ({int(timeout)}s)",
                "timeout",
                model=model,
            )
        except requests.ConnectionError as exc:
            return fail(f"Image endpoint connection error: {exc}", "connection_error", model=model)
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else 0
            detail = ""
            if response is not None:
                try:
                    body = response.json()
                    error = body.get("error") if isinstance(body, dict) else None
                    detail = error.get("message", "") if isinstance(error, dict) else str(error or "")
                except ValueError:
                    detail = response.text[:300]
            return fail(
                f"Image generation failed ({status}): {detail or str(exc)}",
                "api_error",
                model=model,
            )

        try:
            result = response.json()
        except ValueError as exc:
            return fail(f"Image endpoint returned invalid JSON: {exc}", "invalid_response", model=model)

        data = result.get("data") if isinstance(result, dict) else None
        first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
        prefix = f"{PROVIDER_NAME}_{_safe_cache_component(model)}"
        try:
            if isinstance(first.get("b64_json"), str) and first["b64_json"].strip():
                image = save_b64_image(first["b64_json"], prefix=prefix)
            elif isinstance(first.get("url"), str) and first["url"].strip():
                image = save_url_image(first["url"], prefix=prefix)
            else:
                return fail("Image endpoint returned no image data", "empty_response", model=model)
        except Exception as exc:
            logger.debug("Could not cache generated image", exc_info=True)
            return fail(f"Could not cache generated image: {exc}", "io_error", model=model)

        return success_response(
            image=str(image),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={"size": payload["size"]},
        )


def register(ctx: Any) -> None:
    ctx.register_image_gen_provider(OpenAICompatibleImageProvider())

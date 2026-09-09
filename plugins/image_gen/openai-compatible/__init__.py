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
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

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


def _load_image_gen_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        image_gen = config.get("image_gen") if isinstance(config, dict) else None
        return image_gen if isinstance(image_gen, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not load image_gen config: %s", exc)
        return {}


def _load_config() -> Dict[str, Any]:
    compatible = _load_image_gen_config().get("openai_compatible")
    return compatible if isinstance(compatible, dict) else {}


def _configured_model() -> str:
    image_gen = _load_image_gen_config()
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


class OpenAICompatibleImageProvider(ImageGenProvider):
    """User-selected OpenAI Images-compatible endpoint."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        config = _load_config()
        value = config.get("display_name")
        return value.strip() if isinstance(value, str) and value.strip() else "OpenAI-compatible"

    def is_available(self) -> bool:
        config = _load_config()
        base_url = config.get("base_url")
        return bool(isinstance(base_url, str) and base_url.strip() and _read_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        model = _configured_model()
        if model:
            return [{"id": model, "display": model}]
        return []

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        return models[0]["id"] if models else None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "User-configured OpenAI /v1/images/generations endpoint",
            "env_vars": [
                {
                    "key": API_KEY_ENV,
                    "prompt": "OpenAI-compatible image endpoint bearer token",
                }
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        config = _load_config()

        def fail(message: str, error_type: str, *, model: str = "") -> Dict[str, Any]:
            return error_response(
                error=message,
                error_type=error_type,
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

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

        api_key = _read_api_key()
        if not api_key:
            return fail(f"{API_KEY_ENV} is not set", "auth_required")

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

        params = config.get("params")
        if params is not None and not isinstance(params, dict):
            return fail("image_gen.openai_compatible.params must be a mapping", "invalid_config", model=model)
        params = dict(params or {})
        conflicts = sorted(_RESERVED_PARAMS.intersection(params))
        if conflicts:
            return fail(
                "image_gen.openai_compatible.params cannot override reserved fields: "
                + ", ".join(conflicts),
                "invalid_config",
                model=model,
            )

        try:
            timeout = float(config.get("timeout", DEFAULT_TIMEOUT))
            if timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return fail("timeout must be a positive number", "invalid_config", model=model)
        try:
            count = int(config.get("n", 1))
        except (TypeError, ValueError):
            return fail(
                "image_gen.openai_compatible.n must be 1 because Hermes returns one image",
                "invalid_config",
                model=model,
            )
        if count != 1:
            return fail(
                "image_gen.openai_compatible.n must be 1 because Hermes returns one image",
                "invalid_config",
                model=model,
            )

        sizes = config.get("size_by_aspect")
        size = sizes.get(aspect) if isinstance(sizes, dict) else None
        if not isinstance(size, str) or not size.strip():
            configured_size = config.get("size")
            size = configured_size if isinstance(configured_size, str) else _DEFAULT_SIZES[aspect]

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size.strip(),
            "n": count,
            **params,
        }
        response_format = config.get("response_format")
        if isinstance(response_format, str) and response_format.strip():
            payload["response_format"] = response_format.strip()

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
                except Exception:  # noqa: BLE001
                    detail = response.text[:300]
            return fail(
                f"Image generation failed ({status}): {detail or str(exc)}",
                "api_error",
                model=model,
            )

        try:
            result = response.json()
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
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

"""OpenRouter-compatible image generation backend (OpenRouter + Nous Portal).

Both OpenRouter and the Nous Portal inference endpoint speak the same
OpenAI-style ``/chat/completions`` image-generation protocol: send
``modalities: ["image", "text"]`` with an image-output model (e.g.
``google/gemini-2.5-flash-image``), pass reference images as ``image_url``
content parts for grounding, and read the generated images back from
``choices[0].message.images[].image_url.url`` (a ``data:image/...;base64`` URI).

Nous Portal proxies OpenRouter, so one implementation services both — we only
swap the resolved ``(base_url, api_key)``. Credentials are resolved through the
agent's existing :func:`~hermes_cli.runtime_provider.resolve_runtime_provider`,
which already understands OpenRouter's key pool and the Nous OAuth device-code
token, so this plugin never reinvents auth.

Reference grounding is the reason pet sprite generation cares about this
backend: each animation row must stay the same character as the chosen base
frame, which only works on models that accept image input. Gemini Flash Image
("nano-banana") does, so both providers advertise image-to-image support.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Default image-output model. Gemini 2.5 Flash Image ("nano-banana") is GA on
# OpenRouter, accepts reference images for grounding, and honors
# ``image_config.aspect_ratio``.
DEFAULT_MODEL = "google/gemini-2.5-flash-image"

# Semantic aspect ratio (the image_gen contract) → OpenRouter's image_config
# aspect_ratio strings.
_ASPECT_RATIOS = {
    "square": "1:1",
    "landscape": "16:9",
    "portrait": "9:16",
}

# Gemini Flash Image accepts up to 3 input images per prompt; clamp references
# so we never overflow the model's limit.
_MAX_REFERENCE_IMAGES = 3

_REQUEST_TIMEOUT = 180.0


def _load_image_gen_config() -> Dict[str, Any]:
    """Read the ``image_gen`` section from config.yaml (``{}`` on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 - config is best-effort
        logger.debug("could not load image_gen config: %s", exc)
        return {}


def _to_image_url_part(ref: str) -> Optional[str]:
    """Turn a reference (local path or http URL) into an ``image_url`` value.

    Remote URLs pass through unchanged; local files are inlined as base64 data
    URIs so the request is self-contained (the provider endpoint can't reach a
    path on our disk). Returns ``None`` when the reference can't be read.
    """
    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.debug("could not read reference image %s: %s", ref, exc)
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_images(payload: Dict[str, Any]) -> List[str]:
    """Pull generated image URLs from a chat-completions response.

    OpenRouter returns generated images under
    ``choices[0].message.images[].image_url.url`` (typically a base64 data URI).
    """
    out: List[str] = []
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return out
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        images = message.get("images") if isinstance(message, dict) else None
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            image_url = image.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if isinstance(url, str) and url.strip():
                out.append(url.strip())
    return out


class OpenRouterCompatImageProvider(ImageGenProvider):
    """Image generation over an OpenRouter-compatible chat-completions endpoint.

    Instantiated once per backend (OpenRouter, Nous Portal). The two differ only
    in which runtime provider supplies ``(base_url, api_key)`` and in the config
    namespace used for the model override.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        display_name: str,
        runtime_name: str,
        config_key: str,
        model_env_var: str,
        setup_schema: Dict[str, Any],
    ) -> None:
        self._name = provider_name
        self._display = display_name
        self._runtime_name = runtime_name
        self._config_key = config_key
        self._model_env_var = model_env_var
        self._setup_schema = setup_schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display

    def _resolve_runtime(self) -> Dict[str, Any]:
        """Resolve ``(base_url, api_key)`` via the shared runtime resolver."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        return resolve_runtime_provider(requested=self._runtime_name)

    def is_available(self) -> bool:
        try:
            runtime = self._resolve_runtime()
        except Exception as exc:  # noqa: BLE001 - treat resolution failure as unavailable
            logger.debug("%s runtime resolution failed: %s", self._name, exc)
            return False
        return bool(str(runtime.get("api_key") or "").strip())

    def capabilities(self) -> Dict[str, Any]:
        # Both text-to-image and image-to-image (reference grounding) — the
        # latter is what makes this backend usable for pet sprite rows.
        return {
            "modalities": ["text", "image"],
            "max_reference_images": _MAX_REFERENCE_IMAGES,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Gemini 2.5 Flash Image (nano-banana)",
                "strengths": "Reference-grounded edits; aspect-ratio control",
            }
        ]

    def default_model(self) -> Optional[str]:
        return self._resolve_model()

    def get_setup_schema(self) -> Dict[str, Any]:
        return dict(self._setup_schema)

    def _resolve_model(self) -> str:
        """Pick the image model: env override → config → :data:`DEFAULT_MODEL`."""
        env_override = os.environ.get(self._model_env_var, "").strip()
        if env_override:
            return env_override
        cfg = _load_image_gen_config()
        scoped = cfg.get(self._config_key) if isinstance(cfg.get(self._config_key), dict) else {}
        if isinstance(scoped, dict):
            value = scoped.get("model")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return DEFAULT_MODEL

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import requests

        try:
            runtime = self._resolve_runtime()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not resolve {self._display} credentials: {exc}",
                error_type="missing_api_key",
                provider=self._name,
                aspect_ratio=aspect_ratio,
            )
        api_key = str(runtime.get("api_key") or "").strip()
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        if not api_key or not base_url:
            return error_response(
                error=(
                    f"No {self._display} credentials found. "
                    f"Configure {self._display} in `hermes tools` → Image Generation."
                ),
                error_type="missing_api_key",
                provider=self._name,
                aspect_ratio=aspect_ratio,
            )

        model_id = self._resolve_model()
        aspect = resolve_aspect_ratio(aspect_ratio)
        or_aspect = _ASPECT_RATIOS.get(aspect, "1:1")

        # Collect every reference: the pet generator passes local paths via the
        # ``reference_images`` kwarg; the generic tool surface uses ``image_url``
        # / ``reference_image_urls``. Accept all three.
        references: List[str] = []
        for ref in kwargs.get("reference_images") or []:
            references.append(str(ref))
        if image_url:
            references.append(str(image_url))
        for ref in reference_image_urls or []:
            references.append(str(ref))

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for ref in references[:_MAX_REFERENCE_IMAGES]:
            part = _to_image_url_part(ref)
            if part:
                content.append({"type": "image_url", "image_url": {"url": part}})

        payload: Dict[str, Any] = {
            "model": model_id,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": content}],
            "image_config": {"aspect_ratio": or_aspect},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (harmless against Nous Portal).
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
            "X-Title": "Hermes Agent",
        }

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:  # noqa: BLE001
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("%s image gen failed (%d): %s", self._name, status, err_msg)
            return error_response(
                error=f"{self._display} image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"{self._display} image generation timed out "
                f"({int(_REQUEST_TIMEOUT)}s)",
                error_type="timeout",
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"{self._display} connection error: {exc}",
                error_type="connection_error",
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"{self._display} returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        images = _extract_images(result)
        if not images:
            # A response with text but no image usually means the model didn't
            # honor image output (wrong model or modalities); surface that.
            return error_response(
                error=(
                    f"{self._display} returned no image. Ensure the model "
                    f"'{model_id}' supports image output."
                ),
                error_type="empty_response",
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = images[0]
        try:
            if first.startswith("data:"):
                b64 = first.split(",", 1)[1] if "," in first else ""
                saved_path = save_b64_image(b64, prefix=f"{self._name}_gen")
            else:
                saved_path = save_url_image(first, prefix=f"{self._name}_gen")
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not save generated image: {exc}",
                error_type="io_error",
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self._name,
        )


def _build_providers() -> List[OpenRouterCompatImageProvider]:
    return [
        OpenRouterCompatImageProvider(
            provider_name="openrouter",
            display_name="OpenRouter",
            runtime_name="openrouter",
            config_key="openrouter",
            model_env_var="OPENROUTER_IMAGE_MODEL",
            setup_schema={
                "name": "OpenRouter (image)",
                "badge": "paid",
                "tag": "Gemini Flash Image & more via OpenRouter; uses OPENROUTER_API_KEY",
                "env_vars": [
                    {
                        "key": "OPENROUTER_API_KEY",
                        "prompt": "OpenRouter API key",
                        "url": "https://openrouter.ai/keys",
                    }
                ],
            },
        ),
        OpenRouterCompatImageProvider(
            provider_name="nous",
            display_name="Nous Portal",
            runtime_name="nous",
            config_key="nous",
            model_env_var="NOUS_IMAGE_MODEL",
            setup_schema={
                "name": "Nous Portal (image)",
                "badge": "subscription",
                "tag": "Reference-grounded image generation via Nous Portal (OpenRouter-backed)",
                "env_vars": [],
                "requires_nous_auth": True,
            },
        ),
    ]


def register(ctx: Any) -> None:
    """Register the OpenRouter + Nous Portal image gen providers."""
    for provider in _build_providers():
        ctx.register_image_gen_provider(provider)

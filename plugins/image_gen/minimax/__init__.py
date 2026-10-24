"""MiniMax image generation backends (global + China).

Exposes MiniMax's native ``POST /v1/image_generation`` endpoint as
:class:`ImageGenProvider` implementations. Two regional providers are
registered so both the international and China APIs are selectable in the
``hermes tools`` picker:

- ``minimax``    -> https://api.minimax.io/v1/image_generation    (MINIMAX_API_KEY)
- ``minimax-cn`` -> https://api.minimaxi.com/v1/image_generation  (MINIMAX_CN_API_KEY)

Both regions expose the same model catalog:

- ``image-01`` — text-to-image generation.
- ``image-01-live`` — image-to-image / subject editing. When the agent
  passes ``image_url`` (or ``reference_image_urls``), the provider routes
  to the image-to-image flow and sends the source image(s) in the
  ``subject_reference`` request field (``type: "character"``).

The API returns generated content in ``data.image_urls`` (for
``response_format="url"``) or ``data.image_base64`` (for
``response_format="base64"``). URLs expire after 24 hours, so the provider
caches the first returned asset locally at tool-completion time, matching
the ephemeral-URL guard used by the other image-gen plugins.

Docs: https://platform.minimax.io/docs/api-reference/image-generation-t2i
      https://platform.minimax.io/docs/api-reference/image-generation-i2i
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from agent.file_safety import raise_if_read_blocked
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
# Model catalog + regional endpoints
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "image-01"
I2I_MODEL = "image-01-live"
REQUEST_TIMEOUT_SECONDS = 120.0

_MODELS: Dict[str, Dict[str, str]] = {
    "image-01": {
        "display": "Image 01",
        "strengths": "Text-to-image generation",
    },
    "image-01-live": {
        "display": "Image 01 Live",
        "strengths": "Image-to-image / subject editing",
    },
}

# MiniMax aspect ratios — map Hermes' three canonical ratios onto the native
# enum. The API also accepts 4:3, 3:4, 3:2, 2:3 and 21:9, but the tool layer
# only surfaces landscape/square/portrait, so those three are sufficient.
_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

_OUTPUT_FORMATS = {"url", "base64"}

# image-01-live subject editing accepts a small number of subject reference
# images; cap the total so a runaway reference list fails with a clear
# message instead of a generic API 4xx.
MAX_SUBJECT_REFERENCES = 3

_REGIONAL_PROVIDERS: Dict[str, Dict[str, str]] = {
    "minimax": {
        "display_name": "MiniMax",
        "api_key_env": "MINIMAX_API_KEY",
        "endpoint": "https://api.minimax.io/v1/image_generation",
        "docs_url": "https://platform.minimax.io/docs/api-reference/image-generation-i2i",
    },
    "minimax-cn": {
        "display_name": "MiniMax (China)",
        "api_key_env": "MINIMAX_CN_API_KEY",
        "endpoint": "https://api.minimaxi.com/v1/image_generation",
        "docs_url": "https://platform.minimaxi.com/docs/api-reference/image-generation-i2i",
    },
}


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------


def _load_minimax_config() -> Dict[str, Any]:
    """Read ``image_gen.minimax`` from config.yaml (returns {} on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        minimax_section = section.get("minimax") if isinstance(section, dict) else None
        return minimax_section if isinstance(minimax_section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen.minimax config: %s", exc)
        return {}


def _resolve_model(candidate: Any) -> str:
    """Return a supported model id, falling back to the documented default."""
    if isinstance(candidate, str):
        normalized = candidate.strip()
        if normalized in _MODELS:
            return normalized
    return DEFAULT_MODEL


def _split_base64_image(value: str) -> Tuple[str, str]:
    """Return raw base64 data and a cache-file extension from a b64 value.

    Accepts bare base64 or a ``data:image/...;base64,...`` data URL.
    """
    extension = "png"
    if value.startswith("data:image/") and "," in value:
        header, value = value.split(",", 1)
        subtype = header.split("data:image/", 1)[1].split(";", 1)[0].lower()
        if subtype in {"jpeg", "jpg", "png", "webp", "gif"}:
            extension = "jpg" if subtype == "jpeg" else subtype
    return value, extension


def _subject_reference_image(source: str) -> Dict[str, str]:
    """Build a MiniMax ``subject_reference`` entry from a URL or local path.

    ``image_file`` accepts a public HTTPS URL or a base64 data URL; local
    file paths are read and encoded into a data URL (JPG/JPEG/PNG per the
    MiniMax image-to-image spec).
    """
    source = source.strip()
    lower = source.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return {"type": "character", "image_file": source}

    raise_if_read_blocked(source)
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(
            "image_url must be a public HTTPS URL, data URI, or an absolute "
            f"file path (got {source!r})"
        )
    extension = (path.suffix.lstrip(".") or "png").lower()
    if extension == "jpg":
        extension = "jpeg"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "type": "character",
        "image_file": f"data:image/{extension};base64,{encoded}",
    }


def _first_image_value(data: Any, key: str) -> Optional[str]:
    """Return the first image value from a ``data`` dict (list or scalar)."""
    if not isinstance(data, dict):
        return None
    value = data.get(key)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_error_message(response: requests.Response) -> str:
    """Best-effort: pull a human message out of a MiniMax error response."""
    try:
        error_body = response.json()
    except ValueError:
        return str(getattr(response, "text", ""))[:300]
    if not isinstance(error_body, dict):
        return ""
    base_resp = error_body.get("base_resp") or {}
    if isinstance(base_resp, dict) and base_resp.get("status_msg"):
        return str(base_resp["status_msg"])
    error_value = error_body.get("error") or {}
    if isinstance(error_value, dict) and error_value.get("message"):
        return str(error_value["message"])
    return str(error_body.get("message") or "")[:300]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax native ``/v1/image_generation`` backend for one API region."""

    def __init__(self, provider_name: str = "minimax") -> None:
        if provider_name not in _REGIONAL_PROVIDERS:
            raise ValueError(f"Unsupported MiniMax image provider: {provider_name}")
        self._provider_name = provider_name
        self._provider_config = _REGIONAL_PROVIDERS[provider_name]

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def display_name(self) -> str:
        return self._provider_config["display_name"]

    def is_available(self) -> bool:
        return bool(os.environ.get(self._provider_config["api_key_env"], "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": metadata["display"],
                "strengths": metadata["strengths"],
            }
            for model_id, metadata in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        # image-01-live supports image-to-image via subject_reference.
        return {
            "modalities": ["text", "image"],
            "max_reference_images": 2,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        api_key_env = self._provider_config["api_key_env"]
        return {
            "name": self.display_name,
            "badge": "paid",
            "tag": (
                "image-01 / image-01-live — text-to-image and image-to-image "
                "(subject editing)"
            ),
            "env_vars": [
                {
                    "key": api_key_env,
                    "prompt": f"{self.display_name} API key",
                    "url": self._provider_config["docs_url"],
                }
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image, or edit a subject image (image-to-image).

        Routing follows the provider-ABC contract: when ``image_url`` (or
        ``reference_image_urls``) is provided, the call goes to the
        image-to-image flow — the ``image-01-live`` model with the source
        image(s) in the ``subject_reference`` field; otherwise it is plain
        text-to-image with the configured/default model.
        """
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )
        if len(prompt) > 1500:
            return error_response(
                error=(
                    f"Prompt is {len(prompt)} characters; MiniMax image "
                    "generation accepts at most 1500."
                ),
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        api_key_env = self._provider_config["api_key_env"]
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            return error_response(
                error=(
                    f"{api_key_env} not set. Run `hermes tools` → Image "
                    f"Generation → {self.display_name} to configure it."
                ),
                error_type="auth_required",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        response_format = str(kwargs.get("response_format") or "url").strip().lower()
        if response_format not in _OUTPUT_FORMATS:
            return error_response(
                error="response_format must be 'url' or 'base64'",
                error_type="invalid_argument",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)

        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        if is_edit:
            model = I2I_MODEL
            if len(sources) > MAX_SUBJECT_REFERENCES:
                return error_response(
                    error=(
                        "MiniMax image-to-image accepts at most "
                        f"{MAX_SUBJECT_REFERENCES} subject reference images"
                    ),
                    error_type="too_many_references",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            try:
                subject_reference = [
                    _subject_reference_image(source) for source in sources
                ]
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not load source image for editing: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            model = _resolve_model(kwargs.get("model"))
            subject_reference = None

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": _ASPECT_RATIOS.get(aspect, "1:1"),
            "response_format": response_format,
            "n": int(kwargs.get("n", 1)),
        }
        if subject_reference is not None:
            payload["subject_reference"] = subject_reference
        seed = kwargs.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            payload["seed"] = seed
        prompt_optimizer = kwargs.get("prompt_optimizer")
        if prompt_optimizer is not None:
            payload["prompt_optimizer"] = bool(prompt_optimizer)
        if not is_edit:
            # width/height only apply to image-01 text-to-image.
            width = kwargs.get("width")
            height = kwargs.get("height")
            if isinstance(width, int) and not isinstance(width, bool):
                payload["width"] = width
            if isinstance(height, int) and not isinstance(height, bool):
                payload["height"] = height

        try:
            response = requests.post(
                self._provider_config["endpoint"],
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.Timeout:
            return error_response(
                error=f"{self.display_name} image generation timed out",
                error_type="timeout",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            message = _extract_error_message(resp) if resp is not None else str(exc)
            logger.error(
                "MiniMax image generation failed (%d): %s", status, message
            )
            return error_response(
                error=(
                    f"{self.display_name} image generation failed ({status}): "
                    f"{message or 'request failed'}"
                ),
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"{self.display_name} connection error: {exc}",
                error_type="connection_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.RequestException as exc:
            logger.debug("MiniMax image generation request failed", exc_info=True)
            return error_response(
                error=f"{self.display_name} image generation failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if response.status_code >= 400:
            message = _extract_error_message(response)
            return error_response(
                error=(
                    f"{self.display_name} image generation HTTP "
                    f"{response.status_code}: {message or 'request failed'}"
                ),
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            body = response.json()
        except ValueError:
            return error_response(
                error=f"{self.display_name} returned a non-JSON response",
                error_type="invalid_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not isinstance(body, dict):
            return error_response(
                error=f"{self.display_name} returned an invalid response object",
                error_type="invalid_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        base_resp = body.get("base_resp") or {}
        status_code = (
            base_resp.get("status_code") if isinstance(base_resp, dict) else None
        )
        if status_code not in (None, 0):
            status_message = base_resp.get("status_msg") or "request failed"
            return error_response(
                error=f"{self.display_name} error {status_code}: {status_message}",
                error_type="provider_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = body.get("data") or {}
        prefix = f"minimax_{model}"
        if response_format == "base64":
            image_value = _first_image_value(data, "image_base64")
            if not image_value:
                return error_response(
                    error=f"{self.display_name} returned no base64 image data",
                    error_type="empty_response",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            raw_base64, extension = _split_base64_image(image_value)
            try:
                image_ref = str(
                    save_b64_image(raw_base64, prefix=prefix, extension=extension)
                )
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not save generated image to cache: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            image_url_value = _first_image_value(data, "image_urls")
            if not image_url_value:
                return error_response(
                    error=f"{self.display_name} returned no image URLs",
                    error_type="empty_response",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            parsed = urlparse(image_url_value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return error_response(
                    error=f"{self.display_name} returned an invalid image URL",
                    error_type="invalid_response",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            try:
                image_ref = str(save_url_image(image_url_value, prefix=prefix))
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "MiniMax image URL could not be cached; returning the URL: %s",
                    exc,
                )
                image_ref = image_url_value

        metadata = body.get("metadata") or {}
        extra: Dict[str, Any] = {"response_format": response_format}
        if isinstance(metadata, dict):
            for source_key in ("success_count", "failed_count"):
                if metadata.get(source_key) is not None:
                    extra[source_key] = metadata[source_key]

        return success_response(
            image=image_ref,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register both regional MiniMax image providers."""
    for provider_name in _REGIONAL_PROVIDERS:
        ctx.register_image_gen_provider(MiniMaxImageGenProvider(provider_name))

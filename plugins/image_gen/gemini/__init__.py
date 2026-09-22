"""Native Google AI Studio image generation backend.

This backend calls Gemini's ``generateContent`` REST API directly with a
``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``. It intentionally does not depend on
FAL, OpenRouter, or the Gemini text provider's OpenAI-compatible transport.
Generated inline image data is materialized under Hermes' image cache so the
rest of the image-generation pipeline sees the same stable path as other
providers.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-image"
_MAX_REFERENCE_IMAGES = 14
_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}
_MODELS: Dict[str, Dict[str, str]] = {
    "gemini-3.1-flash-lite-image": {
        "display": "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "speed": "fast",
        "strengths": "Lowest latency and cost; efficient image generation and editing",
    },
    "gemini-3.1-flash-image": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "speed": "balanced",
        "strengths": "General-purpose generation, 4K output, and multi-reference editing",
    },
    "gemini-3-pro-image": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "premium",
        "strengths": "Highest fidelity, reasoning, 4K output, and search grounding",
    },
    "gemini-2.5-flash-image": {
        "display": "Nano Banana (Gemini 2.5 Flash Image)",
        "speed": "fast",
        "strengths": "Legacy fast generation and image editing",
    },
}
_MODEL_REFERENCE_LIMITS = {
    "gemini-2.5-flash-image": 3,
}


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml without making configuration fatal."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        section = config.get("image_gen") if isinstance(config, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 - config is best effort
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model(explicit: Optional[str] = None) -> str:
    """Resolve the model: call override, env, scoped config, global config."""
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    env_model = os.environ.get("GEMINI_IMAGE_MODEL", "").strip()
    if env_model:
        return env_model

    config = _load_image_gen_config()
    scoped = config.get("gemini")
    if isinstance(scoped, dict):
        model = scoped.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()

    model = config.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return DEFAULT_MODEL


def _api_key() -> Optional[str]:
    """Return the direct Google key, keeping the text-provider precedence."""
    try:
        return (
            get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY") or ""
        ).strip() or None
    except Exception as exc:  # noqa: BLE001 - scoped-secret failures are auth failures
        logger.debug("Google AI Studio credential lookup failed: %s", exc)
        return None


def _load_reference_image(reference: str) -> Tuple[bytes, str]:
    """Load a URL, data URI, or safe local path as ``(bytes, mime_type)``."""
    reference = reference.strip()
    lower = reference.lower()

    if lower.startswith(("http://", "https://")):
        import requests

        response = requests.get(reference, timeout=60)
        response.raise_for_status()
        content_type = (
            (response.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        )
        return response.content, content_type if content_type.startswith(
            "image/"
        ) else "image/png"

    if lower.startswith("data:"):
        header, separator, encoded = reference.partition(",")
        if not separator:
            raise ValueError("image data URI is missing its payload")
        mime = header[5:].split(";", 1)[0].strip() or "image/png"
        return base64.b64decode(encoded, validate=True), mime

    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(reference)
    path = Path(reference)
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return data, mime if mime.startswith("image/") else "image/png"


def _error_message(response: Any) -> str:
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(body, dict) and body.get("message"):
            return str(body["message"])
    except Exception:
        pass
    return getattr(response, "reason", None) or "unknown Google AI Studio error"


class GeminiImageGenProvider(ImageGenProvider):
    """Direct Gemini image generation through Google AI Studio."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google AI Studio"

    def is_available(self) -> bool:
        return _api_key() is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": metadata["display"],
                "speed": metadata["speed"],
                "strengths": metadata["strengths"],
                "price": "varies",
            }
            for model_id, metadata in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "max_reference_images": _MAX_REFERENCE_IMAGES,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Google AI Studio (direct)",
            "badge": "API key",
            "tag": "Native Gemini image generation with GOOGLE_API_KEY or GEMINI_API_KEY",
            "env_vars": [
                {
                    "key": "GOOGLE_API_KEY",
                    "prompt": "Google AI Studio API key",
                    "url": "https://aistudio.google.com/apikey",
                },
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Gemini API key (alternative)",
                    "url": "https://aistudio.google.com/apikey",
                },
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
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = _resolve_model(kwargs.get("model"))

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                model=model,
                aspect_ratio=aspect,
            )

        api_key = _api_key()
        if not api_key:
            return error_response(
                error=(
                    "GOOGLE_API_KEY or GEMINI_API_KEY not set. Run `hermes tools` "
                    "→ Image Generation → Google AI Studio to configure it."
                ),
                error_type="auth_required",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        reference_limit = _MODEL_REFERENCE_LIMITS.get(model, _MAX_REFERENCE_IMAGES)
        sources = sources[:reference_limit]

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for source in sources:
            try:
                data, mime = _load_reference_image(source)
            except Exception as exc:  # noqa: BLE001 - stable tool error
                return error_response(
                    error=f"Could not load reference image for editing: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            parts.append({
                "inlineData": {
                    "mimeType": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": _ASPECT_RATIOS[aspect]},
            },
        }
        timeout = kwargs.get("timeout", 180)
        try:
            timeout = max(1.0, float(timeout))
        except (TypeError, ValueError):
            timeout = 180

        endpoint = f"{BASE_URL}/models/{quote(model, safe='')}:generateContent"
        try:
            import requests

            response = requests.post(
                endpoint,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else 0
            message = _error_message(response) if response is not None else str(exc)
            error_type = "auth_error" if status in (401, 403) else "api_error"
            return error_response(
                error=f"Google AI Studio image generation failed ({status}): {message}",
                error_type=error_type,
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"Google AI Studio image generation timed out ({int(timeout)}s)",
                error_type="timeout",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.RequestException as exc:
            return error_response(
                error=f"Google AI Studio request failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Google AI Studio returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_data = None
        image_mime = "image/png"
        response_text: List[str] = []
        candidates = body.get("candidates") if isinstance(body, dict) else None
        for candidate in candidates if isinstance(candidates, list) else []:
            content = candidate.get("content") if isinstance(candidate, dict) else None
            response_parts = content.get("parts") if isinstance(content, dict) else None
            for part in response_parts if isinstance(response_parts, list) else []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    response_text.append(part["text"].strip())
                inline = part.get("inlineData") if isinstance(part, dict) else None
                if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                    image_data = inline["data"]
                    image_mime = str(inline.get("mimeType") or image_mime)
                    break
            if image_data:
                break

        if not image_data:
            detail = " ".join(text for text in response_text if text)
            message = "Google AI Studio returned no inline image data"
            if detail:
                message = f"{message}: {detail}"
            return error_response(
                error=message,
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extension = {
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
        }.get(image_mime.lower(), "png")
        try:
            image_path = save_b64_image(
                image_data,
                prefix=f"gemini_{model.replace('/', '_')}",
                extension=extension,
            )
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not save generated image: {exc}",
                error_type="io_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(image_path),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="image" if sources else "text",
            extra={"api": "google-ai-studio"},
        )


def register(ctx: Any) -> None:
    """Register the direct Google AI Studio provider."""
    ctx.register_image_gen_provider(GeminiImageGenProvider())

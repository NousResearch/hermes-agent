"""OpenAI-compatible local proxy image generation backend.

Routes ``image_generate`` through a local OpenAI-compatible image
proxy. Unlike the prompt-only OpenAI/xAI backends, this provider supports an
optional local reference image via ``image_url`` for models whose catalog entry
advertises ``"image"`` in ``input_modalities``.

Configuration precedence:

1. ``OPENAI_COMPATIBLE_IMAGE_MODEL`` env var
2. ``image_gen.openai_compatible.model`` in config.yaml
3. top-level ``image_gen.model`` when present
4. first catalog model, preferring text+image models

Base URL precedence:

1. ``OPENAI_COMPATIBLE_IMAGE_BASE_URL`` env var
2. ``image_gen.openai_compatible.base_url``
3. ``OPENAI_BASE_URL`` env var
4. ``http://localhost:20128/v1``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

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

DEFAULT_BASE_URL = "http://localhost:20128/v1"
DEFAULT_TIMEOUT = 420

_SIZES = {
    "landscape": "1024x1024",
    "square": "1024x1024",
    "portrait": "1024x1024",
}


def _load_openai_compatible_config() -> Dict[str, Any]:
    """Read ``image_gen.openai_compatible`` from config.yaml (returns {} on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        sub = section.get("openai_compatible") if isinstance(section, dict) else None
        return sub if isinstance(sub, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.openai_compatible config: %s", exc)
        return {}


def _load_image_gen_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _base_url() -> str:
    cfg = _load_openai_compatible_config()
    raw = (
        os.getenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL")
        or cfg.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return str(raw).strip().rstrip("/")


def _catalog_url() -> str:
    return f"{_base_url()}/images/generations"


def _request_timeout() -> int:
    """Return image-generation request timeout in seconds.

    OpenAI-compatible image models can legitimately take several minutes. The
    provider default is intentionally longer than generic tool/network defaults,
    while still allowing users to override it in config.yaml.
    """
    cfg = _load_openai_compatible_config()
    top_cfg = _load_image_gen_config()
    raw = cfg.get("timeout") or top_cfg.get("timeout")
    if raw is None:
        return DEFAULT_TIMEOUT
    try:
        timeout = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    return timeout if timeout > 0 else DEFAULT_TIMEOUT


def _normalise_catalog_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("models") or []
    else:
        data = payload
    if not isinstance(data, list):
        return []
    models: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("name")
            if isinstance(model_id, str) and model_id.strip():
                models.append(item)
    return models


def _fetch_catalog() -> List[Dict[str, Any]]:
    response = requests.get(_catalog_url(), timeout=10)
    response.raise_for_status()
    return _normalise_catalog_payload(response.json())


def _model_id(entry: Dict[str, Any]) -> str:
    value = entry.get("id") or entry.get("model") or entry.get("name")
    return str(value)


def _input_modalities(entry: Dict[str, Any]) -> List[str]:
    raw = (
        entry.get("input_modalities")
        or entry.get("inputModalities")
        or entry.get("modalities")
        or []
    )
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _supports_image_input(entry: Dict[str, Any]) -> bool:
    return "image" in {m.lower() for m in _input_modalities(entry)}


def _supported_sizes(entry: Dict[str, Any]) -> List[str]:
    raw = entry.get("supported_sizes") or entry.get("supportedSizes") or []
    if isinstance(raw, list):
        return [str(x) for x in raw if isinstance(x, str) and x]
    return []


def _resolve_model(catalog: List[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    cfg = _load_openai_compatible_config()
    top_cfg = _load_image_gen_config()
    configured = (
        os.getenv("OPENAI_COMPATIBLE_IMAGE_MODEL")
        or cfg.get("model")
        or top_cfg.get("model")
    )
    if isinstance(configured, str) and configured.strip():
        configured = configured.strip()
        for entry in catalog:
            if _model_id(entry) == configured:
                return configured, entry
        # Ignore stale top-level image_gen.model values from other backends;
        # only the OpenAI-compatible image proxy-specific/env override should force an unknown id.
        openai_compatible_specific = os.getenv("OPENAI_COMPATIBLE_IMAGE_MODEL") or cfg.get("model")
        if isinstance(openai_compatible_specific, str) and openai_compatible_specific.strip():
            return configured, None

    if not catalog:
        return "", None

    for entry in catalog:
        if _supports_image_input(entry):
            return _model_id(entry), entry
    return _model_id(catalog[0]), catalog[0]


def _size_for_aspect(aspect: str, entry: Optional[Dict[str, Any]]) -> str:
    preferred = _SIZES.get(aspect, _SIZES["square"])
    supported = _supported_sizes(entry or {})
    if not supported or preferred in supported:
        return preferred
    return supported[0]


def _normalise_reference_uri(image_url: Optional[str]) -> Optional[str]:
    raw = (image_url or "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError("Reference image must be a local file path or file:// URL")
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("Reference image file:// URL must point to localhost")
        path = Path(unquote(parsed.path))
    else:
        path = Path(raw).expanduser()

    resolved = path.resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"Reference image file not found: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise PermissionError(f"Reference image file is not readable: {resolved}")
    return resolved.as_uri()


class OpenAICompatibleImageGenProvider(ImageGenProvider):
    """Local OpenAI-compatible image backend with optional reference-image support."""

    @property
    def name(self) -> str:
        return "openai-compatible"

    @property
    def display_name(self) -> str:
        return "OpenAI-compatible image proxy"

    def is_available(self) -> bool:
        try:
            response = requests.get(_catalog_url(), timeout=5)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            catalog = _fetch_catalog()
        except Exception as exc:
            logger.debug("Could not fetch OpenAI-compatible image model catalog: %s", exc)
            return []

        rows: List[Dict[str, Any]] = []
        for entry in catalog:
            model_id = _model_id(entry)
            modalities = _input_modalities(entry)
            rows.append({
                "id": model_id,
                "display": str(entry.get("display") or entry.get("name") or model_id),
                "speed": str(entry.get("speed") or ""),
                "strengths": " + ".join(modalities) if modalities else "",
                "price": str(entry.get("price") or "local proxy"),
            })
        return rows

    def default_model(self) -> Optional[str]:
        try:
            model_id, _ = _resolve_model(_fetch_catalog())
            return model_id or None
        except Exception:
            return None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI-compatible image proxy",
            "badge": "local",
            "tag": "OpenAI-compatible local image proxy; supports reference images on image-input models",
            "env_vars": [
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_BASE_URL",
                    "prompt": "OpenAI-compatible image base URL",
                    "default": DEFAULT_BASE_URL,
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-compatible",
                aspect_ratio=aspect,
            )

        reference_uri: Optional[str] = None
        raw_reference = kwargs.get("image_url")
        if raw_reference:
            try:
                reference_uri = _normalise_reference_uri(str(raw_reference))
            except Exception as exc:
                return error_response(
                    error=str(exc),
                    error_type="invalid_reference_image",
                    provider="openai-compatible",
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        try:
            catalog = _fetch_catalog()
        except Exception as exc:
            return error_response(
                error=f"Could not fetch OpenAI-compatible image model catalog: {exc}",
                error_type="catalog_error",
                provider="openai-compatible",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        model_id, entry = _resolve_model(catalog)
        if not model_id:
            return error_response(
                error="OpenAI-compatible image catalog is empty",
                error_type="empty_catalog",
                provider="openai-compatible",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if reference_uri and entry is not None and not _supports_image_input(entry):
            return error_response(
                error=f"OpenAI-compatible image model '{model_id}' does not advertise image input support",
                error_type="unsupported_reference_image",
                provider="openai-compatible",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        size = _size_for_aspect(aspect, entry)
        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "size": size,
        }
        if reference_uri:
            payload["image_url"] = reference_uri

        try:
            timeout = _request_timeout()
            response = requests.post(
                f"{_base_url()}/images/generations",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.Timeout:
            return error_response(
                error=f"OpenAI-compatible image generation timed out ({_request_timeout()}s)",
                error_type="timeout",
                provider="openai-compatible",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            return error_response(
                error=f"OpenAI-compatible image generation failed: {exc}",
                error_type="api_error",
                provider="openai-compatible",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or not data:
            return error_response(
                error="OpenAI-compatible image backend returned no image data",
                error_type="empty_response",
                provider="openai-compatible",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        image_ref = None
        if isinstance(first, dict):
            image_ref = first.get("b64_json") or first.get("url") or first.get("image_url")
        if not image_ref:
            return error_response(
                error="OpenAI-compatible image backend response contained no image data (expected url, image_url, or b64_json)",
                error_type="empty_response",
                provider="openai-compatible",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Decode base64 data URIs (e.g. "data:image/png;base64,...") to local files
        # so the image_gen tool can deliver them as attachments.
        if isinstance(image_ref, str) and image_ref.startswith("data:"):
            try:
                header, b64_payload = image_ref.split(",", 1)
                ext = "png"
                if "jpeg" in header or "jpg" in header:
                    ext = "jpg"
                elif "webp" in header:
                    ext = "webp"
                saved = save_b64_image(b64_payload, prefix="openai_compat", extension=ext)
                image_ref = str(saved)
            except Exception as exc:
                return error_response(
                    error=f"Failed to decode data URI from image backend: {exc}",
                    error_type="decode_error",
                    provider="openai-compatible",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif isinstance(first, dict) and first.get("b64_json"):
            # Raw b64 without data URI wrapper
            try:
                saved = save_b64_image(first["b64_json"], prefix="openai_compat")
                image_ref = str(saved)
            except Exception as exc:
                return error_response(
                    error=f"Failed to decode b64_json from image backend: {exc}",
                    error_type="decode_error",
                    provider="openai-compatible",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        return success_response(
            image=str(image_ref),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-compatible",
            extra={"size": size, "reference_image": bool(reference_uri)},
        )


def register(ctx: Any) -> None:
    """Plugin entry point — register the OpenAI-compatible image provider."""
    ctx.register_image_gen_provider(OpenAICompatibleImageGenProvider())

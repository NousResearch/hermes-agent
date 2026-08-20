"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Identical model catalog and tier semantics to the ``openai`` image-gen plugin
(``gpt-image-2`` at low/medium/high quality), but routes the request through
Codex's dedicated Images generation/edit endpoints. This lets users who are already
authenticated with Codex/ChatGPT generate images without configuring a
separate ``OPENAI_API_KEY``.

Selection precedence for the tier (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai-codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``

Output is saved as PNG under ``$HERMES_HOME/cache/images/``. Source images for
image-to-image/editing are sent as Images API data URLs.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# HTTP errors from Codex must surface verbatim. Do not classify a wire-level
# request rejection as an account entitlement problem; see #19505 and #49008.

_MAX_ERROR_BODY_CHARS = 500
_MAX_DIAGNOSTIC_STRING_CHARS = 160
_MAX_DIAGNOSTIC_ITEMS = 8
_MAX_DIAGNOSTIC_DEPTH = 3
_SENSITIVE_DIAGNOSTIC_KEYS = frozenset({
    "api_key",
    "authorization",
    "b64_json",
    "cookie",
    "credential",
    "image",
    "image_url",
    "images",
    "password",
    "refresh_token",
    "secret",
    "token",
})


def _sanitize_diagnostic_text(value: str) -> str:
    """Redact common credential/image encodings from diagnostic text."""
    text = re.sub(
        r"(?i)data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=_-]+",
        "[redacted image data]",
        value,
    )
    text = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer [redacted]", text)
    text = re.sub(r"\b(?:sk|sess)-[a-zA-Z0-9_-]{8,}\b", "[redacted credential]", text)
    text = re.sub(
        r"\beyJ[a-zA-Z0-9_-]{10,}(?:\.[a-zA-Z0-9_-]+){1,2}\b",
        "[redacted credential]",
        text,
    )
    text = re.sub(
        r"(?<![a-zA-Z0-9+/=_-])[a-zA-Z0-9+/=_-]{64,}"
        r"(?![a-zA-Z0-9+/=_-])",
        "[redacted blob]",
        text,
    )
    return text[:_MAX_DIAGNOSTIC_STRING_CHARS]


def _sanitize_diagnostic_value(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded JSON-compatible value safe for error diagnostics."""
    if depth >= _MAX_DIAGNOSTIC_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= _MAX_DIAGNOSTIC_ITEMS:
                break
            safe_key = _sanitize_diagnostic_text(str(key))
            key_lc = safe_key.lower()
            if any(marker in key_lc for marker in _SENSITIVE_DIAGNOSTIC_KEYS):
                sanitized[safe_key] = "[redacted]"
            else:
                sanitized[safe_key] = _sanitize_diagnostic_value(child, depth=depth + 1)
        if len(value) > _MAX_DIAGNOSTIC_ITEMS:
            sanitized["..."] = f"<{len(value) - _MAX_DIAGNOSTIC_ITEMS} more keys>"
        return sanitized
    if isinstance(value, list):
        sanitized_items = [
            _sanitize_diagnostic_value(item, depth=depth + 1)
            for item in value[:_MAX_DIAGNOSTIC_ITEMS]
        ]
        if len(value) > _MAX_DIAGNOSTIC_ITEMS:
            sanitized_items.append(f"<{len(value) - _MAX_DIAGNOSTIC_ITEMS} more items>")
        return sanitized_items
    if isinstance(value, str):
        return _sanitize_diagnostic_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def _summarize_diagnostic_body(body: str) -> str:
    """Return a bounded, sanitized excerpt of an unexpected response body."""
    text = body or ""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        summary = _sanitize_diagnostic_text(text)
    else:
        summary = json.dumps(
            _sanitize_diagnostic_value(payload),
            ensure_ascii=True,
            separators=(",", ":"),
        )
    return summary[:_MAX_ERROR_BODY_CHARS]


def _summarize_error_body(body: str) -> str:
    """Return a bounded, information-preserving summary of an error body.

    Prefers the parsed ``error.message`` field, because a blind head-truncation
    of the raw body can cut the actual message off entirely — Codex error
    payloads sometimes carry hundreds of bytes of leading metadata, so
    ``body[:500]`` yielded a wall of padding and no diagnosis. Falls back to a
    truncated raw body for non-JSON responses.
    """
    text = body or ""
    try:
        payload = json.loads(text)
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message.strip():
            return message.strip()[:_MAX_ERROR_BODY_CHARS]
    except (TypeError, ValueError):
        pass
    return text[:_MAX_ERROR_BODY_CHARS]


# ---------------------------------------------------------------------------
# Model catalog — mirrors the ``openai`` plugin so the picker UX is identical.
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

_MAX_INPUT_IMAGES = 5
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
# gpt-image-2's Images edit endpoint accepts raster formats only. The
# shared magic-byte sniffer also recognizes SVG/TIFF/ICO, which the API
# rejects server-side — gate to this allowlist so unsupported inputs fail
# locally with a clear error instead of an opaque HTTP 400.
_ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


# ---------------------------------------------------------------------------
# Config + auth helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    import os

    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _read_codex_access_token() -> Optional[str]:
    """Return a usable Codex OAuth token, or None.

    Delegates to the canonical reader in ``agent.auxiliary_client`` so token
    expiry, credential pool selection, and JWT decoding stay in one place.
    """
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    """Return a safe raster image MIME from magic bytes (not filename labels).

    Delegates magic-byte detection to the shared sniffer in
    ``agent.image_routing`` (single source of truth), then gates the result
    to :data:`_ACCEPTED_INPUT_MIME` — the raster formats gpt-image-2's
    ``input_image`` actually accepts. SVG/TIFF/ICO (which the shared sniffer
    also recognizes) are rejected here so they fail locally with a clear
    error instead of an opaque server-side HTTP 400.
    """
    from agent.image_routing import _sniff_mime_from_bytes

    mime = _sniff_mime_from_bytes(raw)
    if mime in _ACCEPTED_INPUT_MIME:
        return mime
    return None


def _data_url_to_input_image_url(value: str) -> str:
    """Validate and canonicalize a data:image URL for a Codex image edit."""
    if "," not in value:
        raise ValueError("Image data URL is missing a comma separator")
    header, data = value.split(",", 1)
    header_lc = header.lower()
    if not header_lc.startswith("data:image/") or ";base64" not in header_lc:
        raise ValueError(
            "Only base64 data:image URLs are supported as Codex image inputs"
        )
    raw = base64.b64decode(data, validate=True)
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError("Image data URL exceeds 25MB cap")
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise ValueError("Image data URL does not contain supported image bytes")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _local_image_to_data_url(value: str) -> str:
    """Read a local image path and return a validated data:image URL."""
    try:
        from agent.file_safety import get_read_block_error

        blocked = get_read_block_error(value)
        if blocked:
            raise ValueError(blocked)
    except ValueError:
        raise
    except Exception as exc:
        logger.debug("Codex image input read guard unavailable: %s", exc)

    path = Path(os.path.expanduser(value)).resolve()
    if not path.is_file():
        raise ValueError(f"Image input path does not exist or is not a file: {value}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Image input path is empty: {value}")
    if size > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path exceeds 25MB cap: {value}")
    raw = path.read_bytes()
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise ValueError(f"Image input path is not a supported image: {value}")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _to_input_image_part(value: str) -> Dict[str, str]:
    """Convert a URL/data URL/local path into an Images API input part."""
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("Blank image input")
    lowered = candidate.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        image_url = candidate
    elif lowered.startswith("data:"):
        image_url = _data_url_to_input_image_url(candidate)
    else:
        image_url = _local_image_to_data_url(candidate)
    return {"type": "input_image", "image_url": image_url}


def _normalize_input_images(
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]],
) -> List[Dict[str, str]]:
    """Collect primary + reference images as ordered Images API inputs."""
    values: List[str] = []
    if isinstance(image_url, str) and image_url.strip():
        values.append(image_url.strip())
    for ref in normalize_reference_images(reference_image_urls) or []:
        values.append(ref)
    if len(values) > _MAX_INPUT_IMAGES:
        raise ValueError("Codex image edits accept at most 5 total images")
    return [_to_input_image_part(value) for value in values]


def _build_images_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    input_images: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build a Codex Images generation or edit request body."""
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "background": "opaque",
        "model": API_MODEL,
        "n": 1,
        "quality": quality,
        "size": size,
    }
    if input_images:
        images: List[Dict[str, str]] = []
        for index, part in enumerate(input_images):
            if not isinstance(part, dict) or part.get("type") != "input_image":
                raise ValueError(
                    f"Malformed Codex image input at index {index}: "
                    "expected an input_image object"
                )
            image_url = part.get("image_url")
            if not isinstance(image_url, str) or not image_url.strip():
                raise ValueError(
                    f"Malformed Codex image input at index {index}: "
                    "expected a non-empty string image_url"
                )
            images.append({"image_url": image_url})
        payload["images"] = images
    return payload


def _collect_image_b64(
    token: str,
    *,
    prompt: str,
    size: str,
    quality: str,
    input_images: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Call the dedicated Codex Images endpoint and return its b64 image."""
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers.update({
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-codex-image-turn-id": str(uuid4()),
    })
    payload = _build_images_payload(
        prompt=prompt,
        size=size,
        quality=quality,
        input_images=input_images,
    )
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)

    endpoint = "edits" if input_images else "generations"
    with httpx.Client(timeout=timeout, headers=headers) as http:
        response = http.post(f"{_CODEX_BASE_URL}/images/{endpoint}", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Codex Images API returned HTTP {exc.response.status_code}: "
                f"{_summarize_error_body(exc.response.text)}"
            ) from exc
        try:
            result = response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Codex Images API returned invalid JSON: "
                f"{_summarize_diagnostic_body(response.text)}"
            ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Codex Images API returned unexpected response: "
            f"{_summarize_diagnostic_body(response.text)}"
        )
    data = result.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            "Codex Images API returned unexpected response: "
            f"{_summarize_diagnostic_body(response.text)}"
        )
    if not data:
        return None
    first = data[0]
    image_b64 = first.get("b64_json") if isinstance(first, dict) else None
    if not isinstance(image_b64, str) or not image_b64:
        raise RuntimeError(
            "Codex Images API returned unexpected response: "
            f"{_summarize_diagnostic_body(response.text)}"
        )
    return image_b64


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICodexImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI (Codex auth)"

    def is_available(self) -> bool:
        if not _read_codex_access_token():
            return False
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required; supports text and image inputs",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."
            ),
        }

    def capabilities(self) -> Dict[str, Any]:
        # The Codex Images edit endpoint accepts source/reference images as
        # data URLs. Keep this capability
        # honest so the dynamic `image_generate` schema encourages identity-
        # preserving edits instead of unrelated text-to-image redraws.
        return {
            "modalities": ["text", "image"],
            # The tool schema has no separate total-input capability. Reserve
            # one slot for the primary ``image_url``; calls without a primary
            # image can still pass five items, enforced below at runtime.
            "max_reference_images": _MAX_INPUT_IMAGES - 1,
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

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        if not _read_codex_access_token():
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        try:
            import httpx  # noqa: F401
        except ImportError:
            return error_response(
                error="httpx Python package not installed (pip install httpx)",
                error_type="missing_dependency",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        token = _read_codex_access_token()
        if not token:
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            input_images = _normalize_input_images(image_url, reference_image_urls)
        except Exception as exc:
            return error_response(
                error=f"Invalid image input for Codex image editing: {exc}",
                error_type="invalid_image_input",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            b64 = _collect_image_b64(
                token,
                prompt=prompt,
                size=size,
                quality=meta["quality"],
                input_images=input_images or None,
            )
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation via Codex auth failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex Images response contained no image data",
                error_type="empty_response",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"openai_codex_{tier_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-codex",
            modality="image" if input_images else "text",
            extra={
                "size": size,
                "quality": meta["quality"],
                "input_image_count": len(input_images),
            },
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())

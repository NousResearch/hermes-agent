"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Identical model catalog and tier semantics to the ``openai`` image-gen plugin
(``gpt-image-2`` at low/medium/high quality), but routes the request through
the Codex Responses API ``image_generation`` tool instead of the
``images.generate`` REST endpoint. This lets users who are already
authenticated with Codex/ChatGPT generate images without configuring a
separate ``OPENAI_API_KEY``.

Selection precedence for the tier (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai-codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``

Output is saved as PNG under ``$HERMES_HOME/cache/images/``.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


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
    "16:9": "1824x1024",
    "5:4": "1280x1024",
    "4:3": "1360x1024",
    "3:2": "1536x1024",
    "1:1": "1024x1024",
    "2:3": "1024x1536",
    "3:4": "1024x1360",
    "4:5": "1024x1280",
    "9:16": "1024x1824",
}


_SIZE_RE = re.compile(r"^\s*(\d{2,5})\s*[xX]\s*(\d{2,5})\s*$")
_MIN_IMAGE_PIXELS = 655_360
_MAX_IMAGE_PIXELS = 8_294_400
_MAX_IMAGE_SIDE = 3_839
_MAX_IMAGE_ASPECT = 3.0


def _normalize_image_size(size: Any) -> Optional[str]:
    """Validate and normalize explicit GPT Image 2 size strings."""
    if size is None or not isinstance(size, str):
        return None
    match = _SIZE_RE.match(size)
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    if width % 16 != 0 or height % 16 != 0:
        return None
    if width > _MAX_IMAGE_SIDE or height > _MAX_IMAGE_SIDE:
        return None
    pixels = width * height
    if pixels < _MIN_IMAGE_PIXELS or pixels > _MAX_IMAGE_PIXELS:
        return None
    if max(width / height, height / width) > _MAX_IMAGE_ASPECT:
        return None
    return f"{width}x{height}"


def _resolve_codex_aspect_ratio(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return DEFAULT_ASPECT_RATIO
    normalized = value.strip().lower()
    return normalized if normalized in _SIZES else DEFAULT_ASPECT_RATIO


def _resolve_openai_size(aspect_ratio: str, requested_size: Any = None) -> Optional[str]:
    explicit = _normalize_image_size(requested_size)
    if requested_size is not None:
        return explicit
    return _SIZES.get(aspect_ratio, _SIZES[DEFAULT_ASPECT_RATIO])

# Codex Responses surface used for the request. The chat model itself is only
# the host that calls the ``image_generation`` tool; the actual image work is
# done by ``API_MODEL``.
_CODEX_CHAT_MODEL = "gpt-5.5"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)
_CODEX_EDIT_INSTRUCTIONS = (
    "You are an assistant that must edit the provided reference image by "
    "using the image_generation tool when provided. Preserve visual details "
    "the user did not ask to change."
)

_ALLOWED_REFERENCE_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_REFERENCE_IMAGE_MAX_BYTES = 20 * 1024 * 1024
_DATA_URL_HEADER_RE = re.compile(r"^data:([^;,]+)(?:;[^,]*)*$", re.IGNORECASE)


def _normalize_reference_image_mime(mime: Optional[str]) -> Optional[str]:
    if not isinstance(mime, str):
        return None
    normalized = mime.split(";", 1)[0].strip().lower()
    if normalized == "image/jpg":
        normalized = "image/jpeg"
    return normalized if normalized in _ALLOWED_REFERENCE_IMAGE_MIME_TYPES else None


def _detect_reference_image_mime(raw: bytes) -> Optional[str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate_data_image_url(value: str) -> str:
    header, sep, payload = value.partition(",")
    if not sep:
        raise ValueError("Reference image data URL is missing a payload")
    match = _DATA_URL_HEADER_RE.match(header)
    if not match:
        raise ValueError("Reference image data URL is malformed")
    mime = _normalize_reference_image_mime(match.group(1))
    if mime is None:
        raise ValueError("Reference image data URL must use PNG, JPEG, WebP, or GIF")
    if ";base64" not in header.lower():
        raise ValueError("Reference image data URL must be base64-encoded")
    compact_payload = "".join(payload.split())
    approx_bytes = max(0, (len(compact_payload) * 3) // 4 - compact_payload.count("="))
    if approx_bytes > _REFERENCE_IMAGE_MAX_BYTES:
        raise ValueError(f"Reference image is too large ({approx_bytes} bytes); max is {_REFERENCE_IMAGE_MAX_BYTES} bytes")
    try:
        raw = base64.b64decode(compact_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Reference image data URL contains invalid base64") from exc
    if _detect_reference_image_mime(raw) != mime:
        raise ValueError("Reference image data URL payload is not a valid PNG, JPEG, WebP, or GIF")
    return f"data:{mime};base64,{compact_payload}"


def _allowed_local_reference_roots() -> Tuple[Path, ...]:
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        home = Path.home() / ".hermes"
    return ((home / "cache" / "images").resolve(strict=False), (home / "image_cache").resolve(strict=False))


def _resolve_allowed_local_reference_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    roots = _allowed_local_reference_roots()
    if any(resolved.is_relative_to(root) for root in roots):
        return resolved
    roots_display = ", ".join(str(root) for root in roots)
    raise ValueError(f"Local reference image paths must be under the Hermes image cache ({roots_display}); use an HTTP(S) URL or data:image URL otherwise")


def _read_local_reference_image(path: Path) -> Tuple[bytes, str]:
    size = path.stat().st_size
    if size > _REFERENCE_IMAGE_MAX_BYTES:
        raise ValueError(f"Reference image is too large ({size} bytes); max is {_REFERENCE_IMAGE_MAX_BYTES} bytes")
    raw = path.read_bytes()
    detected = _detect_reference_image_mime(raw)
    if detected is None:
        guessed = _normalize_reference_image_mime(mimetypes.guess_type(str(path))[0])
        hint = f" (guessed {guessed})" if guessed else ""
        raise ValueError(f"Reference image must be a PNG, JPEG, WebP, or GIF{hint}")
    return raw, detected


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


def _resolve_model(model: Optional[str] = None, quality_tier: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    import os

    if isinstance(model, str) and model in _MODELS:
        return model, _MODELS[model]
    if isinstance(quality_tier, str):
        tier = quality_tier.strip().lower()
        if tier in {"low", "medium", "high"}:
            model_id = f"gpt-image-2-{tier}"
            return model_id, _MODELS[model_id]

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


def _resolve_requested_model(kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    model = kwargs.get("model")
    quality_tier = kwargs.get("quality_tier")
    return _resolve_model(model if isinstance(model, str) else None, quality_tier if isinstance(quality_tier, str) else None)


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


def _build_responses_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    content: Optional[List[Dict[str, Any]]] = None,
    instructions: str = _CODEX_INSTRUCTIONS,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Codex Responses request body for an image_generation call."""
    tool: Dict[str, Any] = {
        "type": "image_generation",
        "model": API_MODEL,
        "size": size,
        "quality": quality,
        "output_format": "png",
        "background": "opaque",
        "partial_images": 1,
    }
    if action:
        tool["action"] = action
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": instructions,
        "input": [{
            "type": "message",
            "role": "user",
            "content": content or [{"type": "input_text", "text": prompt}],
        }],
        "tools": [tool],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
        "stream": True,
    }


def _extract_image_b64(value: Any) -> Optional[str]:
    """Return the newest image b64 embedded in a Responses event payload."""
    found: Optional[str] = None
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call":
            result = value.get("result")
            if isinstance(result, str) and result:
                found = result
        partial = value.get("partial_image_b64")
        if isinstance(partial, str) and partial:
            found = partial
        for child in value.values():
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    elif isinstance(value, list):
        for child in value:
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    return found


def _iter_sse_json(response: Any):
    """Yield JSON payloads from an SSE response without OpenAI SDK parsing.

    The ChatGPT/Codex backend can emit image-generation events newer than the
    pinned Python SDK understands. Parsing raw SSE keeps this provider tolerant
    of those event-shape changes.
    """
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        event = event_name
        event_name = None
        data_lines = []
        if not raw or raw == "[DONE]":
            return None
        payload = json.loads(raw)
        if isinstance(payload, dict) and event and "type" not in payload:
            payload["type"] = event
        return payload

    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line)
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    payload = flush()
    if payload is not None:
        yield payload


def _collect_image_b64(token: str, *, prompt: str, size: str, quality: str) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    return _collect_image_b64_from_payload(
        token,
        _build_responses_payload(prompt=prompt, size=size, quality=quality),
    )


def _collect_image_b64_from_payload(token: str, payload: Dict[str, Any]) -> Optional[str]:
    """Stream a Codex Responses payload and return the newest b64 image."""
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers.update({
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)

    image_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as http:
        with http.stream("POST", f"{_CODEX_BASE_URL}/responses", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                exc.response.read()
                body = exc.response.text[:500]
                raise RuntimeError(
                    f"Codex Responses API returned HTTP {exc.response.status_code}: {body}"
                ) from exc
            for event in _iter_sse_json(response):
                found = _extract_image_b64(event)
                if found:
                    image_b64 = found

    return image_b64


def _image_to_input_image_part(image: str) -> Dict[str, str]:
    """Convert a local path, HTTP(S) URL, or data URL into Responses input_image."""
    value = (image or "").strip()
    if not value:
        raise ValueError("image is required")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return {"type": "input_image", "image_url": value}
    if parsed.scheme == "data":
        return {"type": "input_image", "image_url": _validate_data_image_url(value)}
    if parsed.scheme:
        raise ValueError(f"Unsupported reference image URL scheme: {parsed.scheme}")
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Reference image not found: {value}")
    path = _resolve_allowed_local_reference_path(path)
    raw, mime = _read_local_reference_image(path)
    encoded = base64.b64encode(raw).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _collect_edit_reference_images(primary: str, kwargs: Dict[str, Any]) -> List[str]:
    # Preserve caller-provided image order exactly. For the common tool path
    # where `image` is omitted and `images=[img1, img2, ...]` is supplied,
    # `primary` is derived from images[0], so returning `images` keeps the
    # upload/reference order intact. When `image` is supplied separately, treat
    # it as the first/base image and append references in their declared order.
    images = _string_list(kwargs.get("images"))
    if images:
        if images[0] == primary:
            return images
        return [primary, *images]

    reference_images = _string_list(kwargs.get("reference_images"))
    references = _string_list(kwargs.get("references"))
    return [primary, *reference_images, *references]


def _collect_edited_image_b64(token: str, *, prompt: str, image: str, size: str, quality: str, images: Optional[List[str]] = None) -> Optional[str]:
    references = images or [image]
    content = [{"type": "input_text", "text": prompt}]
    content.extend(_image_to_input_image_part(ref) for ref in references)
    payload = _build_responses_payload(
        prompt=prompt,
        size=size,
        quality=quality,
        content=content,
        instructions=_CODEX_EDIT_INSTRUCTIONS,
        action="edit",
    )
    return _collect_image_b64_from_payload(token, payload)


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
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."
            ),
        }

    def supports_edit(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = _resolve_codex_aspect_ratio(aspect_ratio)

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

        tier_id, meta = _resolve_requested_model(kwargs)
        requested_size = kwargs.get("size")
        size = _resolve_openai_size(aspect, requested_size)
        if requested_size is not None and size is None:
            return error_response(
                error=(
                    "Invalid size. Use <width>x<height> with dimensions that are "
                    "multiples of 16, max side < 3840, aspect ratio <= 3:1, and "
                    "total pixels between 655,360 and 8,294,400."
                ),
                error_type="invalid_argument",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

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
            b64 = _collect_image_b64(
                token,
                prompt=prompt,
                size=size,
                quality=meta["quality"],
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
                error="Codex response contained no image_generation_call result",
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
            extra={"size": size, "quality": meta["quality"]},
        )

    def edit(
        self,
        prompt: str,
        image: Any,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = _resolve_codex_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )
        if not isinstance(image, str) or not image.strip():
            return error_response(
                error="A reference image path or URL is required",
                error_type="invalid_argument",
                provider="openai-codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        token = _read_codex_access_token()
        if not token:
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        try:
            import httpx  # noqa: F401
        except ImportError:
            return error_response(
                error="httpx Python package not installed (pip install httpx)",
                error_type="missing_dependency",
                provider="openai-codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_requested_model(kwargs)
        requested_size = kwargs.get("size")
        size = _resolve_openai_size(aspect, requested_size)
        if requested_size is not None and size is None:
            return error_response(
                error=(
                    "Invalid size. Use <width>x<height> with dimensions that are "
                    "multiples of 16, max side < 3840, aspect ratio <= 3:1, and "
                    "total pixels between 655,360 and 8,294,400."
                ),
                error_type="invalid_argument",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            reference_images = _collect_edit_reference_images(image.strip(), kwargs)
            b64 = _collect_edited_image_b64(
                token,
                prompt=prompt,
                image=image.strip(),
                images=reference_images,
                size=size or _SIZES[DEFAULT_ASPECT_RATIO],
                quality=meta["quality"],
            )
        except FileNotFoundError as exc:
            return error_response(
                error=str(exc),
                error_type="not_found",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except ValueError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_argument",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.debug("Codex image edit failed", exc_info=True)
            return error_response(
                error=f"OpenAI image edit via Codex auth failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex response contained no image_generation_call result",
                error_type="empty_response",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"openai_codex_edit_{tier_id}")
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
            extra={"size": size, "quality": meta["quality"], "source_image": image.strip(), "source_images": reference_images},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())

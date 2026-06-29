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

Output is saved as PNG under ``$HERMES_HOME/cache/images/``. Source images for
image-to-image/editing are sent as Responses ``input_image`` content parts.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Codex Responses surface used for the request. The chat model itself is only
# the host that calls the ``image_generation`` tool; the actual image work is
# done by ``API_MODEL``.
_CODEX_CHAT_MODEL = "gpt-5.5"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation and image editing "
    "requests by using the image_generation tool when provided."
)

_MAX_REFERENCE_IMAGES = 16
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
# gpt-image-2's Responses ``input_image`` accepts raster formats only. The
# shared magic-byte sniffer also recognizes SVG/TIFF/ICO, which the API
# rejects server-side — gate to this allowlist so unsupported inputs fail
# locally with a clear error instead of an opaque HTTP 400.
_ACCEPTED_INPUT_MIME = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)
_CODEX_ERROR_BODY_MAX_BYTES = 1024 * 1024
_CODEX_ERROR_PREVIEW_CHARS = 500


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
    """Validate and canonicalize a data:image URL for Responses input_image."""
    if "," not in value:
        raise ValueError("Image data URL is missing a comma separator")
    header, data = value.split(",", 1)
    header_lc = header.lower()
    if not header_lc.startswith("data:image/") or ";base64" not in header_lc:
        raise ValueError("Only base64 data:image URLs are supported as Codex image inputs")
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
    """Convert a URL/data URL/local path into a Responses input_image part."""
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
    """Collect primary + reference images as ordered Responses content parts."""
    values: List[str] = []
    if isinstance(image_url, str) and image_url.strip():
        values.append(image_url.strip())
    for ref in (normalize_reference_images(reference_image_urls) or []):
        values.append(ref)
    values = values[:_MAX_REFERENCE_IMAGES]
    return [_to_input_image_part(value) for value in values]


def _build_responses_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    input_images: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build the Codex Responses request body for an image_generation call."""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if input_images:
        content.extend(input_images)
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [{
            "type": "message",
            "role": "user",
            "content": content,
        }],
        "tools": [{
            "type": "image_generation",
            "model": API_MODEL,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": 1,
        }],
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


def _codex_error_response_preview(response: Any) -> str:
    """Read a bounded preview from a streamed Codex error response."""
    chunks: List[bytes] = []
    total = 0
    limit = _CODEX_ERROR_BODY_MAX_BYTES

    for chunk in response.iter_bytes():
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="replace")
        if not isinstance(chunk, (bytes, bytearray)) or not chunk:
            continue
        remaining = limit + 1 - total
        if remaining <= 0:
            break
        piece = bytes(chunk[:remaining])
        chunks.append(piece)
        total += len(piece)
        if total > limit:
            break

    raw = b"".join(chunks)
    truncated = len(raw) > limit
    if truncated:
        raw = raw[:limit]

    preview = raw.decode("utf-8", errors="replace")[:_CODEX_ERROR_PREVIEW_CHARS]
    if truncated:
        preview += f" ... [truncated after {limit} bytes]"
    return preview


def _collect_image_b64(
    token: str,
    *,
    prompt: str,
    size: str,
    quality: str,
    input_images: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers.update({
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    payload = _build_responses_payload(
        prompt=prompt,
        size=size,
        quality=quality,
        input_images=input_images,
    )
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)

    image_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as http:
        with http.stream("POST", f"{_CODEX_BASE_URL}/responses", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = _codex_error_response_preview(exc.response)
                raise RuntimeError(
                    f"Codex Responses API returned HTTP {exc.response.status_code}: {body}"
                ) from exc
            for event in _iter_sse_json(response):
                found = _extract_image_b64(event)
                if found:
                    image_b64 = found

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
        # The Codex Responses image_generation tool accepts source/reference
        # images as `input_image` message content parts. Keep this capability
        # honest so the dynamic `image_generate` schema encourages identity-
        # preserving edits instead of unrelated text-to-image redraws.
        return {"modalities": ["text", "image"], "max_reference_images": _MAX_REFERENCE_IMAGES}

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
            modality="image" if input_images else "text",
            extra={"size": size, "quality": meta["quality"], "input_image_count": len(input_images)},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())

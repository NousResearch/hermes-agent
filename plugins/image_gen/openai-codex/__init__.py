"""OpenAI image generation — ChatGPT/Codex OAuth variant.

Same catalog/tiers as the ``openai`` plugin (``gpt-image-2`` low/medium/high), posted to the
Codex backend's native ``images/generations`` and ``images/edits`` endpoints, the same route the
official Codex client uses (``codex-rs/ext/image-generation``). No ``OPENAI_API_KEY`` is needed.

There is deliberately NO chat/host model here. An earlier version rode a Responses call with a
hosted ``image_generation`` tool on a pinned chat model (``gpt-5.5``): when OpenAI withdrew that id
from an account cohort every image call 404'd while chat kept working (#105398, #107076), and the
host model was free to answer in text instead of calling the tool. The native route has neither
failure mode.

The backend does not enforce ``model``/``quality``/``size`` — it accepts unknown model ids and
returns its own quality/size (#107233). We send the catalog values and report what came back
(``reported_quality``/``reported_size``) so a request that was not honoured is diagnosable.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, save_b64_image, success_response
from plugins.image_gen._common import (
    GPT_IMAGE_2_API_MODEL as API_MODEL, GPT_IMAGE_2_DEFAULT as DEFAULT_MODEL, GPT_IMAGE_2_TIERS,
    StaticImageGenProvider, collect_source_images, error_factory, prompt_required_error,
    resolve_static_model, size_for)

logger = logging.getLogger(__name__)

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_MAX_ERROR_BODY_CHARS = 500

_MAX_REFERENCE_IMAGES = 16
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
# The edit endpoint accepts raster only; the shared sniffer also knows SVG/TIFF/ICO, which it rejects.
_ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

_NO_AUTH = (
    "No Codex/ChatGPT OAuth credentials available. Run "
    "`hermes auth codex` (or `hermes setup` → Codex) to sign in.")


def _summarize_error_body(body: str) -> str:
    """Bounded summary preferring parsed ``error.message`` (Codex bodies carry leading metadata)."""
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


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    return resolve_static_model(
        GPT_IMAGE_2_TIERS, DEFAULT_MODEL, env_var="OPENAI_IMAGE_MODEL", config_key="openai-codex")


def _read_codex_access_token() -> Optional[str]:
    """Usable Codex OAuth token or None (``agent.auxiliary_client`` owns expiry/pool/JWT)."""
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        return token.strip() if isinstance(token, str) and token.strip() else None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _httpx_available() -> bool:
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    from agent.image_routing import _sniff_mime_from_bytes

    mime = _sniff_mime_from_bytes(raw)
    return mime if mime in _ACCEPTED_INPUT_MIME else None


def _encode_input_image(raw: bytes, too_big: str, unsupported: str) -> str:
    """Size- and MIME-check raw image bytes, then return a canonical ``data:`` URL."""
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(too_big)
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise ValueError(unsupported)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _data_url_to_input_image_url(value: str) -> str:
    if "," not in value:
        raise ValueError("Image data URL is missing a comma separator")
    header, data = value.split(",", 1)
    header_lc = header.lower()
    if not header_lc.startswith("data:image/") or ";base64" not in header_lc:
        raise ValueError("Only base64 data:image URLs are supported as Codex image inputs")
    return _encode_input_image(
        base64.b64decode(data, validate=True),
        "Image data URL exceeds 25MB cap",
        "Image data URL does not contain supported image bytes")


def _remote_image_to_data_url(value: str) -> str:
    """The edit endpoint takes inline data URLs only (as the official client sends), so fetch."""
    import httpx

    response = httpx.get(value, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return _encode_input_image(
        response.content,
        f"Image URL exceeds 25MB cap: {value}",
        f"Image URL did not return a supported image: {value}")


def _local_image_to_data_url(value: str) -> str:
    from agent.file_safety import get_read_block_error

    blocked = get_read_block_error(value)
    if blocked:
        raise ValueError(blocked)
    path = Path(os.path.expanduser(value)).resolve()
    if not path.is_file():
        raise ValueError(f"Image input path does not exist or is not a file: {value}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Image input path is empty: {value}")
    return _encode_input_image(
        path.read_bytes(),
        f"Image input path exceeds 25MB cap: {value}",
        f"Image input path is not a supported image: {value}")


def _to_input_image(value: str) -> Dict[str, str]:
    """Convert a URL/data URL/local path into an ``images[]`` entry for ``images/edits``."""
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("Blank image input")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        image_url = _remote_image_to_data_url(candidate)
    elif lowered.startswith("data:"):
        image_url = _data_url_to_input_image_url(candidate)
    else:
        image_url = _local_image_to_data_url(candidate)
    return {"image_url": image_url}


def _normalize_input_images(
    image_url: Optional[str], reference_image_urls: Optional[List[str]]
) -> List[Dict[str, str]]:
    values = collect_source_images(image_url, reference_image_urls, limit=_MAX_REFERENCE_IMAGES)
    return [_to_input_image(value) for value in values]


def _build_image_request(
    *, prompt: str, size: str, quality: str, input_images: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, Dict[str, Any]]:
    """``(endpoint_path, json_body)`` — ``images/edits`` when sources are present, else
    ``images/generations``. Field set mirrors the official client's ``ImageGenerationRequest`` /
    ``ImageEditRequest``."""
    body: Dict[str, Any] = {
        "prompt": prompt, "model": API_MODEL, "n": 1, "quality": quality, "size": size,
        "background": "opaque",
    }
    if input_images:
        body["images"] = input_images
        return "images/edits", body
    return "images/generations", body


def _post_image_request(
    token: str, *, prompt: str, size: str, quality: str, input_images: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """POST to the native Codex images endpoint; return the decoded JSON body plus
    ``imagegen_request_id`` (backend correlation id, for support tickets)."""
    import httpx
    from agent.codex_headers import codex_cloudflare_headers

    headers = codex_cloudflare_headers(token)
    headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-codex-image-turn-id": str(uuid.uuid4()),
    })
    path, body = _build_image_request(prompt=prompt, size=size, quality=quality, input_images=input_images)
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout, headers=headers) as http:
        response = http.post(f"{_CODEX_BASE_URL}/{path}", json=body)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Codex images API returned HTTP {response.status_code}: "
            f"{_summarize_error_body(response.text)}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Codex images API returned a non-object body")
    payload["imagegen_request_id"] = response.headers.get("x-codex-imagegen-request-id")
    return payload


def _png_pixel_size(raw: bytes) -> Optional[str]:
    """``"{w}x{h}"`` for a PNG payload, or None if not a PNG IHDR."""
    import struct

    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return f"{width}x{height}"


class OpenAICodexImageGenProvider(StaticImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    provider_id = "openai-codex"
    label = "OpenAI (Codex auth)"
    models = GPT_IMAGE_2_TIERS
    default_model_id = DEFAULT_MODEL
    price = "varies"

    def is_available(self) -> bool:
        return bool(_read_codex_access_token()) and _httpx_available()

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required; supports text and image inputs",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."),
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": _MAX_REFERENCE_IMAGES}

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error("openai-codex", aspect)
        token = _read_codex_access_token()
        if not token:
            return error_factory("openai-codex", aspect)(_NO_AUTH, "auth_required")
        if not _httpx_available():
            return error_factory("openai-codex", aspect)(
                "httpx Python package not installed (pip install httpx)", "missing_dependency")

        tier_id, meta = _resolve_model()
        size = size_for(aspect)
        fail = error_factory("openai-codex", aspect, model=tier_id, prompt=prompt)
        try:
            input_images = _normalize_input_images(image_url, reference_image_urls)
        except Exception as exc:
            return fail(f"Invalid image input for Codex image editing: {exc}", "invalid_image_input")

        try:
            payload = _post_image_request(
                token, prompt=prompt, size=size, quality=meta["quality"], input_images=input_images or None)
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return fail(f"OpenAI image generation via Codex auth failed: {exc}", "api_error")

        data = payload.get("data")
        b64 = data[0].get("b64_json") if isinstance(data, list) and data and isinstance(data[0], dict) else None
        if not isinstance(b64, str) or not b64:
            return fail("Codex images API response contained no image data", "empty_response")

        try:
            pixel_size = _png_pixel_size(base64.b64decode(b64))
            saved_path = save_b64_image(b64, prefix=f"openai_codex_{tier_id}")
        except Exception as exc:
            return fail(f"Could not save image to cache: {exc}", "io_error")
        return success_response(
            image=str(saved_path), model=tier_id, prompt=prompt, aspect_ratio=aspect,
            provider="openai-codex", modality="image" if input_images else "text",
            extra={
                "size": size, "quality": meta["quality"], "input_image_count": len(input_images),
                "requested_size": size, "pixel_size": pixel_size,
                "reported_quality": payload.get("quality"), "reported_size": payload.get("size"),
                "imagegen_request_id": payload.get("imagegen_request_id"),
            })


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())

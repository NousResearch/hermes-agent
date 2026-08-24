"""OpenAI image generation — ChatGPT/Codex OAuth variant.

Same catalog/tiers as the ``openai`` plugin (``gpt-image-2`` low/medium/high), routed
through the Codex Responses API ``image_generation`` tool, so no ``OPENAI_API_KEY`` is
needed. Output is PNG; source images travel as Responses ``input_image`` parts.

Do NOT reintroduce an "account capability" classifier keyed on ``Tool choice
'image_generation' not found in 'tools' parameter``: that 400 is a request-shape
rejection for every account, fixed by omitting tool_choice (``_build_responses_payload``);
any remaining HTTP error must surface verbatim.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, save_b64_image, success_response
from plugins.image_gen._common import (
    GPT_IMAGE_2_API_MODEL as API_MODEL, GPT_IMAGE_2_DEFAULT as DEFAULT_MODEL, GPT_IMAGE_2_TIERS,
    StaticImageGenProvider, collect_source_images, error_factory, prompt_required_error,
    resolve_static_model, size_for)

logger = logging.getLogger(__name__)

# NOTE: do NOT reintroduce an "account capability" classifier keyed on ``Tool choice 'image_generation' not
# found in 'tools' parameter``. That HTTP 400 is a *request-shape* rejection (the Codex backend resolves
# tool_choice as a function-tool name and never recognizes hosted-tool entries) — it is emitted for every
# account, including accounts where image generation works. A previous version of this file translated that
# 400 into "Image generation is not enabled for the current Codex account. Switch the image provider to
# OpenAI API key, FAL, or xAI.", which reported a universal bug in our own payload as the user's entitlement
# problem and sent people away from a provider that was never actually tried. The request-shape bug is fixed
# by omitting tool_choice (see ``_build_responses_payload``); any remaining HTTP error must surface verbatim
# so it stays diagnosable. See issues #19505, #49008 and #31335.
_MAX_ERROR_BODY_CHARS = 500

# Hosts the ``image_generation`` tool call; ``API_MODEL`` does the image work.
_CODEX_CHAT_MODEL = "gpt-5.5"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation and image editing "
    "requests by using the image_generation tool when provided.")

_MAX_REFERENCE_IMAGES = 16
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
# ``input_image`` accepts raster only; the shared sniffer also knows SVG/TIFF/ICO, which the API rejects.
_ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# Progressive frames (partial_image_b64) saved as finals produced the "smear" failure mode:
# request 0 partials, never let a partial overwrite a final, only deliver source=final.
_PARTIAL_IMAGES_REQUESTED = 0
_NONFINAL_RETRIES = 1  # content-agnostic retries when the stream yields no final result

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


def _local_image_to_data_url(value: str) -> str:
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
    if path.stat().st_size <= 0:
        raise ValueError(f"Image input path is empty: {value}")
    return _encode_input_image(
        path.read_bytes(),
        f"Image input path exceeds 25MB cap: {value}",
        f"Image input path is not a supported image: {value}")


def _to_input_image_part(value: str) -> Dict[str, str]:
    """Convert a URL/data URL/local path into a Responses input_image part."""
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("Blank image input")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        image_url = candidate
    elif lowered.startswith("data:"):
        image_url = _data_url_to_input_image_url(candidate)
    else:
        image_url = _local_image_to_data_url(candidate)
    return {"type": "input_image", "image_url": image_url}


def _normalize_input_images(
    image_url: Optional[str], reference_image_urls: Optional[List[str]]
) -> List[Dict[str, str]]:
    values = collect_source_images(image_url, reference_image_urls, limit=_MAX_REFERENCE_IMAGES)
    return [_to_input_image_part(value) for value in values]


def _build_responses_payload(
    *, prompt: str, size: str, quality: str, input_images: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Responses body for an image_generation call. No ``tool_choice``: Codex rejects every shape
    for forcing the hosted tool (looks it up as a *function* name), so the host model decides,
    nudged by ``instructions``."""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}, *(input_images or [])]
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [{
            "type": "image_generation",
            "model": API_MODEL,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": _PARTIAL_IMAGES_REQUESTED,
        }],
        # No ``tool_choice`` is sent: the chatgpt.com/backend-api/codex backend rejects every shape we have
        # for forcing the hosted ``image_generation`` tool. ``{"type": "allowed_tools", "mode": "required",
        # "tools": [{"type": "image_generation"}]}`` (and the simpler ``{"type": "image_generation"}`` form)
        # both 400 with ``Tool choice 'image_generation' not found in 'tools' parameter`` — the backend
        # looks up tool_choice as a *function* name and never recognizes hosted-tool entries. Letting the
        # host model decide is the only shape Codex currently accepts; the ``instructions`` above are what
        # nudge it toward the tool. See issue #19505.
        "stream": True,
    }


def _extract_image_candidates(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(final_result_b64, latest_partial_b64)`` from a payload tree; a partial never overwrites
    a final."""
    result_b64: Optional[str] = None
    partial_b64: Optional[str] = None

    def walk(node: Any) -> None:
        nonlocal result_b64, partial_b64
        if isinstance(node, dict):
            result = node.get("result") if node.get("type") == "image_generation_call" else None
            if isinstance(result, str) and result:
                result_b64 = result
            partial = node.get("partial_image_b64")
            if isinstance(partial, str) and partial:
                partial_b64 = partial
        for child in node.values() if isinstance(node, dict) else node if isinstance(node, list) else ():
            walk(child)

    walk(value)
    return result_b64, partial_b64


def _extract_image_b64(value: Any) -> Optional[str]:
    """Image b64 from a payload, preferring a final result over a partial."""
    return next((b64 for b64 in _extract_image_candidates(value) if b64), None)


def _png_pixel_size(raw: bytes) -> Optional[str]:
    """``"{w}x{h}"`` for a PNG payload, or None if not a PNG IHDR."""
    import struct

    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return f"{width}x{height}"


def _iter_sse_json(response: Any):
    """JSON payloads from an SSE response, without SDK parsing (events newer than the SDK still parse)."""
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        event, event_name, data_lines = event_name, None, []
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
        elif line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    payload = flush()
    if payload is not None:
        yield payload


def _collect_image_b64(
    token: str, *, prompt: str, size: str, quality: str, input_images: Optional[List[Dict[str, str]]] = None
) -> Optional[Dict[str, str]]:
    """Stream a Codex Responses image_generation call → ``{"b64", "source": "final"|"partial"}`` or
    ``None``. A partial is kept only when no final arrives; callers must not treat it as success."""
    import httpx
    from agent.codex_headers import codex_cloudflare_headers

    headers = codex_cloudflare_headers(token)
    headers.update({
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    payload = _build_responses_payload(
        prompt=prompt, size=size, quality=quality, input_images=input_images)
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)

    final_b64: Optional[str] = None
    partial_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as http:
        with http.stream("POST", f"{_CODEX_BASE_URL}/responses", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                exc.response.read()
                raise RuntimeError(
                    f"Codex Responses API returned HTTP {exc.response.status_code}: "
                    f"{_summarize_error_body(exc.response.text)}"
                ) from exc
            for event in _iter_sse_json(response):
                result_b64, event_partial = _extract_image_candidates(event)
                final_b64 = result_b64 or final_b64
                partial_b64 = event_partial or partial_b64
    if final_b64:
        return {"b64": final_b64, "source": "final"}
    return {"b64": partial_b64, "source": "partial"} if partial_b64 else None


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
        attempts = _NONFINAL_RETRIES + 1
        try:
            input_images = _normalize_input_images(image_url, reference_image_urls)
        except Exception as exc:
            return fail(f"Invalid image input for Codex image editing: {exc}", "invalid_image_input")

        try:
            collected: Optional[Dict[str, str]] = None
            for attempt in range(attempts):
                collected = _collect_image_b64(
                    token, prompt=prompt, size=size, quality=meta["quality"],
                    input_images=input_images or None)
                if collected and collected.get("source") == "final" and collected.get("b64"):
                    break
                if attempt < _NONFINAL_RETRIES:
                    partial = collected and collected.get("source") == "partial"
                    logger.warning(
                        "Codex image stream ended with %s (attempt %s/%s); "
                        "retrying once before failing closed.",
                        "progressive-only partial frame" if partial else "no image_generation_call result",
                        attempt + 1, attempts)
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return fail(f"OpenAI image generation via Codex auth failed: {exc}", "api_error")

        if not collected or not collected.get("b64"):
            return fail(
                f"Codex response contained no image_generation_call result after {attempts} attempt(s)",
                "empty_response")
        image_source = collected.get("source") or "unknown"
        b64 = collected["b64"]
        # Never deliver a progressive-only frame as success (smeared previews).
        if image_source != "final":
            try:
                pixel_hint = _png_pixel_size(base64.b64decode(b64, validate=False))
            except Exception:
                pixel_hint = None
            detail = (
                "Codex returned only a progressive partial image frame after "
                f"{attempts} attempt(s); refusing to save it as a final deliverable.")
            if pixel_hint:
                detail = f"{detail} partial_pixel_size={pixel_hint}."
            return {
                **fail(detail, "incomplete_image"), "image_source": image_source, "requested_size": size,
                "partial_pixel_size": pixel_hint, "nonfinal_retries": _NONFINAL_RETRIES,
            }

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
                "image_source": image_source, "requested_size": size, "pixel_size": pixel_size,
            })


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())

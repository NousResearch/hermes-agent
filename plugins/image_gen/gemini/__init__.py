"""Google AI Studio Gemini image generation (Nano Banana 2 / Lite / Pro);
base64 inlineData → image cache. Selection: ``GEMINI_IMAGE_MODEL`` → ``image_gen.gemini.model`` →
``image_gen.model`` → :data:`DEFAULT_MODEL`; an id outside the catalog is sent verbatim.
Endpoint: ``image_gen.gemini.base_url`` → the named endpoint ``image_gen.gemini.provider`` →
``GEMINI_BASE_URL`` → :data:`BASE_URL`; key: env named by ``image_gen.gemini.key_env`` → the named
endpoint's credential → ``GOOGLE_API_KEY`` → ``GEMINI_API_KEY``."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from agent.gemini_native_adapter import _usage_from_metadata
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, _URL_IMAGE_CONTENT_TYPES, resolve_aspect_ratio,
    save_b64_image, success_response)
from agent.secret_scope import get_secret
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory, load_image_gen_config,
    post_json, prompt_required_error, record_token_usage, resolve_static_model)

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-image"

MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-3.1-flash-image": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "speed": "Fast",
        "strengths": "Best balance — 14 aspect ratios, 512/1K/2K/4K resolution, up to 14 reference images",
        "api_model": "gemini-3.1-flash-image",
        "resolutions": ("512", "1K", "2K", "4K"),
        "max_refs": 14,
    },
    "gemini-3.1-flash-lite-image": {
        "display": "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "speed": "Fastest",
        "strengths": "Lowest latency & cost; 14 aspect ratios, 1K output, up to 14 reference images",
        "api_model": "gemini-3.1-flash-lite-image",
        "resolutions": ("1K",),
        "max_refs": 14,
    },
    "gemini-3-pro-image": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "Slower",
        "strengths": "Highest fidelity & reasoning — complex compositions, text rendering, 1K/2K/4K",
        "api_model": "gemini-3-pro-image",
        "resolutions": ("1K", "2K", "4K"),
        "max_refs": 14,
    },
}

_ASPECT_RATIOS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
_GEMINI_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
)
_MAX_REFERENCE_IMAGES = 14
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB per reference image
_DEFAULT_TIMEOUT = (15.0, 120.0)


def _strip_google_prefix(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    return raw.split("/", 1)[1].strip() if raw.lower().startswith("google/") else raw


def _resolve_model(explicit: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """``(model_id, meta)`` from :data:`MODELS` via :func:`resolve_static_model` (with ``google/``
    prefix stripping and passthrough for custom Gemini model ids)."""
    cleaned = _strip_google_prefix(explicit)
    model_id, meta = resolve_static_model(
        MODELS, DEFAULT_MODEL, env_var="GEMINI_IMAGE_MODEL", config_key="gemini",
        explicit=cleaned, passthrough=True)
    normalized = _strip_google_prefix(model_id) or model_id
    if normalized in MODELS:
        return normalized, MODELS[normalized]
    return normalized, {
        **meta, "api_model": normalized,
        "resolutions": meta.get("resolutions") or ("1K", "2K", "4K"),
        "max_refs": meta.get("max_refs") or _MAX_REFERENCE_IMAGES,
    }


def _named_endpoint(name: str) -> Tuple[str, str]:
    """``(base_url, api_key)`` of the user-declared custom endpoint *name* (``providers:`` /
    ``custom_providers:``), so image generation reuses a chat endpoint's URL and credential without
    duplicating the key into Gemini variables (#83080). Unknown name → ``("", "")`` with a warning."""
    from hermes_cli.runtime_provider import _get_named_custom_provider

    entry = _get_named_custom_provider(name)
    if not entry:
        logger.warning("image_gen.gemini.provider %r matches no custom endpoint in providers:", name)
        return "", ""
    key_env = str(entry.get("key_env") or "").strip()
    api_key = str(entry.get("api_key") or "").strip() or (get_secret(key_env) if key_env else None) or ""
    return str(entry.get("base_url") or "").strip().rstrip("/"), api_key


def _resolve_endpoint() -> Tuple[str, str]:
    """``(base_url, api_key)`` — ``image_gen.gemini.base_url`` → named endpoint → ``GEMINI_BASE_URL`` →
    :data:`BASE_URL`; the env var named by ``image_gen.gemini.key_env`` → named endpoint →
    ``GOOGLE_API_KEY`` → ``GEMINI_API_KEY``. Shared by ``is_available()`` and ``generate()``."""
    cfg = load_image_gen_config("gemini")
    named = str(cfg.get("provider") or "").strip()
    named_base, named_key = _named_endpoint(named) if named else ("", "")
    base_url = (
        str(cfg.get("base_url") or "").strip().rstrip("/")
        or named_base
        or os.environ.get("GEMINI_BASE_URL", "").strip().rstrip("/")
        or BASE_URL
    )
    key_env = str(cfg.get("key_env") or "").strip()
    api_key = (
        (get_secret(key_env) if key_env else None)
        or named_key
        or get_secret("GOOGLE_API_KEY")
        or get_secret("GEMINI_API_KEY")
        or ""
    ).strip()
    return base_url, api_key


def _resolve_exact_aspect_ratio(
    aspect_ratio: Optional[str], exact_override: Optional[str] = None
) -> Tuple[str, str]:
    """``(semantic_aspect, wire_aspect)`` — exact override (``aspect_ratio_exact`` → exact ``aspect_ratio``
    arg → ``GEMINI_IMAGE_ASPECT_RATIO`` → ``image_gen.gemini.aspect_ratio``) wins when in
    :data:`_GEMINI_RATIOS`; otherwise the semantic mapping applies."""
    semantic = resolve_aspect_ratio(aspect_ratio)
    raw_arg = (aspect_ratio or "").strip()
    cfg = load_image_gen_config("gemini")
    for candidate in (
        exact_override,
        raw_arg if raw_arg in _GEMINI_RATIOS else None,
        os.environ.get("GEMINI_IMAGE_ASPECT_RATIO"),
        cfg.get("aspect_ratio"),
    ):
        if isinstance(candidate, str) and candidate.strip() in _GEMINI_RATIOS:
            exact = candidate.strip()
            w, h = (int(x) for x in exact.split(":"))
            return ("square" if w == h else ("landscape" if w > h else "portrait")), exact
    return semantic, _ASPECT_RATIOS.get(semantic, "16:9")


def _resolve_image_size(
    meta: Dict[str, Any], explicit_size: Optional[str] = None, upscale: bool = False
) -> Optional[str]:
    """Resolve ``imageConfig.imageSize`` (``"512"``, ``"1K"``, ``"2K"``, ``"4K"``) when supported by *meta*."""
    supported: Tuple[str, ...] = tuple(meta.get("resolutions") or ())
    if not supported:
        return None
    cfg = load_image_gen_config("gemini")
    candidates = [
        explicit_size,
        "2K" if upscale and "2K" in supported else ("4K" if upscale and "4K" in supported else None),
        os.environ.get("GEMINI_IMAGE_SIZE"),
        cfg.get("image_size") or cfg.get("resolution"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip().upper()
            if normalized in supported:
                return normalized
    return None


def _resolve_google_search(explicit_search: Optional[bool] = None) -> bool:
    """Resolve whether ``tools: [{"googleSearch": {}}]`` grounding is enabled."""
    if isinstance(explicit_search, bool):
        return explicit_search
    env_val = os.environ.get("GEMINI_IMAGE_GOOGLE_SEARCH", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    if env_val in ("0", "false", "no", "off"):
        return False
    cfg = load_image_gen_config("gemini")
    return bool(cfg.get("google_search") or cfg.get("search_grounding"))


def _sniff_mime(data: bytes, fallback: str = "image/png") -> str:
    """Detect image MIME type from magic bytes; fallback to *fallback*."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return fallback


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load ``(data, mime_type)`` from a URL, data URI or local path; raises on IO/network/security error."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        from tools.url_safety import create_ssrf_safe_client, is_safe_url

        if not is_safe_url(ref):
            raise ValueError(f"Image reference URL failed the SSRF safety check: {ref}")
        with create_ssrf_safe_client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(ref)
        resp.raise_for_status()
        raw = resp.content
        if len(raw) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"Image reference URL exceeds 25MB cap: {ref}")
        header_mime = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        return raw, _sniff_mime(raw, header_mime or "image/png")
    if lower.startswith("data:"):
        header, sep, b64 = ref.partition(",")
        if not sep:
            raise ValueError("image data URI is missing its payload")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError("Image data URI exceeds 25MB cap")
        header_mime = header[5:].split(";", 1)[0].strip() or "image/png"
        return raw, _sniff_mime(raw, header_mime)
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before local bytes

    raise_if_read_blocked(ref)
    path = Path(os.path.expanduser(ref))
    raw = path.read_bytes()
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path exceeds 25MB cap: {ref}")
    guessed_mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return raw, _sniff_mime(raw, guessed_mime)


def _extract_error_message(response: Any, exc: Exception) -> str:
    """Extract Google AI Studio's nested ``error.message`` JSON field on HTTP errors."""
    if response is not None:
        try:
            body = response.json()
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if isinstance(body, dict) and body.get("message"):
                return str(body["message"])
        except Exception:  # noqa: BLE001
            pass
        text = getattr(response, "text", None) or getattr(response, "reason", None)
        if text:
            return str(text)[:300]
    return str(exc)


def _extract_inline_image(body: Dict[str, Any]) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """Return ``((b64_data, extension), text_fallback)`` from a ``generateContent`` response."""
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        block = (body.get("promptFeedback") or {}).get("blockReason")
        return None, (f"Prompt blocked by safety filter ({block})" if block else None)

    texts: List[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parts = ((candidate.get("content") or {}).get("parts")) or []
        for part in parts:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict):
                b64_data = str(inline.get("data") or "").strip()
                if b64_data:
                    mime = str(inline.get("mimeType") or inline.get("mime_type") or "").strip().lower()
                    ext = _URL_IMAGE_CONTENT_TYPES.get(mime)
                    if not ext:
                        try:
                            ext = _URL_IMAGE_CONTENT_TYPES.get(
                                _sniff_mime(base64.b64decode(b64_data[:64])), "png")
                        except Exception:  # noqa: BLE001
                            ext = "png"
                    return (b64_data, ext), None
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        finish_reason = candidate.get("finishReason")
        if finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
            texts.append(f"finishReason={finish_reason}")
    return None, ("; ".join(texts) if texts else None)


class GeminiImageGenProvider(StaticImageGenProvider):
    """Google AI Studio ``models/{model}:generateContent`` backend for Nano Banana image generation."""

    provider_id = "gemini"
    label = "Google AI Studio"
    models = MODELS
    default_model_id = DEFAULT_MODEL
    price = "varies"

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Google AI Studio (direct)",
            "badge": "API key",
            "tag": "Native Gemini (Nano Banana) image generation & editing with GOOGLE_API_KEY or GEMINI_API_KEY",
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

    def is_available(self) -> bool:
        return bool(_resolve_endpoint()[1])

    def capabilities(self) -> Dict[str, Any]:
        _, meta = _resolve_model()
        return {
            "modalities": ["text", "image"],
            "max_reference_images": int(meta.get("max_refs") or _MAX_REFERENCE_IMAGES),
            "supports_upscale": True,
        }

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect, wire_aspect = _resolve_exact_aspect_ratio(
            aspect_ratio, kwargs.get("aspect_ratio_exact"))
        if not prompt:
            return prompt_required_error("gemini", aspect)
        base_url, api_key = _resolve_endpoint()
        if not api_key:
            return error_factory("gemini", aspect)(
                "Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set (and image_gen.gemini.key_env is empty). "
                "Run `hermes tools` → Image Generation → Google AI Studio to configure.",
                "auth_required")

        model_id, meta = _resolve_model(kwargs.get("model"))
        max_refs = int(meta.get("max_refs") or _MAX_REFERENCE_IMAGES)
        sources = collect_source_images(image_url, reference_image_urls, limit=max_refs)
        is_edit = bool(sources)
        fail = error_factory("gemini", aspect, model=model_id, prompt=prompt)

        parts: List[Dict[str, Any]] = []
        for ref in sources:
            try:
                img_bytes, mime = _load_image_bytes(ref)
            except Exception as exc:  # noqa: BLE001
                return fail(f"Could not load reference image: {exc}", "invalid_argument")
            parts.append({
                "inlineData": {"mimeType": mime, "data": base64.b64encode(img_bytes).decode("ascii")},
            })
        parts.append({"text": prompt})

        image_config: Dict[str, Any] = {"aspectRatio": wire_aspect}
        image_size = _resolve_image_size(
            meta, kwargs.get("image_size") or kwargs.get("resolution"), bool(kwargs.get("upscale")))
        if image_size:
            image_config["imageSize"] = image_size

        payload: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": image_config},
        }
        use_google_search = _resolve_google_search(kwargs.get("google_search"))
        if use_google_search:
            payload["tools"] = [{"googleSearch": {}}]

        url = f"{base_url}/models/{quote(meta['api_model'], safe='')}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        body, failure = post_json(
            url, headers=headers, payload=payload, timeout=_DEFAULT_TIMEOUT,
            label="Google AI Studio", error_message=_extract_error_message,
            catch_request_exception=True)
        if failure is not None:
            err_type = "auth_required" if failure.status in (401, 403) else failure.error_type
            return fail(failure.error, err_type)
        if not isinstance(body, dict):
            return fail("Google AI Studio returned a non-object JSON payload", "invalid_response")

        # Gemini bills per text/image token; recorded before extraction so a billed
        # HTTP 200 with a text refusal / empty image still lands in session accounting.
        if isinstance(body.get("usageMetadata"), dict):
            record_token_usage(
                _usage_from_metadata(body["usageMetadata"]),
                model=meta["api_model"], provider="gemini", base_url=base_url)

        extracted, reason = _extract_inline_image(body)
        if not extracted:
            detail = f": {reason}" if reason else ""
            return fail(f"Google AI Studio returned no image data{detail}", "empty_response")

        b64_data, ext = extracted
        try:
            saved_path = save_b64_image(b64_data, prefix=f"gemini_{model_id}", extension=ext)
        except Exception as exc:  # noqa: BLE001
            return fail(f"Google AI Studio returned undecodable image data: {exc}", "invalid_response")

        extra: Dict[str, Any] = {"api": "google-ai-studio", "exact_aspect_ratio": wire_aspect}
        if image_size:
            extra["image_size"] = image_size
        if kwargs.get("upscale") and image_size in ("2K", "4K"):
            extra["upscaled"] = True
        if use_google_search:
            extra["google_search"] = True
        return success_response(
            image=str(saved_path), model=model_id, prompt=prompt, aspect_ratio=aspect,
            provider="gemini", modality="image" if is_edit else "text", extra=extra)


def register(ctx) -> None:
    """Plugin entry point — wire ``GeminiImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(GeminiImageGenProvider())

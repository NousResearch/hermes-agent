"""Google AI Studio Gemini image generation (Nano Banana 2 / Lite / Pro);
base64 inlineData → image cache. Selection: the caller's ``model`` (honoured only when it names a
catalog entry) → ``GEMINI_IMAGE_MODEL`` → ``image_gen.gemini.model`` → :data:`DEFAULT_MODEL`. The
shared top-level ``image_gen.model`` is deliberately NOT honoured: it is provider-agnostic and may
hold another backend's id.
Endpoint: ``image_gen.gemini.base_url`` → the named endpoint ``image_gen.gemini.provider`` →
``GEMINI_BASE_URL`` → :data:`BASE_URL`; key: env named by ``image_gen.gemini.key_env`` → the named
endpoint's credential → ``GOOGLE_API_KEY`` → ``GEMINI_API_KEY``.
Resolution (``imageSize``), an exact aspect ratio and Google Search grounding are configured
through ``image_gen.gemini.*`` or the matching ``GEMINI_IMAGE_*`` env vars — the shared
``image_generate`` tool schema carries no arguments for them.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, save_b64_image, success_response)
from agent.secret_scope import get_secret
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory, load_image_gen_config,
    post_json, prompt_required_error, record_token_usage, resolve_static_model)

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-image"

_ASPECT_RATIOS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
_GEMINI_RATIOS_10 = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
)
_GEMINI_RATIOS_14 = _GEMINI_RATIOS_10 + ("1:4", "4:1", "1:8", "8:1")
_MAX_REFERENCE_IMAGES = 14
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB per reference image
# Gemini caps an inline (non-Files-API) request at ~100 MB in total, so a per-image limit alone is
# not enough: 14 references at 25 MB would assemble ~470 MB of base64 before the server rejects it.
# Measured against the encoded payload, which runs ~4/3 of the raw bytes.
_MAX_INLINE_REQUEST_BYTES = 80 * 1024 * 1024
_DEFAULT_TIMEOUT = (15.0, 120.0)

# mime → cache-file extension. Local copy: a plugin should not import a private core name, and the
# set of formats Gemini can return inline is small and stable.
_MIME_TO_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/webp": "webp", "image/gif": "gif",
}

MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-3.1-flash-image": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "speed": "Fast",
        "strengths": "Best balance — 14 aspect ratios, 512/1K/2K/4K resolution, Google Search, up to 14 refs",
        "api_model": "gemini-3.1-flash-image",
        "aspect_ratios": _GEMINI_RATIOS_14,
        "resolutions": ("512", "1K", "2K", "4K"),
        "supports_search": True,
        "max_refs": 14,
    },
    "gemini-3.1-flash-lite-image": {
        "display": "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "speed": "Fastest",
        "strengths": "Lowest latency & cost; 10 aspect ratios, 1K output, up to 14 reference images",
        "api_model": "gemini-3.1-flash-lite-image",
        "aspect_ratios": _GEMINI_RATIOS_10,
        "resolutions": ("1K",),
        "supports_search": False,
        "max_refs": 14,
    },
    "gemini-3-pro-image": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "Slower",
        "strengths": "Highest fidelity & reasoning — 10 aspect ratios, 1K/2K/4K, Google Search, up to 14 refs",
        "api_model": "gemini-3-pro-image",
        "aspect_ratios": _GEMINI_RATIOS_10,
        "resolutions": ("1K", "2K", "4K"),
        "supports_search": True,
        "max_refs": 14,
    },
}


def _strip_google_prefix(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    return raw.split("/", 1)[1].strip() if raw.lower().startswith("google/") else raw


def _custom_model_meta(model_id: str) -> Dict[str, Any]:
    """Conservative metadata for an uncatalogued Gemini id: the ratio set common to every image
    model, no ``imageSize`` and no search grounding.

    A model we do not know may not accept the newer ``imageConfig``/``tools`` fields, and sending
    them earns a remote 400 that reads like a Hermes bug. Omitting a capability the model happens
    to support only costs the user that extra, and mirrors ``resolve_static_model`` dropping
    ``quality`` on passthrough (#97928).
    """
    return {
        "display": model_id,
        "api_model": model_id,
        "aspect_ratios": _GEMINI_RATIOS_10,
        "resolutions": (),
        "supports_search": False,
        "max_refs": _MAX_REFERENCE_IMAGES,
    }


def _resolve_model(caller_model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """``(model_id, meta)`` — caller ``model`` (only when it names a catalog entry) →
    ``GEMINI_IMAGE_MODEL`` → ``image_gen.gemini.model`` → :data:`DEFAULT_MODEL`.

    ``image_generate`` passes the shared top-level ``image_gen.model`` down as the ``model`` kwarg,
    and that key is provider-agnostic: with ``image_gen: {provider: gemini, model: gpt-image-2}``
    it names an OpenAI model. Honouring it verbatim would POST to ``models/gpt-image-2`` and return
    a remote 404 instead of a local fallback, so membership is checked first — the same rule the
    xAI backend applies to its own caller kwarg. Every other source is resolved by
    :func:`resolve_static_model`, which already refuses the shared top-level key while still
    allowing a deliberately provider-scoped custom id through.
    """
    cleaned = _strip_google_prefix(caller_model)
    if cleaned and cleaned in MODELS:
        return cleaned, MODELS[cleaned]
    if cleaned:
        logger.debug(
            "Ignoring image model id %r: not a Gemini model. Set image_gen.gemini.model or "
            "GEMINI_IMAGE_MODEL to choose one.", cleaned)
    model_id, _ = resolve_static_model(
        MODELS, DEFAULT_MODEL, env_var="GEMINI_IMAGE_MODEL", config_key="gemini", passthrough=True)
    normalized = _strip_google_prefix(model_id) or model_id
    return normalized, MODELS.get(normalized) or _custom_model_meta(normalized)


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


def _resolve_endpoint(cfg: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """``(base_url, api_key)`` — ``image_gen.gemini.base_url`` → named endpoint → ``GEMINI_BASE_URL`` →
    :data:`BASE_URL`; the env var named by ``image_gen.gemini.key_env`` → named endpoint →
    ``GOOGLE_API_KEY`` → ``GEMINI_API_KEY``. Shared by ``is_available()`` and ``generate()``."""
    cfg = load_image_gen_config("gemini") if cfg is None else cfg
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
    meta: Dict[str, Any], aspect_ratio: Optional[str], cfg: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """``(semantic_aspect, wire_aspect)`` — an exact ratio (the ``aspect_ratio`` argument itself →
    ``GEMINI_IMAGE_ASPECT_RATIO`` → ``image_gen.gemini.aspect_ratio``) wins when listed in
    ``meta['aspect_ratios']``; otherwise the semantic landscape/square/portrait mapping applies."""
    semantic = resolve_aspect_ratio(aspect_ratio)
    supported: Tuple[str, ...] = tuple(meta.get("aspect_ratios") or _GEMINI_RATIOS_10)
    raw_arg = (aspect_ratio or "").strip()
    cfg = load_image_gen_config("gemini") if cfg is None else cfg
    for candidate in (
        raw_arg if raw_arg in supported else None,
        os.environ.get("GEMINI_IMAGE_ASPECT_RATIO"),
        cfg.get("aspect_ratio"),
    ):
        if isinstance(candidate, str) and candidate.strip() in supported:
            exact = candidate.strip()
            w, h = (int(x) for x in exact.split(":"))
            return ("square" if w == h else ("landscape" if w > h else "portrait")), exact
    # resolve_aspect_ratio only ever returns one of the three semantic names, so the mapping hits.
    return semantic, _ASPECT_RATIOS[semantic]


def _resolve_image_size(
    meta: Dict[str, Any], upscale: bool = False, cfg: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """``imageConfig.imageSize`` (``"512"``/``"1K"``/``"2K"``/``"4K"``) when *meta* supports it:
    ``upscale`` → ``GEMINI_IMAGE_SIZE`` → ``image_gen.gemini.image_size``/``.resolution``.
    ``None`` leaves the field off, which is also what an uncatalogued model gets."""
    supported: Tuple[str, ...] = tuple(meta.get("resolutions") or ())
    if not supported:
        return None
    cfg = load_image_gen_config("gemini") if cfg is None else cfg
    for candidate in (
        "2K" if upscale and "2K" in supported else ("4K" if upscale and "4K" in supported else None),
        os.environ.get("GEMINI_IMAGE_SIZE"),
        cfg.get("image_size") or cfg.get("resolution"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip().upper()
            if normalized in supported:
                return normalized
    return None


def _resolve_google_search(meta: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Whether ``tools: [{"googleSearch": {}}]`` grounding is enabled, when *meta* supports it:
    ``GEMINI_IMAGE_GOOGLE_SEARCH`` → ``image_gen.gemini.google_search``/``.search_grounding``."""
    if not meta.get("supports_search"):
        return False
    env_val = os.environ.get("GEMINI_IMAGE_GOOGLE_SEARCH", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    if env_val in ("0", "false", "no", "off"):
        return False
    cfg = load_image_gen_config("gemini") if cfg is None else cfg
    return bool(cfg.get("google_search") or cfg.get("search_grounding"))


def _sniff_mime(data: bytes) -> str:
    """The image MIME type of *data*, read from its magic bytes.

    Magic bytes rather than the declared ``Content-Type`` or the file extension: for the formats
    Gemini accepts inline those bytes are authoritative, so a payload whose label disagrees with
    its content is mislabelled by definition. Raising here names the offending reference, where a
    trusted label would inline an HTML error page and earn an opaque remote 400 instead.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise ValueError("not a recognised image (expected PNG, JPEG, WebP or GIF)")


def _cap_message(ref: str) -> str:
    return f"exceeds the {_MAX_INPUT_IMAGE_BYTES // (1024 * 1024)}MB per-image cap: {ref}"


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load ``(data, mime_type)`` from a URL, data URI or local path; raises on IO/network/security
    error, on a payload that is not a supported image, or past :data:`_MAX_INPUT_IMAGE_BYTES`."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        from tools.url_safety import create_ssrf_safe_client, is_safe_url

        if not is_safe_url(ref):
            raise ValueError(f"Image reference URL failed the SSRF safety check: {ref}")
        # Streamed so an oversized body is dropped mid-flight instead of being buffered in full and
        # only then measured. Redirect hops need no further check here: the SSRF-safe client guards
        # at the TCP connect layer, validating (and dialling) a vetted IP on every hop.
        with create_ssrf_safe_client(timeout=60.0, follow_redirects=True) as client:
            with client.stream("GET", ref) as resp:
                resp.raise_for_status()
                declared = (resp.headers.get("Content-Length") or "").strip()
                if declared.isdigit() and int(declared) > _MAX_INPUT_IMAGE_BYTES:
                    raise ValueError(f"Image reference URL {_cap_message(ref)}")
                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > _MAX_INPUT_IMAGE_BYTES:
                        raise ValueError(f"Image reference URL {_cap_message(ref)}")
                    chunks.append(chunk)
        raw = b"".join(chunks)
        return raw, _sniff_mime(raw)
    if lower.startswith("data:"):
        _, sep, b64 = ref.partition(",")
        if not sep:
            raise ValueError("image data URI is missing its payload")
        # base64 inflates ~4/3, so the encoded length bounds the decode before it allocates.
        if len(b64) > (_MAX_INPUT_IMAGE_BYTES // 3) * 4 + 4:
            raise ValueError(f"Image data URI {_cap_message('data:')}")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"Image data URI {_cap_message('data:')}")
        return raw, _sniff_mime(raw)
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before local bytes

    raise_if_read_blocked(ref)
    path = Path(os.path.expanduser(ref))
    # stat() first so an oversized file is refused without reading it into memory.
    if path.is_file() and path.stat().st_size > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path {_cap_message(ref)}")
    raw = path.read_bytes()
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path {_cap_message(ref)}")
    return raw, _sniff_mime(raw)


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
                    ext = _MIME_TO_EXT.get(mime)
                    if not ext:
                        # No/unknown mimeType: sniff the leading bytes rather than assume PNG, so a
                        # WebP or JPEG still lands in the cache under its true extension.
                        try:
                            ext = _MIME_TO_EXT.get(
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


def _usage_from_gemini_metadata(usage_meta: Dict[str, Any]) -> SimpleNamespace:
    """Gemini's camelCase ``usageMetadata`` → the snake_case shape ``record_token_usage`` reads.

    ``thoughtsTokenCount`` is folded into the completion count. Nano Banana Pro reasons before it
    draws, and ``totalTokenCount`` already counts those tokens, so omitting them records a row
    where prompt + completion does not reconcile with the total — on a pro image call the
    shortfall is most of the billed work. Core's converter still leaves them out (upstream
    #103002 / #103205 propose the same fix there); this is a deliberate, local divergence rather
    than a fork of core behaviour.
    """
    def count(key: str) -> int:
        value = usage_meta.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    return SimpleNamespace(
        prompt_tokens=count("promptTokenCount"),
        completion_tokens=count("candidatesTokenCount") + count("thoughtsTokenCount"),
        total_tokens=count("totalTokenCount"),
        prompt_tokens_details=SimpleNamespace(cached_tokens=count("cachedContentTokenCount")),
    )


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
        resolutions = set(meta.get("resolutions") or ())
        return {
            "modalities": ["text", "image"],
            "max_reference_images": int(meta.get("max_refs") or _MAX_REFERENCE_IMAGES),
            # Advertise the knob only when this model has a rung above 1K to climb to. The lite
            # model is 1K-only, so offering `upscale` there would accept the argument and quietly
            # do nothing; the tool hides the parameter entirely when this is False.
            "supports_upscale": bool(resolutions & {"2K", "4K"}),
        }

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        # One snapshot for the whole call: load_image_gen_config deep-copies on every read, and the
        # resolvers below would otherwise re-read it four times and could disagree if the file
        # changed underneath a long generation.
        cfg = load_image_gen_config("gemini")
        model_id, meta = _resolve_model(kwargs.get("model"))
        aspect, wire_aspect = _resolve_exact_aspect_ratio(meta, aspect_ratio, cfg)
        if not prompt:
            return prompt_required_error("gemini", aspect)
        base_url, api_key = _resolve_endpoint(cfg)
        if not api_key:
            return error_factory("gemini", aspect)(
                "Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set (and image_gen.gemini.key_env is empty). "
                "Run `hermes tools` → Image Generation → Google AI Studio to configure.",
                "auth_required")

        max_refs = int(meta.get("max_refs") or _MAX_REFERENCE_IMAGES)
        sources = collect_source_images(image_url, reference_image_urls, limit=max_refs)
        is_edit = bool(sources)
        fail = error_factory("gemini", aspect, model=model_id, prompt=prompt)

        parts: List[Dict[str, Any]] = []
        encoded_total = 0
        for ref in sources:
            try:
                img_bytes, mime = _load_image_bytes(ref)
            except Exception as exc:  # noqa: BLE001
                return fail(f"Could not load reference image: {exc}", "invalid_argument")
            encoded = base64.b64encode(img_bytes).decode("ascii")
            # Each image is individually under the per-image cap, but the request as a whole still
            # has to fit Gemini's inline ceiling. Fail here, naming the reference we stopped on,
            # rather than assembling hundreds of MB only for the server to refuse it.
            encoded_total += len(encoded)
            if encoded_total > _MAX_INLINE_REQUEST_BYTES:
                limit_mb = _MAX_INLINE_REQUEST_BYTES // (1024 * 1024)
                return fail(
                    f"Reference images exceed the {limit_mb}MB total request limit (stopped at "
                    f"{ref}). Use fewer or smaller references.", "invalid_argument")
            parts.append({"inlineData": {"mimeType": mime, "data": encoded}})
        parts.append({"text": prompt})

        image_config: Dict[str, Any] = {"aspectRatio": wire_aspect}
        image_size = _resolve_image_size(meta, bool(kwargs.get("upscale")), cfg)
        if image_size:
            image_config["imageSize"] = image_size

        payload: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": image_config},
        }
        use_google_search = _resolve_google_search(meta, cfg)
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
                _usage_from_gemini_metadata(body["usageMetadata"]),
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

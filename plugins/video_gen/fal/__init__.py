"""FAL.ai video generation backend.

The user picks a **model family** (e.g. "Pixverse v6"); the plugin routes to its text-to-video endpoint without
``image_url`` and to its image-to-video endpoint otherwise (gemini-omni-flash is i2v only). Active-family precedence:
tool ``model=`` → ``FAL_VIDEO_MODEL`` env → ``video_gen.fal.model`` → ``video_gen.model`` (family id or an endpoint
path containing one) → ``DEFAULT_MODEL``. Auth via ``FAL_KEY`` or the managed Nous gateway; output is an HTTPS URL.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import VideoGenProvider, error_response, success_response

logger = logging.getLogger(__name__)

# Family catalog. Capability flags gate which keys reach the payload — keys a family doesn't advertise are never sent (the
# managed gateway forwards everything verbatim). Enums default to None (endpoint decides), flags to False. ``durations`` is an
# enum tuple OR a ``(min, max)`` range (2 ints with gap > 1). Extras: audio_native (always on; description line only),
# duration_int (JSON int, default queue-API string), duration_suffix ("4s"), image_param_key (i2v key when not `image_url`),
# image_drop_keys (i2v endpoint rejects), resolution_aliases (tool value → endpoint enum), static_payload (always required).
def _family(display: str, speed: str, tier: str, strengths: str, text: Optional[str], image: str, **caps: Any) -> Dict[str, Any]:
    return {"display": display, "speed": speed, "price": tier, "tier": tier, "strengths": strengths, "text_endpoint": text, "image_endpoint": image,
            "aspect_ratios": None, "resolutions": None, "durations": None, "audio": False, "negative": False, "seed": False, **caps}


_SIX_ASPECTS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
# MiniMax H3 uses capitalized/2K-style resolution enums; aliases map the tool's usual values. Max tops out at 768P.
_H3_ALIASES = {"480p": "768P", "540p": "768P", "720p": "768P", "768p": "768P", "1080p": "2K", "2k": "2K", "4k": "4K", "2160p": "4K"}
_H3_MAX_ALIASES = {"480p": "480P", "540p": "480P", "720p": "768P", "768p": "768P", "1080p": "768P", "2k": "768P", "4k": "768P", "2160p": "768P"}

FAL_FAMILIES: Dict[str, Dict[str, Any]] = {
    # ─── Cheap / fast tier ─────────────────────────────────────────────
    "ltx-2.3": _family("LTX 2.3 (22B)", "~30-60s", "cheap", "22B model with native audio generation. Affordable.",  # docs expose no enums
                       "fal-ai/ltx-2.3-22b/text-to-video", "fal-ai/ltx-2.3-22b/image-to-video", audio=True, negative=True, seed=True),
    "pixverse-v6": _family("Pixverse v6", "~30-90s", "cheap", "Affordable. Negative prompts. 1-15s durations.", "fal-ai/pixverse/v6/text-to-video",
                           "fal-ai/pixverse/v6/image-to-video", resolutions=("360p", "540p", "720p", "1080p"), durations=(1, 15), audio=True, negative=True, seed=True),
    "seedance-2.0-mini": _family("Seedance 2.0 Mini", "~30-90s", "cheap", "ByteDance. Faster/cheaper Seedance tier, audio + lip-sync, 4-15s.",
                                 "bytedance/seedance-2.0/mini/text-to-video", "bytedance/seedance-2.0/mini/image-to-video", aspect_ratios=_SIX_ASPECTS,
                                 resolutions=("480p", "720p"), durations=(4, 15), audio=True),
    # ─── Expensive / premium tier ──────────────────────────────────────
    "veo3.1": _family("Veo 3.1", "~60-120s", "premium", "Google DeepMind. Cinematic, native audio, strong prompt adherence.", "fal-ai/veo3.1",
                      "fal-ai/veo3.1/image-to-video", aspect_ratios=("16:9", "9:16"), resolutions=("720p", "1080p", "4k"), durations=(4, 6, 8),
                      duration_suffix="s", audio=True, negative=True, seed=True),  # wants "4s" not "4"
    "seedance-2.0": _family("Seedance 2.0", "~60-120s", "premium", "ByteDance. Cinematic, synchronized audio + lip-sync, 4-15s.",  # no "auto" aspect, no `seed`
                            "bytedance/seedance-2.0/text-to-video", "bytedance/seedance-2.0/image-to-video", aspect_ratios=_SIX_ASPECTS,
                            resolutions=("480p", "720p", "1080p"), durations=(4, 15), audio=True),
    "seedance-2.5": _family("Seedance 2.5", "~60-180s", "premium", "ByteDance flagship. Native 30s single-pass, audio in the same latent space, lip-sync.",
                            "bytedance/seedance-2.5/text-to-video", "bytedance/seedance-2.5/image-to-video", aspect_ratios=_SIX_ASPECTS,
                            image_drop_keys=("aspect_ratio",), resolutions=("480p", "720p"), durations=(4, 30), audio=True),  # i2v aspect is "auto" only
    "minimax-h3": _family("MiniMax H3", "~60-180s", "premium", "MiniMax frontier. Native 2K (up to 4K), 5-15s, seven aspect ratios.",
                          "minimax/h3/text-to-video", "minimax/h3/image-to-video", duration_int=True, image_drop_keys=("aspect_ratio",),  # i2v follows image
                          aspect_ratios=_SIX_ASPECTS, resolutions=("768P", "2K", "4K"), resolution_aliases=_H3_ALIASES, durations=(5, 15), audio_native=True),
    # i2v schema doesn't declare aspect_ratio; unlike base H3, Max declares `seed` on both endpoints; static key is in the schema's required array.
    "minimax-h3-max": _family("MiniMax H3 Max (fal post-train)", "~5-30s", "premium", "fal's post-trained MiniMax H3. Top-ranked quality/prompt "
                              "adherence/aesthetics, 768p in seconds, 5-15s.", "minimax/h3-max/text-to-video", "minimax/h3-max/image-to-video",
                              duration_int=True, image_drop_keys=("aspect_ratio",), aspect_ratios=_SIX_ASPECTS, resolutions=("480P", "768P"),
                              resolution_aliases=_H3_MAX_ALIASES, durations=(5, 15), static_payload={"prompt_expansion_mode": "balanced"}, audio_native=True, seed=True),
    "flux-3": _family("FLUX 3 (via FAL)", "~60-120s", "premium", "Black Forest Labs frontier video. Native audio, 5-20s, 8 aspect ratios.",
                      "blackforestlabs/flux-3/text-to-video", "blackforestlabs/flux-3/image-to-video", duration_int=True,  # enum "auto" | 5..20 ints
                      aspect_ratios=("21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16"), resolutions=("720p", "1080p"), durations=(5, 20), audio=True),
    "grok-imagine-1.5": _family("Grok Imagine 1.5 (via FAL)", "~30-90s", "premium", "xAI. Fast stylized video with audio, 1-15s, cheap per second.",
                                "xai/grok-imagine-video/v1.5/text-to-video", "xai/grok-imagine-video/v1.5/image-to-video", duration_int=True,
                                image_drop_keys=("aspect_ratio",), aspect_ratios=("16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"),  # aspect is t2v-only
                                resolutions=("480p", "720p", "1080p"), durations=(1, 15), audio_native=True),
    "gemini-omni-flash": _family("Gemini Omni Flash (via FAL)", "~60-120s", "premium", "Google. Image-to-video with audio, physics-grounded motion, 3-10s.",
                                 None, "google/gemini-omni-flash/image-to-video", duration_int=True, aspect_ratios=("16:9", "9:16"), durations=(3, 10), audio_native=True),
    "kling-v3-4k": _family("Kling v3 4K", "~120-300s", "premium", "4K output, native audio (Chinese/English), 3-15s.", "fal-ai/kling-video/v3/4k/text-to-video",
                           "fal-ai/kling-video/v3/4k/image-to-video", image_param_key="start_image_url", aspect_ratios=("16:9", "9:16", "1:1"),
                           durations=(3, 15), audio=True, negative=True, seed=True),
    "happy-horse": _family("Happy Horse 1.0", "~60-120s", "premium", "Alibaba. New model, sparse public docs — conservative defaults.",
                           "alibaba/happy-horse/text-to-video", "alibaba/happy-horse/image-to-video", audio_native=True, seed=True),
}

DEFAULT_MODEL = "pixverse-v6"  # cheap, both modalities, sane defaults


def _clamp_duration(durations: Tuple[int, ...], duration: Optional[int]) -> Optional[int]:
    """Clamp into a ``(min, max)`` range (None stays None: endpoint default) or snap to the nearest enum entry (None → first).
    Range heuristic: a 2-tuple of ints with a gap > 1."""
    if len(durations) == 2 and all(isinstance(d, int) for d in durations) and durations[1] - durations[0] > 1:
        return None if duration is None else max(durations[0], min(durations[1], duration))
    return durations[0] if duration is None else min(durations, key=lambda d: abs(d - duration))


def _modalities(meta: Dict[str, Any]) -> List[str]:
    return [m for m in ("text", "image") if meta[f"{m}_endpoint"]]


def _normalize_family_key(c: str) -> Optional[str]:
    """Known family ID from a bare id, full endpoint path, truncated stem (``minimax/h3``) or provider-prefixed name."""
    c = c.strip()
    if not c or c in FAL_FAMILIES:
        return c or None
    endpoints = [(fid, ep) for fid, meta in FAL_FAMILIES.items()
                 for ep in (meta["text_endpoint"], meta["image_endpoint"]) if isinstance(ep, str)]
    # Exact declared endpoint beats any segment scan (which would see "seedance-2.0" inside ".../seedance-2.0/mini/...").
    # Truncated stem: the segment after ``c`` must be a modality leaf so "bytedance/seedance-2.0" skips Mini's deeper path.
    # Last resort: longest family-id path-segment match (prefers seedance-2.0-mini over seedance-2.0 when both appear).
    hit = ([fid for fid, ep in endpoints if c == ep]
           or [fid for fid, ep in endpoints if ep.startswith(c + "/") and ep[len(c) + 1:].split("/", 1)[0] in ("text-to-video", "image-to-video")]
           or sorted((fid for fid in FAL_FAMILIES if fid in c.split("/")), key=len, reverse=True))
    return hit[0] if hit else None


def _resolve_family(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Decide which FAL family to use. Returns ``(family_id, meta)``."""
    import os
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("Could not load video_gen config: %s", exc)
        cfg = None
    cfg = cfg.get("video_gen") if isinstance(cfg, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    fal_cfg = cfg.get("fal") if isinstance(cfg.get("fal"), dict) else {}
    for c in (explicit, os.environ.get("FAL_VIDEO_MODEL"), fal_cfg.get("model"), cfg.get("model")):
        fid = _normalize_family_key(c) if isinstance(c, str) else None
        if fid:
            return fid, FAL_FAMILIES[fid]
    return DEFAULT_MODEL, FAL_FAMILIES[DEFAULT_MODEL]


def _build_payload(family: Dict[str, Any], *, prompt: str, image_url: Optional[str], duration: Optional[int], aspect_ratio: str,
                   resolution: str, negative_prompt: Optional[str], audio: Optional[bool], seed: Optional[int]) -> Dict[str, Any]:
    """Build a family-specific payload, dropping keys the family doesn't declare (unsupported enums → endpoint default)."""
    resolved = (family.get("resolution_aliases") or {}).get((resolution or "").lower(), resolution)
    clamped = _clamp_duration(family["durations"], duration) if family["durations"] else None
    payload: Dict[str, Any] = {key: value for ok, key, value in (
        (prompt, "prompt", prompt),
        (image_url, family.get("image_param_key") or "image_url", image_url),
        # Newer endpoints declare no `seed` and the managed gateway forwards whatever we send — gate on the family.
        (seed is not None and family.get("seed", True), "seed", seed),
        (family["aspect_ratios"] and aspect_ratio in family["aspect_ratios"], "aspect_ratio", aspect_ratio),
        (family["resolutions"] and resolved in family["resolutions"], "resolution", resolved),
        # FAL's queue API types duration as a string ("8" not 8) unless the family says int; veo3.1 also wants a unit suffix.
        (clamped is not None, "duration", clamped if family.get("duration_int") else f"{clamped}{family.get('duration_suffix', '')}"),
        (family["audio"] and audio is not None, "generate_audio", bool(audio)),
        (family["negative"] and negative_prompt, "negative_prompt", negative_prompt),
    ) if ok}
    for key in family.get("image_drop_keys", ()) if image_url else ():  # keys the i2v endpoint rejects outright
        payload.pop(key, None)
    for key, value in (family.get("static_payload") or {}).items():  # constants the endpoint always requires
        payload.setdefault(key, value)
    return payload


def _video_url_from_result(result: Any) -> Tuple[Any, Optional[str]]:
    """Return ``(video_field, url)`` from a FAL result dict (url None if absent)."""
    video = result.get("video") if isinstance(result, dict) else None
    url = video.get("url") if isinstance(video, dict) else video if isinstance(video, str) else None
    return video, url or None


_fal_client: Any = None
_fal_client_lock = threading.Lock()

_managed_fal_video_client: Any = None
_managed_fal_video_client_config: Any = None
_managed_fal_video_client_lock = threading.Lock()


def _load_fal_client() -> Any:
    """Lazy-load ``fal_client`` once via ``tools.fal_common``."""
    global _fal_client
    with _fal_client_lock:
        if _fal_client is None:
            from tools.fal_common import import_fal_client
            _fal_client = import_fal_client()
        return _fal_client


def _resolve_managed_fal_video_gateway():
    """Resolve the FAL video route from the stored ``video_gen`` selection: ``"nous"`` → managed only (unentitled ⇒
    selection-naming error); other stored provider → direct only (missing FAL_KEY ⇒ error); never-configured → autodetect."""
    from tools.managed_tool_gateway import resolve_managed_tool_gateway
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, fal_key_is_configured, read_selection, selection_error
    selected = read_selection("video_gen")
    if selected == NOUS_MANAGED_PROVIDER:
        gateway = resolve_managed_tool_gateway("fal-queue")
        if gateway is None:
            raise ValueError(selection_error("video_gen", NOUS_MANAGED_PROVIDER, "the Nous Tool Gateway is not available (not entitled or unreachable)"))
        return gateway
    if selected is not None:
        if not fal_key_is_configured():
            raise ValueError(selection_error("video_gen", selected, "FAL_KEY is not set"))
        return None
    return None if fal_key_is_configured() else resolve_managed_tool_gateway("fal-queue")


def _fal_video_available() -> bool:
    """True if the selected (or, never-configured, any) FAL backend is reachable; raises on a stored-but-broken selection."""
    from tools.tool_backend_helpers import fal_key_is_configured
    return _resolve_managed_fal_video_gateway() is not None or fal_key_is_configured()


def _get_managed_fal_video_client(managed_gateway):
    """Reuse the managed FAL client so its internal httpx.Client is not leaked per call."""
    global _managed_fal_video_client, _managed_fal_video_client_config
    from tools.fal_common import _ManagedFalSyncClient
    client_config = (managed_gateway.gateway_origin.rstrip("/"), managed_gateway.nous_user_token)
    with _managed_fal_video_client_lock:
        if _managed_fal_video_client is None or _managed_fal_video_client_config != client_config:
            _managed_fal_video_client = _ManagedFalSyncClient(_load_fal_client(), key=managed_gateway.nous_user_token,
                                                              queue_run_origin=managed_gateway.gateway_origin)
            _managed_fal_video_client_config = client_config
        return _managed_fal_video_client


def _submit_fal_video_request(endpoint: str, arguments: Dict[str, Any]):
    """Submit via direct credentials or the managed queue gateway; ``.get()`` blocks."""
    client = _load_fal_client()
    headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_video_gateway()
    if managed_gateway is None:
        return client.submit(endpoint, arguments=arguments, headers=headers)
    try:
        return _get_managed_fal_video_client(managed_gateway).submit(endpoint, arguments=arguments, headers=headers)
    except Exception as exc:
        from tools.fal_common import _extract_http_status
        status = _extract_http_status(exc)
        if status is not None and 400 <= status < 500:
            raise ValueError(f"Nous Subscription gateway rejected endpoint '{endpoint}' (HTTP {status}). This model may not yet be enabled "
                             f"on the Nous Portal's FAL proxy. Either:\n  • Set FAL_KEY in your environment to use FAL.ai directly, or\n"
                             f"  • Pick a different model via `hermes tools` → Video Generation.") from exc
        raise


# ByteDance SeedVR2 on FAL: $0.001/megapixel of output; a 5s 720p→1440p 2x pass is roughly $0.44.
UPSCALER_ENDPOINT = "fal-ai/seedvr/upscale/video"
UPSCALER_FACTOR = 2


def _upscale_video(video_url: str, source_request_id: Optional[str] = None) -> Optional[str]:
    """Best-effort SeedVR2 upscale; returns the new URL or None (never raises)."""
    try:
        logger.info("Upscaling video with SeedVR2 (%dx)...", UPSCALER_FACTOR)
        arguments: Dict[str, Any] = {"video_url": video_url, "upscale_mode": "factor", "upscale_factor": UPSCALER_FACTOR}
        if _resolve_managed_fal_video_gateway() is not None:
            if not source_request_id:
                raise RuntimeError("Managed SeedVR upscale requires the source FAL request id")
            arguments["source_request_id"] = source_request_id
        result = _submit_fal_video_request(UPSCALER_ENDPOINT, arguments).get()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Video upscale failed: %s", exc)
        return None
    _video, url = _video_url_from_result(result)
    if not url:
        logger.warning("Video upscaler returned no URL")
    return url


_NO_BACKEND_MSG = ("No FAL backend available. Either set FAL_KEY (run `hermes tools` → Video Generation → FAL to configure) "
                   "or sign in to Nous (`hermes setup`) for managed gateway access.")
_MODALITY_MISSING_MSG = {
    "image": "FAL family {fid} has no image-to-video endpoint. Pick a family with image-to-video support via `hermes tools` → Video Generation.",
    "text": "FAL family {fid} has no text-to-video endpoint. Pass an image_url to use its image-to-video endpoint, or pick a different family.",
}


def _fal_error(error: str, error_type: str, prompt: str, model: str = "", aspect_ratio: str = "") -> Dict[str, Any]:
    return error_response(error=error, error_type=error_type, provider="fal", model=model, prompt=prompt, aspect_ratio=aspect_ratio)


class FALVideoGenProvider(VideoGenProvider):
    """FAL.ai multi-family backend; routes t2v/i2v on ``image_url`` presence."""

    name = "fal"
    display_name = "FAL"

    def is_available(self) -> bool:
        # A stored-but-broken selection raises the selection-naming ValueError; report unavailable, never break the picker.
        try:
            return _fal_video_available()
        except Exception:  # noqa: BLE001
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": fid, **{k: meta[k] for k in ("display", "speed", "strengths", "price", "tier")}, "modalities": _modalities(meta),
                 **({"min_duration": min(d), "max_duration": max(d)} if (d := meta["durations"]) else {})}
                for fid, meta in FAL_FAMILIES.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": "FAL", "badge": "paid", "env_vars": [{"key": "FAL_KEY", "prompt": "FAL.ai API key", "url": "https://fal.ai/dashboard/keys"}],
                "tag": "LTX, Pixverse, Seedance 2.0/2.5/Mini, Veo 3.1, MiniMax H3, FLUX 3, Kling 4K, Happy Horse, Grok Imagine, "
                       "Gemini Omni — text-to-video & image-to-video"}

    def capabilities(self) -> Dict[str, Any]:
        # RESOLVED family's surface so the dynamic tool schema gates params on what the selected model honors; union fallback (never raises).
        try:
            # Falls back to the cross-family union if resolution fails (never raises). See #97057.
            _family_id, family = _resolve_family(None)
        except Exception:  # noqa: BLE001
            family = None
        if family:
            durations = family["durations"] or (1, 1)
            return {"modalities": _modalities(family) or ["text"], "aspect_ratios": list(family["aspect_ratios"] or []),
                    "resolutions": list(family["resolutions"] or []), "max_duration": max(durations), "min_duration": min(durations),
                    "supports_audio": bool(family["audio"]), "audio_always_on": bool(family.get("audio_native")),  # no toggle: description only
                    "supports_negative_prompt": bool(family["negative"]), "supports_seed": bool(family["seed"]),
                    "supports_upscale": True, "max_reference_images": 0}  # SeedVR chains for any family
        spans = [m["durations"] for m in FAL_FAMILIES.values() if m["durations"]]
        return {"modalities": ["text", "image"], "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["360p", "540p", "720p", "1080p"],
                "max_duration": max([1] + [max(d) for d in spans]), "min_duration": min([min(d) for d in spans], default=1),
                "supports_audio": True, "supports_negative_prompt": True, "supports_seed": True, "supports_upscale": True, "max_reference_images": 0}

    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None, aspect_ratio: str = "16:9", resolution: str = "720p", negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None, seed: Optional[int] = None, upscale: Optional[bool] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        try:  # a stored selection that cannot run gets the honest selection-naming error from the strict resolver
            if not _fal_video_available():
                return _fal_error(_NO_BACKEND_MSG, "auth_required", prompt)
        except ValueError as exc:
            return _fal_error(str(exc), "auth_required", prompt)
        try:
            _load_fal_client()
        except ImportError:
            return _fal_error("fal_client Python package not installed (pip install fal-client)", "missing_dependency", prompt)
        prompt = (prompt or "").strip()
        family_id, family = _resolve_family(model)
        image_url_norm = (image_url or "").strip() or None
        modality_used = "image" if image_url_norm else "text"  # routes to the i2v vs t2v endpoint
        endpoint = family[f"{modality_used}_endpoint"]
        if not endpoint:
            return _fal_error(_MODALITY_MISSING_MSG[modality_used].format(fid=family_id), "modality_unsupported", prompt, model=family_id)
        if not prompt:
            return _fal_error("prompt is required.", "missing_prompt", prompt, model=family_id)
        payload = _build_payload(family, prompt=prompt, image_url=image_url_norm, duration=duration, aspect_ratio=aspect_ratio,
                                 resolution=resolution, negative_prompt=negative_prompt, audio=audio, seed=seed)
        try:
            handle = _submit_fal_video_request(endpoint, payload)
            source_request_id = getattr(handle, "request_id", None)
            video, url = _video_url_from_result(handle.get())
        except Exception as exc:
            logger.warning("FAL video gen failed (family=%s, endpoint=%s): %s", family_id, endpoint, exc, exc_info=True)
            return _fal_error(f"FAL video generation failed: {exc}", "api_error", prompt, model=family_id, aspect_ratio=aspect_ratio)
        if not url:
            return _fal_error("FAL returned no video URL in response", "empty_response", prompt, model=family_id)
        # Optional SeedVR2 pass — explicit opt-in, best-effort: failure falls back to the native video.
        upscaled_url = _upscale_video(url, source_request_id) if upscale else None
        upscaled = bool(upscaled_url)
        if upscale and not upscaled:
            logger.warning("Video upscale pass failed — returning native-resolution video")
        url = upscaled_url or url
        extra: Dict[str, Any] = {"endpoint": endpoint, "upscaled": upscaled, **({"upscale_factor": UPSCALER_FACTOR} if upscaled else {})}
        if isinstance(video, dict):  # native-resolution file_size no longer applies after an upscale
            extra.update({k: video[k] for k in (("content_type",) if upscaled else ("file_size", "content_type")) if video.get(k)})
        return success_response(
            video=url, model=family_id, prompt=prompt, modality=modality_used, provider="fal", extra=extra,
            aspect_ratio=aspect_ratio if "aspect_ratio" in payload else "",
            duration=int("".join(c for c in str(payload["duration"]) if c.isdigit()) or "0") if "duration" in payload else 0,
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``FALVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(FALVideoGenProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

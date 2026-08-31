"""FAL.ai video generation backend.

User-facing surface: pick a **model family** (e.g. "Pixverse v6",
"Veo 3.1", "Seedance 2.0", "Kling v3 4K", "LTX 2.3", "Happy Horse").
The plugin auto-routes to the family's text-to-video endpoint when
called without ``image_url``, and to its image-to-video endpoint when
``image_url`` is provided. The agent never sees the routing — it just
calls ``video_generate(prompt=..., image_url=...)``.

Model families (most expose both t2v + i2v; gemini-omni-flash is image-to-video only):

  Cheap tier:
    ltx-2.3            fal-ai/ltx-2.3-22b/text-to-video           /  fal-ai/ltx-2.3-22b/image-to-video
    pixverse-v6        fal-ai/pixverse/v6/text-to-video           /  fal-ai/pixverse/v6/image-to-video
    seedance-2.0-mini  bytedance/seedance-2.0/mini/text-to-video  /  bytedance/seedance-2.0/mini/image-to-video

  Premium tier:
    veo3.1             fal-ai/veo3.1                              /  fal-ai/veo3.1/image-to-video
    seedance-2.0       bytedance/seedance-2.0/text-to-video       /  bytedance/seedance-2.0/image-to-video
    seedance-2.5       bytedance/seedance-2.5/text-to-video       /  bytedance/seedance-2.5/image-to-video
    minimax-h3         minimax/h3/text-to-video                   /  minimax/h3/image-to-video
    minimax-h3-max     minimax/h3-max/text-to-video               /  minimax/h3-max/image-to-video
    flux-3             blackforestlabs/flux-3/text-to-video       /  blackforestlabs/flux-3/image-to-video
    grok-imagine-1.5   xai/grok-imagine-video/v1.5/text-to-video  /  xai/grok-imagine-video/v1.5/image-to-video
    kling-v3-4k        fal-ai/kling-video/v3/4k/text-to-video     /  fal-ai/kling-video/v3/4k/image-to-video
    happy-horse        alibaba/happy-horse/text-to-video          /  alibaba/happy-horse/image-to-video

  Image-to-video only (no text_endpoint):
    gemini-omni-flash  google/gemini-omni-flash/image-to-video

Selection precedence for the active family:
    1. ``model=`` arg from the tool call
    2. ``FAL_VIDEO_MODEL`` env var
    3. ``video_gen.fal.model`` in ``config.yaml``
    4. ``video_gen.model`` in ``config.yaml`` (when it's one of our family IDs
       or a full endpoint path that contains a family ID)
    5. ``DEFAULT_MODEL``

Authentication via ``FAL_KEY`` or the managed Nous gateway. Output is an
HTTPS URL from FAL's CDN; the gateway downloads and delivers it.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import VideoGenProvider, error_response, success_response

logger = logging.getLogger(__name__)

# Family catalog. Capability flags gate which keys reach the payload — keys a family doesn't advertise are never sent (the
# managed gateway forwards everything verbatim). Enums default to None (endpoint decides), flags to False. ``durations`` is always a
# ``(min, max)`` range (clamp); a family whose endpoint only accepts discrete values adds ``duration_enum`` (snap to nearest; None
# stays None so the endpoint default applies); ``duration_cap_by_resolution`` lowers the ceiling per resolution enum (applied after
# the snap/clamp). Extras: audio_native (always on; description line only),
# duration_int (JSON int, default queue-API string), duration_suffix ("4s"), image_param_key (i2v key when not `image_url`),
# image_drop_keys (i2v endpoint rejects), audio_param_key (toggle key when not `generate_audio`), resolution_aliases (tool value → endpoint enum), static_payload (always required).
def _family(display: str, speed: str, tier: str, strengths: str, text: Optional[str], image: str, **caps: Any) -> Dict[str, Any]:
    return {"display": display, "speed": speed, "price": tier, "tier": tier, "strengths": strengths, "text_endpoint": text, "image_endpoint": image,
            "aspect_ratios": None, "resolutions": None, "durations": None, "audio": False, "negative": False, "seed": False, **caps}


_SIX_ASPECTS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
# MiniMax H3 uses capitalized/2K-style resolution enums; aliases map the tool's usual values. Max tops out at 768P.
_H3_ALIASES = {"480p": "768P", "540p": "768P", "720p": "768P", "768p": "768P", "1080p": "2K", "2k": "2K", "4k": "4K", "2160p": "4K"}
_H3_MAX_ALIASES = {"480p": "480P", "540p": "480P", "720p": "768P", "768p": "768P", "1080p": "768P", "2k": "768P", "4k": "768P", "2160p": "768P"}
_H3_MAX_TURBO_ALIASES = {"480p": "480P", "540p": "480P", "720p": "768P", "768p": "768P", "1080p": "1080P", "2k": "1080P", "4k": "1080P", "2160p": "1080P"}

FAL_FAMILIES: Dict[str, Dict[str, Any]] = {
    # ─── Cheap / fast tier ─────────────────────────────────────────────
    "ltx-2.3": {
        "display": "LTX 2.3 (22B)",
        "speed": "~30-60s",
        "price": "cheap",
        "strengths": "22B model with native audio generation. Affordable.",
        "tier": "cheap",
        "text_endpoint": "fal-ai/ltx-2.3-22b/text-to-video",
        "image_endpoint": "fal-ai/ltx-2.3-22b/image-to-video",
        # LTX docs don't expose duration/aspect/resolution enums — leave
        # blank so we don't send unrecognized payload keys.
        "aspect_ratios": None,
        "resolutions": None,
        "durations": None,
        "audio": True,
        "negative": True,
        "seed": True,
    },
    "pixverse-v6": {
        "display": "Pixverse v6",
        "speed": "~30-90s",
        "price": "cheap",
        "strengths": "Affordable. Negative prompts. 1-15s durations.",
        "tier": "cheap",
        "text_endpoint": "fal-ai/pixverse/v6/text-to-video",
        "image_endpoint": "fal-ai/pixverse/v6/image-to-video",
        "aspect_ratios": None,
        "resolutions": ("360p", "540p", "720p", "1080p"),
        "durations": (1, 15),
        "audio": True,
        "negative": True,
        "seed": True,
    },
    "seedance-2.0-mini": {
        "display": "Seedance 2.0 Mini",
        "speed": "~30-90s",
        "price": "cheap",
        "strengths": "ByteDance. Faster/cheaper Seedance tier, audio + lip-sync, 4-15s.",
        "tier": "cheap",
        "text_endpoint": "bytedance/seedance-2.0/mini/text-to-video",
        "image_endpoint": "bytedance/seedance-2.0/mini/image-to-video",
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p"),
        "durations": (4, 15),
        "audio": True,
        "negative": False,
        "seed": False,
    },
    # ─── Expensive / premium tier ──────────────────────────────────────
    "veo3.1": {
        "display": "Veo 3.1",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Google DeepMind. Cinematic, native audio, strong prompt adherence.",
        "tier": "premium",
        "text_endpoint": "fal-ai/veo3.1",
        "image_endpoint": "fal-ai/veo3.1/image-to-video",
        "aspect_ratios": ("16:9", "9:16"),
        "resolutions": ("720p", "1080p", "4k"),
        "durations": (4, 6, 8),
        "duration_suffix": "s",  # FAL veo3.1 wants "4s" not "4"
        "audio": True,
        "negative": True,
        "seed": True,
    },
    "seedance-2.0": {
        "display": "Seedance 2.0",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "ByteDance. Cinematic, synchronized audio + lip-sync, 4-15s.",
        "tier": "premium",
        "text_endpoint": "bytedance/seedance-2.0/text-to-video",
        "image_endpoint": "bytedance/seedance-2.0/image-to-video",
        # Seedance accepts "auto" too — we omit it from the enum so the
        # agent can't pass it; the endpoint defaults handle the rest.
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p", "1080p"),
        "durations": (4, 15),
        "audio": True,
        "negative": False,
        # FAL input schema has no `seed` (only returned on output).
        "seed": False,
    },
    "seedance-2.5": {
        "display": "Seedance 2.5",
        "speed": "~60-180s",
        "price": "premium",
        "strengths": "ByteDance flagship. Native 30s single-pass, audio in the same latent space, lip-sync.",
        "tier": "premium",
        "text_endpoint": "bytedance/seedance-2.5/text-to-video",
        "image_endpoint": "bytedance/seedance-2.5/image-to-video",
        # i2v accepts only "auto" for aspect_ratio (it follows the input
        # image), so aspect_ratio is dropped for image jobs via
        # image_drop_keys.
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p"),
        "durations": (4, 30),
        "audio": True,
        "negative": False,
        "seed": False,
    },
    "minimax-h3": {
        "display": "MiniMax H3",
        "speed": "~60-180s",
        "price": "premium",
        "strengths": "MiniMax frontier. Native 2K (up to 4K), 5-15s, seven aspect ratios.",
        "tier": "premium",
        "text_endpoint": "minimax/h3/text-to-video",
        "image_endpoint": "minimax/h3/image-to-video",
        # H3 takes duration as a JSON integer, not the stringified form
        # most FAL endpoints use.
        "duration_int": True,
        # i2v derives the aspect ratio from the input image and rejects
        # the key entirely.
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        # H3 uses capitalized/2K-style resolution enums — mapped from the
        # tool's usual 720p/1080p-style values via resolution_aliases.
        "resolutions": ("768P", "2K", "4K"),
        "resolution_aliases": {
            "480p": "768P", "540p": "768P", "720p": "768P", "768p": "768P",
            "1080p": "2K", "2k": "2K", "4k": "4K", "2160p": "4K",
        },
        "durations": (5, 15),
        "audio": False,  # no generate_audio TOGGLE — audio is always on
        "audio_native": True,  # native audio in every generation (fal docs)  # audio is native/always-on; no generate_audio key
        "negative": False,
        "seed": False,
    },
    "minimax-h3-max": {
        "display": "MiniMax H3 Max (fal post-train)",
        "speed": "~5-30s",
        "price": "premium",
        "strengths": "fal's post-trained MiniMax H3. Top-ranked quality/prompt adherence/aesthetics, 768p in seconds, 5-15s.",
        "tier": "premium",
        "text_endpoint": "minimax/h3-max/text-to-video",
        "image_endpoint": "minimax/h3-max/image-to-video",
        # Same wire quirks as base H3: integer duration, i2v derives the
        # aspect ratio from the input image (t2v-only key on Max: the i2v
        # schema doesn't declare aspect_ratio at all).
        "duration_int": True,
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        # Max tops out at 768P (no 2K/4K tiers like base H3); map the
        # tool's usual values onto the two capitalized enums.
        "resolutions": ("480P", "768P"),
        "resolution_aliases": {
            "480p": "480P", "540p": "480P",
            "720p": "768P", "768p": "768P", "1080p": "768P",
            "2k": "768P", "4k": "768P", "2160p": "768P",
        },
        "durations": (5, 15),
        # `prompt_expansion_mode` is in the schema's required array (with a
        # "balanced" default) — always send it.
        "static_payload": {"prompt_expansion_mode": "balanced"},
        "audio": False,  # no generate_audio TOGGLE — audio is always on
        "audio_native": True,  # native audio in every generation (fal docs)  # audio is native/always-on; no generate_audio key
        "negative": False,
        "seed": True,
        # Unlike base H3, Max declares `seed` on both endpoints.
    },
    "flux-3": {
        "display": "FLUX 3 (via FAL)",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Black Forest Labs frontier video. Native audio, 5-20s, 8 aspect ratios.",
        "tier": "premium",
        "text_endpoint": "blackforestlabs/flux-3/text-to-video",
        "image_endpoint": "blackforestlabs/flux-3/image-to-video",
        # FLUX 3 duration enum is "auto" | 5..20 as JSON integers.
        "duration_int": True,
        "aspect_ratios": ("21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("720p", "1080p"),
        "durations": (5, 20),
        "audio": True,
        "negative": False,
        "seed": False,
    },
    "grok-imagine-1.5": {
        "display": "Grok Imagine 1.5 (via FAL)",
        "speed": "~30-90s",
        "price": "premium",
        "strengths": "xAI. Fast stylized video with audio, 1-15s, cheap per second.",
        "tier": "premium",
        "text_endpoint": "xai/grok-imagine-video/v1.5/text-to-video",
        "image_endpoint": "xai/grok-imagine-video/v1.5/image-to-video",
        "duration_int": True,
        # i2v derives aspect from the input image; the key is t2v-only.
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"),
        "resolutions": ("480p", "720p", "1080p"),
        "durations": (1, 15),
        "audio": False,  # no generate_audio TOGGLE — audio is always on
        "audio_native": True,  # native audio in every generation (fal docs)  # audio is native; no generate_audio key
        "negative": False,
        "seed": False,
    },
    "gemini-omni-flash": {
        "display": "Gemini Omni Flash (via FAL)",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Google. Image-to-video with audio, physics-grounded motion, 3-10s.",
        "tier": "premium",
        # No text-to-video endpoint on FAL — image/reference only.
        "text_endpoint": None,
        "image_endpoint": "google/gemini-omni-flash/image-to-video",
        "duration_int": True,
        "aspect_ratios": ("16:9", "9:16"),
        "resolutions": None,
        "durations": (3, 10),
        "audio": False,  # no generate_audio TOGGLE — audio is always on
        "audio_native": True,  # native audio in every generation (fal docs)  # audio is native; no generate_audio key
        "negative": False,
        "seed": False,
    },
    "kling-v3-4k": {
        "display": "Kling v3 4K",
        "speed": "~120-300s",
        "price": "premium",
        "strengths": "4K output, native audio (Chinese/English), 3-15s.",
        "tier": "premium",
        "text_endpoint": "fal-ai/kling-video/v3/4k/text-to-video",
        "image_endpoint": "fal-ai/kling-video/v3/4k/image-to-video",
        # Kling 4K image-to-video uses `start_image_url` instead of
        # `image_url`. Handled in _build_payload via image_param_key.
        "image_param_key": "start_image_url",
        "aspect_ratios": ("16:9", "9:16", "1:1"),
        "resolutions": None,  # 4K is implicit
        "durations": (3, 15),
        "audio": True,
        "negative": True,
        "seed": True,
    },
    "happy-horse": {
        "display": "Happy Horse 1.0",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Alibaba. New model, sparse public docs — conservative defaults.",
        "tier": "premium",
        "text_endpoint": "alibaba/happy-horse/text-to-video",
        "image_endpoint": "alibaba/happy-horse/image-to-video",
        # Docs don't expose duration/aspect/resolution — let the endpoint
        # apply its own defaults.
        "aspect_ratios": None,
        "resolutions": None,
        "durations": None,
        "audio": False,  # no generate_audio TOGGLE — audio is always on
        "audio_native": True,  # native audio in every generation (fal docs)
        "negative": False,
        "seed": True,
    },
}

DEFAULT_MODEL = "pixverse-v6"  # cheap, both modalities, sane defaults


def _clamp_duration(family: Dict[str, Any], duration: Optional[int], resolution: Optional[str] = None) -> Optional[int]:
    """Snap to the nearest ``duration_enum`` entry when the family declares one, else clamp into the ``durations``
    ``(min, max)`` range; None stays None (endpoint default). A ``duration_cap_by_resolution`` ceiling for the resolved
    *resolution* is applied last (fal rejects LTX 2.5 >10s at 1440p/2160p)."""
    if duration is None:
        return None
    enum = family.get("duration_enum")
    if enum:
        clamped = min(enum, key=lambda d: abs(d - duration))
    else:
        lo, hi = family["durations"]
        clamped = max(lo, min(hi, duration))
    cap = (family.get("duration_cap_by_resolution") or {}).get(resolution)
    return clamped if cap is None else min(clamped, cap)


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


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _build_payload(
    family: Dict[str, Any],
    *,
    prompt: str,
    image_url: Optional[str],
    duration: Optional[int],
    aspect_ratio: str,
    resolution: str,
    negative_prompt: Optional[str],
    audio: Optional[bool],
    seed: Optional[int],
) -> Dict[str, Any]:
    """Build a family-specific payload, dropping keys the family doesn't declare."""
    payload: Dict[str, Any] = {}

    if prompt:
        payload["prompt"] = prompt
    if image_url:
        # Some endpoints (e.g. Kling v3 4K image-to-video) expect
        # `start_image_url` instead of `image_url`. The family entry can
        # declare an override.
        key = family.get("image_param_key") or "image_url"
        payload[key] = image_url
    # Several newer endpoints (seedance 2.x, minimax h3, flux-3, grok, gemini)
    # declare no `seed` field, and the managed gateway forwards whatever we
    # send — so gate it on the family rather than leaking an unknown key.
    if seed is not None and family.get("seed", True):
        payload["seed"] = seed

    if family.get("aspect_ratios"):
        if aspect_ratio in family["aspect_ratios"]:
            payload["aspect_ratio"] = aspect_ratio
        # otherwise let the endpoint auto-crop / use its default

    if family.get("resolutions"):
        # Some families use non-standard resolution enums (e.g. MiniMax H3's
        # "768P"/"2K"/"4K"); resolution_aliases maps the tool's usual
        # 720p/1080p-style values onto them.
        aliases = family.get("resolution_aliases") or {}
        resolved = aliases.get((resolution or "").lower(), resolution)
        if resolved in family["resolutions"]:
            payload["resolution"] = resolved
        # else: let the endpoint default

    clamped = _clamp_duration(family, duration)
    if clamped is not None and family.get("durations"):
        if family.get("duration_int"):
            # A few endpoints (MiniMax H3) require duration as a JSON integer.
            payload["duration"] = clamped
        else:
            # FAL exposes duration as a string in the queue API ("8" not 8).
            # Some families (e.g. veo3.1) require a unit suffix ("4s" not "4").
            suffix = family.get("duration_suffix", "")
            payload["duration"] = f"{clamped}{suffix}"

    if family.get("audio") and audio is not None:
        payload["generate_audio"] = bool(audio)

    if family.get("negative") and negative_prompt:
        payload["negative_prompt"] = negative_prompt

    # Keys the family's image-to-video endpoint rejects outright (e.g.
    # Seedance 2.5 / MiniMax H3 derive aspect_ratio from the input image).
    if image_url:
        for key in family.get("image_drop_keys", ()):  # type: ignore[assignment]
            payload.pop(key, None)

    # Constant keys the endpoint requires on every request (e.g. MiniMax
    # H3 Max lists `prompt_expansion_mode` in its required array).
    for key, value in (family.get("static_payload") or {}).items():
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
    """Resolve the FAL video route from the stored selection.

    Plain switch on the stored ``video_gen`` provider string — mirrors the
    image FAL resolver: ``"nous"`` (or legacy ``use_gateway: true``) →
    managed only (unentitled ⇒ selection-naming error); any other stored
    provider → direct only (missing FAL_KEY ⇒ selection-naming error);
    never-configured category → legacy credential autodetect.
    """
    from tools.managed_tool_gateway import resolve_managed_tool_gateway
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER,
        fal_key_is_configured,
        read_selection,
        selection_error,
    )

    selected = read_selection("video_gen")
    if selected == NOUS_MANAGED_PROVIDER:
        gateway = resolve_managed_tool_gateway("fal-queue")
        if gateway is None:
            raise ValueError(selection_error(
                "video_gen",
                NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or "
                "unreachable)",
            ))
        return gateway
    if selected is not None:
        if not fal_key_is_configured():
            raise ValueError(selection_error(
                "video_gen",
                selected,
                "FAL_KEY is not set",
            ))
        return None
    # Never-configured category: legacy credential autodetect (do NOT persist).
    if fal_key_is_configured():
        return None
    return resolve_managed_tool_gateway("fal-queue")


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
        from tools.fal_common import _extract_http_status, _managed_fal_billing_error
        status = _extract_http_status(exc)
        if status is not None and 400 <= status < 500:
            billing = _managed_fal_billing_error(exc, "endpoint")
            if billing is not None:
                raise ValueError(
                    f"Nous Subscription gateway rejected endpoint '{endpoint}' (HTTP {status}): {billing}") from exc
            raise ValueError(f"Nous Subscription gateway rejected endpoint '{endpoint}' (HTTP {status}). This model may not yet be enabled "
                             f"on the Nous Portal's FAL proxy. Either:\n  • Set FAL_KEY in your environment to use FAL.ai directly, or\n"
                             f"  • Pick a different model via `hermes tools` → Video Generation.") from exc
        raise


def _check_fal_video_available() -> bool:
    """True if the FAL video backend selected via `hermes tools` (or, on a
    never-configured install, any FAL backend) is reachable.

    Never raises — a stored-but-broken selection reports False here; the
    honest selection-naming error surfaces at call time from
    ``_resolve_managed_fal_video_gateway``.
    """
    from tools.managed_tool_gateway import resolve_managed_tool_gateway
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER,
        fal_key_is_configured,
        read_selection,
    )

    selected = read_selection("video_gen")
    if selected == NOUS_MANAGED_PROVIDER:
        return resolve_managed_tool_gateway("fal-queue") is not None
    if selected is not None:
        return fal_key_is_configured()
    if fal_key_is_configured():
        return True
    return resolve_managed_tool_gateway("fal-queue") is not None


# ---------------------------------------------------------------------------
# Upscaler (SeedVR2 — video upscale pass)
# ---------------------------------------------------------------------------

# ByteDance SeedVR2 on FAL: $0.001/megapixel of output video. A 5s 720p→1440p
# 2x pass is roughly $0.44. Faithful restoration-style upscaler (the same
# model family Krea exposes as its "SeedVR2" video enhancer).
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
                "tag": "LTX 2.3/2.5, Pixverse, Seedance 2.0/2.5/Mini, Veo 3.1, MiniMax H3, FLUX 3, Kling 3.0/4K/O3, Wan 3.0, Happy Horse, Grok Imagine, "
                       "Gemini Omni — text-to-video & image-to-video"}

    def capabilities(self) -> Dict[str, Any]:
        # Active-model-aware (mirrors the image_gen fal plugin, #97057):
        # report the RESOLVED family's actual surface so the dynamic tool
        # schema gates params on what the selected model honors, not a
        # union that overstates every axis. Falls back to the cross-family
        # union if resolution fails (never raises).
        try:
            _family_id, family = _resolve_family(None)
        except Exception:  # noqa: BLE001
            family = None
        if family:
            modalities = []
            if family.get("text_endpoint"):
                modalities.append("text")
            if family.get("image_endpoint"):
                modalities.append("image")
            durs = family.get("durations") or (1, 1)
            if _is_duration_range(durs):
                lo, hi = durs
            else:
                lo, hi = min(durs), max(durs)
            return {
                "modalities": modalities or ["text"],
                "aspect_ratios": list(family.get("aspect_ratios") or []),
                "resolutions": list(family.get("resolutions") or []),
                "max_duration": hi,
                "min_duration": lo,
                "supports_audio": bool(family.get("audio")),
                # Always-on native audio (no toggle): surfaces as a
                # description line, not a param. Verified per-family
                # against fal model pages (H3, Grok 1.5, Happy Horse,
                # Gemini Omni Flash all return native audio every run).
                "audio_always_on": bool(family.get("audio_native")),
                "supports_negative_prompt": bool(family.get("negative")),
                # Explicit per-family key (contract-tested); absent would
                # mean a catalog bug, so fail closed here.
                "supports_seed": bool(family.get("seed", False)),
                # SeedVR upscaler chains for any FAL video family.
                "supports_upscale": True,
                "max_reference_images": 0,
            }
        # Fallback: union across families (legacy shape).
        max_dur = 1
        min_dur: Optional[int] = None
        for meta in FAL_FAMILIES.values():
            durs = meta.get("durations")
            if not durs:
                continue
            if _is_duration_range(durs):
                lo, hi = durs
            else:
                lo, hi = min(durs), max(durs)
            max_dur = max(max_dur, hi)
            min_dur = lo if min_dur is None else min(min_dur, lo)
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "resolutions": ["360p", "540p", "720p", "1080p"],
            "max_duration": max_dur,
            "min_duration": min_dur if min_dur is not None else 1,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "supports_seed": True,
            "supports_upscale": True,
            "max_reference_images": 0,
        }

    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None, aspect_ratio: str = "16:9", resolution: str = "720p", negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None, seed: Optional[int] = None, upscale: Optional[bool] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        if not _check_fal_video_available():
            from tools.tool_backend_helpers import read_selection

            if read_selection("video_gen") is not None:
                # A stored selection that cannot run gets the honest
                # selection-naming error from the strict resolver.
                try:
                    _resolve_managed_fal_video_gateway()
                except ValueError as exc:
                    return error_response(
                        error=str(exc),
                        error_type="auth_required",
                        provider="fal",
                        prompt=prompt,
                    )
            return error_response(
                error=(
                    "No FAL backend available. Either set FAL_KEY "
                    "(run `hermes tools` → Video Generation → FAL to configure) "
                    "or sign in to Nous (`hermes setup`) for managed gateway access."
                ),
                error_type="auth_required",
                provider="fal",
                prompt=prompt,
            )

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

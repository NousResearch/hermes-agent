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
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Family catalog
# ---------------------------------------------------------------------------
#
# Each family declares both endpoints (when available) plus a per-family
# capability sheet derived from FAL's OpenAPI schemas. Capability flags
# drive which keys get added to the request payload — keys a family doesn't
# advertise are dropped before send.
#
# Capabilities:
#   aspect_ratios  : tuple of supported ratios (None = endpoint decides)
#   resolutions    : tuple of supported resolutions (None = endpoint decides)
#   durations      : tuple of supported durations OR (min, max) range
#                    (heuristic: 2-element with gap > 1 is a range)
#   audio          : True if generate_audio is supported
#   negative       : True if negative_prompt is supported
#   seed           : False when the endpoint declares no `seed` field
#                    (absent = True, so existing families keep sending it)
#   duration_int   : True when FAL types duration as an integer rather than
#                    the usual queue-API string

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
        "audio": False,  # audio is native/always-on; no generate_audio key
        "negative": False,
        "seed": False,
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
        "audio": False,  # audio is native; no generate_audio key
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
        "audio": False,  # audio is native; no generate_audio key
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
        "audio": False,
        "negative": False,
    },
}

DEFAULT_MODEL = "pixverse-v6"  # cheap, both modalities, sane defaults


def _is_duration_range(durations: Any) -> bool:
    """Heuristic: a 2-tuple of ints with a gap > 1 is treated as ``(min, max)``."""
    if not isinstance(durations, tuple) or len(durations) != 2:
        return False
    if not all(isinstance(d, int) for d in durations):
        return False
    return durations[1] - durations[0] > 1


def _clamp_duration(family: Dict[str, Any], duration: Optional[int]) -> Optional[int]:
    durations = family.get("durations")
    if not durations:
        return duration
    if duration is None:
        # Range families (e.g. pixverse-v6 (1,15)) should omit the field so
        # the FAL endpoint applies its own default rather than receiving the
        # minimum value.  Enum families (e.g. veo3.1 (4,6,8)) keep sending
        # their first entry as the default.
        return None if _is_duration_range(durations) else durations[0]
    if _is_duration_range(durations):
        lo, hi = durations
        return max(lo, min(hi, duration))
    # enum
    if duration in durations:
        return duration
    return min(durations, key=lambda d: abs(d - duration))


# ---------------------------------------------------------------------------
# Config / model resolution
# ---------------------------------------------------------------------------


def _load_video_gen_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load video_gen config: %s", exc)
        return {}


_ENDPOINT_MODALITY_LEAVES = frozenset({"text-to-video", "image-to-video"})


def _normalize_family_key(c: str) -> Optional[str]:
    """Try to extract a known family ID from a model string.

    Handles bare IDs (``seedance-2.5``), full endpoint paths
    (``bytedance/seedance-2.5/text-to-video``), truncated endpoint stems
    (``minimax/h3``, ``bytedance/seedance-2.0/mini``), and provider-prefixed
    names (``bytedance/seedance-2.5``).
    """
    c = c.strip()
    if not c:
        return None
    if c in FAL_FAMILIES:
        return c

    # Exact declared endpoint — unambiguous, and beats any segment scan
    # that would otherwise see "seedance-2.0" inside ".../seedance-2.0/mini/...".
    for fid, meta in FAL_FAMILIES.items():
        if c in (meta.get("text_endpoint"), meta.get("image_endpoint")):
            return fid

    # Truncated stem of a declared endpoint: "minimax/h3" or
    # "bytedance/seedance-2.0/mini". The next path segment after ``c`` must
    # be a modality leaf so "bytedance/seedance-2.0" does not also match the
    # Mini family's deeper ".../seedance-2.0/mini/text-to-video" path.
    stem_hits: List[Tuple[int, str]] = []
    for fid, meta in FAL_FAMILIES.items():
        for endpoint in (meta.get("text_endpoint"), meta.get("image_endpoint")):
            if not isinstance(endpoint, str):
                continue
            if not endpoint.startswith(c + "/"):
                continue
            first = endpoint[len(c) + 1:].split("/", 1)[0]
            if first in _ENDPOINT_MODALITY_LEAVES:
                stem_hits.append((len(c), fid))
                break
    if stem_hits:
        stem_hits.sort(key=lambda item: item[0], reverse=True)
        return stem_hits[0][1]

    # Longest family-id path-segment match ("bytedance/seedance-2.5" →
    # seedance-2.5; prefers seedance-2.0-mini over seedance-2.0 when both
    # somehow appear).
    parts = set(c.split("/"))
    best_fid: Optional[str] = None
    best_len = -1
    for fid in FAL_FAMILIES:
        if fid in parts and len(fid) > best_len:
            best_fid = fid
            best_len = len(fid)
    return best_fid


def _resolve_family(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Decide which FAL family to use. Returns ``(family_id, meta)``."""
    candidates: List[Optional[str]] = []
    candidates.append(explicit)
    candidates.append(os.environ.get("FAL_VIDEO_MODEL"))

    cfg = _load_video_gen_section()
    fal_cfg = cfg.get("fal") if isinstance(cfg.get("fal"), dict) else {}
    if isinstance(fal_cfg, dict):
        candidates.append(fal_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for c in candidates:
        if isinstance(c, str) and c.strip():
            fid = _normalize_family_key(c)
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

    return payload


# ---------------------------------------------------------------------------
# fal_client lazy import (shared with image_generation_tool via fal_common)
# ---------------------------------------------------------------------------

_fal_client: Any = None
_fal_client_lock = threading.Lock()


def _load_fal_client() -> Any:
    """Lazy-load the ``fal_client`` SDK and cache it on this module.

    Delegates the actual import to :func:`tools.fal_common.import_fal_client`
    so the ``lazy_deps`` ensure-install handling stays in one place.

    Thread-safe via double-checked locking: concurrent first calls import
    the SDK exactly once instead of each racing thread re-running the import.
    """
    global _fal_client
    if _fal_client is not None:
        return _fal_client
    with _fal_client_lock:
        if _fal_client is not None:  # re-check inside the lock
            return _fal_client
        from tools.fal_common import import_fal_client
        _fal_client = import_fal_client()
        return _fal_client


# ---------------------------------------------------------------------------
# Managed FAL gateway (Nous Subscription)
# ---------------------------------------------------------------------------

_managed_fal_video_client: Any = None
_managed_fal_video_client_config: Any = None
_managed_fal_video_client_lock = threading.Lock()


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

    client_config = (
        managed_gateway.gateway_origin.rstrip("/"),
        managed_gateway.nous_user_token,
    )
    with _managed_fal_video_client_lock:
        if _managed_fal_video_client is not None and _managed_fal_video_client_config == client_config:
            return _managed_fal_video_client

        _load_fal_client()
        _managed_fal_video_client = _ManagedFalSyncClient(
            _fal_client,
            key=managed_gateway.nous_user_token,
            queue_run_origin=managed_gateway.gateway_origin,
        )
        _managed_fal_video_client_config = client_config
        return _managed_fal_video_client


def _submit_fal_video_request(endpoint: str, arguments: Dict[str, Any]):
    """Submit a FAL video request using direct credentials or the managed queue gateway.

    Returns a request handle whose ``.get()`` blocks until the result is ready.
    """
    _load_fal_client()
    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_video_gateway()
    if managed_gateway is None:
        return _fal_client.submit(endpoint, arguments=arguments, headers=request_headers)

    managed_client = _get_managed_fal_video_client(managed_gateway)
    try:
        return managed_client.submit(
            endpoint,
            arguments=arguments,
            headers=request_headers,
        )
    except Exception as exc:
        from tools.fal_common import _extract_http_status

        status = _extract_http_status(exc)
        if status is not None and 400 <= status < 500:
            raise ValueError(
                f"Nous Subscription gateway rejected endpoint '{endpoint}' "
                f"(HTTP {status}). This model may not yet be enabled on "
                f"the Nous Portal's FAL proxy. Either:\n"
                f"  • Set FAL_KEY in your environment to use FAL.ai directly, or\n"
                f"  • Pick a different model via `hermes tools` → Video Generation."
            ) from exc
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


def _upscale_video(
    video_url: str,
    source_request_id: Optional[str] = None,
) -> Optional[str]:
    """Upscale a generated video via SeedVR2; return the new URL or None.

    Best-effort: any failure logs and returns ``None`` so the caller falls
    back to the native-resolution video — an upscale failure must never
    destroy an already-successful generation.
    """
    try:
        logger.info("Upscaling video with SeedVR2 (%dx)...", UPSCALER_FACTOR)
        arguments: Dict[str, Any] = {
            "video_url": video_url,
            "upscale_mode": "factor",
            "upscale_factor": UPSCALER_FACTOR,
        }
        if _resolve_managed_fal_video_gateway() is not None:
            if not source_request_id:
                raise RuntimeError("Managed SeedVR upscale requires the source FAL request id")
            arguments["source_request_id"] = source_request_id
        handle = _submit_fal_video_request(UPSCALER_ENDPOINT, arguments)
        result = handle.get()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Video upscale failed: %s", exc)
        return None

    video = (result or {}).get("video") if isinstance(result, dict) else None
    if isinstance(video, dict) and video.get("url"):
        return video["url"]
    if isinstance(video, str) and video:
        return video
    logger.warning("Video upscaler returned no URL")
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class FALVideoGenProvider(VideoGenProvider):
    """FAL.ai multi-family video generation backend.

    Routes between text-to-video and image-to-video endpoints automatically
    based on whether ``image_url`` was provided.
    """

    @property
    def name(self) -> str:
        return "fal"

    @property
    def display_name(self) -> str:
        return "FAL"

    def is_available(self) -> bool:
        try:
            return _check_fal_video_available()
        except Exception:  # noqa: BLE001 — never break the picker
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fid, meta in FAL_FAMILIES.items():
            modalities: List[str] = []
            if meta.get("text_endpoint"):
                modalities.append("text")
            if meta.get("image_endpoint"):
                modalities.append("image")
            entry: Dict[str, Any] = {
                "id": fid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
                "tier": meta.get("tier", "premium"),
                "modalities": modalities,
            }
            durs = meta.get("durations")
            if durs:
                if _is_duration_range(durs):
                    entry["min_duration"], entry["max_duration"] = durs
                else:
                    entry["min_duration"] = min(durs)
                    entry["max_duration"] = max(durs)
            out.append(entry)
        return out

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "FAL",
            "badge": "paid",
            "tag": "LTX, Pixverse, Seedance 2.0/2.5/Mini, Veo 3.1, MiniMax H3, FLUX 3, Kling 4K, Happy Horse, Grok Imagine, Gemini Omni — text-to-video & image-to-video",
            "env_vars": [
                {
                    "key": "FAL_KEY",
                    "prompt": "FAL.ai API key",
                    "url": "https://fal.ai/dashboard/keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Union across families so the tool schema doesn't understate the
        # longest-running models (Seedance 2.5 = 30s, FLUX 3 = 20s).
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
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        upscale: Optional[bool] = None,
        **kwargs: Any,
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
            return error_response(
                error="fal_client Python package not installed (pip install fal-client)",
                error_type="missing_dependency",
                provider="fal",
                prompt=prompt,
            )

        prompt = (prompt or "").strip()
        family_id, family = _resolve_family(model)

        # Route: image_url → image-to-video endpoint; else → text-to-video.
        image_url_norm = (image_url or "").strip() or None
        if image_url_norm:
            endpoint = family.get("image_endpoint")
            modality_used = "image"
            if not endpoint:
                return error_response(
                    error=(
                        f"FAL family {family_id} has no image-to-video "
                        f"endpoint. Pick a family with image-to-video support "
                        f"via `hermes tools` → Video Generation."
                    ),
                    error_type="modality_unsupported",
                    provider="fal", model=family_id, prompt=prompt,
                )
        else:
            endpoint = family.get("text_endpoint")
            modality_used = "text"
            if not endpoint:
                return error_response(
                    error=(
                        f"FAL family {family_id} has no text-to-video "
                        f"endpoint. Pass an image_url to use its "
                        f"image-to-video endpoint, or pick a different family."
                    ),
                    error_type="modality_unsupported",
                    provider="fal", model=family_id, prompt=prompt,
                )

        if not prompt:
            return error_response(
                error="prompt is required.",
                error_type="missing_prompt",
                provider="fal", model=family_id, prompt=prompt,
            )

        payload = _build_payload(
            family,
            prompt=prompt,
            image_url=image_url_norm,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            negative_prompt=negative_prompt,
            audio=audio,
            seed=seed,
        )

        try:
            handle = _submit_fal_video_request(endpoint, payload)
            source_request_id = getattr(handle, "request_id", None)
            result = handle.get()
        except Exception as exc:
            logger.warning(
                "FAL video gen failed (family=%s, endpoint=%s): %s",
                family_id, endpoint, exc, exc_info=True,
            )
            return error_response(
                error=f"FAL video generation failed: {exc}",
                error_type="api_error",
                provider="fal", model=family_id, prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        video = (result or {}).get("video") if isinstance(result, dict) else None
        url: Optional[str] = None
        if isinstance(video, dict):
            url = video.get("url")
        elif isinstance(video, str):
            url = video

        if not url:
            return error_response(
                error="FAL returned no video URL in response",
                error_type="empty_response",
                provider="fal", model=family_id, prompt=prompt,
            )

        # Optional high-resolution pass (SeedVR2). Explicit agent/user opt-in
        # only; best-effort — failure falls back to the native-resolution
        # video rather than failing the generation.
        upscaled = False
        if upscale:
            upscaled_url = _upscale_video(url, source_request_id)
            if upscaled_url:
                url = upscaled_url
                upscaled = True
            else:
                logger.warning(
                    "Video upscale pass failed — returning native-resolution video"
                )

        extra: Dict[str, Any] = {"endpoint": endpoint, "upscaled": upscaled}
        if upscaled:
            extra["upscale_factor"] = UPSCALER_FACTOR
        if isinstance(video, dict):
            if video.get("file_size") and not upscaled:
                extra["file_size"] = video["file_size"]
            if video.get("content_type"):
                extra["content_type"] = video["content_type"]

        return success_response(
            video=url,
            model=family_id,
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio if "aspect_ratio" in payload else "",
            duration=int("".join(c for c in str(payload["duration"]) if c.isdigit()) or "0") if "duration" in payload else 0,
            provider="fal",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``FALVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(FALVideoGenProvider())

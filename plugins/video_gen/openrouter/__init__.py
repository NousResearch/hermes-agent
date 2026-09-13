"""OpenRouter video generation backend.

OpenRouter fronts every major video model (Veo, Sora, Kling, Seedance, Wan, Hailuo, Grok
Imagine, FLUX Video, …) behind one asynchronous API: ``POST /api/v1/videos`` → poll
``GET /api/v1/videos/{id}`` → download ``GET /api/v1/videos/{id}/content`` with the
bearer key. The catalog and each model's limits come live from ``GET /api/v1/videos/models``
(public, no key), so new releases are selectable without a patch and a retired model drops
out on its own. ``capabilities()`` reflects the *selected* model so the dynamic
``video_generate`` schema advertises only what that model honours.

Docs: https://openrouter.ai/docs/guides/overview/multimodal/video-generation
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import VideoGenProvider, error_response, save_url_video, success_response

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "minimax/hailuo-3-max"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_CATALOG_TTL_S = 300.0
_CATALOG_TIMEOUT_S = 10.0
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled", "expired"})
# Every value the API accepts (docs "Supported Resolutions / Aspect Ratios"); the union fallback
# when the live catalog is unreachable. Heights drive nearest-match clamping.
_RESOLUTION_HEIGHT = {"480p": 480, "540p": 540, "720p": 720, "768p": 768, "1080p": 1080, "1K": 1024, "2K": 1440, "4K": 2160}
_ALL_ASPECT_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21")
_MAX_REFERENCE_IMAGES = 3  # image references are accepted by every provider per the API schema

# Offline snapshot so the picker/default work before the first successful catalog fetch.
_FALLBACK_CATALOG: List[Dict[str, Any]] = [
    {"id": DEFAULT_MODEL, "name": "MiniMax: Hailuo 3 Max", "supported_durations": list(range(5, 16)),
     "supported_resolutions": ["768p", "480p"], "supported_aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
     "supported_frame_images": ["first_frame", "last_frame"], "generate_audio": False, "seed": False,
     "pricing_skus": {"duration_seconds_480p": "0.05", "duration_seconds_768p": "0.08"}},
]


def _price_label(skus: Any) -> str:
    """Compact per-second price from ``pricing_skus`` (``$0.10/s`` or ``$0.05–0.28/s``); ``""`` when the
    SKU shape is token- or megapixel-priced (Seedance, FLUX upscale) — a wrong number is worse than none."""
    if not isinstance(skus, dict):
        return ""
    per_second: List[float] = []
    for key, value in skus.items():
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if key.startswith("duration_seconds"):
            per_second.append(amount)
        elif key.startswith(("cents_per_second_output", "cents_per_video_output_second")):
            per_second.append(amount / 100)
    if not per_second:
        return ""
    lo, hi = min(per_second), max(per_second)
    return f"${lo:.2f}/s" if lo == hi else f"${lo:.2f}–{hi:.2f}/s"


def _ratio(value: Any) -> Optional[float]:
    try:
        w, h = str(value).split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return None


def _nearest(value: Any, supported: List[Any], height: Optional[Dict[str, int]] = None) -> Any:
    """Closest supported value (durations by distance, resolutions by pixel height, aspect ratios by
    numeric ratio); the request is otherwise a guaranteed 400 because the tool always sends a default
    resolution/aspect ratio."""
    if not supported or value in supported:
        return value
    if height is not None:
        target = height.get(str(value))
        if target is None:
            return supported[0]
        scored = [(abs(height[str(s)] - target), i, s) for i, s in enumerate(supported) if str(s) in height]
        return min(scored)[2] if scored else supported[0]
    if ":" in str(value):
        target = _ratio(value)
        if target is None:
            return supported[0]
        scored = [(abs(r - target), i, s) for i, s in enumerate(supported) if (r := _ratio(s)) is not None]
        return min(scored)[2] if scored else supported[0]
    try:
        return min(supported, key=lambda s: (abs(int(s) - int(value)), s))
    except (TypeError, ValueError):
        return supported[0]


def _is_generative(entry: Dict[str, Any]) -> bool:
    """Text/image-to-video models declare durations; edit, upscale and avatar models (video/audio input)
    do not and are outside the unified ``video_generate`` surface."""
    return bool(entry.get("supported_durations"))


def _entry_capabilities(entry: Dict[str, Any]) -> Dict[str, Any]:
    durations = [int(d) for d in entry.get("supported_durations") or [] if isinstance(d, (int, float))]
    return {
        "modalities": ["text", "image"] if entry.get("supported_frame_images") else ["text"],
        "aspect_ratios": list(entry.get("supported_aspect_ratios") or _ALL_ASPECT_RATIOS),
        "resolutions": list(entry.get("supported_resolutions") or _RESOLUTION_HEIGHT),
        "max_duration": max(durations) if durations else 30, "min_duration": min(durations) if durations else 1,
        "supports_audio": bool(entry.get("generate_audio")), "supports_negative_prompt": False,
        "supports_seed": bool(entry.get("seed")), "supports_upscale": False,
        "max_reference_images": _MAX_REFERENCE_IMAGES,
    }


def _image_part(url: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _acceptable_image_ref(value: str) -> bool:
    """OpenRouter fetches the frame itself, so it needs a public HTTPS URL or an inline ``data:image/`` URL
    (what the sandbox confinement chokepoint hands us for local files)."""
    lowered = value.lower()
    return lowered.startswith("https://") or lowered.startswith("data:image/")


def _build_payload(
    entry: Dict[str, Any], *, model: str, prompt: str, image_url: Optional[str], reference_image_urls: Optional[List[str]],
    duration: Optional[int], aspect_ratio: str, resolution: str, audio: Optional[bool], seed: Optional[int],
) -> Dict[str, Any]:
    """Unified inputs → OpenRouter body, clamped to the model's live limits; unsupported toggles are dropped
    rather than sent (the API 400s on ``seed``/``generate_audio`` for models that lack them)."""
    payload: Dict[str, Any] = {"model": model, "prompt": prompt}
    if aspect_ratio:
        payload["aspect_ratio"] = _nearest(aspect_ratio, list(entry.get("supported_aspect_ratios") or []))
    if resolution:
        payload["resolution"] = _nearest(resolution, list(entry.get("supported_resolutions") or []), _RESOLUTION_HEIGHT)
    if duration:
        payload["duration"] = int(_nearest(int(duration), list(entry.get("supported_durations") or [])))
    if image_url:
        payload["frame_images"] = [{**_image_part(image_url), "frame_type": "first_frame"}]
    if reference_image_urls:
        payload["input_references"] = [_image_part(u) for u in reference_image_urls[:_MAX_REFERENCE_IMAGES]]
    if audio is not None and entry.get("generate_audio"):
        payload["generate_audio"] = bool(audio)
    if seed is not None and entry.get("seed"):
        payload["seed"] = int(seed)
    return payload


class OpenRouterVideoGenProvider(VideoGenProvider):
    """Every generative model on OpenRouter's video API, catalog and limits discovered live."""

    name = "openrouter"
    display_name = "OpenRouter"
    _poll_interval_s = 10.0
    _poll_deadline_s = 900.0
    _request_timeout_s = 60.0

    def __init__(self) -> None:
        self._catalog_cache: Optional[Tuple[List[Dict[str, Any]], float]] = None

    # ---- credentials / transport -------------------------------------------------------------
    def _api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    def _base_url(self) -> str:
        return (os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/NousResearch/hermes-agent", "X-Title": "Hermes Agent"}

    def _session(self) -> Any:
        import requests
        return requests.Session()

    def is_available(self) -> bool:
        return bool(self._api_key())

    # ---- catalog -------------------------------------------------------------------------------
    def _catalog(self) -> List[Dict[str, Any]]:
        """Live ``/videos/models`` entries (public endpoint), cached per TTL; the snapshot when unreachable."""
        if self._catalog_cache and time.monotonic() - self._catalog_cache[1] < _CATALOG_TTL_S:
            return self._catalog_cache[0]
        entries: List[Dict[str, Any]] = []
        try:
            import requests
            response = requests.get(f"{self._base_url()}/videos/models", timeout=_CATALOG_TIMEOUT_S)
            response.raise_for_status()
            data = response.json().get("data")
            entries = [e for e in (data if isinstance(data, list) else []) if isinstance(e, dict) and e.get("id")]
        except Exception as exc:  # noqa: BLE001 — offline picker keeps working on the snapshot
            logger.debug("OpenRouter video catalog unavailable: %s", exc)
        if not entries:
            return _FALLBACK_CATALOG
        self._catalog_cache = (entries, time.monotonic())
        return entries

    def _entry(self, model_id: str) -> Dict[str, Any]:
        """Catalog row for *model_id*; ``{}`` for an unknown id (request passes through unclamped so a
        brand-new model works before our cache refreshes — the API validates)."""
        return next((e for e in self._catalog() if e.get("id") == model_id), {})

    def _configured_model(self) -> str:
        try:
            from hermes_cli.config import cfg_get, load_config
            value = cfg_get(load_config(), "video_gen", "model")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read video_gen.model: %s", exc)
            value = None
        return value.strip() if isinstance(value, str) and value.strip() else DEFAULT_MODEL

    def list_models(self) -> List[Dict[str, Any]]:
        rows = []
        for entry in self._catalog():
            if not _is_generative(entry):
                continue
            caps = _entry_capabilities(entry)
            extras = [f"{caps['min_duration']}-{caps['max_duration']}s", "/".join(caps["resolutions"])]
            extras += ["audio"] if caps["supports_audio"] else []
            extras += ["i2v"] if "image" in caps["modalities"] else []
            rows.append({"id": entry["id"], "display": entry.get("name") or entry["id"], "strengths": ", ".join(extras),
                         "price": _price_label(entry.get("pricing_skus")), "modalities": caps["modalities"],
                         "min_duration": caps["min_duration"], "max_duration": caps["max_duration"]})
        return rows

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        """The selected model's live surface; the API-wide union when the id is unknown or the catalog is down.
        ``supports_seed``/``supports_audio`` are per-model (Veo/Wan/Seedance: yes; Hailuo/Grok: no)."""
        entry = self._entry(self._configured_model())
        if entry:
            return _entry_capabilities(entry)
        return {"modalities": ["text", "image"], "aspect_ratios": list(_ALL_ASPECT_RATIOS),
                "resolutions": list(_RESOLUTION_HEIGHT), "max_duration": 30, "min_duration": 1,
                "supports_audio": True, "supports_negative_prompt": False, "supports_seed": True,
                "supports_upscale": False, "max_reference_images": _MAX_REFERENCE_IMAGES}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": "OpenRouter", "badge": "paid",
                "tag": "Veo 3.1, Sora 2 Pro, Kling 3, Seedance 2, Wan 3, Hailuo 3, Grok Imagine & more — live catalog; "
                       "text-to-video, image-to-video & reference-to-video; uses OPENROUTER_API_KEY",
                "env_vars": [{"key": "OPENROUTER_API_KEY", "prompt": "OpenRouter API key", "url": "https://openrouter.ai/settings/keys"}]}

    # ---- generation ----------------------------------------------------------------------------
    def _poll(self, session: Any, job_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self._poll_deadline_s
        url = f"{self._base_url()}/videos/{job_id}"
        last_status = "unknown"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"video job {job_id} did not finish within {int(self._poll_deadline_s)}s (last status={last_status})")
            response = session.get(url, headers=self._headers(), timeout=max(0.001, min(self._request_timeout_s, remaining)))
            response.raise_for_status()
            payload = response.json()
            last_status = str(payload.get("status") or "").lower() or "unknown"
            if last_status in _TERMINAL_STATUSES:
                return payload
            time.sleep(min(self._poll_interval_s, max(0.0, deadline - time.monotonic())))

    def _save_completed_video(self, job_id: str) -> str:
        # The content endpoint is derived from our configured origin, never from ``unsigned_urls``: the
        # bearer key must only ever be sent to the host the operator selected.
        return str(save_url_video(f"{self._base_url()}/videos/{job_id}/content", prefix="openrouter",
                                  headers=self._headers(), require_video_content_type=True))

    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None, duration: Optional[int] = None,
        aspect_ratio: str = "16:9", resolution: str = "720p", negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None, seed: Optional[int] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        del negative_prompt, kwargs  # no top-level negative_prompt on this API; unknown kwargs are ignored per the ABC
        prompt = (prompt or "").strip()
        model_id = (model or "").strip() or self._configured_model()

        def fail(error: str, error_type: str) -> Dict[str, Any]:
            return error_response(error=error, error_type=error_type, provider=self.name, model=model_id, prompt=prompt,
                                  aspect_ratio=aspect_ratio)

        if not prompt:
            return fail("prompt is required", "invalid_request")
        if not self._api_key():
            return fail("OPENROUTER_API_KEY is not set", "missing_credentials")
        image_url = (image_url or "").strip() or None
        refs = [r.strip() for r in (reference_image_urls or []) if isinstance(r, str) and r.strip()]
        for ref in ([image_url] if image_url else []) + refs:
            if not _acceptable_image_ref(ref):
                return fail("image inputs must be public HTTPS URLs or data:image/ URLs (OpenRouter fetches them itself)",
                            "invalid_request")

        entry = self._entry(model_id)
        payload = _build_payload(entry, model=model_id, prompt=prompt, image_url=image_url, reference_image_urls=refs,
                                 duration=duration, aspect_ratio=aspect_ratio, resolution=resolution, audio=audio, seed=seed)
        session = self._session()
        try:
            submitted = session.post(f"{self._base_url()}/videos", headers=self._headers(), json=payload,
                                     timeout=self._request_timeout_s)
            if submitted.status_code >= 400:
                detail = ""
                try:
                    detail = str((submitted.json().get("error") or {}).get("message") or "")
                except Exception:  # noqa: BLE001 — non-JSON error body
                    detail = ""
                return fail(f"OpenRouter rejected the request (HTTP {submitted.status_code}): {detail or submitted.text[:300]}",
                            "api_error")
            job_id = str(submitted.json().get("id") or "").strip()
            if not job_id:
                return fail("OpenRouter submit response did not contain a job id", "api_error")
            job = self._poll(session, job_id)
            status = str(job.get("status") or "").lower()
            if status != "completed":
                return fail(str(job.get("error") or f"video job ended with status={status!r}"), "job_failed")
            video_path = self._save_completed_video(job_id)
        except Exception as exc:  # noqa: BLE001 — normalize transport/timeout failures for tool callers
            logger.debug("OpenRouter video generation failed", exc_info=True)
            return fail(f"OpenRouter video generation failed: {exc}", "api_error")
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()

        raw_usage = job.get("usage")
        usage: Dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        extra: Dict[str, Any] = {"job_id": job_id, **({"cost": usage["cost"]} if usage.get("cost") is not None else {})}
        return success_response(
            video=video_path, model=model_id, prompt=prompt, modality="image" if image_url else "text",
            aspect_ratio=str(payload.get("aspect_ratio") or ""), duration=int(payload.get("duration") or 0),
            provider=self.name, extra=extra)


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenRouterVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(OpenRouterVideoGenProvider())

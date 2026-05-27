"""Krea image generation backend.

Exposes Krea's `Krea 2` foundation image model family — Krea 2 Medium and
Krea 2 Large — as an :class:`ImageGenProvider` implementation.

Krea's API is asynchronous: the generate endpoint returns a ``job_id``
that you poll at ``GET /jobs/{job_id}``. This provider hides that
roundtrip behind the synchronous ``generate()`` contract: submit, poll
every 2s with light backoff, materialise the result URL to local cache,
return the success/error dict like every other backend.

Selection precedence (first hit wins):

1. ``KREA_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.krea.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our IDs)
4. :data:`DEFAULT_MODEL` — ``krea-2-medium`` (Krea's "start here" recommendation)

Docs: https://docs.krea.ai/developers/krea-2/overview
API:  https://docs.krea.ai/api-reference/krea/krea-2-large
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.krea.ai"

# Map our short model IDs to Krea's URL path segment.
_MODELS: Dict[str, Dict[str, Any]] = {
    "krea-2-medium": {
        "display": "Krea 2 Medium",
        "speed": "~15-25s",
        "strengths": "Illustration, anime, painting, expressive styles. Faster + cheaper.",
        "price": "$0.030 (text) / $0.035 (style refs) / $0.040 (moodboards)",
        "path": "medium",
    },
    "krea-2-large": {
        "display": "Krea 2 Large",
        "speed": "~25-60s",
        "strengths": "Photorealism, raw textured looks (motion blur, grain), expressive styles.",
        "price": "$0.060 (text) / $0.065 (style refs) / $0.070 (moodboards)",
        "path": "large",
    },
}

DEFAULT_MODEL = "krea-2-medium"

# Hermes uses 3 abstract aspect ratios. Map to Krea's enum (which is wider).
# Krea accepts: 1:1, 4:3, 3:2, 16:9, 2.35:1, 4:5, 2:3, 9:16
_ASPECT_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# Only resolution Krea currently supports.
DEFAULT_RESOLUTION = "1K"

# Valid creativity levels per Krea docs. Default is "medium".
_VALID_CREATIVITY = {"raw", "low", "medium", "high"}

# Polling cadence. Krea recommends 2-5s; we start at 2s and back off to 5s
# for long jobs (Large can take ~1min). Total ceiling matches Krea's
# hosted-tool timeout of 3 minutes.
_POLL_INITIAL_INTERVAL = 2.0
_POLL_MAX_INTERVAL = 5.0
_POLL_BACKOFF = 1.3
_POLL_TIMEOUT_SECONDS = 180.0

# HTTP statuses worth retrying during the poll loop. Everything else (401,
# 402, 403, 404, other 4xx) is a permanent failure — surface it immediately
# instead of burning the 180s deadline retrying a request that will never
# succeed.
_RETRYABLE_POLL_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_TERMINAL_STATES = {"completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_krea_config() -> Dict[str, Any]:
    """Read ``image_gen.krea`` (with fallthrough to ``image_gen``) from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("KREA_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_krea_config()
    krea_cfg = cfg.get("krea") if isinstance(cfg.get("krea"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(krea_cfg, dict):
        value = krea_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_creativity(value: Optional[str]) -> str:
    """Coerce ``creativity`` kwarg to a valid Krea value (default ``medium``)."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _VALID_CREATIVITY:
            return v
    cfg = _load_krea_config()
    krea_cfg = cfg.get("krea") if isinstance(cfg.get("krea"), dict) else {}
    cfg_value = krea_cfg.get("creativity") if isinstance(krea_cfg, dict) else None
    if isinstance(cfg_value, str) and cfg_value.strip().lower() in _VALID_CREATIVITY:
        return cfg_value.strip().lower()
    return "medium"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class KreaImageGenProvider(ImageGenProvider):
    """Krea ``Krea 2`` foundation image model backend (Medium + Large)."""

    @property
    def name(self) -> str:
        return "krea"

    @property
    def display_name(self) -> str:
        return "Krea"

    def is_available(self) -> bool:
        return bool(os.environ.get("KREA_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Krea",
            "badge": "paid",
            "tag": "Krea 2 foundation model — Medium ($0.03) + Large ($0.06). Strong style transfer + moodboards.",
            "env_vars": [
                {
                    "key": "KREA_API_KEY",
                    "prompt": "Krea API key",
                    "url": "https://www.krea.ai/settings/api-tokens",
                },
            ],
        }

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        krea_ar = _ASPECT_MAP.get(aspect, "1:1")

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="krea",
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("KREA_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "KREA_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → Krea to configure, or get a key at "
                    "https://www.krea.ai/settings/api-tokens."
                ),
                error_type="auth_required",
                provider="krea",
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        creativity = _resolve_creativity(kwargs.get("creativity"))

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": krea_ar,
            "resolution": DEFAULT_RESOLUTION,
            "creativity": creativity,
        }

        # Optional forward-compat passthroughs — the Krea API accepts these
        # but they're not required and most agent calls won't supply them.
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            payload["seed"] = seed

        styles = kwargs.get("styles")
        if isinstance(styles, list) and styles:
            payload["styles"] = styles

        image_style_references = kwargs.get("image_style_references")
        if isinstance(image_style_references, list) and image_style_references:
            # Krea caps at 10 refs per request.
            payload["image_style_references"] = image_style_references[:10]

        moodboards = kwargs.get("moodboards")
        if isinstance(moodboards, list) and moodboards:
            # Krea currently caps at 1 moodboard per request.
            payload["moodboards"] = moodboards[:1]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent/1.0 (krea-image-gen)",
        }

        # 1. Submit job.
        submit_url = f"{BASE_URL}/generate/image/krea/krea-2/{meta['path']}"
        try:
            response = requests.post(
                submit_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                body = resp.json() if resp is not None else {}
                err_msg = (
                    body.get("error", {}).get("message")
                    if isinstance(body.get("error"), dict)
                    else body.get("message") or body.get("detail")
                ) or (resp.text[:300] if resp is not None else str(exc))
            except Exception:  # noqa: BLE001
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("Krea submit failed (%d): %s", status, err_msg)
            return error_response(
                error=f"Krea image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="Krea submit timed out (30s)",
                error_type="timeout",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"Krea connection error: {exc}",
                error_type="connection_error",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            submit_body = response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Krea returned invalid JSON on submit: {exc}",
                error_type="invalid_response",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        job_id = submit_body.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            return error_response(
                error="Krea submit response missing job_id",
                error_type="invalid_response",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # 2. Poll for completion.
        job_url = f"{BASE_URL}/jobs/{job_id}"
        poll_headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Hermes-Agent/1.0 (krea-image-gen)",
        }
        interval = _POLL_INITIAL_INTERVAL
        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        last_status: Optional[str] = None

        while True:
            time.sleep(interval)
            interval = min(interval * _POLL_BACKOFF, _POLL_MAX_INTERVAL)

            try:
                poll_resp = requests.get(job_url, headers=poll_headers, timeout=30)
                poll_resp.raise_for_status()
            except requests.HTTPError as exc:
                resp = exc.response
                status = resp.status_code if resp is not None else 0
                logger.error("Krea poll failed (%d) for job %s", status, job_id)
                # Fail fast for non-retryable statuses (auth/billing/not-found,
                # other permanent 4xx) so callers don't wait the full 180s
                # deadline on a request that will never succeed. Only retry
                # transient statuses such as 408/409/425/429/5xx.
                if status not in _RETRYABLE_POLL_STATUSES or time.monotonic() >= deadline:
                    return error_response(
                        error=f"Krea poll failed ({status}) for job {job_id}",
                        error_type="api_error",
                        provider="krea",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                # Otherwise keep trying — transient 5xx (and a few retryable
                # 4xx like 408/409/425/429) are common on async jobs.
                continue
            except (requests.Timeout, requests.ConnectionError) as exc:
                logger.warning("Krea poll transient error for job %s: %s", job_id, exc)
                if time.monotonic() >= deadline:
                    return error_response(
                        error=f"Krea poll timed out for job {job_id}: {exc}",
                        error_type="timeout",
                        provider="krea",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                continue

            try:
                job = poll_resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Krea poll returned invalid JSON for job %s: %s", job_id, exc)
                if time.monotonic() >= deadline:
                    return error_response(
                        error=f"Krea poll returned invalid JSON: {exc}",
                        error_type="invalid_response",
                        provider="krea",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                continue

            status_str = job.get("status") if isinstance(job, dict) else None
            if isinstance(status_str, str):
                last_status = status_str
                if status_str in _TERMINAL_STATES:
                    break

            # ``completed_at`` is a backstop terminal marker even when the
            # ``status`` enum is unfamiliar (Krea adds new pending states
            # over time — backlogged/scheduled/sampling — and we don't
            # want to mis-handle a future one).
            if isinstance(job, dict) and job.get("completed_at"):
                break

            if time.monotonic() >= deadline:
                return error_response(
                    error=(
                        f"Krea job {job_id} did not complete within "
                        f"{int(_POLL_TIMEOUT_SECONDS)}s (last status: {last_status or 'unknown'})"
                    ),
                    error_type="timeout",
                    provider="krea",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        # 3. Terminal — extract result.
        if not isinstance(job, dict):
            return error_response(
                error="Krea returned non-dict job body",
                error_type="invalid_response",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if last_status == "failed":
            err = (job.get("result") or {}).get("error") if isinstance(job.get("result"), dict) else None
            return error_response(
                error=f"Krea job {job_id} failed: {err or 'unknown error'}",
                error_type="api_error",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if last_status == "cancelled":
            return error_response(
                error=f"Krea job {job_id} was cancelled",
                error_type="cancelled",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Successful path — pull URL out of the result.
        result = job.get("result")
        if not isinstance(result, dict):
            return error_response(
                error="Krea job completed but result was missing",
                error_type="empty_response",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Per Krea's job-lifecycle docs the completed payload exposes
        # ``result.urls`` (an array). Fall back to a single ``url`` field
        # for forward/backward compatibility.
        image_url: Optional[str] = None
        urls = result.get("urls")
        if isinstance(urls, list) and urls:
            for candidate in urls:
                if isinstance(candidate, str) and candidate.strip():
                    image_url = candidate.strip()
                    break
        if image_url is None:
            single = result.get("url")
            if isinstance(single, str) and single.strip():
                image_url = single.strip()

        if image_url is None:
            return error_response(
                error="Krea result contained no image URL",
                error_type="empty_response",
                provider="krea",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Materialise locally — Krea result URLs may expire, mirroring
        # what we do for xAI / OpenAI URL responses (#26942).
        try:
            saved_path = save_url_image(image_url, prefix=f"krea_{model_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Krea image URL %s could not be cached (%s); falling back to bare URL.",
                image_url,
                exc,
            )
            image_ref = image_url
        else:
            image_ref = str(saved_path)

        extra: Dict[str, Any] = {
            "krea_aspect_ratio": krea_ar,
            "resolution": DEFAULT_RESOLUTION,
            "creativity": creativity,
            "job_id": job_id,
        }
        if isinstance(job.get("completed_at"), str):
            extra["completed_at"] = job["completed_at"]

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="krea",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``KreaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(KreaImageGenProvider())

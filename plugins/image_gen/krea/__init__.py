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
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
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
    "krea-2-medium-turbo": {
        "display": "Krea 2 Medium Turbo",
        "speed": "~8-15s",
        "strengths": "Fastest Krea 2 — medium quality at lower latency / cost.",
        "price": "$0.015 (text) / $0.0175 (style refs)",
        "path": "medium-turbo",
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

# Krea's image_style_references entries are objects ({"url", "strength"}), not
# bare URL strings. When the caller supplies a URL without an explicit strength
# we apply Krea's recommended starting value. Range per Krea docs is -2..2.
_DEFAULT_STYLE_REFERENCE_STRENGTH = 0.6

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


def _resolve_model(explicit: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``.

    Precedence: explicit caller override (e.g. managed-mode routing or a direct
    ``model`` kwarg) → ``KREA_IMAGE_MODEL`` env → ``image_gen.krea.model`` →
    ``image_gen.model`` → :data:`DEFAULT_MODEL`.
    """
    if isinstance(explicit, str) and explicit.strip() in _MODELS:
        return explicit.strip(), _MODELS[explicit.strip()]

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


def _resolve_managed_krea_gateway():
    """Return managed Krea gateway config when the user is on the managed path.

    Mirrors ``_resolve_managed_fal_gateway`` in ``tools/image_generation_tool.py``:
    the Nous-hosted Krea gateway wins when it is resolvable AND either no direct
    ``KREA_API_KEY`` is configured or the user explicitly opted into the gateway
    for ``image_gen``. Returns ``None`` (direct/BYO path) otherwise, and never
    raises — plugin discovery and availability scans must stay robust.
    """
    try:
        from tools.managed_tool_gateway import resolve_managed_tool_gateway
        from tools.tool_backend_helpers import prefers_gateway
    except Exception as exc:  # noqa: BLE001
        logger.debug("Managed Krea gateway resolution unavailable: %s", exc)
        return None

    if os.environ.get("KREA_API_KEY") and not prefers_gateway("image_gen"):
        return None

    try:
        return resolve_managed_tool_gateway("krea")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Managed Krea gateway resolution failed: %s", exc)
        return None


def _managed_krea_gateway_ready() -> bool:
    """Cheap, offline-friendly probe for managed Krea availability."""
    try:
        from tools.managed_tool_gateway import is_managed_tool_gateway_ready
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(is_managed_tool_gateway_ready("krea"))
    except Exception:  # noqa: BLE001
        return False


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
        # Available with a direct Krea key OR via the managed Nous gateway
        # (Nous Subscription), so portal users with no Krea key can still
        # reach Krea 2 through the gateway.
        return bool(os.environ.get("KREA_API_KEY")) or _managed_krea_gateway_ready()

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
            "tag": "Krea 2 foundation model — Medium ($0.03), Large ($0.06), Medium Turbo ($0.015). Style transfer, moodboards, reference-guided generation. Direct key or managed Nous Subscription gateway.",
            "env_vars": [
                {
                    "key": "KREA_API_KEY",
                    "prompt": "Krea API key",
                    "url": "https://www.krea.ai/settings/api-tokens",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Krea supports reference-guided generation (image-to-image style
        # transfer) via image_style_references — up to 10 refs.
        return {"modalities": ["text", "image"], "max_reference_images": 10, "url_input": True}

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

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
        krea_ar = _ASPECT_MAP.get(aspect, "1:1")

        # Collect reference images for reference-guided generation (image-to-
        # image style transfer). Sources, in order:
        #   1. unified image_url (primary source) + reference_image_urls (strings)
        #   2. legacy image_style_references kwarg — may be plain URL strings OR
        #      Krea's richer ref objects (e.g. {"url": ..., "strength": ...}),
        #      which are passed through verbatim for backward compatibility.
        style_refs: List[Any] = []
        if isinstance(image_url, str) and image_url.strip():
            style_refs.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            style_refs.append(ref)
        legacy_refs = kwargs.get("image_style_references")
        if isinstance(legacy_refs, list):
            for ref in legacy_refs:
                if isinstance(ref, str):
                    if ref.strip():
                        style_refs.append(ref.strip())
                elif ref:
                    # Non-string ref object (dict, etc.) — pass through as-is.
                    style_refs.append(ref)
        # Dedupe string entries while preserving order (dict refs aren't
        # hashable, so they're kept verbatim); Krea caps at 10.
        seen: set = set()
        deduped: List[Any] = []
        for r in style_refs:
            if isinstance(r, str):
                if r in seen:
                    continue
                seen.add(r)
            deduped.append(r)
        style_refs = deduped[:10]
        modality = "image" if style_refs else "text"

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="krea",
                aspect_ratio=aspect,
            )

        # Route through the managed Nous gateway (Nous Subscription) when the
        # user is on the managed path; otherwise use the direct Krea API with a
        # BYO ``KREA_API_KEY``. The gateway owns the shared Krea credential and
        # meters/bills per generation, so the caller token is the Nous access
        # token, not a Krea key.
        managed = _resolve_managed_krea_gateway()
        if managed is not None:
            base_url = managed.gateway_origin.rstrip("/")
            auth_token = managed.nous_user_token
        else:
            base_url = BASE_URL
            auth_token = os.environ.get("KREA_API_KEY")
            if not auth_token:
                return error_response(
                    error=(
                        "KREA_API_KEY not set. Run `hermes tools` → Image "
                        "Generation → Krea to configure, get a key at "
                        "https://www.krea.ai/settings/api-tokens, or sign in to "
                        "a Nous account with the managed Krea gateway enabled "
                        "(`hermes setup`)."
                    ),
                    error_type="auth_required",
                    provider="krea",
                    aspect_ratio=aspect,
                )

        model_id, meta = _resolve_model(kwargs.get("model"))
        creativity = _resolve_creativity(kwargs.get("creativity"))

        # The managed gateway only prices base text-to-image and URL
        # ``image_style_references`` tiers. Trained styles (LoRAs) and
        # moodboards have no managed price and are rejected at the gateway, so
        # fail fast here with actionable guidance instead of a raw 400.
        if managed is not None:
            if isinstance(kwargs.get("styles"), list) and kwargs.get("styles"):
                return error_response(
                    error=(
                        "Managed Krea (Nous Subscription) does not support "
                        "trained styles (LoRAs). Set KREA_API_KEY to use Krea "
                        "directly, or omit `styles`."
                    ),
                    error_type="unsupported_argument",
                    provider="krea",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            if isinstance(kwargs.get("moodboards"), list) and kwargs.get("moodboards"):
                return error_response(
                    error=(
                        "Managed Krea (Nous Subscription) does not support "
                        "moodboards. Set KREA_API_KEY to use Krea directly, or "
                        "omit `moodboards`."
                    ),
                    error_type="unsupported_argument",
                    provider="krea",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

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

        if style_refs:
            # Reference-guided generation (image-to-image style transfer).
            # Krea requires each entry to be an object ({"url", "strength"}),
            # NOT a bare URL string — a string yields a 422 "Expected object,
            # received string". Convert URL strings to the object form and pass
            # already-object refs through verbatim (clamped to 10 above).
            normalized_refs: List[Any] = []
            for ref in style_refs:
                if isinstance(ref, str):
                    normalized_refs.append(
                        {"url": ref, "strength": _DEFAULT_STYLE_REFERENCE_STRENGTH}
                    )
                else:
                    normalized_refs.append(ref)
            payload["image_style_references"] = normalized_refs

        moodboards = kwargs.get("moodboards")
        if isinstance(moodboards, list) and moodboards:
            # Krea currently caps at 1 moodboard per request.
            payload["moodboards"] = moodboards[:1]

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent/1.0 (krea-image-gen)",
        }
        if managed is not None:
            # The gateway derives the per-generation billing idempotency
            # boundary from this header (else it falls back to a body
            # fingerprint). A fresh key per submit keeps each generation a
            # distinct billable execution.
            headers["x-idempotency-key"] = str(uuid.uuid4())

        # 1. Submit job.
        submit_url = f"{base_url}/generate/image/krea/krea-2/{meta['path']}"
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
            # On a managed 4xx, surface actionable remediation mirroring the
            # FAL managed gateway path: the model may not be enabled/priced on
            # the Nous Portal, or the gateway's shared Krea key hit its
            # concurrency cap (429).
            if managed is not None and 400 <= status < 500:
                hint = (
                    "Krea's shared-key concurrency cap was hit — retry shortly."
                    if status == 429
                    else (
                        f"Model '{model_id}' may not be enabled/priced on the "
                        "Nous Portal's Krea gateway. Set KREA_API_KEY to use "
                        "Krea directly, or pick a different model via "
                        "`hermes tools` → Image Generation."
                    )
                )
                return error_response(
                    error=(
                        f"Nous Subscription Krea gateway rejected '{model_id}' "
                        f"(HTTP {status}): {err_msg}. {hint}"
                    ),
                    error_type="api_error",
                    provider="krea",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
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

        # 2. Poll for completion. Status/result polling is bound to the same
        # principal at the gateway, so the managed path polls the gateway's
        # ``/jobs/{id}`` with the Nous token (404 on cross-user/unknown jobs).
        job_url = f"{base_url}/jobs/{job_id}"
        poll_headers = {
            "Authorization": f"Bearer {auth_token}",
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
        result_image_url: Optional[str] = None
        urls = result.get("urls")
        if isinstance(urls, list) and urls:
            for candidate in urls:
                if isinstance(candidate, str) and candidate.strip():
                    result_image_url = candidate.strip()
                    break
        if result_image_url is None:
            single = result.get("url")
            if isinstance(single, str) and single.strip():
                result_image_url = single.strip()

        if result_image_url is None:
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
            saved_path = save_url_image(result_image_url, prefix=f"krea_{model_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Krea image URL %s could not be cached (%s); falling back to bare URL.",
                result_image_url,
                exc,
            )
            image_ref = result_image_url
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
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``KreaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(KreaImageGenProvider())

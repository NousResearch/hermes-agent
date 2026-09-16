"""Krea image generation backend (``Krea 2`` Medium / Large / Medium Turbo).

Krea's API is asynchronous: submit returns a ``job_id`` polled at ``GET
/jobs/{job_id}``; ``generate()`` hides that (submit, poll every 2s with light
backoff, cache the result URL locally). Selection: ``model`` kwarg →
``KREA_IMAGE_MODEL`` → ``image_gen.krea.model`` → ``image_gen.model`` (when one
of our IDs) → :data:`DEFAULT_MODEL`. Docs: https://docs.krea.ai/developers/krea-2/overview
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from agent.secret_scope import get_secret
from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, save_url_image, success_response
from plugins.image_gen._common import (
    ErrorFn, StaticImageGenProvider, collect_source_images, error_factory, load_image_gen_config, post_json,
    prompt_required_error, resolve_static_model)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.krea.ai"

# ``path`` is Krea's URL segment. ``upscale`` (Enhance pass) is opt-in for every tier:
# default-on enhance degraded output quality, and Large is 2K native anyway.
_MODELS: Dict[str, Dict[str, Any]] = {
    "krea-2-medium": {
        "display": "Krea 2 Medium", "speed": "~15-25s",
        "strengths": "Illustration, anime, painting, expressive styles. Faster + cheaper.",
        "price": "$0.030 (text) / $0.035 (style refs) / $0.040 (moodboards)",
        "path": "medium", "upscale": False,
    },
    "krea-2-large": {
        "display": "Krea 2 Large", "speed": "~25-60s",
        "strengths": "Photorealism, raw textured looks (motion blur, grain), expressive styles.",
        "price": "$0.060 (text) / $0.065 (style refs) / $0.070 (moodboards)",
        "path": "large", "upscale": False,
    },
    "krea-2-medium-turbo": {
        "display": "Krea 2 Medium Turbo", "speed": "~8-15s",
        "strengths": "Fastest Krea 2 — medium quality at lower latency / cost.",
        "price": "$0.015 (text) / $0.0175 (style refs)", "path": "medium-turbo", "upscale": False,
    },
}

DEFAULT_MODEL = "krea-2-medium"

# Hermes' 3 abstract ratios → Krea's enum (1:1, 4:3, 3:2, 16:9, 2.35:1, 4:5, 2:3, 9:16).
_ASPECT_MAP = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
DEFAULT_RESOLUTION = "1K"  # only resolution Krea currently supports
# Style refs are objects ({"url", "strength"}); bare URLs get Krea's recommended start (range -2..2).
_DEFAULT_STYLE_REFERENCE_STRENGTH = 0.6
_MAX_STYLE_REFERENCES = 10
_VALID_CREATIVITY = {"raw", "low", "medium", "high"}

# Polling: Krea recommends 2-5s; 2s backing off to 5s (Large ~1min); ceiling = Krea's 3 min tool timeout.
_POLL_INITIAL_INTERVAL = 2.0
_POLL_MAX_INTERVAL = 5.0
_POLL_BACKOFF = 1.3
_POLL_TIMEOUT_SECONDS = 180.0
# Retryable poll statuses; other 4xx are permanent — surface them instead of burning the deadline.
_RETRYABLE_POLL_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TERMINAL_STATES = {"completed", "failed", "cancelled"}

# Krea Enhance — the optional ``upscale`` pass after generation (max 8K).
_ENHANCE_PATH = "/generate/enhance/krea/enhance"
_ENHANCE_SCALE_FACTOR = 2
_USER_AGENT = "Hermes-Agent/1.0 (krea-image-gen)"

# Fatal poll outcome (``_poll_krea_job`` ``kind``) → (error_type, message builder).
_POLL_FAILURES: Dict[str, Tuple[str, Callable[[str, Any], str]]] = {
    "http": ("api_error", lambda job_id, detail: f"Krea poll failed ({detail}) for job {job_id}"),
    "timeout": ("timeout", lambda job_id, detail: f"Krea poll timed out for job {job_id}: {detail}"),
    "invalid_json": ("invalid_response", lambda job_id, detail: f"Krea poll returned invalid JSON: {detail}"),
    "deadline": ("timeout", lambda job_id, detail: (
        f"Krea job {job_id} did not complete within "
        f"{int(_POLL_TIMEOUT_SECONDS)}s (last status: {detail or 'unknown'})")),
}
# Managed gateway prices base text-to-image and URL style references only.
_MANAGED_UNSUPPORTED = (("trained styles (LoRAs)", "styles"), ("moodboards", "moodboards"))


def _load_krea_config() -> Dict[str, Any]:
    """Read ``image_gen`` (the krea section lives under ``image_gen.krea``)."""
    return load_image_gen_config()


def _krea_section() -> Dict[str, Any]:
    section = _load_krea_config().get("krea")
    return section if isinstance(section, dict) else {}


def _resolve_model(explicit: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    return resolve_static_model(
        _MODELS, DEFAULT_MODEL, env_var="KREA_IMAGE_MODEL", config_key="krea", explicit=explicit,
        config=_load_krea_config())


def _resolve_managed_krea_gateway():
    """Managed gateway config on the managed path, else ``None``. Managed when the stored
    ``image_gen`` selection is ``nous`` (or legacy ``use_gateway: true``), or never-configured with
    no ``KREA_API_KEY``; an explicit vendor selection pins direct. Never raises (discovery scans)."""
    try:
        from tools.managed_tool_gateway import resolve_managed_tool_gateway
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection
    except Exception as exc:  # noqa: BLE001
        logger.debug("Managed Krea gateway resolution unavailable: %s", exc)
        return None
    try:
        selected = read_selection("image_gen")
    except Exception:  # noqa: BLE001
        selected = None
    if selected is not None and selected != NOUS_MANAGED_PROVIDER:
        return None
    if selected is None and get_secret("KREA_API_KEY"):
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

        return bool(is_managed_tool_gateway_ready("krea"))
    except Exception:  # noqa: BLE001
        return False


def _resolve_creativity(value: Optional[str]) -> str:
    """Coerce ``creativity`` kwarg (then config) to a valid Krea value; default ``medium``."""
    for candidate in (value, _krea_section().get("creativity")):
        if isinstance(candidate, str) and candidate.strip().lower() in _VALID_CREATIVITY:
            return candidate.strip().lower()
    return "medium"


def _headers(auth_token: str, *, managed: bool, json_body: bool) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {auth_token}", "User-Agent": _USER_AGENT}
    if json_body:
        headers["Content-Type"] = "application/json"
    if managed:
        # Gateway billing idempotency boundary: a fresh key per submit = one billable execution.
        headers["x-idempotency-key"] = str(uuid.uuid4())
    return headers


def _submit_error_message(resp: Any, exc: Exception) -> str:
    fallback = resp.text[:300] if resp is not None else str(exc)
    try:
        body = resp.json() if resp is not None else {}
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        else:
            message = body.get("message") or body.get("detail")
        return message or fallback
    except Exception:  # noqa: BLE001
        return fallback


def _is_terminal(job: Any) -> bool:
    """``completed_at`` is a backstop terminal marker for unfamiliar ``status`` enums (Krea adds
    pending states — backlogged/scheduled/sampling — over time)."""
    return isinstance(job, dict) and (job.get("status") in _TERMINAL_STATES or bool(job.get("completed_at")))


def _poll_krea_job(
    base_url: str, auth_token: str, job_id: str, *, timeout_seconds: float = _POLL_TIMEOUT_SECONDS,
    on_error: Optional[Any] = None,
) -> Any:
    """Poll ``/jobs/{job_id}`` until terminal; returns the job dict or ``None`` when it gave up.

    With ``on_error(kind, detail)`` (main path) a fatal poll failure returns that callback's
    result; without it (best-effort Enhance) failures only log. ``kind`` ∈ ``http`` (detail =
    status) / ``timeout`` / ``invalid_json`` / ``deadline`` (detail = last status seen).
    """
    job_url = f"{base_url}/jobs/{job_id}"
    headers = _headers(auth_token, managed=False, json_body=False)
    interval = _POLL_INITIAL_INTERVAL
    deadline = time.monotonic() + timeout_seconds
    last_status: Optional[str] = None
    enhance = on_error is None

    def give_up(kind: str, detail: Any, warning: str, *warn_args: Any) -> Any:
        if on_error is None:
            logger.warning(warning, *warn_args)
            return None
        return on_error(kind, detail)

    while True:
        time.sleep(interval)
        interval = min(interval * _POLL_BACKOFF, _POLL_MAX_INTERVAL)
        try:
            resp = requests.get(job_url, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if not enhance:
                logger.error("Krea poll failed (%d) for job %s", status, job_id)
            # Fail fast on permanent statuses; retry transient ones.
            if status not in _RETRYABLE_POLL_STATUSES or time.monotonic() >= deadline:
                return give_up("http", status, "Krea enhance poll failed (%d) for job %s", status, job_id)
            continue
        except (requests.Timeout, requests.ConnectionError) as exc:
            if not enhance:
                logger.warning("Krea poll transient error for job %s: %s", job_id, exc)
            if time.monotonic() >= deadline:
                return give_up("timeout", exc, "Krea enhance poll gave up for job %s: %s", job_id, exc)
            continue
        except Exception as exc:  # noqa: BLE001 — enhance-only: any other failure is best-effort
            if not enhance:
                raise
            if time.monotonic() >= deadline:
                logger.warning("Krea enhance poll gave up for job %s: %s", job_id, exc)
                return None
            continue

        try:
            job = resp.json()
        except Exception as exc:  # noqa: BLE001
            if not enhance:
                logger.warning("Krea poll returned invalid JSON for job %s: %s", job_id, exc)
            if time.monotonic() >= deadline:
                return give_up("invalid_json", exc, "Krea enhance poll gave up for job %s: %s", job_id, exc)
            continue

        if isinstance(job, dict) and isinstance(job.get("status"), str):
            last_status = job["status"]
        if _is_terminal(job):
            return job
        if time.monotonic() >= deadline:
            return give_up(
                "deadline", last_status,
                "Krea enhance job %s did not finish in %ds", job_id, int(timeout_seconds))


def _extract_result_url(job: Optional[Dict[str, Any]]) -> Optional[str]:
    """First result URL: ``result.urls[]`` per Krea's job docs, else ``result.url``."""
    result = job.get("result") if isinstance(job, dict) else None
    if not isinstance(result, dict):
        return None
    urls = result.get("urls")
    for candidate in [*(urls if isinstance(urls, list) else []), result.get("url")]:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _enhance_image(
    base_url: str, auth_token: str, image_url: str, prompt: str, *, managed: bool
) -> Optional[str]:
    """Krea Enhance on ``image_url`` → enhanced URL, or ``None`` on any failure (best-effort: an
    upscale failure must never destroy an already-successful generation)."""
    # The prompt guides detail; default ai_strength (0.4) adds detail without redrawing.
    payload = {"image_url": image_url, "image_scaling_factor": _ENHANCE_SCALE_FACTOR, "prompt": prompt}
    try:
        resp = requests.post(
            f"{base_url}{_ENHANCE_PATH}", headers=_headers(auth_token, managed=managed, json_body=True),
            json=payload, timeout=30)
        resp.raise_for_status()
        job_id = (resp.json() or {}).get("job_id")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Krea Enhance submit failed: %s", exc)
        return None
    if not isinstance(job_id, str) or not job_id:
        logger.warning("Krea Enhance submit response missing job_id")
        return None
    job = _poll_krea_job(base_url, auth_token, job_id)
    if not isinstance(job, dict) or job.get("status") in {"failed", "cancelled"}:
        logger.warning("Krea Enhance job %s did not complete successfully", job_id)
        return None
    return _extract_result_url(job)


def _collect_style_refs(
    image_url: Optional[str], reference_image_urls: Optional[List[str]], legacy_refs: Any
) -> List[Any]:
    """``image_url`` + ``reference_image_urls`` first, then legacy ``image_style_references``
    (URL strings or Krea ref objects, passed through); strings deduped in order; capped at 10."""
    refs: List[Any] = collect_source_images(image_url, reference_image_urls)
    for ref in legacy_refs if isinstance(legacy_refs, list) else []:
        if isinstance(ref, str):
            if ref.strip():
                refs.append(ref.strip())
        elif ref:
            refs.append(ref)
    seen: set = set()
    deduped: List[Any] = []
    for r in refs:
        if isinstance(r, str):
            if r in seen:
                continue
            seen.add(r)
        deduped.append(r)
    return deduped[:_MAX_STYLE_REFERENCES]


def _build_payload(
    prompt: str, krea_ar: str, creativity: str, style_refs: List[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt, "aspect_ratio": krea_ar, "resolution": DEFAULT_RESOLUTION, "creativity": creativity,
    }
    if isinstance(kwargs.get("seed"), int):
        payload["seed"] = kwargs["seed"]
    styles, moodboards = kwargs.get("styles"), kwargs.get("moodboards")
    if isinstance(styles, list) and styles:
        payload["styles"] = styles
    if style_refs:
        # Krea requires objects — a bare string yields 422 "Expected object, received string".
        payload["image_style_references"] = [
            {"url": ref, "strength": _DEFAULT_STYLE_REFERENCE_STRENGTH} if isinstance(ref, str) else ref
            for ref in style_refs
        ]
    if isinstance(moodboards, list) and moodboards:
        payload["moodboards"] = moodboards[:1]  # Krea caps at 1 moodboard per request.
    return payload


def _submit_job(
    base_url: str, auth_token: str, model_path: str, payload: Dict[str, Any], managed: bool, model_id: str,
    fail: ErrorFn,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """POST the generation request; ``(job_id, None)`` or ``(None, error)``."""
    submit_body, failure = post_json(
        f"{base_url}/generate/image/krea/krea-2/{model_path}",
        headers=_headers(auth_token, managed=managed, json_body=True),
        payload=payload, timeout=30, label="Krea", error_message=_submit_error_message)
    if failure is not None:
        if failure.kind == "http":
            status, err_msg = failure.status, failure.message
            logger.error("Krea submit failed (%d): %s", status, err_msg)
            # Managed 4xx: model not enabled/priced on the Portal, or shared-key concurrency cap (429).
            if managed and 400 <= status < 500:
                hint = (
                    "Krea's shared-key concurrency cap was hit — retry shortly." if status == 429 else
                    f"Model '{model_id}' may not be enabled/priced on the Nous Portal's Krea gateway. "
                    "Set KREA_API_KEY to use Krea directly, or pick a different model via "
                    "`hermes tools` → Image Generation.")
                return None, fail(
                    f"Nous Subscription Krea gateway rejected '{model_id}' "
                    f"(HTTP {status}): {err_msg}. {hint}",
                    "api_error")
            return None, fail(failure.error, "api_error")
        if failure.kind == "timeout":
            return None, fail("Krea submit timed out (30s)", "timeout")
        if failure.kind == "invalid_json":
            return None, fail(f"Krea returned invalid JSON on submit: {failure.message}", "invalid_response")
        return None, fail(failure.error, failure.error_type)
    job_id = submit_body.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return None, fail("Krea submit response missing job_id", "invalid_response")
    return job_id, None


def _terminal_result_url(
    job: Dict[str, Any], job_id: str, fail: ErrorFn
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Result URL of a terminal job; ``(url, None)`` or ``(None, error)``."""
    result = job.get("result")
    if job.get("status") == "failed":
        err = result.get("error") if isinstance(result, dict) else None
        return None, fail(f"Krea job {job_id} failed: {err or 'unknown error'}", "api_error")
    if job.get("status") == "cancelled":
        return None, fail(f"Krea job {job_id} was cancelled", "cancelled")
    if not isinstance(result, dict):
        return None, fail("Krea job completed but result was missing", "empty_response")
    result_image_url = _extract_result_url(job)
    if result_image_url is None:
        return None, fail("Krea result contained no image URL", "empty_response")
    return result_image_url, None


def _upscale_requested(explicit: Any, meta: Dict[str, Any]) -> bool:
    """Precedence: explicit kwarg > ``image_gen.krea.upscale`` config > per-model catalog default."""
    if isinstance(explicit, bool):
        return explicit
    cfg_upscale = _krea_section().get("upscale")
    return cfg_upscale if isinstance(cfg_upscale, bool) else bool(meta.get("upscale", False))


class KreaImageGenProvider(StaticImageGenProvider):
    """Krea ``Krea 2`` foundation image model backend (Medium + Large)."""

    provider_id = "krea"
    label = "Krea"
    models = _MODELS
    default_model_id = DEFAULT_MODEL
    setup = dict(
        name="Krea", badge="paid",
        tag="Krea 2 foundation model — Medium ($0.03), Large ($0.06), Medium Turbo ($0.015). Style transfer, moodboards, reference-guided generation. Direct key or managed Nous Subscription gateway.",
        key="KREA_API_KEY", prompt="Krea API key", url="https://www.krea.ai/settings/api-tokens")

    def is_available(self) -> bool:
        # Direct key OR managed Nous gateway (portal users without a Krea key).
        return bool(get_secret("KREA_API_KEY")) or _managed_krea_gateway_ready()

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"], "max_reference_images": _MAX_STYLE_REFERENCES,
            "supports_upscale": True,
        }

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        krea_ar = _ASPECT_MAP.get(aspect, "1:1")
        style_refs = _collect_style_refs(
            image_url, reference_image_urls, kwargs.get("image_style_references"))
        if not prompt:
            return prompt_required_error("krea", aspect)

        # Managed gateway owns the shared Krea credential and meters per generation (token =
        # Nous access token); otherwise direct Krea with a BYO ``KREA_API_KEY``.
        managed = _resolve_managed_krea_gateway()
        if managed is not None:
            base_url = managed.gateway_origin.rstrip("/")
            auth_token = managed.nous_user_token
        else:
            base_url = BASE_URL
            auth_token = get_secret("KREA_API_KEY")
            if not auth_token:
                return error_factory("krea", aspect)(
                    "KREA_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → Krea to configure, get a key at "
                    "https://www.krea.ai/settings/api-tokens, or sign in to "
                    "a Nous account with the managed Krea gateway enabled "
                    "(`hermes setup`).",
                    "auth_required")

        model_id, meta = _resolve_model(kwargs.get("model"))
        creativity = _resolve_creativity(kwargs.get("creativity"))
        fail = error_factory("krea", aspect, model=model_id, prompt=prompt)
        payload = _build_payload(prompt, krea_ar, creativity, style_refs, kwargs)

        # LoRAs/moodboards are rejected by the managed gateway: fail fast with guidance, not a raw 400.
        if managed is not None:
            for what, arg in _MANAGED_UNSUPPORTED:
                if arg in payload:
                    return fail(
                        f"Managed Krea (Nous Subscription) does not support {what}. "
                        f"Set KREA_API_KEY to use Krea directly, or omit `{arg}`.",
                        "unsupported_argument")

        # 1. Submit job.
        job_id, err = _submit_job(
            base_url, auth_token, meta["path"], payload, managed is not None, model_id, fail)
        if err is not None:
            return err

        # 2. Poll — same principal as submit, so the managed path polls the gateway with the Nous token.
        poll_errors: List[Dict[str, Any]] = []

        def poll_error(kind: str, detail: Any) -> Dict[str, Any]:
            error_type, build = _POLL_FAILURES.get(kind, _POLL_FAILURES["deadline"])
            poll_errors.append(fail(build(job_id, detail), error_type))
            return poll_errors[-1]

        job = _poll_krea_job(base_url, auth_token, job_id, on_error=poll_error)
        if poll_errors:
            return poll_errors[0]
        if not isinstance(job, dict):
            return fail("Krea returned non-dict job body", "invalid_response")

        # 3. Terminal — extract result.
        result_image_url, err = _terminal_result_url(job, job_id, fail)
        if err is not None:
            return err

        # Krea Enhance pass — best-effort: failure falls back to the original image.
        upscaled = False
        if _upscale_requested(kwargs.get("upscale"), meta):
            enhanced_url = _enhance_image(
                base_url, auth_token, result_image_url, prompt, managed=managed is not None)
            if enhanced_url:
                result_image_url = enhanced_url
                upscaled = True
            else:
                logger.warning("Krea Enhance pass failed — returning native-resolution image")

        # Materialise locally — Krea result URLs may expire.
        try:
            # See #26942.
            image_ref = str(save_url_image(result_image_url, prefix=f"krea_{model_id}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Krea image URL %s could not be cached (%s); falling back to bare URL.", result_image_url, exc,
            )
            image_ref = result_image_url
        extra: Dict[str, Any] = {
            "krea_aspect_ratio": krea_ar, "resolution": DEFAULT_RESOLUTION, "creativity": creativity,
            "job_id": job_id, "upscaled": upscaled,
        }
        if upscaled:
            extra["upscale_factor"] = _ENHANCE_SCALE_FACTOR
        if isinstance(job.get("completed_at"), str):
            extra["completed_at"] = job["completed_at"]
        return success_response(
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect, provider="krea",
            modality="image" if style_refs else "text", extra=extra)


def register(ctx) -> None:
    """Plugin entry point — wire ``KreaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(KreaImageGenProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'ImageGenProvider': ('agent.image_gen_provider', 'ImageGenProvider'),
    'error_response': ('agent.image_gen_provider', 'error_response'),
    'normalize_reference_images': ('agent.image_gen_provider', 'normalize_reference_images'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

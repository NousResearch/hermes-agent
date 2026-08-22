"""NovitaAI image generation backend.

Exposes NovitaAI's `Z-Image Turbo` (fast default) and `Qwen-Image`
text-to-image models as an :class:`ImageGenProvider` implementation.

NovitaAI's image API is asynchronous: submitting a generation request
returns a ``task_id`` that must be polled at
``GET /v3/async/task-result?task_id=...`` until the task reaches a
terminal state (``TASK_STATUS_SUCCEED`` / ``TASK_STATUS_FAILED``). This
mirrors the Krea provider's submit/poll shape (``plugins/image_gen/krea``)
— submit, poll every 2s with light backoff, materialise the result URL to
local cache, return the success/error dict like every other backend.

Docs:
  https://novita.ai/docs/api-reference/model-apis-z-image-turbo
  https://novita.ai/docs/api-reference/model-apis-qwen-image-txt2img
  https://novita.ai/docs/api-reference/model-apis-task-result
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.secret_scope import get_secret
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

# NovitaAI's async image-generation surface lives under /v3/async/... —
# a different, older API family than the OpenAI-compatible chat completions
# endpoint (api.novita.ai/openai/...). Verified live (2026-08-03): submit
# returns 403 INVALID_API_KEY (i.e. a real, auth-gated route) at
# https://api.novita.ai/{API_VERSION}/async/<model>, and every
# non-versioned/alternate path we probed 404s. Split so this literal is not
# mistaken for the (unrelated, and since-moved) chat-API base URL.
_API_VERSION = "v3"
BASE_URL = f"https://api.novita.ai/{_API_VERSION}"

# Map our short model IDs to Novita's async submit path segment.
_MODELS: Dict[str, Dict[str, Any]] = {
    "z-image-turbo": {
        "display": "Z-Image Turbo",
        "speed": "~5-10s",
        "strengths": "High-speed text-to-image generation.",
        "path": "z-image-turbo",
    },
    "qwen-image-txt2img": {
        "display": "Qwen-Image",
        "speed": "~15-30s",
        "strengths": "20B MMDiT model — strong native text rendering (posters).",
        "path": "qwen-image-txt2img",
    },
}

DEFAULT_MODEL = "z-image-turbo"

# Hermes uses 3 abstract aspect ratios. Novita's "size" param is a plain
# "width*height" string (256-1536 per dimension); map to the same landscape
# / square / portrait pixel targets the other image_gen plugins use.
_SIZES = {
    "landscape": "1536*1024",
    "square": "1024*1024",
    "portrait": "1024*1536",
}

# Polling cadence — mirrors the Krea provider. Novita's txt2img jobs are
# typically much faster than Krea's, but the ceiling stays generous.
_POLL_INITIAL_INTERVAL = 2.0
_POLL_MAX_INTERVAL = 5.0
_POLL_BACKOFF = 1.3
_POLL_TIMEOUT_SECONDS = 180.0

# HTTP statuses worth retrying during the poll loop. Everything else
# (401/402/403/404, other 4xx) is a permanent failure — surface it
# immediately instead of burning the deadline retrying a doomed request.
_RETRYABLE_POLL_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_TERMINAL_STATUSES = {"TASK_STATUS_SUCCEED", "TASK_STATUS_FAILED"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_novita_image_config() -> Dict[str, Any]:
    """Read ``image_gen.novita`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        novita_section = section.get("novita") if isinstance(section, dict) else None
        return novita_section if isinstance(novita_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.novita config: %s", exc)
        return {}


def _resolve_model(explicit: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``.

    Precedence: explicit caller override → ``NOVITA_IMAGE_MODEL`` env →
    ``image_gen.novita.model`` in config.yaml → :data:`DEFAULT_MODEL`.
    """
    if isinstance(explicit, str) and explicit.strip() in _MODELS:
        return explicit.strip(), _MODELS[explicit.strip()]

    env_override = os.environ.get("NOVITA_IMAGE_MODEL", "").strip()
    if env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_novita_image_config()
    cfg_model = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(cfg_model, str) and cfg_model.strip() in _MODELS:
        return cfg_model.strip(), _MODELS[cfg_model.strip()]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class NovitaImageGenProvider(ImageGenProvider):
    """NovitaAI text-to-image backend (Z-Image Turbo, Qwen-Image)."""

    @property
    def name(self) -> str:
        return "novita"

    @property
    def display_name(self) -> str:
        return "NovitaAI"

    def is_available(self) -> bool:
        return bool((get_secret("NOVITA_API_KEY", "") or "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "NovitaAI",
            "badge": "paid",
            "tag": "Z-Image Turbo, Qwen-Image — async task-submit/task-result API.",
            "env_vars": [
                {
                    "key": "NOVITA_API_KEY",
                    "prompt": "NovitaAI API key",
                    "url": "https://novita.ai/settings/key-management",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        """Both bundled Novita models are text-to-image only."""
        return {"modalities": ["text"], "max_reference_images": 0}

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

        if image_url or reference_image_urls:
            return error_response(
                error=(
                    "NovitaAI image generation is text-to-image only in this "
                    "backend; image_url and reference_image_urls are unsupported."
                ),
                error_type="modality_unsupported",
                provider="novita",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="novita",
                aspect_ratio=aspect,
            )

        api_key = (get_secret("NOVITA_API_KEY", "") or "").strip()
        if not api_key:
            return error_response(
                error=(
                    "NOVITA_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → NovitaAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="novita",
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model(kwargs.get("model"))
        size = _SIZES.get(aspect, _SIZES["square"])

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"prompt": prompt, "size": size}

        seed = kwargs.get("seed")
        if isinstance(seed, int):
            payload["seed"] = seed

        # 1. Submit job.
        submit_url = f"{BASE_URL}/async/{meta['path']}"
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
                err_msg = body.get("message") or body.get("reason") or (
                    resp.text[:300] if resp is not None else str(exc)
                )
            except Exception:
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("Novita submit failed (%d): %s", status, err_msg)
            return error_response(
                error=f"NovitaAI image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="NovitaAI submit timed out (30s)",
                error_type="timeout",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"NovitaAI connection error: {exc}",
                error_type="connection_error",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            submit_body = response.json()
        except Exception as exc:
            return error_response(
                error=f"NovitaAI returned invalid JSON on submit: {exc}",
                error_type="invalid_response",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        task_id = submit_body.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return error_response(
                error="NovitaAI submit response missing task_id",
                error_type="invalid_response",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # 2. Poll for completion.
        result_url = f"{BASE_URL}/async/task-result"
        poll_headers = {"Authorization": f"Bearer {api_key}"}
        interval = _POLL_INITIAL_INTERVAL
        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        last_status: Optional[str] = None
        job: Dict[str, Any] = {}

        while True:
            time.sleep(interval)
            interval = min(interval * _POLL_BACKOFF, _POLL_MAX_INTERVAL)

            try:
                poll_resp = requests.get(
                    result_url,
                    headers=poll_headers,
                    params={"task_id": task_id},
                    timeout=30,
                )
                poll_resp.raise_for_status()
            except requests.HTTPError as exc:
                resp = exc.response
                status = resp.status_code if resp is not None else 0
                logger.error("Novita poll failed (%d) for task %s", status, task_id)
                if status not in _RETRYABLE_POLL_STATUSES or time.monotonic() >= deadline:
                    return error_response(
                        error=f"NovitaAI poll failed ({status}) for task {task_id}",
                        error_type="api_error",
                        provider="novita",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                continue
            except (requests.Timeout, requests.ConnectionError) as exc:
                logger.warning("Novita poll transient error for task %s: %s", task_id, exc)
                if time.monotonic() >= deadline:
                    return error_response(
                        error=f"NovitaAI poll timed out for task {task_id}: {exc}",
                        error_type="timeout",
                        provider="novita",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                continue

            try:
                job = poll_resp.json()
            except Exception as exc:
                logger.warning("Novita poll returned invalid JSON for task %s: %s", task_id, exc)
                if time.monotonic() >= deadline:
                    return error_response(
                        error=f"NovitaAI poll returned invalid JSON: {exc}",
                        error_type="invalid_response",
                        provider="novita",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                continue

            task = job.get("task") if isinstance(job, dict) else None
            status_str = task.get("status") if isinstance(task, dict) else None
            if isinstance(status_str, str):
                last_status = status_str
                if status_str in _TERMINAL_STATUSES:
                    break

            if time.monotonic() >= deadline:
                return error_response(
                    error=(
                        f"NovitaAI task {task_id} did not complete within "
                        f"{int(_POLL_TIMEOUT_SECONDS)}s (last status: {last_status or 'unknown'})"
                    ),
                    error_type="timeout",
                    provider="novita",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        # 3. Terminal — extract result.
        if last_status == "TASK_STATUS_FAILED":
            task = job.get("task") if isinstance(job, dict) else {}
            reason = task.get("reason") if isinstance(task, dict) else None
            return error_response(
                error=f"NovitaAI task {task_id} failed: {reason or 'unknown error'}",
                error_type="api_error",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        images = job.get("images") if isinstance(job, dict) else None
        result_image_url: Optional[str] = None
        if isinstance(images, list):
            for candidate in images:
                if isinstance(candidate, dict):
                    url = candidate.get("image_url")
                    if isinstance(url, str) and url.strip():
                        result_image_url = url.strip()
                        break

        if result_image_url is None:
            return error_response(
                error="NovitaAI task completed but returned no image URL",
                error_type="empty_response",
                provider="novita",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Materialise locally — Novita result URLs are TTL-bound (see
        # ``image_url_ttl`` in the task-result payload), mirroring what we
        # do for Krea / xAI / OpenAI URL responses (#26942).
        try:
            saved_path = save_url_image(result_image_url, prefix=f"novita_{model_id}")
        except Exception as exc:
            logger.warning(
                "Novita image URL %s could not be cached (%s); falling back to bare URL.",
                result_image_url,
                exc,
            )
            image_ref = result_image_url
        else:
            image_ref = str(saved_path)

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="novita",
            extra={"size": size, "task_id": task_id},
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``NovitaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(NovitaImageGenProvider())

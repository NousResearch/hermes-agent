"""MiniMax image generation backend.

Wraps the MiniMax ``/v1/image_generation`` endpoint (model ``image-01``)
as an :class:`ImageGenProvider` implementation.

MiniMax returns image URLs (not base64) in ``data.image_urls[0]`` with
a default ``response_format`` of ``url``. To stay compatible with the
provider-ABC ``success_response`` contract (which expects a local path or
final URL we can hand to the user), the plugin downloads the URL bytes
into the Hermes images cache via :func:`save_url_image` — same pattern
xAI and OpenAI fall-back use for ephemeral URLs.

Authentication uses a MiniMax **Subscription Key** (``sk-cp-...``)
stored in the ``MINIMAX_API_KEY`` environment variable. This is the
key type issued for Token Plan / credits calls; the legacy pay-as-you-go
key (``sk-api-...``) is also accepted by MiniMax but requires credits.

Selection precedence (first hit wins):

1. ``MINIMAX_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.minimax.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL` — ``image-01`` (the only model in the public docs)

Docs: https://platform.minimax.io/docs/guides/image-generation
API:  POST https://api.minimax.io/v1/image_generation
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

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


DEFAULT_MODEL = "image-01"
DEFAULT_BASE_URL = "https://api.minimax.io/v1"
IMAGE_ENDPOINT = "/image_generation"

REQUEST_TIMEOUT_SECONDS = 120

VALID_ASPECT_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4", "21:9")

# MiniMax ratio strings → tool-level "landscape" / "square" / "portrait".
# Tool side defaults to "landscape"; we map square and portrait explicitly.
_RATIO_FOR_ASPECT = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def _resolve_model() -> str:
    """Pick the model id per the precedence rules in the module docstring."""
    env = os.environ.get("MINIMAX_IMAGE_MODEL")
    if env and env.strip():
        return env.strip()
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            mm = section.get("minimax")
            if isinstance(mm, dict):
                m = mm.get("model")
                if isinstance(m, str) and m.strip():
                    return m.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read image_gen.minimax.model from config: %s", exc)
    return DEFAULT_MODEL


def _resolve_base_url() -> str:
    """Pick the API base URL — env override, else default."""
    env = os.environ.get("MINIMAX_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    return DEFAULT_BASE_URL


def _resolve_api_key() -> Optional[str]:
    """Read the API key from env, trimming blanks and ignoring empty values."""
    for name in ("MINIMAX_API_KEY", "MINIMAX_SUBSCRIPTION_KEY", "MINIMAX_SUBS_KEY"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _aspect_to_ratio(aspect: str) -> str:
    """Map tool-level aspect_ratio names onto MiniMax's accepted values."""
    if aspect in VALID_ASPECT_RATIOS:
        return aspect
    return _RATIO_FOR_ASPECT.get(aspect, "16:9")


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """Image generation backend that talks to MiniMax's REST API."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        # The MiniMax public image-generation docs only document `image-01`
        # at the moment. If MiniMax adds more, plumb them through here.
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "image-01 (MiniMax)",
                "speed": "~10-15s",
                "strengths": "photorealistic, anime, product shots, character consistency (subject_reference)",
                "price": "included in Token Plan / credits",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "subscription",
            "tag": "image-01 (text-to-image + subject_reference) — billed against your MiniMax Token Plan / credits.",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax Subscription Key (sk-cp-...) for Token Plan / credits calls",
                    "url": "https://platform.minimax.io",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # MiniMax supports text-to-image and image-to-image (subject_reference
        # accepts a single reference image). Reference grounding beyond one
        # image is not part of the documented surface.
        return {
            "modalities": ["text", "image"],
            "max_reference_images": 1,
        }

    # -- generation --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate or edit an image via MiniMax's image_generation endpoint.

        Forward-compat kwargs accepted: ``model``, ``num_images``, ``seed``.
        Unknown keys are ignored per the ABC contract — implementations MUST
        NOT TypeError on extras.
        """
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt must not be empty.",
                error_type="invalid_argument",
                provider=self.name,
                model="",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        ratio = _aspect_to_ratio(aspect)

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "MINIMAX_API_KEY is not set. Add your MiniMax Subscription "
                    "Key (sk-cp-...) to $HERMES_HOME/.env or configure it via "
                    "Hermes tools setup."
                ),
                error_type="missing_credentials",
                provider=self.name,
                model="",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        model = kwargs.get("model") or _resolve_model()
        base_url = _resolve_base_url()
        url = f"{base_url}{IMAGE_ENDPOINT}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": ratio,
            "response_format": "url",
        }

        # image-to-image: MiniMax uses `subject_reference` with a single
        # reference image. We accept either `image_url` or the first entry
        # of `reference_image_urls` (the ABC declares 1 max).
        ref_url = image_url
        if ref_url is None:
            refs = normalize_reference_images(reference_image_urls)
            if refs:
                ref_url = refs[0]

        if ref_url:
            payload["subject_reference"] = [
                {"type": "character", "image_file": ref_url}
            ]

        # Forward optional numeric/seed knobs without breaking on unknown keys.
        for key in ("num_images", "seed"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        logger.info(
            "MiniMax image_generate: model=%s aspect=%s ratio=%s has_ref=%s",
            model,
            aspect,
            ratio,
            bool(ref_url),
        )

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            logger.warning("MiniMax request failed: %s", exc, exc_info=True)
            return error_response(
                error=f"MiniMax request failed: {exc}",
                error_type="network_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if response.status_code >= 400:
            try:
                body = response.text[:300]
            except Exception:  # noqa: BLE001
                body = "<unavailable>"
            return error_response(
                error=f"MiniMax HTTP error {response.status_code}: {body}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # MiniMax returns HTTP 200 even for business errors — check
        # base_resp.status_code for the real verdict.
        try:
            data = response.json()
        except ValueError:
            return error_response(
                error=(
                    f"MiniMax returned non-JSON (HTTP {response.status_code}): "
                    f"{response.text[:200]!r}"
                ),
                error_type="invalid_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        status_msg = base_resp.get("status_msg", "")

        if status_code != 0:
            logger.warning(
                "MiniMax business error: status_code=%s status_msg=%s",
                status_code,
                status_msg,
            )
            return error_response(
                error=f"MiniMax error {status_code}: {status_msg or 'unknown'}",
                error_type="provider_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_urls = ((data.get("data") or {}).get("image_urls")) or []
        if not image_urls:
            return error_response(
                error="MiniMax returned success but no image_urls.",
                error_type="empty_result",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        raw_url = image_urls[0]
        modality = "image" if ref_url else "text"

        # MiniMax returns a signed Aliyun OSS URL with a TTL. The TTL is
        # generous enough (~12 hours observed) that we could ship the URL
        # as-is, but downloading + caching here means the path on disk
        # works even after the signed URL expires, and downstream tools
        # (Telegram send_photo, browser fetch) get a stable local file.
        try:
            cached_path = save_url_image(raw_url, prefix="minimax")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to cache MiniMax image locally (%s); returning raw URL",
                exc,
            )
            return success_response(
                image=raw_url,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
                provider=self.name,
                modality=modality,
                extra={"remote_url": raw_url, "cache_status": "skipped"},
            )

        return success_response(
            image=str(cached_path),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality=modality,
            extra={
                "cache_status": "cached",
                "generation_id": data.get("id"),
            },
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())

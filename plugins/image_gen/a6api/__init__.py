"""a6api image generation backend.

OpenAI-compatible relay that exposes ``gpt-image-2`` as an
:class:`ImageGenProvider`. It talks to the user's a6api endpoint
(``https://api.a6api.com/v1``) using the ``A6API_API_KEY`` secret, so no
OpenAI account or extra key is required.

Model IDs (mirrors the openai provider's three quality tiers):

    gpt-image-2-low     ~fast      iteration
    gpt-image-2-medium  ~balanced  default
    gpt-image-2-high    ~high      highest fidelity

All three hit the same underlying ``gpt-image-2`` API model with a
different ``quality`` parameter. Output base64 JSON is saved under
``$HERMES_HOME/cache/images/``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


API_MODEL = "gpt-image-2"

# Default relay endpoint. A user can override via a6api's base_url in
# config.yaml custom_providers (same value used by the main model).
DEFAULT_BASE_URL = "https://api.a6api.com/v1"
# Secret env var that carries the a6api key.
KEY_VAR = "A6API_API_KEY"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    """Read ``image_gen`` and the a6api custom-provider entry from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        if not isinstance(cfg, dict):
            return {}
        image_section = cfg.get("image_gen") if isinstance(cfg.get("image_gen"), dict) else {}
        base = DEFAULT_BASE_URL
        key = None
        for p in (cfg.get("custom_providers") or []):
            if isinstance(p, dict) and (p.get("name") == "a6api" or "a6api.com" in str(p.get("base_url", ""))):
                base = str(p.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
                if p.get("api_key"):
                    key = str(p.get("api_key"))
        return {
            "image_gen": image_section,
            "base_url": base,
            "config_key": key,
        }
    except Exception as exc:
        logger.debug("Could not load image_gen/a6api config: %s", exc)
        return {}


def _api_key() -> Optional[str]:
    """Resolve the a6api key: env/.env secret first, then config custom_providers."""
    from agent.secret_scope import UnscopedSecretError

    try:
        secret = get_secret(KEY_VAR)
        if secret:
            return secret
    except UnscopedSecretError:
        pass
    except Exception:
        pass
    return (_load_config().get("config_key") or None)


def _base_url() -> str:
    return str(_load_config().get("base_url") or DEFAULT_BASE_URL).rstrip("/")


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    env_override = os.environ.get("A6API_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_config().get("image_gen") or {}
    a6api_cfg = cfg.get("a6api") if isinstance(cfg.get("a6api"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(a6api_cfg, dict):
        value = a6api_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load image bytes from a URL, data URI, or local file path."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        import requests

        resp = requests.get(ref, timeout=60)
        resp.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return resp.content, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = "png"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "png"
        return base64.b64decode(b64), f"image.{ext}"
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    return data, os.path.basename(ref) or "image.png"


class A6ApiImageGenProvider(ImageGenProvider):
    """a6api relay ``images.generate`` / ``images.edit`` backend — gpt-image-2."""

    @property
    def name(self) -> str:
        return "a6api"

    @property
    def display_name(self) -> str:
        return "a6api (gpt-image-2)"

    def is_available(self) -> bool:
        if not _api_key():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "a6api relay",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "a6api",
            "badge": "relay",
            "tag": "gpt-image-2 through your a6api relay endpoint — text-to-image & image editing",
            "env_vars": [
                {
                    "key": KEY_VAR,
                    "prompt": "a6api API key",
                    "url": "https://api.a6api.com",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 16}

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

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="a6api",
                aspect_ratio=aspect,
            )

        api_key = _api_key()
        if not api_key:
            return error_response(
                error=(
                    f"{KEY_VAR} not set. Run `hermes tools` → Image "
                    "Generation → a6api to configure."
                ),
                error_type="auth_required",
                provider="a6api",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="a6api",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)
        sources = sources[:16]
        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        client = openai.OpenAI(api_key=api_key, base_url=_base_url())

        if is_edit:
            import io

            try:
                files = []
                for ref in sources:
                    data, fname = _load_image_bytes(ref)
                    bio = io.BytesIO(data)
                    bio.name = fname
                    files.append(bio)
            except Exception as exc:
                return error_response(
                    error=f"Could not load source image for editing: {exc}",
                    error_type="io_error",
                    provider="a6api",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            try:
                response = client.images.edit(
                    model=API_MODEL,
                    image=files if len(files) > 1 else files[0],
                    prompt=prompt,
                    size=size,  # type: ignore[arg-type]
                    quality=meta["quality"],
                    n=1,
                )
            except Exception as exc:
                logger.debug("a6api image edit failed", exc_info=True)
                return error_response(
                    error=f"a6api image editing failed: {exc}",
                    error_type="api_error",
                    provider="a6api",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            payload: Dict[str, Any] = {
                "model": API_MODEL,
                "prompt": prompt,
                "size": size,
                "n": 1,
                "quality": meta["quality"],
            }

            try:
                response = client.images.generate(**payload)
            except Exception as exc:
                logger.debug("a6api image generation failed", exc_info=True)
                return error_response(
                    error=f"a6api image generation failed: {exc}",
                    error_type="api_error",
                    provider="a6api",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="a6api returned no image data",
                error_type="empty_response",
                provider="a6api",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"a6api_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="a6api",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            try:
                saved_path = save_url_image(url, prefix=f"a6api_{tier_id}")
            except Exception as exc:
                logger.warning("a6api image URL %s could not be cached (%s); falling back to bare URL.", url, exc)
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="a6api response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="a6api",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size, "quality": meta["quality"]}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="a6api",
            modality=modality,
            extra=extra,
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``A6ApiImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(A6ApiImageGenProvider())

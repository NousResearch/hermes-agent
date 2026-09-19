"""Azure Foundry image generation backend.

Exposes GPT Image deployments hosted on Azure AI Foundry at three quality
tiers as an :class:`ImageGenProvider` implementation, mirroring the bundled
OpenAI backend. Azure Foundry exposes OpenAI-compatible image APIs
(``images.generate`` / ``images.edit``); the user supplies their own
resource endpoint and deployment name since those are per-resource.

The tiers are implemented as three virtual model IDs so the ``hermes tools``
model picker and the ``image_gen.model`` config key behave like any other
multi-model backend:

    azure-gpt-image-low     ~15s   fastest, good for iteration
    azure-gpt-image-medium  ~40s   default — balanced
    azure-gpt-image-high    ~2min  slowest, highest fidelity

All three hit the same underlying Azure deployment (``image_gen.azure_foundry.
deployment``, default ``gpt-image-2``) with a different ``quality`` parameter.
Output is base64 JSON → saved under ``$HERMES_HOME/cache/images/``.

Authentication supports both contracts Azure Foundry offers:

* **API key** — ``AZURE_FOUNDRY_IMAGE_API_KEY`` (dedicated secret; the LLM
  provider's ``AZURE_FOUNDRY_API_KEY`` is intentionally NOT reused so the
  two surfaces stay independently scoped).
* **Microsoft Entra ID** — ``auth_mode: entra_id`` under
  ``image_gen.azure_foundry``. Uses :func:`agent.azure_identity_adapter.
  build_token_provider` (``DefaultAzureCredential`` chain) exactly like the
  LLM provider; the OpenAI SDK calls the returned callable before every
  request, so token refresh is transparent.

Selection precedence (first hit wins):

1. ``AZURE_FOUNDRY_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.azure_foundry.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``azure-gpt-image-medium``

Example configuration:

.. code-block:: yaml

    image_gen:
      provider: azure-foundry
      model: azure-gpt-image-medium
      azure_foundry:
        endpoint: https://YOUR-RESOURCE.openai.azure.com
        deployment: gpt-image-2
        auth_mode: api_key        # or: entra_id
        entra:
          scope: https://ai.azure.com/.default
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.azure_identity_adapter import (
    SCOPE_AI_AZURE_DEFAULT,
    EntraIdentityConfig,
    build_token_provider,
    has_azure_identity_installed,
)
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
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# All three IDs resolve to the same underlying Azure deployment with a
# different ``quality`` setting. ``api_model`` (the deployment name) comes
# from config; ``quality`` is the knob that changes generation time and
# output fidelity.

DEFAULT_DEPLOYMENT = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "azure-gpt-image-low": {
        "display": "GPT Image (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "azure-gpt-image-medium": {
        "display": "GPT Image (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "azure-gpt-image-high": {
        "display": "GPT Image (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "azure-gpt-image-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_azure_config() -> Dict[str, Any]:
    """Read ``image_gen.azure_foundry`` from config.yaml ({} on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            return {}
        azure_cfg = section.get("azure_foundry")
        return azure_cfg if isinstance(azure_cfg, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.azure_foundry config: %s", exc)
        return {}


def _build_base_url(endpoint: str) -> str:
    """Normalize a user-supplied Azure resource endpoint into the OpenAI SDK
    base URL for the image APIs.

    Accepts either the resource root (``https://RESOURCE.openai.azure.com``)
    or an already-suffixed form (``.../openai`` or ``.../openai/v1``).
    """
    url = str(endpoint or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/openai/v1"):
        return url
    if url.endswith("/openai"):
        return f"{url}/v1"
    return f"{url}/openai/v1"


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("AZURE_FOUNDRY_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_azure_config()
    candidate: Optional[str] = None
    value = cfg.get("model")
    if isinstance(value, str) and value in _MODELS:
        candidate = value
    if candidate is None:
        try:
            from hermes_cli.config import load_config

            top = load_config().get("image_gen")
            if isinstance(top, dict):
                value = top.get("model")
                if isinstance(value, str) and value in _MODELS:
                    candidate = value
        except Exception:
            pass

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_deployment() -> str:
    """Return the Azure deployment name to send as the API ``model``."""
    cfg = _load_azure_config()
    deployment = str(cfg.get("deployment") or "").strip()
    return deployment or DEFAULT_DEPLOYMENT


def _resolve_auth_mode() -> str:
    """Return ``entra_id`` or ``api_key`` (default)."""
    cfg = _load_azure_config()
    return str(cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"


def _load_azure_api_key() -> str:
    """Return the dedicated Azure Foundry image API key, if set."""
    try:
        return str(get_secret("AZURE_FOUNDRY_IMAGE_API_KEY", "") or "").strip()
    except Exception:
        return str(os.getenv("AZURE_FOUNDRY_IMAGE_API_KEY", "") or "").strip()


# ---------------------------------------------------------------------------
# Source-image loading (for image-to-image / edit)
# ---------------------------------------------------------------------------


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load image bytes from a URL or local file path.

    Returns ``(data, filename)``. Raises on any network / IO error so the
    caller can surface a clean error_response.
    """
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
    # Local file path — enforce the shared credential-read guard before reading.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    name = os.path.basename(ref) or "image.png"
    return data, name


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AzureFoundryImageGenProvider(ImageGenProvider):
    """Azure Foundry ``images.generate`` / ``images.edit`` backend (GPT Image
    deployments via Azure's OpenAI-compatible image APIs)."""

    @property
    def name(self) -> str:
        return "azure-foundry"

    @property
    def display_name(self) -> str:
        return "Azure Foundry"

    def is_available(self) -> bool:
        if _load_azure_api_key():
            return True
        # Entra ID mode: available when configured and azure-identity present.
        # No token is minted here — keeps CLI/picker startup latency flat.
        if _resolve_auth_mode() == "entra_id":
            return has_azure_identity_installed()
        return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Azure Foundry",
            "badge": "paid",
            "tag": "GPT Image deployments at low/medium/high quality tiers — text-to-image & image editing (API key or Entra ID)",
            "env_vars": [
                {
                    "key": "AZURE_FOUNDRY_IMAGE_API_KEY",
                    "prompt": "Azure Foundry image API key (skip when using Entra ID auth)",
                    "url": "https://ai.azure.com/",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # GPT Image deployments support editing via images.edit() with up to
        # 16 source images — same surface as the OpenAI backend.
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
                provider="azure-foundry",
                aspect_ratio=aspect,
            )

        cfg = _load_azure_config()
        endpoint = str(cfg.get("endpoint") or "").strip()
        if not endpoint:
            return error_response(
                error=(
                    "image_gen.azure_foundry.endpoint is not set. Configure "
                    "your Azure resource endpoint in config.yaml (e.g. "
                    "https://YOUR-RESOURCE.openai.azure.com)."
                ),
                error_type="auth_required",
                provider="azure-foundry",
                aspect_ratio=aspect,
            )
        base_url = _build_base_url(endpoint)
        deployment = _resolve_deployment()

        auth_mode = _resolve_auth_mode()
        api_key: Any = _load_azure_api_key()
        if auth_mode == "entra_id" and not api_key:
            try:
                entra_cfg = cfg.get("entra")
                scope = ""
                if isinstance(entra_cfg, dict):
                    scope = str(entra_cfg.get("scope") or "").strip()
                identity_config = EntraIdentityConfig(
                    scope=scope or SCOPE_AI_AZURE_DEFAULT,
                )
                api_key = build_token_provider(config=identity_config)
            except ImportError as exc:
                return error_response(
                    error=(
                        "Azure Foundry Entra ID auth requires the "
                        "'azure-identity' package. Install it with: "
                        f"pip install azure-identity (import failed: {exc})"
                    ),
                    error_type="missing_dependency",
                    provider="azure-foundry",
                    aspect_ratio=aspect,
                )

        if not api_key:
            return error_response(
                error=(
                    "AZURE_FOUNDRY_IMAGE_API_KEY not set and Entra ID auth is "
                    "not configured. Run `hermes tools` → Image Generation → "
                    "Azure Foundry to configure, or set "
                    "image_gen.azure_foundry.auth_mode: entra_id in config.yaml."
                ),
                error_type="auth_required",
                provider="azure-foundry",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="azure-foundry",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        # Collect source images (primary + references) for image-to-image.
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)
        sources = sources[:16]  # GPT Image edit caps at 16 images
        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        if is_edit:
            # images.edit() expects file-like objects. Download/read each
            # source into a named BytesIO so the SDK sends correct multipart.
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
                    provider="azure-foundry",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            try:
                response = client.images.edit(
                    model=deployment,
                    image=files if len(files) > 1 else files[0],
                    prompt=prompt,
                    size=size,  # type: ignore[arg-type]  # _SIZES values are valid GPT Image sizes
                    quality=meta["quality"],
                    n=1,
                )
            except Exception as exc:
                logger.debug("Azure Foundry image edit failed", exc_info=True)
                return error_response(
                    error=f"Azure Foundry image editing failed: {exc}",
                    error_type="api_error",
                    provider="azure-foundry",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            # GPT Image returns b64_json unconditionally and REJECTS
            # ``response_format`` as an unknown parameter. Don't send it.
            payload: Dict[str, Any] = {
                "model": deployment,
                "prompt": prompt,
                "size": size,
                "n": 1,
                "quality": meta["quality"],
            }

            try:
                response = client.images.generate(**payload)
            except Exception as exc:
                logger.debug("Azure Foundry image generation failed", exc_info=True)
                return error_response(
                    error=f"Azure Foundry image generation failed: {exc}",
                    error_type="api_error",
                    provider="azure-foundry",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="Azure Foundry returned no image data",
                error_type="empty_response",
                provider="azure-foundry",
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
                saved_path = save_b64_image(b64, prefix=f"azure_foundry_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="azure-foundry",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — GPT Image returns b64 today, but Azure's
            # OpenAI-compatible API may return URLs. Cache the bytes locally
            # so the gateway never tries to fetch an ephemeral / signed URL
            # after it expires — same rationale as the xAI provider (#26942).
            try:
                saved_path = save_url_image(url, prefix=f"azure_foundry_{tier_id}")
            except Exception as exc:
                logger.warning(
                    "Azure Foundry image URL %s could not be cached (%s); falling back to bare URL.",
                    url,
                    exc,
                )
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="Azure Foundry response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="azure-foundry",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "size": size,
            "quality": meta["quality"],
            "deployment": deployment,
        }
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="azure-foundry",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``AzureFoundryImageGenProvider`` into the
    registry."""
    ctx.register_image_gen_provider(AzureFoundryImageGenProvider())

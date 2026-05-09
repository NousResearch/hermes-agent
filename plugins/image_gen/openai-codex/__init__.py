"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Identical model catalog and tier semantics to the ``openai`` image-gen plugin
(``gpt-image-2`` at low/medium/high quality), but routes the request through
the Codex Responses API ``image_generation`` tool instead of the
``images.generate`` REST endpoint. This lets users who are already
authenticated with Codex/ChatGPT generate images without configuring a
separate ``OPENAI_API_KEY``.

Selection precedence for the tier (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai-codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``

Output is saved as PNG under ``$HERMES_HOME/cache/images/``.
"""

from __future__ import annotations

import logging
import base64
import binascii
import ipaddress
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog — mirrors the ``openai`` plugin so the picker UX is identical.
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"

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

# Codex Responses surface used for the request. The chat model itself is only
# the host that calls the ``image_generation`` tool; the actual image work is
# done by ``API_MODEL``.
_CODEX_CHAT_MODEL = "gpt-5.4"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)
_MAX_REFERENCE_IMAGES = 16
_MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024
_ALLOWED_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}
_DATA_IMAGE_RE = re.compile(r"^data:(image/(?:png|jpeg|jpg|webp|gif));base64,", re.I)


# ---------------------------------------------------------------------------
# Config + auth helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    import os

    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _read_codex_access_token() -> Optional[str]:
    """Return a usable Codex OAuth token, or None.

    Delegates to the canonical reader in ``agent.auxiliary_client`` so token
    expiry, credential pool selection, and JWT decoding stay in one place.
    """
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _build_codex_client():
    """Return an OpenAI client pointed at the ChatGPT/Codex backend, or None."""
    token = _read_codex_access_token()
    if not token:
        return None
    try:
        import openai
        from agent.auxiliary_client import _codex_cloudflare_headers

        return openai.OpenAI(
            api_key=token,
            base_url=_CODEX_BASE_URL,
            default_headers=_codex_cloudflare_headers(token),
        )
    except Exception as exc:
        logger.debug("Could not build Codex image client: %s", exc)
        return None


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_reference_roots() -> List[Path]:
    """Local roots that may be read as image references.

    The image tool is exposed to gateway sessions, so local references must
    not become an arbitrary file-read primitive. Allow generated/uploaded
    image locations and Xiaoyaner/Xiaomiao photo libraries, not the whole
    Hermes home where config.yaml, auth.json, logs, and state.db live.
    """
    try:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home().expanduser().resolve()
    except Exception:
        hermes_home = Path.home() / ".hermes"

    roots = [
        hermes_home / "cache",
        hermes_home / "xiaoyaner" / "photos",
        hermes_home / "xiaomiao" / "photos",
    ]
    return [root.expanduser().resolve() for root in roots]


def _assert_allowed_local_reference(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not any(_path_is_relative_to(resolved, root) for root in _allowed_reference_roots()):
        raise ValueError(
            "Local reference images must live under Hermes cache/upload "
            "directories or Xiaoyaner/Xiaomiao photo libraries; refusing "
            "to read arbitrary local path"
        )


def _detect_image_mime(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate_data_image_url(ref: str) -> str:
    match = _DATA_IMAGE_RE.match(ref)
    if not match:
        raise ValueError("data reference_images must be base64 PNG/JPEG/WEBP/GIF data:image URLs")
    encoded = ref[match.end():]
    if len(encoded) > _MAX_REFERENCE_IMAGE_BYTES * 4 // 3 + 4096:
        raise ValueError("Reference image data URL is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("Reference image data URL is not valid base64") from exc
    if len(raw) > _MAX_REFERENCE_IMAGE_BYTES:
        raise ValueError("Reference image data URL is too large")
    detected = _detect_image_mime(raw[:64])
    declared = match.group(1).lower().replace("image/jpg", "image/jpeg")
    if detected not in _ALLOWED_IMAGE_MIMES or detected != declared:
        raise ValueError("Reference image data URL content does not match its declared image type")
    return ref


def _validate_http_image_url(ref: str) -> str:
    parsed = urlparse(ref)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("reference_images URLs must be valid http(s) URLs")
    if host.lower() in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost reference image URLs are not allowed")
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return ref
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise ValueError("Private/internal reference image URLs are not allowed")
    return ref


def _image_ref_to_input_item(reference: str) -> Dict[str, str]:
    """Convert a local path or URL into a Responses ``input_image`` item."""
    if not isinstance(reference, str):
        raise ValueError("reference_images entries must be strings")
    ref = str(reference or "").strip()
    if not ref:
        raise ValueError("reference_images entries must be non-empty strings")

    if ref.startswith("data:image/"):
        image_url = _validate_data_image_url(ref)
    elif ref.startswith(("http://", "https://")):
        image_url = _validate_http_image_url(ref)
    else:
        path = Path(ref).expanduser()
        if not path.exists() or not path.is_file():
            raise ValueError(f"Reference image not found: {ref}")
        _assert_allowed_local_reference(path)
        if path.stat().st_size > _MAX_REFERENCE_IMAGE_BYTES:
            raise ValueError("Reference image file is too large")
        raw = path.read_bytes()
        detected = _detect_image_mime(raw[:64])
        guessed = mimetypes.guess_type(path.name)[0]
        if detected not in _ALLOWED_IMAGE_MIMES:
            raise ValueError("Reference file is not a supported image (PNG/JPEG/WEBP/GIF)")
        if guessed in _ALLOWED_IMAGE_MIMES and guessed != detected:
            raise ValueError("Reference image extension does not match its content")
        encoded = base64.b64encode(raw).decode("ascii")
        image_url = f"data:{detected};base64,{encoded}"

    return {"type": "input_image", "image_url": image_url}


def _build_input_content(prompt: str, reference_images: Optional[Sequence[str]]) -> List[Dict[str, str]]:
    content: List[Dict[str, str]] = [{"type": "input_text", "text": prompt}]
    if isinstance(reference_images, str):
        raise ValueError("reference_images must be a list of image references, not a string")
    refs = list(reference_images or [])
    if len(refs) > _MAX_REFERENCE_IMAGES:
        raise ValueError(f"gpt-image reference-image workflows support at most {_MAX_REFERENCE_IMAGES} images")
    content.extend(_image_ref_to_input_item(ref) for ref in refs)
    return content


def _collect_image_b64(
    client: Any,
    *,
    prompt: str,
    size: str,
    quality: str,
    reference_images: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    image_b64: Optional[str] = None

    input_content = _build_input_content(prompt, reference_images)
    tool: Dict[str, Any] = {
        "type": "image_generation",
        "model": API_MODEL,
        "size": size,
        "quality": quality,
        "output_format": "png",
        "background": "opaque",
        "partial_images": 1,
    }
    if reference_images:
        tool["action"] = "edit"

    with client.responses.stream(
        model=_CODEX_CHAT_MODEL,
        store=False,
        instructions=_CODEX_INSTRUCTIONS,
        input=[{
            "type": "message",
            "role": "user",
            "content": input_content,
        }],
        tools=[tool],
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
    ) as stream:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "image_generation_call":
                    result = getattr(item, "result", None)
                    if isinstance(result, str) and result:
                        image_b64 = result
            elif event_type == "response.image_generation_call.partial_image":
                partial = getattr(event, "partial_image_b64", None)
                if isinstance(partial, str) and partial:
                    image_b64 = partial
        final = stream.get_final_response()

    # Final-response sweep covers the case where the stream finished before
    # we observed the ``output_item.done`` event for the image call.
    for item in getattr(final, "output", None) or []:
        if getattr(item, "type", None) == "image_generation_call":
            result = getattr(item, "result", None)
            if isinstance(result, str) and result:
                image_b64 = result

    return image_b64


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICodexImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI (Codex auth)"

    def is_available(self) -> bool:
        if not _read_codex_access_token():
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
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."
            ),
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        if not _read_codex_access_token():
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        try:
            import openai  # noqa: F401
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])
        reference_images = kwargs.get("reference_images")

        client = _build_codex_client()
        if client is None:
            return error_response(
                error="Could not initialize Codex image client",
                error_type="auth_required",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            b64 = _collect_image_b64(
                client,
                prompt=prompt,
                size=size,
                quality=meta["quality"],
                reference_images=reference_images,
            )
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation via Codex auth failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex response contained no image_generation_call result",
                error_type="empty_response",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"openai_codex_{tier_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-codex",
            extra={"size": size, "quality": meta["quality"]},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())

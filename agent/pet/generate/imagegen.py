"""Thin image-generation layer for pet sprites.

Wraps the active :class:`~agent.image_gen_provider.ImageGenProvider` with the
two things sprite generation needs that the agent-facing ``image_generate`` tool
doesn't expose: **N variants** (loop) and **reference-image grounding** (so each
animation row stays the same character as the chosen base).

Reference grounding only works on providers that support it — currently OpenAI
``gpt-image-2`` (image edits) and Krea (style references). We resolve to one of
those and surface a clear, actionable error otherwise rather than silently
producing an ungrounded, drifting pet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Providers that can ground generation on a reference image.
# openrouter / nous reach Gemini Flash Image (and friends) over the
# OpenRouter-compatible chat-completions image protocol, which accepts
# reference images for grounding. Nous Portal proxies OpenRouter, so both
# qualify.
_REF_CAPABLE = ("openai", "openai-codex", "krea", "openrouter", "nous")


class GenerationError(RuntimeError):
    """Raised on any image-generation failure (no provider, API error, IO)."""


@dataclass(frozen=True)
class SpriteProvider:
    """Resolved provider plus whether it can take reference images."""

    name: str
    provider: object
    supports_references: bool


def _discover() -> None:
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        logger.debug("image-gen plugin discovery failed: %s", exc)


def resolve_provider(*, require_references: bool = True) -> SpriteProvider:
    """Pick the image provider to use for sprite work.

    Preference: the configured provider when it's reference-capable, else the
    first available reference-capable provider. With *require_references* off we
    fall back to any available provider (used for prompt-only base drafts).
    """
    _discover()
    from agent.image_gen_registry import get_active_provider, get_provider

    # Configured / active provider first.
    active = None
    try:
        active = get_active_provider()
    except Exception:  # noqa: BLE001
        active = None
    if active is not None:
        name = getattr(active, "name", "")
        if name in _REF_CAPABLE and active.is_available():
            return SpriteProvider(name=name, provider=active, supports_references=True)

    # Any available reference-capable provider.
    for name in _REF_CAPABLE:
        provider = get_provider(name)
        if provider is not None and provider.is_available():
            return SpriteProvider(name=name, provider=provider, supports_references=True)

    if not require_references and active is not None and active.is_available():
        return SpriteProvider(
            name=getattr(active, "name", "unknown"), provider=active, supports_references=False
        )

    raise GenerationError(
        "Pet generation needs an image backend that supports reference images. "
        "Open `hermes tools` → Image Generation and configure OpenRouter, Nous "
        "Portal, or OpenAI (gpt-image-2) with an API key."
    )


def _save_local(image_ref: str, *, prefix: str) -> Path:
    """Return a local path for *image_ref*, downloading it if it's a URL."""
    if image_ref.startswith(("http://", "https://")):
        from agent.image_gen_provider import save_url_image

        return Path(save_url_image(image_ref, prefix=prefix))
    return Path(image_ref)


def _rejected_background(error: str) -> bool:
    """True when a provider error is specifically about the ``background`` param.

    Transparent backgrounds are a per-model capability (e.g. some gpt-image tiers
    reject ``background=transparent`` outright). We detect that one rejection so
    we can retry without the flag rather than failing the whole pet — our chroma
    key pass makes the result transparent regardless.
    """
    lowered = (error or "").lower()
    return "background" in lowered and ("not supported" in lowered or "transparent" in lowered)


def generate(
    prompt: str,
    *,
    n: int = 1,
    reference_images: list[Path] | None = None,
    provider: SpriteProvider | None = None,
    prefix: str = "pet_gen",
) -> list[Path]:
    """Generate *n* square sprite images and return their local paths.

    *reference_images* grounds the output on a base image (required for rows).
    We *ask* for a transparent background, but fall back to an opaque generation
    (cleaned up downstream by the chroma-key pass) on models that reject the
    flag. Raises :class:`GenerationError` if nothing usable comes back.
    """
    sprite = provider or resolve_provider(require_references=bool(reference_images))
    if reference_images and not sprite.supports_references:
        raise GenerationError(
            f"image backend '{sprite.name}' cannot use reference images; "
            "configure OpenAI gpt-image-2 or Krea for pet generation"
        )

    refs = [str(p) for p in (reference_images or [])]

    def _run(extra: dict) -> tuple[Path | None, str]:
        kwargs: dict = {"aspect_ratio": "square", **extra}
        if refs:
            # Providers disagree on the ref kwarg name: our OpenRouter/Nous
            # backends read ``reference_images``, OpenAI's gpt-image-2 reads
            # ``reference_image_urls``. Send both; each ignores the other.
            kwargs["reference_images"] = refs
            kwargs["reference_image_urls"] = refs
        try:
            result = sprite.provider.generate(prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize provider crashes
            logger.debug("provider.generate crashed: %s", exc)
            return None, str(exc)
        if not isinstance(result, dict) or not result.get("success"):
            return None, (result or {}).get("error", "unknown error") if isinstance(result, dict) else "no result"
        image_ref = result.get("image")
        if not image_ref:
            return None, "provider returned no image"
        try:
            return _save_local(str(image_ref), prefix=prefix), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"could not save generated image: {exc}"

    out: list[Path] = []
    last_error = ""
    allow_transparent = True
    for _ in range(max(1, n)):
        path, err = _run({"background": "transparent"} if allow_transparent else {})
        # Model doesn't support the transparent flag → drop it for this and every
        # remaining variant (no point re-probing a capability we just disproved).
        if path is None and allow_transparent and _rejected_background(err):
            allow_transparent = False
            path, err = _run({})
        if path is not None:
            out.append(path)
        else:
            last_error = err

    if not out:
        raise GenerationError(last_error or "image generation produced no output")
    return out

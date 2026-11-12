"""Thin image-generation layer for pet sprites.

Wraps the active ``ImageGenProvider`` with what the ``image_generate`` tool
doesn't expose: N variants and reference-image grounding (each animation row
stays the same character as the chosen base). Grounding needs a ref-capable
provider; we resolve to one or raise an actionable error rather than drift.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from hermes_agent.agent.image_gen_provider import ImageGenProvider
from hermes_agent.tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER

logger = logging.getLogger(__name__)

# Providers that can ground generation on a reference image, in preference order.
# OpenRouter/Nous run a quality-first model chain and may fall back depending on
# account access, so fidelity can vary by configured backend.
_REF_CAPABLE = ("nous", "openai", "openai-codex", "openrouter", "krea")
# Friendly display label per reference-capable provider (desktop pet-gen picker).
_PROVIDER_LABELS = {"nous": "Nous Portal", "openrouter": "OpenRouter", "openai": "OpenAI", "openai-codex": "OpenAI (Codex)", "krea": "Krea"}


class GenerationError(RuntimeError):
    """Raised on any image-generation failure (no provider, API error, IO)."""


@dataclass(frozen=True)
class SpriteProvider:
    """Logical provider choice, the backend serving it, and a model only a managed redirect pins."""

    name: str
    provider: ImageGenProvider
    supports_references: bool
    model: str | None = None


def _discover() -> None:
    try:
        from hermes_agent.hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:
        logger.debug("image-gen plugin discovery failed: %s", exc)


def _available(name: str):
    """The registered provider *name* if it exists and has credentials, else ``None``."""
    from hermes_agent.agent.image_gen_registry import get_provider

    provider = get_provider(name)
    return provider if provider is not None and provider.is_available() else None


def _managed_krea_model() -> str | None:
    """The stored native Krea id when the stored selection routes to the Krea gateway (a managed
    ``nous`` pick, or no provider set), else ``None``."""
    from hermes_agent.plugins.image_gen._common import load_image_gen_config
    from hermes_agent.tools.image_generation_managed import KREA, managed_route

    cfg = load_image_gen_config()
    model = cfg.get("model")
    if not isinstance(model, str) or managed_route(cfg.get("provider"), model) != KREA:
        return None
    return model.strip()


def _candidate(name: str) -> SpriteProvider | None:
    """The sprite backend for the logical choice *name*, or ``None`` when it cannot serve.

    Nous Subscription is one choice spanning several gateways: a native Krea id under it goes to the
    Krea backend (the Portal 404s on it), pinned so ``KREA_IMAGE_MODEL`` cannot swap the model, as the
    image tool's managed route does; with Krea unavailable, Nous cannot serve. Every other choice keeps
    its own model resolution (env override, ``image_gen.<provider>.model``): the top-level
    ``image_gen.model`` may be another backend's id.
    """
    if (provider := _available(name)) is None:
        return None
    if name == NOUS_MANAGED_PROVIDER and (model := _managed_krea_model()):
        krea = _available("krea")
        return None if krea is None else SpriteProvider(name=name, provider=krea, supports_references=True, model=model)
    return SpriteProvider(name=name, provider=provider, supports_references=True)


def resolve_provider(*, require_references: bool = True, prefer: str | None = None) -> SpriteProvider:
    """Pick the image provider for sprite work.

    Preference: ``HERMES_PET_IMAGE_PROVIDER`` (QA override, unknown values ignored),
    then *prefer* (desktop picker), then the active provider, then the first available
    — each only if ref-capable and configured (Nous with a native Krea model only while the
    Krea backend is available). With *require_references* off, any available active
    provider is accepted (prompt-only base drafts).
    """
    _discover()
    from hermes_agent.agent.image_gen_registry import get_active_provider

    forced = os.environ.get("HERMES_PET_IMAGE_PROVIDER", "").strip().lower()
    for name in (forced, prefer):
        if name in _REF_CAPABLE and (sprite := _candidate(name)) is not None:
            return sprite
    try:
        active = get_active_provider()
    except Exception:
        active = None
    active_name = getattr(active, "name", "") if active is not None else ""
    if active_name in _REF_CAPABLE and (sprite := _candidate(active_name)) is not None:
        return sprite
    for name in _REF_CAPABLE:
        if (sprite := _candidate(name)) is not None:
            return sprite
    # The scan above found Nous unable to serve; when it is configured, that means its Krea model
    # lacks the Krea backend. Raised before the prompt-only fallback, which would send the Krea id
    # to another backend (FAL swaps in its default model; the Portal 404s on it).
    if _available(NOUS_MANAGED_PROVIDER) is not None:
        raise GenerationError(
            "The selected Nous image model needs the Krea gateway, but that backend is "
            "unavailable. Check Nous Subscription under `hermes tools` → Image Generation."
        )
    if not require_references and active is not None and active.is_available():
        return SpriteProvider(name=getattr(active, "name", "unknown"), provider=active, supports_references=False)
    raise GenerationError(
        "Pet generation needs an image backend that supports reference images. "
        "Open `hermes tools` → Image Generation and configure Nous Portal, "
        "OpenRouter, or OpenAI (gpt-image-2) with an API key."
    )


def list_sprite_providers() -> list[dict]:
    """``[{name, label, default}]`` per configured ref-capable provider, in preference order; empty hides the picker."""
    _discover()
    try:
        default_name = resolve_provider(require_references=True).name
    except GenerationError:
        default_name = ""
    return [
        {"name": name, "label": _PROVIDER_LABELS.get(name, name), "default": name == default_name}
        for name in _REF_CAPABLE
        if _candidate(name) is not None
    ]


def _save_local(image_ref: str, *, prefix: str) -> Path:
    """Return a local path for *image_ref*, downloading it if it's a URL."""
    if image_ref.startswith(("http://", "https://")):
        from hermes_agent.agent.image_gen_provider import save_url_image

        return Path(save_url_image(image_ref, prefix=prefix))
    return Path(image_ref)


def _rejected_background(error: str) -> bool:
    """True when a provider error is specifically about ``background=transparent`` (a per-model capability; we retry without it)."""
    lowered = (error or "").lower()
    return "background" in lowered and ("not supported" in lowered or "transparent" in lowered)


def generate(
    prompt: str,
    *,
    n: int = 1,
    reference_images: list[Path] | None = None,
    provider: SpriteProvider | None = None,
    prefix: str = "pet_gen",
    aspect_ratio: str = "square",
) -> list[Path]:
    """Generate *n* sprite images and return their local paths.

    *reference_images* grounds the output on a base image (required for rows).
    *aspect_ratio* ``"landscape"`` (row strips) gives every frame horizontal room so
    winged poses needn't shrink. We *ask* for a transparent background but fall
    back to opaque on models that reject the flag. Raises :class:`GenerationError`
    if nothing usable comes back.
    """
    sprite = provider or resolve_provider(require_references=bool(reference_images))
    if reference_images and not sprite.supports_references:
        raise GenerationError(
            f"image backend '{sprite.name}' cannot use reference images; "
            "configure OpenAI gpt-image-2 or Krea for pet generation"
        )

    refs = [str(p) for p in (reference_images or [])]

    # Providers disagree on the ref kwarg name: our OpenRouter/Nous backends read
    # ``reference_images``, OpenAI's gpt-image-2 reads ``reference_image_urls``.
    # Send both; each ignores the other.
    ref_kwargs = {"reference_images": refs, "reference_image_urls": refs} if refs else {}
    model_kwargs = {"model": sprite.model} if sprite.model else {}

    def _run(extra: dict) -> tuple[Path | None, str]:
        try:
            result = sprite.provider.generate(
                prompt, aspect_ratio=aspect_ratio, **extra, **ref_kwargs, **model_kwargs
            )
        except Exception as exc:
            logger.debug("provider.generate crashed: %s", exc)
            return None, str(exc)
        if not isinstance(result, dict):
            return None, "no result"
        if not result.get("success"):
            return None, result.get("error", "unknown error")
        if not (image_ref := result.get("image")):
            return None, "provider returned no image"
        try:
            return _save_local(str(image_ref), prefix=prefix), ""
        except Exception as exc:
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

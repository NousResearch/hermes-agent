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
from typing import Any

logger = logging.getLogger(__name__)

# Friendly display labels for shipped providers. Third-party providers fall back to
# their registry display_name so the pet generator stays plugin-extensible.
_BUILTIN_REF_PROVIDER_ORDER = ("nous", "openai", "openai-codex", "openrouter", "krea")
_LOCAL_REFERENCE_INCOMPATIBLE_PROVIDERS = {"fal"}

_PROVIDER_LABELS = {
    "nous": "Nous Portal",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "openai-codex": "OpenAI (Codex)",
    "krea": "Krea",
}


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


def _available(name: str | None):
    """The registered provider *name* if it exists and has credentials, else ``None``."""
    if not isinstance(name, str) or not name.strip():
        return None
    from agent.image_gen_registry import get_provider

    raw_name = name.strip()
    provider = get_provider(raw_name)
    if provider is None and raw_name != raw_name.lower():
        provider = get_provider(raw_name.lower())
    if provider is None:
        return None
    try:
        return provider if provider.is_available() else None
    except Exception:  # noqa: BLE001 - provider availability probes must not break the picker
        return None


def _supports_references(provider: Any) -> bool:
    """Whether an image provider advertises image/reference inputs."""
    if getattr(provider, "name", "") in _LOCAL_REFERENCE_INCOMPATIBLE_PROVIDERS:
        return False
    try:
        caps = provider.capabilities()
    except Exception:  # noqa: BLE001 - provider capability probes must not break the picker
        return False
    modalities = caps.get("modalities") if isinstance(caps, dict) else None
    try:
        max_refs = int(caps.get("max_reference_images") or 0) if isinstance(caps, dict) else 0
    except (TypeError, ValueError):
        max_refs = 0
    return "image" in (modalities or []) and max_refs > 0


def _available_reference_provider(name: str | None):
    provider = _available(name)
    return provider if provider is not None and _supports_references(provider) else None


def _provider_label(provider: Any) -> str:
    name = getattr(provider, "name", "")
    return _PROVIDER_LABELS.get(name, str(getattr(provider, "display_name", name) or name))


def _registered_reference_providers() -> list[Any]:
    from agent.image_gen_registry import list_providers

    providers_by_name = {provider.name: provider for provider in list_providers()}
    ordered: list[Any] = []
    seen: set[str] = set()

    def add_if_reference(provider: Any | None) -> None:
        if provider is None or provider.name in seen:
            return
        try:
            if provider.is_available() and _supports_references(provider):
                ordered.append(provider)
                seen.add(provider.name)
        except Exception:  # noqa: BLE001 - one broken plugin must not hide the rest
            return

    for name in _BUILTIN_REF_PROVIDER_ORDER:
        add_if_reference(providers_by_name.get(name))
    for provider in providers_by_name.values():
        add_if_reference(provider)
    return ordered


def resolve_provider(*, require_references: bool = True, prefer: str | None = None) -> SpriteProvider:
    """Pick the image provider for sprite work.

    Preference: ``HERMES_PET_IMAGE_PROVIDER`` (QA override), then *prefer*
    (desktop picker), then the active provider, then the first registered provider
    that advertises image/reference inputs. With *require_references* off, any
    available active provider is accepted for prompt-only base drafts.
    """
    _discover()
    from agent.image_gen_registry import get_active_provider

    forced = os.environ.get("HERMES_PET_IMAGE_PROVIDER", "").strip()
    for name in (forced, prefer):
        if (chosen := _available_reference_provider(name)) is not None:
            return SpriteProvider(name=chosen.name, provider=chosen, supports_references=True)
    try:
        active = get_active_provider()
    except Exception:  # noqa: BLE001
        active = None
    if active is not None:
        try:
            if active.is_available() and _supports_references(active):
                return SpriteProvider(name=active.name, provider=active, supports_references=True)
        except Exception:  # noqa: BLE001
            pass
    for provider in _registered_reference_providers():
        return SpriteProvider(name=provider.name, provider=provider, supports_references=True)
    if not require_references and active is not None and active.is_available():
        return SpriteProvider(name=getattr(active, "name", "unknown"), provider=active, supports_references=False)
    raise GenerationError(
        "Pet generation needs an image backend that supports reference images. "
        "Open `hermes tools` → Image Generation and configure a backend whose "
        "capabilities include image/reference inputs."
    )


def list_sprite_providers() -> list[dict]:
    """``[{name, label, default}]`` per configured reference-capable provider; empty hides the picker."""
    _discover()
    try:
        default_name = resolve_provider(require_references=True).name
    except GenerationError:
        default_name = ""
    return [
        {"name": provider.name, "label": _provider_label(provider), "default": provider.name == default_name}
        for provider in _registered_reference_providers()
    ]


def _save_local(image_ref: str, *, prefix: str) -> Path:
    """Return a local path for *image_ref*, downloading it if it's a URL."""
    if image_ref.startswith(("http://", "https://")):
        from agent.image_gen_provider import save_url_image

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

    def _run(extra: dict) -> tuple[Path | None, str]:
        try:
            result = sprite.provider.generate(prompt, aspect_ratio=aspect_ratio, **extra, **ref_kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize provider crashes
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

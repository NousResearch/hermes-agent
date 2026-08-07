"""Thin image-generation layer for pet sprites.

Wraps the active :class:`~agent.image_gen_provider.ImageGenProvider` with the
two things sprite generation needs that the agent-facing ``image_generate`` tool
doesn't expose: **N variants** (loop) and **reference-image grounding** (so each
animation row stays the same character as the chosen base).

Reference grounding only works on providers that support it. A provider
advertises that by including ``"image"`` in ``capabilities()["modalities"]``
(image-to-image / editing), so user plugins qualify automatically; the built-in
names in :data:`_REF_CAPABLE` are additionally trusted as a fallback and set the
preference order. We resolve to a reference-capable provider and surface a
clear, actionable error otherwise rather than silently producing an ungrounded,
drifting pet.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Built-in providers known to ground generation on a reference image, in
# preference order (Nous Portal → OpenAI → OpenRouter → …). OpenRouter/Nous run
# a quality-first model chain and may fall back depending on account access and
# endpoint behavior, so fidelity can vary by configured backend + model
# availability. This tuple is an ordering hint plus a fallback for providers
# that don't override ``capabilities()`` — any other registered provider that
# declares the ``"image"`` modality (e.g. a local img2img plugin) is
# reference-capable too; see :func:`_supports_references`.
_REF_CAPABLE = ("nous", "openai", "openai-codex", "openrouter", "krea")

# Friendly display label per reference-capable provider, surfaced in the desktop
# pet-gen picker.
_PROVIDER_LABELS: dict[str, str] = {
    "nous": "Nous Portal",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "openai-codex": "OpenAI (Codex)",
    "krea": "Krea",
}


def _supports_references(name: str, provider: object) -> bool:
    """True when *provider* can ground generation on reference images.

    Primary signal: the provider's own ``capabilities()`` declares the
    ``"image"`` modality (image-to-image / editing) — this is what lets user
    plugins participate without being hardcoded here. Fallback: the name is in
    :data:`_REF_CAPABLE`, which keeps built-ins working even if a provider
    doesn't override ``capabilities()`` (the base-class default is text-only).
    """
    caps = getattr(provider, "capabilities", None)
    if callable(caps):
        try:
            modalities = (caps() or {}).get("modalities") or ()
        except Exception as exc:  # noqa: BLE001 - a broken override shouldn't break resolution
            logger.debug("capabilities() failed for image provider '%s': %s", name, exc)
        else:
            if "image" in modalities:
                return True
    return name in _REF_CAPABLE


def _ref_capable_names() -> list[str]:
    """Names of reference-capable providers, in preference order.

    The built-ins from :data:`_REF_CAPABLE` come first (their tuple order is
    the preference order), followed by any other registered provider that
    declares itself reference-capable, in the registry's name-sorted order.
    """
    from agent.image_gen_registry import list_providers

    names = list(_REF_CAPABLE)
    try:
        registered = list_providers()
    except Exception as exc:  # noqa: BLE001 - registry hiccups shouldn't break resolution
        logger.debug("image provider listing failed: %s", exc)
        registered = []
    for provider in registered:
        name = getattr(provider, "name", "")
        if name and name not in names and _supports_references(name, provider):
            names.append(name)
    return names


def _provider_label(name: str, provider: object) -> str:
    """Display label for the desktop pet-gen picker."""
    label = _PROVIDER_LABELS.get(name)
    if label:
        return label
    return str(getattr(provider, "display_name", "") or name)


def _forced_provider_from_env() -> str | None:
    """Optional QA override to force a pet-gen backend.

    `HERMES_PET_IMAGE_PROVIDER=<name>` (e.g. `openrouter`) bypasses the normal
    active/default provider resolution for pet generation only. Any registered
    reference-capable provider qualifies, including user plugins. Unknown values
    are ignored so existing users are unaffected.
    """
    forced = os.environ.get("HERMES_PET_IMAGE_PROVIDER", "").strip().lower()
    if not forced:
        return None
    if forced in _REF_CAPABLE:
        return forced
    from agent.image_gen_registry import get_provider

    provider = get_provider(forced)
    if provider is not None and _supports_references(forced, provider):
        return forced
    return None


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


def resolve_provider(*, require_references: bool = True, prefer: str | None = None) -> SpriteProvider:
    """Pick the image provider to use for sprite work.

    Preference: an explicit *prefer* choice (the desktop pet-gen picker) when it's
    reference-capable and configured, then the configured/active provider when
    it's reference-capable, else the first available reference-capable provider.
    With *require_references* off we fall back to any available provider (used for
    prompt-only base drafts).
    """
    _discover()
    from agent.image_gen_registry import get_active_provider, get_provider

    # QA override: force one provider for pet-gen iteration regardless of the
    # globally active image_gen backend.
    forced = _forced_provider_from_env()
    if forced:
        chosen = get_provider(forced)
        if chosen is not None and chosen.is_available():
            return SpriteProvider(name=forced, provider=chosen, supports_references=True)

    # An explicit user pick wins when it's reference-capable and has credentials;
    # otherwise we ignore it and fall through to the normal resolution.
    if prefer:
        chosen = get_provider(prefer)
        if chosen is not None and _supports_references(prefer, chosen) and chosen.is_available():
            return SpriteProvider(name=prefer, provider=chosen, supports_references=True)

    # Configured / active provider first.
    active = None
    try:
        active = get_active_provider()
    except Exception:  # noqa: BLE001
        active = None
    if active is not None:
        name = getattr(active, "name", "")
        if _supports_references(name, active) and active.is_available():
            return SpriteProvider(name=name, provider=active, supports_references=True)

    # Any available reference-capable provider.
    for name in _ref_capable_names():
        provider = get_provider(name)
        if provider is not None and provider.is_available():
            return SpriteProvider(name=name, provider=provider, supports_references=True)

    if not require_references and active is not None and active.is_available():
        return SpriteProvider(
            name=getattr(active, "name", "unknown"), provider=active, supports_references=False
        )

    raise GenerationError(
        "Pet generation needs an image backend that supports reference images. "
        "Open `hermes tools` → Image Generation and configure Nous Portal, "
        "OpenRouter, or OpenAI (gpt-image-2) with an API key. Any image-gen "
        "plugin that supports image-to-image works too."
    )


def list_sprite_providers() -> list[dict]:
    """The reference-capable providers available to pick for pet generation.

    Returns ``[{name, label, default}]`` for every ref-capable provider the user
    actually has credentials for, in preference order, marking the one
    :func:`resolve_provider` would choose with no explicit preference. Empty when
    none is configured (the picker hides itself). Best-effort: discovery hiccups
    yield an empty list.
    """
    _discover()
    from agent.image_gen_registry import get_provider

    try:
        default_name = resolve_provider(require_references=True).name
    except GenerationError:
        default_name = ""

    out: list[dict] = []
    for name in _ref_capable_names():
        provider = get_provider(name)
        if provider is None or not provider.is_available():
            continue
        out.append(
            {
                "name": name,
                "label": _provider_label(name, provider),
                "default": name == default_name,
            }
        )
    return out


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
    aspect_ratio: str = "square",
) -> list[Path]:
    """Generate *n* sprite images and return their local paths.

    *reference_images* grounds the output on a base image (required for rows).
    *aspect_ratio* picks the canvas: ``"square"`` for single-character base
    drafts, ``"landscape"`` for multi-frame row strips (the wider 1536px canvas
    gives every frame real horizontal room so winged poses don't have to be
    shrunk to avoid touching their neighbors).
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
        kwargs: dict = {"aspect_ratio": aspect_ratio, **extra}
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

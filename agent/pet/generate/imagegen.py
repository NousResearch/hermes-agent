"""Thin image-generation layer for pet sprites.

Wraps the active :class:`~agent.image_gen_provider.ImageGenProvider` with the
two things sprite generation needs that the agent-facing ``image_generate`` tool
doesn't expose: **N variants** (loop) and **reference-image grounding** (so each
animation row stays the same character as the chosen base).

Reference grounding only works on providers and models that advertise image
input/editing support. We resolve against that capability contract, including
third-party providers, and surface a clear error rather than silently producing
an ungrounded, drifting pet.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Stable fallback preference after the user's configured/explicit provider.
# Capability checks, not this tuple, decide whether a provider is usable; names
# absent here are appended alphabetically so third-party providers work too.
_PROVIDER_PREFERENCE = (
    "nous",
    "openai",
    "openai-codex",
    "openrouter",
    "xai",
    "fal",
    "krea",
)
# Backward-compatible preference export. Capability checks remain authoritative;
# callers must not treat this tuple as an allowlist.
_REF_CAPABLE = _PROVIDER_PREFERENCE

# Friendly display label per reference-capable provider, surfaced in the desktop
# pet-gen picker.
_PROVIDER_LABELS: dict[str, str] = {
    "nous": "Nous Portal",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "openai-codex": "OpenAI (Codex)",
    "xai": "xAI Grok Imagine",
    "fal": "FAL.ai",
    "krea": "Krea",
}


def _forced_provider_from_env() -> str | None:
    """Return the existing QA-only pet provider override, if set."""
    return os.environ.get("HERMES_PET_IMAGE_PROVIDER", "").strip().lower() or None


class GenerationError(RuntimeError):
    """Raised on any image-generation failure (no provider, API error, IO)."""


@dataclass(frozen=True)
class SpriteProvider:
    """Resolved provider plus whether it can take reference images."""

    name: str
    provider: object
    supports_references: bool
    supports_seed: bool = False
    supports_model_override: bool = False
    model: str | None = None


def _discover() -> None:
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        logger.debug("image-gen plugin discovery failed: %s", exc)


def _capabilities(provider: object, model: str | None = None) -> dict[str, Any]:
    """Return model-specific capabilities, falling back to provider-wide data."""
    base: dict[str, Any] = {}
    try:
        raw = provider.capabilities()
        if isinstance(raw, dict):
            base.update(raw)
    except Exception as exc:  # noqa: BLE001 - a provider cannot break discovery
        logger.debug("image provider capabilities failed: %s", exc)

    # Compatibility for pre-capability providers shipped before the unified
    # image contract. New/third-party providers must advertise capabilities;
    # this only preserves the historically-known built-ins.
    if not base and getattr(provider, "name", "") in _REF_CAPABLE:
        base.update({"modalities": ["text", "image"], "max_reference_images": 1})

    if model:
        try:
            for entry in provider.list_models() or []:
                if isinstance(entry, dict) and entry.get("id") == model:
                    for key in (
                        "modalities",
                        "max_reference_images",
                        "supports_seed",
                        "supports_model_override",
                    ):
                        if key in entry:
                            base[key] = entry[key]
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("image provider model catalog failed: %s", exc)
    return base


def _model_entries(provider: object) -> list[dict[str, Any]]:
    """Return the provider's normalized model catalog, best-effort."""
    try:
        return [
            dict(entry)
            for entry in (provider.list_models() or [])
            if isinstance(entry, dict) and entry.get("id")
        ]
    except Exception as exc:  # noqa: BLE001 - a provider cannot break discovery
        logger.debug("image provider model catalog failed: %s", exc)
        return []


def _model_entry(provider: object, model: str) -> dict[str, Any] | None:
    return next((entry for entry in _model_entries(provider) if str(entry.get("id")) == model), None)


def _is_available(provider: object) -> bool:
    try:
        return bool(provider.is_available())
    except Exception as exc:  # noqa: BLE001
        logger.debug("image provider availability failed: %s", exc)
        return False


def _sprite(provider: object, *, model: str | None = None) -> SpriteProvider:
    caps = _capabilities(provider, model)
    modalities = set(caps.get("modalities") or ["text"])
    catalogued = bool(model and _model_entry(provider, model))
    return SpriteProvider(
        name=str(getattr(provider, "name", "") or ""),
        provider=provider,
        supports_references="image" in modalities and int(caps.get("max_reference_images") or 0) > 0,
        supports_seed=bool(caps.get("supports_seed")),
        supports_model_override=bool(caps.get("supports_model_override")) or catalogued,
        model=model,
    )


def _ordered_providers(providers: list[object]) -> list[object]:
    rank = {name: index for index, name in enumerate(_PROVIDER_PREFERENCE)}
    return sorted(providers, key=lambda p: (rank.get(getattr(p, "name", ""), len(rank)), getattr(p, "name", "")))


def resolve_provider(
    *,
    require_references: bool = True,
    prefer: str | None = None,
    model: str | None = None,
) -> SpriteProvider:
    """Pick the image provider to use for sprite work.

    Preference: an explicit *prefer* choice (the desktop pet-gen picker) when it's
    reference-capable and configured, then the configured/active provider when
    it's reference-capable, else the first available reference-capable provider.
    With *require_references* off we fall back to any available provider (used for
    prompt-only base drafts).
    """
    _discover()
    from agent.image_gen_registry import get_active_provider, get_provider, list_providers

    def acceptable(candidate: object | None, candidate_model: str | None = None) -> SpriteProvider | None:
        if candidate is None or not _is_available(candidate):
            return None

        if candidate_model:
            entries = _model_entries(candidate)
            entry = next((item for item in entries if str(item.get("id")) == candidate_model), None)
            if entries and entry is None:
                return None
            resolved = _sprite(candidate, model=candidate_model)
            if not resolved.supports_model_override:
                return None
            if require_references and not resolved.supports_references:
                return None
            return resolved

        resolved = _sprite(candidate)
        if not require_references or resolved.supports_references:
            return resolved

        # Some providers (notably FAL) have a text-only configured default but
        # expose reference-capable models in the same catalog. Pick a compatible
        # model explicitly instead of declaring the whole provider unavailable.
        for entry in _model_entries(candidate):
            candidate_id = str(entry["id"])
            model_resolved = _sprite(candidate, model=candidate_id)
            if model_resolved.supports_references and model_resolved.supports_model_override:
                return model_resolved
        return None

    # Preserve the long-standing QA override. Invalid/unavailable overrides are
    # intentionally ignored because this is an internal test knob, not a
    # user-facing persisted selection.
    forced = _forced_provider_from_env()
    if forced:
        resolved = acceptable(get_provider(forced), model)
        if resolved is not None:
            return resolved

    # Explicit choices are contracts, not hints. A typo, missing credential, or
    # incompatible model must never spend money on a different provider.
    if prefer:
        chosen = get_provider(prefer)
        if chosen is None:
            raise GenerationError(f"Unknown image provider '{prefer}'.")
        if not _is_available(chosen):
            raise GenerationError(f"Image provider '{prefer}' is not configured or available.")
        resolved = acceptable(chosen, model)
        if resolved is not None:
            return resolved
        if model:
            raise GenerationError(
                f"Image model '{model}' is not available for provider '{prefer}'"
                + (" with reference-image editing." if require_references else ".")
            )
        raise GenerationError(
            f"Image provider '{prefer}' has no available model that supports reference-image editing."
        )

    # Configured / active provider first.
    active = None
    try:
        active = get_active_provider()
    except Exception:  # noqa: BLE001
        active = None
    if model:
        if active is None or not _is_available(active):
            raise GenerationError("A model override requires an available active image provider or --provider.")
        resolved = acceptable(active, model)
        if resolved is not None:
            return resolved
        active_name = str(getattr(active, "name", "active provider"))
        raise GenerationError(
            f"Image model '{model}' is not available for provider '{active_name}'"
            + (" with reference-image editing." if require_references else ".")
        )

    if active is not None:
        resolved = acceptable(active)
        if resolved is not None:
            return resolved

    # Any available reference-capable provider.
    for provider in _ordered_providers(list_providers()):
        resolved = acceptable(provider)
        if resolved is not None:
            return resolved

    raise GenerationError(
        "Pet generation needs an image backend/model that supports reference images. "
        "Open `hermes tools` → Image Generation and configure a compatible provider."
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
    from agent.image_gen_registry import get_provider, list_providers

    discovered = {str(getattr(provider, "name", "")): provider for provider in list_providers()}
    # Compatibility with registries/tests that only implement name lookup;
    # capability-based third-party providers still enter through list_providers.
    for name in _PROVIDER_PREFERENCE:
        provider = get_provider(name)
        if provider is not None:
            discovered.setdefault(name, provider)

    try:
        default_name = resolve_provider(require_references=True).name
    except GenerationError:
        default_name = ""

    out: list[dict] = []
    for provider in _ordered_providers(list(discovered.values())):
        if not _is_available(provider):
            continue
        name = str(getattr(provider, "name", "") or "")
        models: list[dict[str, Any]] = []
        for entry in _model_entries(provider):
            candidate = _sprite(provider, model=str(entry["id"]))
            if not candidate.supports_references or not candidate.supports_model_override:
                continue
            models.append(
                {
                    "id": str(entry["id"]),
                    "display": str(entry.get("display") or entry.get("name") or entry["id"]),
                    "supportsSeed": candidate.supports_seed,
                }
            )

        resolved = _sprite(provider)
        if not resolved.supports_references and not models:
            continue
        default_model = ""
        try:
            provider_default = str(provider.default_model() or "")
        except Exception:  # noqa: BLE001
            provider_default = ""
        if provider_default and any(item["id"] == provider_default for item in models):
            default_model = provider_default
        elif not resolved.supports_references and models:
            default_model = str(models[0]["id"])
        default_sprite = _sprite(provider, model=default_model or None)
        out.append(
            {
                "name": name,
                "label": _PROVIDER_LABELS.get(name, getattr(provider, "display_name", name)),
                "default": name == default_name,
                "models": models,
                "defaultModel": default_model,
                "supportsSeed": default_sprite.supports_seed,
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
    model: str | None = None,
    seed: int | None = None,
) -> list[Path]:
    """Generate *n* sprite images and return their local paths.

    *reference_images* grounds the output on a base image (required for poses).
    *aspect_ratio* picks the canvas: ``"square"`` for single-character base
    drafts and one-pose edits, ``"landscape"`` for two-pose edits (the wider
    canvas gives both complete bodies room around a clean center gutter).
    We *ask* for a transparent background, but fall back to an opaque generation
    (cleaned up downstream by the chroma-key pass) on models that reject the
    flag. Raises :class:`GenerationError` if nothing usable comes back.
    """
    sprite = provider or resolve_provider(require_references=bool(reference_images))
    if reference_images and not sprite.supports_references:
        raise GenerationError(
            f"image backend '{sprite.name}' cannot use reference images; "
            "choose a model that supports image editing"
        )

    effective_model = model or sprite.model
    if model and sprite.model and model != sprite.model:
        raise GenerationError(
            f"Image model '{model}' does not match resolved model '{sprite.model}' for provider '{sprite.name}'."
        )
    if effective_model and not sprite.supports_model_override:
        raise GenerationError(f"Image provider '{sprite.name}' does not support per-run model overrides.")
    if seed is not None and not sprite.supports_seed:
        raise GenerationError(
            f"Image provider '{sprite.name}'"
            + (f" model '{effective_model}'" if effective_model else "")
            + " does not support deterministic seeds."
        )

    refs = [str(p) for p in (reference_images or [])]

    def _run(extra: dict) -> tuple[Path | None, str]:
        kwargs: dict = {"aspect_ratio": aspect_ratio, **extra}
        if effective_model:
            kwargs["model"] = effective_model
        if seed is not None:
            kwargs["seed"] = seed
        if refs:
            # Providers disagree on the ref kwarg name: our OpenRouter/Nous
            # backends read ``reference_images``, OpenAI's gpt-image-2 reads
            # ``reference_image_urls``. Send both; each ignores the other.
            kwargs["image_url"] = refs[0]
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

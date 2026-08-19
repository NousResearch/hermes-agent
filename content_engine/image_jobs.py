"""Provider-free request planning for the converged image-generation path.

This module deliberately prepares a request only. It cannot call Codex,
ComfyUI, FAL, Pollinations, a gateway or a publishing path.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit

from blog.art_director import STYLE_LIBRARY
from image_router import LAYOUT_REGISTRY


class ImageRequestError(ValueError):
    """Raised when an ad-hoc image request violates the public contract."""


_ALLOWED_BACKENDS = frozenset({"codex", "local"})
_REFERENCE_ROLES = frozenset({"written_inspiration", "visual_reference"})
MAX_REFERENCES = 8


@dataclass(frozen=True)
class ReferenceRequest:
    """A user-supplied source retained for later safe staging."""

    url: str
    requested_roles: frozenset[str] = _REFERENCE_ROLES


@dataclass(frozen=True)
class StyleProfile:
    """Canonical style metadata retaining its existing layout grammar."""

    style_id: str
    label: str
    kind: str
    layout_options: tuple[str, ...]


@dataclass(frozen=True)
class PreparedImageRequest:
    """Validated request with no backend side effects."""

    prompt: str
    style_id: str
    backend: str
    references: tuple[ReferenceRequest, ...]
    # Optional extended style menu: neutral trait fragment resolved from the
    # ~9,000-style registry (single style or deterministic blend). Never
    # contains protected names or SREF; purely additive to the existing
    # style/layout resolution.
    registry_traits: str | None = None
    registry_slugs: tuple[str, ...] = ()
    registry_seed: int | None = None


def _style_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


_LEGACY_SOCIAL_STUDIO_ALIASES = {
    "mythic-tech-codex-illustration": "mythic-tech-codex",
    "cosmic-postcard-atelier": "cosmic-postcard",
    "ink-ember-studio": "ink-ember-studio",
    "saga-noir-studio": "saga-noir",
    "ninth-observatory": "ninth-observatory",
    "chromatic-institute": "chromatic-institute",
    "dark-cyberpunk-hud": "signal-hud",
}


def _split_layout_options(raw: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r",|;|\bor\b", raw)
        if item.strip()
    )


def _style_profiles() -> dict[str, StyleProfile]:
    return {
        str(style["id"]): StyleProfile(
            style_id=str(style["id"]),
            label=str(style["label"]),
            kind=str(style["kind"]),
            layout_options=_split_layout_options(str(style["layout"])),
        )
        for style in STYLE_LIBRARY
    }


def _style_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for profile in _style_profiles().values():
        for candidate in (profile.style_id, profile.label):
            aliases[_style_key(candidate)] = profile.style_id
    aliases.update({
        _style_key(legacy): style_id
        for legacy, style_id in _LEGACY_SOCIAL_STUDIO_ALIASES.items()
    })
    return aliases


def get_style_profile(value: str) -> StyleProfile:
    """Resolve a canonical or legacy style name into retained metadata."""

    style_id = _resolve_style(value)
    return _style_profiles()[style_id]


def canonical_layouts() -> frozenset[str]:
    """Return the non-destructive union of blog and legacy-social layouts."""

    blog_layouts = {
        layout
        for profile in _style_profiles().values()
        for layout in profile.layout_options
    }
    return frozenset(blog_layouts | set(LAYOUT_REGISTRY))


def _resolve_style(value: str) -> str:
    key = _style_key(value)
    style_id = _style_aliases().get(key)
    if style_id is None:
        raise ImageRequestError(f"unknown style: {value!r}")
    return style_id


def _is_unsafe_ip(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return not address.is_global


def _validate_reference_url(value: str) -> str:
    text = str(value).strip()
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"}:
        raise ImageRequestError("reference URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ImageRequestError("reference URL must not include credentials")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ImageRequestError("reference URL must include a hostname")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ImageRequestError("reference URL must use a public host")
    if _is_unsafe_ip(hostname):
        raise ImageRequestError("reference URL must not target a private address")
    return text


def _normalise_references(values: Iterable[str]) -> tuple[ReferenceRequest, ...]:
    references: list[ReferenceRequest] = []
    seen: set[str] = set()
    for value in values:
        url = _validate_reference_url(value)
        if url not in seen:
            references.append(ReferenceRequest(url=url))
            if len(references) > MAX_REFERENCES:
                raise ImageRequestError(f"at most {MAX_REFERENCES} distinct reference URLs are permitted")
            seen.add(url)
    return tuple(references)


def prepare_image_request(
    *,
    prompt: str,
    style: str,
    backend: str = "codex",
    references: Iterable[str] = (),
    blend: Iterable[str] = (),
    registry_seed: int | None = None,
    registry_style: str | None = None,
) -> PreparedImageRequest:
    """Validate a request without selecting or invoking a generator.

    ``codex`` is the explicit policy default. ``local`` is available only when
    chosen by the caller. No unsupported provider can silently become a fallback.

    ``blend`` (optional) is the extended style menu: two or more registry
    slugs are deterministically blended into a neutral trait fragment that is
    added to the request's ``registry_traits``. A single ``style`` may be
    supplied alone (existing behaviour); if ``blend`` is supplied, ``style``
    is still required as the canonical base style (backward compatible).
    Unknown slugs fail fast — never a silent fallback.
    """

    clean_prompt = str(prompt).strip()
    if not clean_prompt:
        raise ImageRequestError("prompt is required")
    clean_backend = str(backend).strip().lower() or "codex"
    if clean_backend not in _ALLOWED_BACKENDS:
        raise ImageRequestError(f"backend must be one of: {', '.join(sorted(_ALLOWED_BACKENDS))}")

    blend_slugs = tuple(str(slug).strip() for slug in blend if str(slug).strip())
    registry_traits: str | None = None
    resolved_seed: int | None = registry_seed
    # A single registry style (no blend) also contributes a neutral trait
    # fragment. blend (if present) wins; otherwise the single style is used.
    active_slugs: tuple[str, ...] = blend_slugs
    if not active_slugs and registry_style:
        active_slugs = (str(registry_style).strip(),)
    if active_slugs:
        from style_registry import RegistryError, resolve_fragment

        try:
            # Seed may be passed explicitly (caller-derived, cross-post
            # variation) or default to the blend engine's built-in seed.
            if len(active_slugs) >= 2:
                registry_traits = resolve_fragment(
                    style_slug=None,
                    blend_slugs=list(active_slugs),
                    seed=resolved_seed,
                )
            else:
                registry_traits = resolve_fragment(
                    style_slug=active_slugs[0],
                    blend_slugs=None,
                    seed=resolved_seed,
                )
        except RegistryError as exc:
            raise ImageRequestError(f"invalid blend: {exc}") from exc

    return PreparedImageRequest(
        prompt=clean_prompt,
        style_id=_resolve_style(str(style)),
        backend=clean_backend,
        references=_normalise_references(references),
        registry_traits=registry_traits,
        registry_slugs=active_slugs,
        registry_seed=resolved_seed,
    )

"""Security policy helpers for embedding the dashboard chat surface."""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import urlsplit

_EMBED_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ORIGIN_NETLOC_RE = re.compile(r"^(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]{1,5})?$")


class EmbedPolicyError(ValueError):
    """The requested embedded-chat scope is not allowed by dashboard config."""


def _dashboard(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("dashboard")
    return value if isinstance(value, Mapping) else {}


def configured_embed_parent_origins(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unique exact HTTP(S) origins accepted as embed parents."""
    raw = _dashboard(config).get("embed_parent_origins")
    if not isinstance(raw, list):
        return ()

    result: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        # This value is interpolated into a CSP response header. Keep the
        # accepted surface ASCII-origin-only and reject percent-encoded control
        # sequences rather than relying on downstream header normalization.
        if "%" in candidate or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in candidate):
            continue
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if not _ORIGIN_NETLOC_RE.fullmatch(parsed.netloc):
            continue
        try:
            _ = parsed.port
        except ValueError:
            continue
        if parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
            continue
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin == candidate and origin not in result:
            result.append(origin)
    return tuple(result)


def configured_embed_profiles(config: Mapping[str, Any]) -> dict[str, str]:
    """Return validated embed IDs mapped to normalized profile scopes."""
    raw = _dashboard(config).get("embed_profiles")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, str] = {}
    for raw_id, raw_profile in raw.items():
        if not isinstance(raw_id, str) or not _EMBED_ID_RE.fullmatch(raw_id):
            continue
        if not isinstance(raw_profile, str):
            continue
        profile = raw_profile.strip().lower()
        if profile == "default":
            profile = ""
        elif not _PROFILE_RE.fullmatch(profile):
            continue
        result[raw_id] = profile
    return result


def resolve_embedded_profile(
    config: Mapping[str, Any], embed_id: str, requested_profile: str | None
) -> str | None:
    """Resolve an embed ID to its configured immutable profile scope."""
    embed_id = (embed_id or "").strip()
    if not _EMBED_ID_RE.fullmatch(embed_id):
        raise EmbedPolicyError("embed id is invalid")

    profiles = configured_embed_profiles(config)
    if embed_id not in profiles:
        raise EmbedPolicyError(f"embed id {embed_id!r} is not configured")

    pinned = profiles[embed_id]

    requested = (requested_profile or "").strip().lower()
    if requested == "default":
        requested = ""
    if requested != pinned:
        raise EmbedPolicyError(
            f"embed id {embed_id!r} does not permit profile {requested_profile or 'default'!r}"
        )
    return pinned or None

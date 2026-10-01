"""Pure identities shared by custom-provider menus and runtime resolution.

Never resolves secrets, runs key commands, or reads configuration. Multiple model
entries on the same credential route share one slug; separate routes receive
collision-free slugs in configuration order.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Iterator


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def credential_identity(entry: dict) -> str:
    """Opaque identity of all configured credential sources (env names are case sensitive)."""
    key = str(entry.get("api_key") or "").strip()
    env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key.startswith("${") and key.endswith("}"):
        env, key = env or key[2:-1].strip(), ""
    return _fingerprint([str(entry.get("key_cmd") or "").strip(), key, env])


def display_prefix(name: str) -> str:
    return next((name.split(sep)[0].strip() for sep in ("—", " - ") if sep in name), name)


def custom_provider_group_key(entry: dict) -> tuple:
    """Identity of a model group, independent of its model list and display suffix."""
    url = str(entry.get("base_url") or entry.get("url") or entry.get("api") or "").strip().rstrip("/").lower()
    name = display_prefix(str(entry.get("name") or entry.get("provider_key") or "").strip())
    mode = str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()
    return (url, credential_identity(entry), mode, _fingerprint(entry.get("extra_headers") or {}), name.lower())


def iter_custom_provider_routes(entries: Iterable[dict]) -> Iterator[tuple[str, dict]]:
    from hermes_cli.providers import custom_provider_slug

    groups: dict[tuple, str] = {}
    used: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = custom_provider_group_key(entry)
        if not key[0] or not key[-1]:
            continue
        slug = groups.get(key)
        if slug is None:
            base = custom_provider_slug(key[-1], str(entry.get("provider_key") or ""))
            slug, suffix = base, 2
            while slug in used:
                slug = f"{base}-{suffix}"
                suffix += 1
            groups[key] = slug
            used.add(slug)
        yield slug, entry


def match_custom_provider_route(requested: str, entries: Iterable[dict]) -> tuple[str, dict] | None:
    from hermes_cli.providers import custom_provider_aliases

    requested = str(requested or "").strip().lower()
    routes = list(iter_custom_provider_routes(entries))
    # An allocated slug wins over another entry's ambiguous display-name alias.
    for slug, entry in routes:
        if requested == slug:
            return slug, entry
    for slug, entry in routes:
        if requested in custom_provider_aliases(str(entry.get("name") or ""), str(entry.get("provider_key") or "")):
            return slug, entry
    return None

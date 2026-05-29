"""Reasoning-effort capability resolution helpers.

Hermes accepts reasoning-effort values as user intent.  Provider APIs expose
smaller, provider-specific wire vocabularies, so profiles should resolve the
requested intent against a declared capability set before sending it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def resolve_reasoning_effort(
    requested: str | None,
    *,
    allowed: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a requested reasoning effort to a provider-supported wire value.

    ``requested`` is Hermes' internal/user-facing intent. ``allowed`` is the
    provider/model's wire vocabulary. ``aliases`` maps internal intents such as
    ``max`` to the nearest supported wire value for that capability set.

    Returns ``None`` when the request cannot be represented safely.  Callers
    should then omit the field and let the provider default rather than guess.
    """
    effort = str(requested or "").strip().lower()
    if not effort:
        return None

    normalized_aliases = {
        str(src).strip().lower(): str(dst).strip().lower()
        for src, dst in (aliases or {}).items()
        if str(src).strip() and str(dst).strip()
    }
    effort = normalized_aliases.get(effort, effort)

    normalized_allowed = {
        str(item).strip().lower()
        for item in allowed
        if str(item).strip()
    }
    if effort in normalized_allowed:
        return effort
    return None

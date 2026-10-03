"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import logging
from typing import Any

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

# Sections that never carry a fallback chain but read as though they should. The CLI reference
# documented ``fallback_providers`` "under ``model:``" until #19691 corrected it, and that issue
# closed on the docs fix alone — so configs written from the old docs still nest the chain where
# no loader looks. ``delegation.fallback_providers`` is deliberately absent: it IS a real key.
_MISPLACED_FALLBACK_SECTIONS: tuple[str, ...] = ("model",)

# Keep only the active diagnostic fingerprint for each profile/config key. The gateway serves
# multiple profiles in one process, while several callers reload config on every turn; profile
# scope plus a non-secret route fingerprint avoids both cross-profile suppression and log spam.
_warned_misplaced_fallback: dict[
    tuple[str, str, str], tuple[tuple[str, str, str, str], ...]
] = {}


def _misplaced_fallback_fingerprint(raw: Any) -> tuple[tuple[str, str, str, str], ...]:
    """Return non-secret route identity for diagnostic de-duplication."""
    candidates = raw if isinstance(raw, list) else [raw]
    fingerprint: list[tuple[str, str, str, str]] = []
    for entry in candidates:
        if isinstance(entry, dict):
            fingerprint.append((
                "entry",
                str(entry.get("provider") or "").strip(),
                str(entry.get("model") or "").strip(),
                _normalized_base_url(entry.get("base_url")),
            ))
        else:
            fingerprint.append((type(entry).__name__, "", "", ""))
    return tuple(fingerprint)


def _warn_misplaced_fallback_keys(config: dict[str, Any]) -> None:
    """Warn for a fallback chain nested under a section no loader reads.

    ``get_fallback_chain`` reads the TOP-LEVEL keys only. A chain written as
    ``model.fallback_providers`` resolves to an EMPTY chain while the config looks configured,
    so the primary's exhaustion kills the session with no fallback and no explanation. Entry
    values are never logged — fallback dicts may carry ``api_key``/``extra_headers``.
    """
    profile_key = hermes_home_key()
    for section_name in _MISPLACED_FALLBACK_SECTIONS:
        section = config.get(section_name)
        for key in ("fallback_providers", "fallback_model"):
            marker = (profile_key, section_name, key)
            raw = section.get(key) if isinstance(section, dict) else None
            if not raw:
                _warned_misplaced_fallback.pop(marker, None)
                continue
            fingerprint = _misplaced_fallback_fingerprint(raw)
            if _warned_misplaced_fallback.get(marker) == fingerprint:
                continue
            _warned_misplaced_fallback[marker] = fingerprint
            logger.warning(
                "'%s.%s' is not read by any loader and is IGNORED — the fallback chain is read "
                "from the TOP-LEVEL '%s' in config.yaml. Move the list to the top level (or run "
                "`hermes fallback add`); until then the effective chain from this section is EMPTY.",
                section_name, key, key)


def _normalized_base_url(value: Any) -> str:
    return value.strip().rstrip("/") if isinstance(value, str) else ""


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``api_key_env`` accepted as alias); None when neither
    yields a value so ``resolve_runtime_provider`` falls through to standard credential resolution.
    ``key_env`` goes through ``agent.secret_scope.get_secret``, not raw ``os.getenv``: in a
    multiplexed gateway a bare env read ignores the active profile's scope and can return another
    profile's credential.
    """
    if not isinstance(entry, dict):
        return None
    if inline := str(entry.get("api_key") or "").strip():
        return inline
    if key_env := str(entry.get("key_env") or entry.get("api_key_env") or "").strip():
        from agent.secret_scope import get_secret
        return (get_secret(key_env) or "").strip() or None
    return None


def effective_runtime_provider(
    entry: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> str:
    """Provider identity to persist/display for a resolved fallback entry.

    ``resolve_runtime_provider`` returns the bare billing class ``"custom"``
    for every named ``providers:`` / ``custom_providers:`` entry; the entry's
    configured id only survives in ``requested_provider``. Fallback resolvers
    that persist ``runtime["provider"]`` as the agent identity therefore label
    sessions/billing rows ``custom`` instead of the configured provider name —
    while the manual ``/model`` switch path correctly persists the named id
    (#98739). Same class as the delegation fix in ``tools/delegate_tool.py``.

    Returns the entry's requested identity when the resolved provider is the
    bare ``custom`` class; a genuinely ad-hoc endpoint (requested provider IS
    ``custom``) keeps the bare class unchanged.
    """
    runtime = runtime or {}
    resolved = str(runtime.get("provider") or "").strip()
    if resolved.lower() != "custom":
        return resolved
    requested = str(
        runtime.get("requested_provider")
        or (entry or {}).get("provider")
        or ""
    ).strip()
    if requested and requested.lower() != "custom":
        return requested
    return resolved


def pre_agent_fallback_notice(
    primary_provider: Any, primary_model: Any, fallback_provider: Any, fallback_model: Any
) -> str:
    """User-visible one-shot line for a provider switch made during credential resolution, before
    any AIAgent exists (#74349). Shared by the messaging gateway, the TUI/Desktop gateway and cron
    so the three pre-agent fallback paths cannot drift in wording."""
    primary_desc = "/".join(str(p).strip() for p in (primary_provider, primary_model) if p) or "primary"
    fallback_desc = "/".join(str(p).strip() for p in (fallback_provider, fallback_model) if p) or "fallback"
    return f"⚠️ Provider fallback: {primary_desc} unavailable; using {fallback_desc} for this response."



def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    candidates = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue
        normalized = {**entry, "provider": provider, "model": model}
        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url
        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its order. Legacy
    ``fallback_model`` entries are appended afterwards unless they target the same
    provider/model/base_url route as an earlier entry. The returned list always contains fresh dict
    copies.
    """
    config = config or {}
    _warn_misplaced_fallback_keys(config)
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain


def scoped_fallback_chain(
    inherited: list[dict[str, Any]] | None, declared: Any, *, pinned: bool, owner: str,
) -> list[dict[str, Any]] | None:
    """Fallback chain for a route owner that can pin its own primary (delegated child, cron job).

    A pinned owner (explicit provider, endpoint or model) never borrows the *inherited* chain: the
    operator chose that route, and a chain entry is a different provider and usually a different
    model. Predictability beats liveness for an explicit pin. An unpinned owner inherits the chain
    when *declared* is absent/None. An explicit ``[]`` disables fallback either way; any other
    *declared* value is the owner's own chain, normalized by :func:`get_fallback_chain` (malformed
    entries are dropped; nothing usable left falls back to the pinned/inherited default).
    """
    default = None if pinned else (inherited or None)
    if declared is None:
        return default
    if declared == []:
        return None
    normalized = get_fallback_chain({"fallback_providers": declared})
    if not normalized:
        logger.warning("%s fallback_providers has no usable routes; using the %s default",
                       owner, "pinned" if pinned else "inherited")
    return normalized or default

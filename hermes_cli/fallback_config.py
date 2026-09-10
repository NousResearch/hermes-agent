"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any, NamedTuple


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
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain


class FallbackResolution(NamedTuple):
    """Non-secret result of a successful fallback-chain walk.

    ``configured_provider`` is the literal ``provider`` key from the matched config entry — not
    ``runtime.get("provider")``, which reflects the resolved runtime *category* (e.g. an Ollama
    entry resolves through the OpenAI-compatible path and would read back as ``"openrouter"``).
    Callers that log or display which fallback fired must use ``configured_provider`` to keep the
    message consistent with what the operator actually configured (#32790). ``runtime`` itself may
    still carry credentials (api_key, etc.) — never log it wholesale.
    """

    runtime: dict[str, Any]
    model: str | None
    configured_provider: str | None


def resolve_first_available_fallback(
    config: dict[str, Any] | None, *, logger: Any = None,
) -> FallbackResolution | None:
    """Walk the configured fallback chain and return a :class:`FallbackResolution` for the first
    entry that resolves; ``None`` when no chain is configured or every entry fails to resolve.

    Shared by every startup credential-fallback caller (gateway, oneshot) so there is exactly one
    fallback-chain-walking implementation: it reuses :func:`get_fallback_chain`,
    :func:`resolve_entry_api_key`, and ``resolve_runtime_provider`` — never re-derive fallback
    resolution elsewhere. An entry that fails to resolve (missing/invalid credentials, network
    error) is skipped in favor of the next; nothing here logs API keys or tokens.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider

    for entry in get_fallback_chain(config):
        try:
            runtime = resolve_runtime_provider(
                requested=entry.get("provider"), explicit_base_url=entry.get("base_url"),
                explicit_api_key=resolve_entry_api_key(entry))
        except Exception as fb_exc:
            if logger is not None:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
            continue
        return FallbackResolution(runtime, entry.get("model"), entry.get("provider"))
    return None

"""Fail-closed URL identity normalization for model/provider routes."""

from __future__ import annotations

from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_route_base_url(base_url: Any) -> str:
    """Canonicalize only proven-equivalent endpoint URL components."""
    raw = str(base_url or "")
    if not raw:
        return ""
    if any(ord(char) <= 0x20 for char in raw):
        return raw
    had_query_delimiter = "?" in raw.split("#", 1)[0]
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return raw
        scheme = parsed.scheme.lower()
        if "%" in hostname:
            address, zone = hostname.split("%", 1)
            host = f"{address.lower()}%{zone}"
        else:
            host = hostname.lower()
        port = parsed.port
    except (TypeError, ValueError):
        return raw
    route_host = parsed.netloc.rsplit("@", 1)[-1]
    if route_host.startswith("[") or ":" in host:
        host = f"[{host}]"
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        host = f"{host}:{port}"
    if "@" in parsed.netloc:
        host = f"{parsed.netloc.rsplit('@', 1)[0]}@{host}"
    path = parsed.path
    if path.endswith("/") and not had_query_delimiter:
        path = path[:-1]
    normalized = urlunsplit((scheme, host, path, parsed.query, ""))
    if had_query_delimiter and not parsed.query:
        normalized += "?"
    return normalized


# Provider ids whose runtime is resolved first-hand (never a named custom provider).
_RUNTIME_FIRST_PROVIDER_IDS = {
    "auto", "moa", "vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai",
}


def normalize_custom_provider_name(value: Any) -> str:
    """Mirror runtime normalization for a requested custom-provider identity."""
    return str(value or "").strip().lower().replace(" ", "-")


def custom_provider_runtime_ids(value: Any) -> set[str]:
    """Return raw/menu identities that runtime accepts for a configured name."""
    normalized = normalize_custom_provider_name(value)
    if not normalized:
        return set()
    return {normalized, f"custom:{normalized}"}


def custom_provider_configured_base_url(
    configured_provider: str, agent_cfg: Any, custom_providers: Any
) -> str:
    """Base URL of a named custom provider (``providers.<name>`` first, then
    ``custom_providers``), normalized for route comparison; "" if unknown.
    Disabled ``providers.*`` entries also mask their ``custom_providers`` twin.
    """
    wanted = normalize_custom_provider_name(configured_provider)
    user_providers = agent_cfg.get("providers")
    disabled_ids: set[str] = set()
    if isinstance(user_providers, dict):
        from hermes_cli.config import is_provider_enabled
        for key, entry in user_providers.items():
            if not isinstance(entry, dict):
                continue
            ids = custom_provider_runtime_ids(key) | custom_provider_runtime_ids(entry.get("name"))
            if not is_provider_enabled(entry):
                disabled_ids.update(ids)
                continue
            if wanted in ids:
                url = normalize_route_base_url(
                    entry.get("api") or entry.get("url") or entry.get("base_url")
                )
                if url:
                    return url
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        key_ids = custom_provider_runtime_ids(entry.get("provider_key"))
        if key_ids & disabled_ids:
            continue
        if wanted in key_ids | custom_provider_runtime_ids(entry.get("name")):
            url = normalize_route_base_url(entry.get("base_url"))
            if url:
                return url
    return ""


def configured_default_base_url(agent_cfg: Any, model_cfg: Any, custom_providers: Any) -> str:
    """Normalized route of the configured default model: ``model.base_url`` when set, else the URL of
    the named custom provider in ``model.provider`` (``""`` when neither identifies a route).

    Every caller that asks whether the ``model.context_length`` pin still matches "the route it was
    written for" must resolve through this first. A custom endpoint is declared under
    ``providers.<name>`` with ``model.base_url`` left empty (the block owns the URL), and its runtime
    identity collapses to the bare ``custom`` billing class while ``model.provider`` still names the
    entry — so comparing the raw config value against the runtime route finds neither a URL nor a
    matching provider id, and the pin is dropped for an unchanged route.
    """
    base_url = normalize_route_base_url(model_cfg.get("base_url"))
    configured_provider = str(model_cfg.get("provider") or "").strip()
    norm = normalize_custom_provider_name(configured_provider)
    candidate = bool(norm)
    if norm in _RUNTIME_FIRST_PROVIDER_IDS:
        candidate = False
    elif candidate and norm != "custom" and not norm.startswith("custom:"):
        with suppress(Exception):
            from hermes_cli.auth import resolve_provider as resolve_auth_provider
            candidate = str(resolve_auth_provider(norm) or "").strip().lower() != norm
    if base_url or not candidate:
        return base_url
    return custom_provider_configured_base_url(configured_provider, agent_cfg, custom_providers)


def should_clear_context_pin(configured_model: Any, active_model: Any, configured_base_url: Any, active_base_url: Any,
                             configured_provider: Any, active_provider: Any) -> bool:
    """True when a configured ``model.context_length`` pin no longer matches its runtime route.
    Fail-closed: any error during route comparison returns ``True`` (drop the pin) so a stale window
    never silently inflates the compression threshold."""
    configured_model = str(configured_model or "").strip()
    if configured_model and configured_model != str(active_model or "").strip():
        return True
    try:
        from agent.agent_init import _context_route_mismatch
        return _context_route_mismatch(configured_base_url, active_base_url, configured_provider, active_provider)
    except Exception:
        return True


async def should_clear_context_pin_async(*args: Any) -> bool:
    """``should_clear_context_pin`` on a worker thread so async gateway handlers never run it on the
    event loop — the resolution chain is cache-only (``allow_network=False``) but can still do
    cold-start disk I/O."""
    import asyncio
    return await asyncio.to_thread(should_clear_context_pin, *args)

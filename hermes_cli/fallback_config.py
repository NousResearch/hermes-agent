"""Helpers for reading and enforcing the fallback provider policy."""

from __future__ import annotations

import os
from typing import Any


FALLBACK_POLICIES = ("off", "local-only", "any")
DEFAULT_FALLBACK_POLICY = "any"
_BUILTIN_LOCAL_PROVIDER_IDS = frozenset({"custom", "local", "lmstudio"})


def get_fallback_policy(config: dict[str, Any] | None) -> str:
    """Return the configured fallback policy, failing closed on invalid input.

    ``any`` is the compatibility default for configs written before the policy
    existed. Invalid explicit values are treated as ``off`` at runtime; config
    validation reports the mistake so a typo can never silently widen routing.
    """
    config = config or {}
    if "fallback_policy" not in config:
        return DEFAULT_FALLBACK_POLICY
    value = config.get("fallback_policy")
    if isinstance(value, str) and value in FALLBACK_POLICIES:
        return value
    return "off"


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

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


def _resolve_fallback_provider_definition(
    entry: dict[str, Any],
    config: dict[str, Any],
):
    provider = str(entry.get("provider") or "").strip()
    if not provider:
        return None
    try:
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.providers import resolve_provider_full

        return resolve_provider_full(
            provider,
            user_providers=(
                config.get("providers")
                if isinstance(config.get("providers"), dict)
                else None
            ),
            custom_providers=get_compatible_custom_providers(config),
        )
    except Exception:
        return None


def fallback_entry_endpoint(
    entry: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """Resolve an entry's endpoint from explicit or provider metadata.

    This intentionally never infers locality from a model name. An explicit
    ``base_url`` wins, followed by a provider-specific environment override,
    then the canonical provider definition (including user providers).
    """
    explicit = _normalized_base_url(entry.get("base_url"))
    if explicit:
        return explicit

    config = config or {}
    resolved = _resolve_fallback_provider_definition(entry, config)
    if resolved is None:
        return ""

    env_name = str(getattr(resolved, "base_url_env_var", "") or "").strip()
    if env_name:
        env_url = _normalized_base_url(os.getenv(env_name))
        if env_url:
            return env_url
    return _normalized_base_url(getattr(resolved, "base_url", ""))


def fallback_entry_is_local(
    entry: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> bool:
    """Return whether provider metadata proves that an entry is local."""
    config = config or {}
    endpoint = fallback_entry_endpoint(entry, config)
    if not endpoint:
        return False
    try:
        from agent.model_metadata import is_local_endpoint

        if not is_local_endpoint(endpoint):
            return False

        # A built-in cloud provider remains a remote route even if a stale or
        # hostile environment variable points its client at localhost. Users
        # who deliberately redefine that provider under ``providers:`` get a
        # ``user-config`` definition and are judged by that endpoint instead.
        resolved = _resolve_fallback_provider_definition(entry, config)
        if (
            resolved is not None
            and getattr(resolved, "source", "") != "user-config"
            and str(getattr(resolved, "id", "") or "").strip().lower()
            not in _BUILTIN_LOCAL_PROVIDER_IDS
        ):
            return False
        canonical_endpoint = _normalized_base_url(
            getattr(resolved, "base_url", "") if resolved is not None else ""
        )
        if (
            canonical_endpoint
            and getattr(resolved, "source", "") != "user-config"
            and not is_local_endpoint(canonical_endpoint)
        ):
            return False
        return True
    except Exception:
        return False


def get_configured_fallback_chain(
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return the ordered configured chain before policy filtering."""
    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)
    return chain


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the chain eligible under ``off | local-only | any``."""
    config = config or {}
    policy = get_fallback_policy(config)
    if policy == "off":
        return []

    chain = get_configured_fallback_chain(config)
    if policy == "local-only":
        return [entry for entry in chain if fallback_entry_is_local(entry, config)]
    return chain


def filter_fallback_chain_for_policy(
    entries: Any,
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Filter a caller-supplied chain through the effective routing policy.

    Some runtime entry points resolve the primary provider before constructing
    an :class:`AIAgent` and therefore cannot use :func:`get_fallback_chain`
    directly: the ordered entries have already been captured by the caller.
    This helper applies the same ``off | local-only | any`` boundary to that
    captured chain so init/auth fallback can never bypass the configured
    policy.  Returned entries are normalized fresh copies.
    """
    config = config or {}
    policy = get_fallback_policy(config)
    if policy == "off":
        return []

    chain = _iter_fallback_entries(entries)
    if policy == "local-only":
        return [entry for entry in chain if fallback_entry_is_local(entry, config)]
    return chain

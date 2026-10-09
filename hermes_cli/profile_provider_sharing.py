"""Opt-in model-provider capability sharing contract for profiles.

This module intentionally contains only the schema parser and capability-scope
helpers for issue #91572. Runtime wiring that loads another profile's files or
changes provider resolution is a follow-up so the public config surface can be
reviewed independently while existing profile isolation remains the default.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

logger_name = __name__

_SHAREABLE_CAPABILITY_KEYS = frozenset({
    "provider_env",
    "provider_base_urls",
    "providers",
    "excluded_providers",
})
_DEFAULT_CAPABILITY_KEYS = _SHAREABLE_CAPABILITY_KEYS
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ProfileProviderSharingError(ValueError):
    """Raised when the profile provider-sharing config is malformed."""


@dataclass(frozen=True)
class ProfileProviderSharingConfig:
    """Resolved profile model-provider sharing settings."""

    enabled: bool = False
    source_profile: str = "default"
    capabilities: frozenset[str] = frozenset()


def _profile_sharing_block(config: Mapping[str, Any] | None) -> Any:
    if not isinstance(config, Mapping):
        return None
    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping):
        return None
    return profiles.get("share_model_providers")


def is_profile_provider_sharing_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether ``profiles.share_model_providers`` is explicitly enabled."""

    return parse_profile_provider_sharing_config(config).enabled


def parse_profile_provider_sharing_config(
    config: Mapping[str, Any] | None,
) -> ProfileProviderSharingConfig:
    """Parse the opt-in model-provider sharing block.

    Accepted forms::

        profiles:
          share_model_providers: true

        profiles:
          share_model_providers:
            enabled: true
            source_profile: default
            capabilities:
              - provider_env
              - provider_base_urls
              - providers
              - excluded_providers

    Missing, ``false``, and empty mapping values all resolve to the disabled
    shape. Unknown capability names and malformed source profiles fail fast.
    """

    block = _profile_sharing_block(config)
    if block is None or block is False:
        return ProfileProviderSharingConfig()
    if block is True:
        return ProfileProviderSharingConfig(
            enabled=True,
            source_profile="default",
            capabilities=frozenset(_DEFAULT_CAPABILITY_KEYS),
        )
    if not isinstance(block, Mapping):
        raise ProfileProviderSharingError(
            "profiles.share_model_providers must be a boolean or mapping"
        )

    enabled = block.get("enabled", False)
    if enabled is False or enabled is None:
        return ProfileProviderSharingConfig()
    if enabled is not True:
        raise ProfileProviderSharingError(
            "profiles.share_model_providers.enabled must be true or false"
        )

    source_profile = block.get("source_profile", "default")
    if not isinstance(source_profile, str) or not source_profile.strip():
        raise ProfileProviderSharingError(
            "profiles.share_model_providers.source_profile must be a non-empty string"
        )
    source_profile = source_profile.strip()
    if source_profile in {".", ".."} or not _PROFILE_NAME_RE.match(source_profile):
        raise ProfileProviderSharingError(
            "profiles.share_model_providers.source_profile contains invalid characters"
        )

    raw_capabilities = block.get("capabilities", tuple(_DEFAULT_CAPABILITY_KEYS))
    capabilities = _parse_capabilities(raw_capabilities)
    if not capabilities:
        raise ProfileProviderSharingError(
            "profiles.share_model_providers.capabilities must include at least one value"
        )

    return ProfileProviderSharingConfig(
        enabled=True,
        source_profile=source_profile,
        capabilities=frozenset(capabilities),
    )


def _parse_capabilities(raw: Any) -> set[str]:
    if isinstance(raw, str):
        items: Iterable[Any] = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray, Mapping)):
        items = raw
    else:
        raise ProfileProviderSharingError(
            "profiles.share_model_providers.capabilities must be a list or comma string"
        )

    capabilities: set[str] = set()
    unknown: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise ProfileProviderSharingError(
                "profiles.share_model_providers.capabilities entries must be strings"
            )
        key = item.strip()
        if not key:
            continue
        if key not in _SHAREABLE_CAPABILITY_KEYS:
            unknown.add(key)
            continue
        capabilities.add(key)
    if unknown:
        raise ProfileProviderSharingError(
            "profiles.share_model_providers.capabilities contains unknown values: "
            + ", ".join(sorted(unknown))
        )
    return capabilities


def builtin_provider_share_env_vars(provider_ids: Iterable[str]) -> set[str]:
    """Return API-key and base-url env vars for built-in model providers.

    Network catalog refresh is deliberately disabled. Sharing decisions must be
    deterministic at config-load time and based on the checked-in provider
    metadata/overlays.
    """

    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import get_provider, normalize_provider

    env_vars: set[str] = set()
    for provider_id in provider_ids:
        if not isinstance(provider_id, str) or not provider_id.strip():
            continue
        canonical = normalize_provider(provider_id)
        auth_provider = PROVIDER_REGISTRY.get(canonical)
        if auth_provider is not None:
            env_vars.update(name for name in auth_provider.api_key_env_vars if name)
            if auth_provider.base_url_env_var:
                env_vars.add(auth_provider.base_url_env_var)
        provider = get_provider(canonical, allow_network=False)
        if provider is None:
            continue
        env_vars.update(name for name in provider.api_key_env_vars if name)
        if provider.base_url_env_var:
            env_vars.add(provider.base_url_env_var)
    return env_vars


def custom_provider_share_env_vars(
    providers: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> set[str]:
    """Return key/base-url env vars declared by custom provider entries."""

    if providers is None:
        return set()
    if isinstance(providers, Mapping):
        entries = providers.values()
    else:
        entries = providers

    env_vars: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key in ("key_env", "api_key_env"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                env_vars.add(value.strip())
        base_url = entry.get("base_url") or entry.get("url") or entry.get("api")
        if isinstance(base_url, str):
            match = _ENV_PLACEHOLDER_RE.match(base_url.strip())
            if match:
                env_vars.add(match.group(1))
    return env_vars

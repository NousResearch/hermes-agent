"""Pure configured-provider identity and route facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from providers.identity import custom_provider_aliases, custom_provider_slug, normalize_provider
from providers.registry import get_provider_profile
from providers.routing import canonicalize_api_mode

_FALSE_WORDS = frozenset({"false", "0", "no", "off"})
_DIRECT_API_BASE_URLS = {"openai": "https://api.openai.com/v1"}


@dataclass(frozen=True)
class ConfiguredProvider:
    """Normalized facts for one matched user-configured provider declaration."""

    identity: str
    source: str
    name: str
    provider_key: str
    base_url: str
    api_mode: str
    model: str
    capabilities: dict[str, bool]
    extra_body: dict[str, Any]
    extra_headers: dict[str, str]
    raw: Mapping[str, Any]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalized_name(value: Any) -> str:
    return _clean(value).lower().replace(" ", "-")


def _entry_url(entry: Mapping[str, Any]) -> str:
    return _clean(entry.get("api") or entry.get("url") or entry.get("base_url"))


def _enabled(entry: Mapping[str, Any]) -> bool:
    flag = entry.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() not in _FALSE_WORDS
    return bool(flag)


def _shadowed_by_builtin(requested: str) -> bool:
    if requested == "custom" or requested.startswith("custom:"):
        return False
    profile = get_provider_profile(requested)
    return bool(profile and _normalized_name(profile.name) == requested)


def _project(
    entry: Mapping[str, Any], *, provider_key: str = "", source: str
) -> ConfiguredProvider:
    name = _clean(entry.get("name")) or provider_key
    base_url = _entry_url(entry)
    api_mode = canonicalize_api_mode(_clean(entry.get("api_mode") or entry.get("transport")))
    model = _clean(entry.get("model") or entry.get("default_model"))
    capabilities = entry.get("capabilities")
    extra_body = entry.get("extra_body")
    extra_headers = entry.get("extra_headers")
    return ConfiguredProvider(
        identity=custom_provider_slug(name, provider_key),
        source=source,
        name=name,
        provider_key=provider_key,
        base_url=base_url,
        api_mode=api_mode,
        model=model,
        capabilities={
            key: value
            for key, value in (capabilities.items() if isinstance(capabilities, dict) else ())
            if isinstance(key, str) and isinstance(value, bool)
        },
        extra_body=dict(extra_body) if isinstance(extra_body, dict) else {},
        extra_headers={
            str(key): str(value)
            for key, value in (extra_headers.items() if isinstance(extra_headers, dict) else ())
            if value is not None
        },
        raw=entry,
    )


def match_configured_provider(
    requested_provider: str,
    *,
    providers: Mapping[Any, Any] | None = None,
    custom_providers: Sequence[Any] | None = None,
) -> ConfiguredProvider | None:
    """Match a configured route, preferring providers mapping over legacy entries."""

    requested = _normalized_name(requested_provider)
    if not requested or requested == "auto" or _shadowed_by_builtin(requested):
        return None

    if isinstance(providers, Mapping):
        for stored_key, entry in providers.items():
            if not isinstance(entry, Mapping) or not _enabled(entry):
                continue
            provider_key = _clean(stored_key)
            name = _clean(entry.get("name")) or provider_key
            if requested not in custom_provider_aliases(name, provider_key):
                continue
            if not _entry_url(entry):
                continue
            return _project(entry, provider_key=provider_key, source="providers")

    if isinstance(custom_providers, Sequence) and not isinstance(custom_providers, (str, bytes)):
        for entry in custom_providers:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            base_url = entry.get("base_url")
            if not isinstance(name, str) or not isinstance(base_url, str):
                continue
            provider_key = _clean(entry.get("provider_key"))
            if requested not in custom_provider_aliases(name, provider_key):
                continue
            return _project(entry, provider_key=provider_key, source="custom_providers")
    return None


def configured_custom_identity(
    *,
    base_url: str = "",
    model: str = "",
    config_provider: str = "",
    providers: Mapping[Any, Any] | None = None,
    custom_providers: Sequence[Any] | None = None,
) -> str:
    """Recover the durable identity of a configured custom route from supplied facts."""

    target_url = _clean(base_url).rstrip("/").lower()
    target_model = _clean(model).lower()

    def serves_model(entry: Mapping[str, Any]) -> bool:
        if not target_model:
            return False
        if target_model in {
            _clean(entry.get("model")).lower(),
            _clean(entry.get("default_model")).lower(),
        }:
            return True
        values = entry.get("models")
        if isinstance(values, Mapping):
            return any(_clean(value).lower() == target_model for value in values)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                candidate = (
                    _clean(value.get("id") or value.get("name"))
                    if isinstance(value, Mapping)
                    else _clean(value)
                )
                if candidate.lower() == target_model:
                    return True
        return False

    if isinstance(providers, Mapping):
        for stored_key, entry in providers.items():
            if not isinstance(entry, Mapping) or not _enabled(entry):
                continue
            projected = _project(entry, provider_key=_clean(stored_key), source="providers")
            if (
                target_url
                and projected.base_url.rstrip("/").lower() == target_url
            ) or serves_model(entry):
                return projected.identity

    if isinstance(custom_providers, Sequence) and not isinstance(
        custom_providers, (str, bytes)
    ):
        for entry in custom_providers:
            if not isinstance(entry, Mapping):
                continue
            name = _clean(entry.get("name"))
            if not name:
                continue
            projected = _project(
                entry,
                provider_key=_clean(entry.get("provider_key")),
                source="custom_providers",
            )
            if (
                target_url
                and projected.base_url.rstrip("/").lower() == target_url
            ) or serves_model(entry):
                return projected.identity

    candidate = _normalized_name(config_provider)
    if candidate and candidate not in {"custom", "auto", "openrouter"}:
        match = match_configured_provider(
            candidate,
            providers=providers,
            custom_providers=custom_providers,
        )
        if match is not None:
            return match.identity
    return ""


def resolves_to_custom_provider(provider: str) -> bool:
    """Whether a registered provider alias resolves to the generic custom profile."""

    name = _normalized_name(provider)
    return bool(name and name not in {"auto", "main"} and normalize_provider(name) == "custom")


def expand_direct_api_alias(
    provider: str | None,
    existing_base: str | None,
    *,
    configured_provider: bool = False,
    preferred_base_url: str = "",
) -> tuple[str | None, str | None]:
    """Expand direct REST aliases using caller-supplied configuration facts."""

    if not provider:
        return provider, existing_base
    target_base = _DIRECT_API_BASE_URLS.get(_normalized_name(provider))
    if target_base is None or configured_provider:
        return provider, existing_base
    return (
        "custom",
        _clean(existing_base) or _clean(preferred_base_url).rstrip("/") or target_base,
    )

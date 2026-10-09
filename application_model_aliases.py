"""Application projection of configured model aliases.

Alias syntax and credential lookup are application configuration concerns. Model
identity is represented as :class:`models.ModelRef`; provider/model selection
and invocation semantics remain owned by the lower domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from models import ModelRef, parse_configured_provider_ref
from providers import (
    custom_provider_aliases,
    custom_provider_slug,
    list_providers,
)


@dataclass(frozen=True, slots=True)
class ConfiguredModelAlias:
    name: str
    ref: ModelRef
    base_url: str = ""
    api_key: str = ""
    key_env: str = ""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def provider_reference_context(config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Known provider IDs plus configured named-custom canonical identities."""
    known = {
        _clean(profile.name).lower()
        for profile in list_providers()
        if _clean(profile.name)
    }
    named_custom: set[str] = set()
    providers = config.get("providers")
    if isinstance(providers, Mapping):
        for key, entry in providers.items():
            if not isinstance(entry, Mapping):
                continue
            raw_key = _clean(key).lower()
            if raw_key:
                known.add(raw_key)
            named_custom.add(
                custom_provider_slug(_clean(entry.get("name")) or _clean(key), _clean(key))
            )
    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        for entry in legacy:
            if isinstance(entry, Mapping) and _clean(entry.get("name")):
                named_custom.add(custom_provider_slug(_clean(entry.get("name"))))
    return tuple(sorted(known)), tuple(sorted(named_custom))


def configured_provider_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Configured provider request IDs used by provider/model startup syntax."""
    ids: set[str] = set()
    providers = config.get("providers")
    if isinstance(providers, Mapping):
        ids.update(_clean(name).lower() for name in providers if _clean(name))
    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        ids.update(
            custom_provider_slug(_clean(entry.get("name")))
            for entry in legacy
            if isinstance(entry, Mapping) and _clean(entry.get("name"))
        )
    return tuple(sorted(ids))


def _configured_alias_identities(config: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    providers = config.get("providers")
    if not isinstance(providers, Mapping):
        return result
    for key, entry in providers.items():
        if not isinstance(entry, Mapping):
            continue
        display = _clean(entry.get("name")) or _clean(key)
        slug = custom_provider_slug(display, _clean(key))
        for alias in custom_provider_aliases(display, _clean(key)):
            result[alias] = slug
    return result


def model_aliases_from_config(config: Mapping[str, Any]) -> dict[str, ConfiguredModelAlias]:
    """Project `model_aliases` / `model.aliases` without owning model semantics."""
    known, named_custom = provider_reference_context(config)
    alias_provider_ids = (*known, *named_custom)
    configured_identities = _configured_alias_identities(config)

    def alias_ref(provider: Any, model: Any) -> ModelRef:
        raw_provider = _clean(provider)
        provider_id = configured_identities.get(raw_provider.lower(), raw_provider)
        return ModelRef(provider_id, _clean(model))

    aliases: dict[str, ConfiguredModelAlias] = {}
    user_aliases = config.get("model_aliases")
    if isinstance(user_aliases, Mapping):
        for name, entry in user_aliases.items():
            key = _clean(name).lower()
            if not key or not isinstance(entry, Mapping) or not _clean(entry.get("model")):
                continue
            aliases[key] = ConfiguredModelAlias(
                key,
                alias_ref(entry.get("provider", "custom"), entry.get("model")),
                _clean(entry.get("base_url")),
                _clean(entry.get("api_key")),
                _clean(entry.get("key_env")),
            )

    model_section = config.get("model")
    simple = model_section.get("aliases") if isinstance(model_section, Mapping) else None
    current_provider = _clean(model_section.get("provider")) if isinstance(model_section, Mapping) else ""
    if isinstance(simple, Mapping):
        for name, value in simple.items():
            key = _clean(name).lower()
            if not key or key in aliases:
                continue
            if isinstance(value, Mapping):
                model = _clean(value.get("model"))
                if model:
                    aliases[key] = ConfiguredModelAlias(
                        key,
                        alias_ref(_clean(value.get("provider")) or current_provider or "custom", model),
                        _clean(value.get("base_url")),
                        _clean(value.get("api_key")),
                        _clean(value.get("key_env")),
                    )
            elif isinstance(value, str) and _clean(value):
                raw = _clean(value)
                ref = parse_configured_provider_ref(raw, alias_provider_ids)
                if ref is not None:
                    prefix = raw.split("/", 1)[0].strip().lower()
                    configured = configured_identities.get(prefix)
                    if configured:
                        ref = ModelRef(configured, ref.model)
                aliases[key] = ConfiguredModelAlias(
                    key,
                    ref or alias_ref(current_provider, raw),
                )
    return aliases


def alias_api_key(alias: ConfiguredModelAlias) -> str:
    """Resolve only the alias's declared credential through the active profile scope."""
    raw = _clean(alias.api_key)
    if raw.startswith("${") and raw.endswith("}"):
        return _scoped_env(raw[2:-1].strip())
    if raw:
        return raw
    return _scoped_env(alias.key_env)


def _scoped_env(name: str) -> str:
    if not name:
        return ""
    try:
        from agent.secret_scope import current_secret_scope, get_secret, is_multiplex_active
        if current_secret_scope() is not None or is_multiplex_active():
            return _clean(get_secret(name, ""))
        from agent.credential_pool import get_env_prefer_dotenv
        return _clean(get_env_prefer_dotenv(name))
    except Exception:
        return ""


__all__ = [
    "ConfiguredModelAlias",
    "alias_api_key",
    "configured_provider_ids",
    "model_aliases_from_config",
    "provider_reference_context",
]

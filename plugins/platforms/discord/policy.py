"""Deterministic Discord scope-policy resolution.

The resolver deliberately knows nothing about Discord objects or network state.  A
missing override returns ``None`` for every field, allowing the adapter to retain
its existing environment/configuration behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


class ScopePolicyValidationError(ValueError):
    """A Discord scope-policy schema error that must abort gateway startup."""


_POLICY_FIELDS = ("require_mention", "allow_bots", "allow_humans", "conversation_trust")
_TRUST_VALUES = frozenset(("legacy", "public", "private", "full_trusted"))


@dataclass(frozen=True)
class ResolvedDiscordPolicy:
    """Effective field values and their configuration provenance."""

    require_mention: Optional[bool] = None
    allow_bots: Optional[bool] = None
    allow_humans: Optional[bool] = None
    conversation_trust: Optional[str] = None
    sources: Mapping[str, str] | None = None


def _scope_id(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise ScopePolicyValidationError(f"{label} must be a numeric Discord ID")
    text = str(value)
    if not text or not text.isdigit():
        raise ScopePolicyValidationError(f"{label} must be a numeric Discord ID: {text!r}")
    return text


def _validate_fields(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ScopePolicyValidationError(f"{path} must be a mapping")
    unknown = sorted(set(value) - set(_POLICY_FIELDS))
    if unknown:
        raise ScopePolicyValidationError(f"{path} contains unknown field(s): {', '.join(map(str, unknown))}")
    result: dict[str, Any] = {}
    for field in _POLICY_FIELDS:
        if field not in value:
            continue
        item = value[field]
        if field in ("require_mention", "allow_bots", "allow_humans"):
            # bool is intentionally strict: 0/1 and string booleans are not schema booleans.
            if not isinstance(item, bool):
                raise ScopePolicyValidationError(f"{path}.{field} must be a boolean")
        elif not isinstance(item, str) or item not in _TRUST_VALUES:
            raise ScopePolicyValidationError(
                f"{path}.{field} must be one of: {', '.join(sorted(_TRUST_VALUES))}"
            )
        result[field] = item
    return result


def validate_scope_policies(raw: Any) -> dict[str, Any] | None:
    """Validate and return a normalized copy of the optional policy schema.

    Validation is deterministic and fail-closed.  IDs are normalized to strings
    so YAML integer keys and quoted snowflakes resolve identically.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ScopePolicyValidationError("scope_policies must be a mapping")
    allowed = {"version", "platform", "guilds"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ScopePolicyValidationError(f"scope_policies contains unknown key(s): {', '.join(map(str, unknown))}")
    version = raw.get("version", 1)
    if type(version) is not int or version != 1:
        raise ScopePolicyValidationError("scope_policies.version must be 1")
    platform = raw.get("platform", {})
    if not isinstance(platform, Mapping):
        raise ScopePolicyValidationError("scope_policies.platform must be a mapping")
    platform_unknown = sorted(set(platform) - {"defaults"})
    if platform_unknown:
        raise ScopePolicyValidationError(f"scope_policies.platform contains unknown key(s): {', '.join(map(str, platform_unknown))}")
    out: dict[str, Any] = {"version": 1, "platform": _validate_fields(platform.get("defaults"), "scope_policies.platform.defaults"), "guilds": {}}
    guilds = raw.get("guilds", {})
    if not isinstance(guilds, Mapping):
        raise ScopePolicyValidationError("scope_policies.guilds must be a mapping")
    for guild_key in sorted(guilds, key=lambda item: str(item)):
        guild_id = _scope_id(guild_key, "scope_policies.guilds key")
        entry = guilds[guild_key]
        if not isinstance(entry, Mapping):
            raise ScopePolicyValidationError(f"scope_policies.guilds.{guild_id} must be a mapping")
        extra = sorted(set(entry) - {"defaults", "channels", "threads"})
        if extra:
            raise ScopePolicyValidationError(f"scope_policies.guilds.{guild_id} contains unknown key(s): {', '.join(map(str, extra))}")
        normalized = {
            "defaults": _validate_fields(entry.get("defaults"), f"guilds.{guild_id}.defaults"),
            "channels": {},
            "threads": {},
        }
        for kind in ("channels", "threads"):
            collection = entry.get(kind, {})
            if not isinstance(collection, Mapping):
                raise ScopePolicyValidationError(f"guilds.{guild_id}.{kind} must be a mapping")
            for key in sorted(collection, key=lambda item: str(item)):
                ident = _scope_id(key, f"guilds.{guild_id}.{kind} key")
                normalized[kind][ident] = _validate_fields(
                    collection[key], f"guilds.{guild_id}.{kind}.{ident}"
                )
        out["guilds"][guild_id] = normalized
    return out


def resolve_scope_policy(
    raw: Any,
    guild_id: Any,
    channel_id: Any,
    parent_channel_id: Any = None,
) -> ResolvedDiscordPolicy:
    """Resolve platform → guild → parent channel → concrete scope inheritance.

    ``parent_channel_id`` denotes a Discord thread's parent.  For a normal
    channel it should be omitted.  Unknown guilds/scopes simply return platform
    defaults (or all ``None`` when no defaults exist).
    """
    config = validate_scope_policies(raw)
    if config is None or guild_id is None:
        return ResolvedDiscordPolicy()
    values: dict[str, Any] = {field: None for field in _POLICY_FIELDS}
    sources: dict[str, str] = {}

    def apply(fields: Mapping[str, Any], source: str) -> None:
        for field in _POLICY_FIELDS:
            if field in fields:
                values[field] = fields[field]
                sources[field] = source

    apply(config["platform"], "platform")
    guild_key = None if guild_id is None else str(guild_id)
    guild = config["guilds"].get(guild_key)
    if guild is not None:
        apply(guild["defaults"], f"guild:{guild_key}")
        if parent_channel_id is not None:
            parent_key = str(parent_channel_id)
            apply(guild["channels"].get(parent_key, {}), f"channel:{parent_key}")
            thread_key = str(channel_id) if channel_id is not None else ""
            apply(guild["threads"].get(thread_key, {}), f"thread:{thread_key}")
        elif channel_id is not None:
            channel_key = str(channel_id)
            apply(guild["channels"].get(channel_key, {}), f"channel:{channel_key}")
    return ResolvedDiscordPolicy(
        require_mention=values["require_mention"],
        allow_bots=values["allow_bots"],
        allow_humans=values["allow_humans"],
        conversation_trust=values["conversation_trust"],
        sources=sources or None,
    )

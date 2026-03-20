"""Discord configuration and policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DiscordPolicyConfig:
    """Typed Discord policy snapshot for a single adapter instance."""
    allowed_users: set[str] = field(default_factory=set)
    bot_filter_policy: str = "none"
    free_response_channels: set[str] = field(default_factory=set)
    require_mention: bool = True
    auto_thread: bool = True


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    return bool(value)


def _normalize_bot_filter_policy(value: Any, *, default: str = "none") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"none", "mentions", "all"}:
            return normalized
    return default


def _split_entries(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [entry.strip() for entry in value.split(",") if entry.strip()]
    if isinstance(value, Iterable):
        entries: list[str] = []
        for entry in value:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                entries.append(text)
        return entries
    text = str(value).strip()
    return [text] if text else []


def clean_discord_id(entry: str) -> str:
    """Normalize a Discord user ID or username entry."""
    entry = entry.strip()
    if entry.startswith("<@") and entry.endswith(">"):
        entry = entry.lstrip("<@!").rstrip(">")
    if entry.lower().startswith("user:"):
        entry = entry[5:]
    return entry.strip()


def parse_allowed_users(value: Any) -> set[str]:
    """Parse Discord allowed-user config into cleaned entries."""
    return {
        clean_discord_id(entry)
        for entry in _split_entries(value)
        if clean_discord_id(entry)
    }


def parse_free_response_channels(value: Any) -> set[str]:
    """Parse Discord free-response channel config into channel IDs."""
    return {entry for entry in _split_entries(value) if entry}


def load_policy_config(
    config: Any | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> DiscordPolicyConfig:
    """Load a typed Discord policy snapshot from config.extra with env fallback."""
    env_map = env or os.environ
    override_map = overrides or {}
    extra = getattr(config, "extra", None)
    if not isinstance(extra, dict):
        extra = {}

    allowed_users = extra.get("allowed_users")
    if allowed_users is None:
        allowed_users = env_map.get("DISCORD_ALLOWED_USERS", "")

    bot_filter_policy = extra.get("allow_bots")
    if bot_filter_policy is None:
        bot_filter_policy = env_map.get("DISCORD_ALLOW_BOTS", "none")

    free_response_channels = extra.get("free_response_channels")
    if free_response_channels is None:
        free_response_channels = env_map.get("DISCORD_FREE_RESPONSE_CHANNELS", "")

    require_mention = extra.get("require_mention")
    if require_mention is None:
        require_mention = env_map.get("DISCORD_REQUIRE_MENTION")

    auto_thread = extra.get("auto_thread")
    if auto_thread is None:
        auto_thread = env_map.get("DISCORD_AUTO_THREAD")

    if "allowed_users" in override_map:
        allowed_users = override_map["allowed_users"]
    if "allow_bots" in override_map:
        bot_filter_policy = override_map["allow_bots"]
    if "free_response_channels" in override_map:
        free_response_channels = override_map["free_response_channels"]
    if "require_mention" in override_map:
        require_mention = override_map["require_mention"]
    if "auto_thread" in override_map:
        auto_thread = override_map["auto_thread"]

    return DiscordPolicyConfig(
        allowed_users=parse_allowed_users(allowed_users),
        bot_filter_policy=_normalize_bot_filter_policy(bot_filter_policy, default="none"),
        free_response_channels=parse_free_response_channels(free_response_channels),
        require_mention=_coerce_bool(require_mention, default=True),
        auto_thread=_coerce_bool(auto_thread, default=True),
    )


def get_bot_filter_policy(config: Any | None = None, *, env: Mapping[str, str] | None = None) -> str:
    """Return the effective bot-message filtering policy."""
    return load_policy_config(config, env=env).bot_filter_policy


def get_free_response_channels(
    config: Any | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> set[str]:
    """Return the effective free-response channel ID set."""
    return load_policy_config(config, env=env).free_response_channels


def is_mention_required(config: Any | None = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Return whether Discord server messages require an explicit mention."""
    return load_policy_config(config, env=env).require_mention


def is_auto_thread_enabled(config: Any | None = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Return whether Discord auto-threading is enabled."""
    return load_policy_config(config, env=env).auto_thread

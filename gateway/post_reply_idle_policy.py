"""Pure per-chat policy for post-reply idle compaction (no scheduling or I/O).

Rules use exact platform + stable chat_id. Optional thread_id outranks scope_id,
which outranks runtime profile, which outranks transport_profile. A chat rule
also covers its threads and every per-user session under that chat. The caller
passes effective raw config and the pinned routing identity, not session keys or
mutable display names. No matching rule (or a matching zero) means disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gateway.config import Platform

_RULE_FIELDS = frozenset({"platform", "chat_id", "after_seconds", "profile",
                          "transport_profile", "scope_id", "thread_id"})
_OPTIONAL_IDS = ("profile", "transport_profile", "scope_id", "thread_id")
_NON_CHAT_PLATFORMS = frozenset({"local", "api_server", "webhook", "msgraph_webhook", "relay"})


def _stable_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def validate_post_reply_idle_policy(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return validated rules or raise ValueError with the offending config path."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    compression = config.get("compression", {})
    if not isinstance(compression, Mapping):
        raise ValueError("compression must be a mapping")
    policy = compression.get("post_reply_idle", {})
    if not isinstance(policy, Mapping) or set(policy) - {"channels"}:
        raise ValueError("compression.post_reply_idle must contain only channels")
    channels = policy.get("channels", [])
    if not isinstance(channels, list):
        raise ValueError("compression.post_reply_idle.channels must be a list")
    seen: set[tuple[Any, ...]] = set()
    rules = []
    for index, rule in enumerate(channels):
        path = f"compression.post_reply_idle.channels[{index}]"
        if not isinstance(rule, Mapping) or set(rule) - _RULE_FIELDS:
            raise ValueError(f"{path}: expected a rule with supported fields only")
        platform = rule.get("platform")
        if not isinstance(platform, str) or not _stable_id(platform) or platform != platform.lower():
            raise ValueError(f"{path}.platform must be a canonical platform name")
        try:
            Platform(platform)
        except ValueError as exc:
            raise ValueError(f"{path}.platform is not a known platform") from exc
        if platform in _NON_CHAT_PLATFORMS:
            raise ValueError(f"{path}.platform must be a messaging platform")
        if not _stable_id(rule.get("chat_id")):
            raise ValueError(f"{path}.chat_id must be a nonempty stable string ID")
        duration = rule.get("after_seconds")
        if type(duration) is not int or duration < 0:
            raise ValueError(f"{path}.after_seconds must be a nonnegative integer")
        for field in _OPTIONAL_IDS:
            if field in rule and not _stable_id(rule[field]):
                raise ValueError(f"{path}.{field} must be a nonempty stable string ID")
        key = (platform, rule["chat_id"], *(rule.get(field) for field in _OPTIONAL_IDS))
        if key in seen:
            raise ValueError(f"duplicate post_reply_idle.channels rule at {path}")
        seen.add(key)
        rules.append(dict(rule))
    return rules


def resolve_post_reply_idle_policy(config: Mapping[str, Any], source: Any, identity: Any) -> int | None:
    """Resolve seconds for a pinned gateway source, or None when disabled/unmatched.

    Validates the whole policy on every call, including rules for other chats,
    so malformed entries cannot silently depend on which message arrived first.
    """
    rules = validate_post_reply_idle_policy(config)
    if config.get("compression", {}).get("enabled", True) is False or source is None or identity is None:
        return None
    platform = getattr(source, "platform", None)
    platform = getattr(platform, "value", platform)
    if platform in _NON_CHAT_PLATFORMS:
        return None
    scope = getattr(source, "scope_id", None) or getattr(source, "guild_id", None)
    thread = getattr(source, "thread_id", None)
    parent = getattr(source, "parent_chat_id", None)
    if parent and not thread:
        thread = getattr(source, "chat_id", None)
    chat = parent or getattr(source, "chat_id", None)
    best: tuple[tuple[bool, ...], int] | None = None
    for rule in rules:
        if rule["platform"] != platform or rule["chat_id"] != chat:
            continue
        selectors = (("thread_id", thread), ("scope_id", scope),
                     ("profile", getattr(identity, "runtime_profile", None)),
                     ("transport_profile", getattr(identity, "transport_profile", None)))
        if any(field in rule and rule[field] != value for field, value in selectors):
            continue
        rank = tuple(field in rule for field, _ in selectors)
        if best is None or rank > best[0]:
            best = (rank, rule["after_seconds"])
    return best[1] or None if best is not None else None

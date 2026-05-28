"""Transactional gateway permission snapshots.

This module keeps permission reload narrow: it models only authorization and
platform permission knobs, not model/tool/session/runtime configuration.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Mapping

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.session import SessionSource


def _string_set(raw: Any) -> frozenset[str]:
    """Normalize list/tuple/set/CSV/single values to a frozenset of strings."""
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = raw
    else:
        values = [raw]
    return frozenset(str(value).strip() for value in values if str(value).strip())


def _bool_value(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes", "on", "all"}
    return bool(raw)


def _extra_or_env_set(extra: Mapping[str, Any], key: str, env_name: str) -> frozenset[str]:
    """Use explicit config values over env fallback, even when empty.

    ``load_gateway_config`` bridges YAML values into ``os.environ`` for legacy
    callers. During hot reload, those bridged env vars can be stale after an
    operator removes an allowlist entry from YAML. Presence in ``extra`` means
    the config loader saw an explicit current value, so do not union the env.
    """
    if key in extra:
        return _string_set(extra.get(key))
    return _string_set(os.getenv(env_name))


def _extra_or_env_value(extra: Mapping[str, Any], key: str, env_name: str) -> Any:
    if key in extra:
        return extra.get(key)
    return os.getenv(env_name)


def _pattern_values(raw: Any) -> tuple[str, ...]:
    """Normalize mention pattern config, including JSON env values."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            if "\n" in text:
                values = text.splitlines()
            else:
                values = text.split(",")
        else:
            values = loaded if isinstance(loaded, list) else [loaded]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = raw
    else:
        values = [raw]
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class PlatformPermissionSnapshot:
    """Immutable permission state for one gateway platform."""

    platform: Platform
    approved_users: Any = field(default_factory=frozenset)
    allowed_users: Any = field(default_factory=frozenset)
    group_allowed_users: Any = field(default_factory=frozenset)
    allowed_chats: Any = field(default_factory=frozenset)
    group_allowed_chats: Any = field(default_factory=frozenset)
    allowed_topics: Any = field(default_factory=frozenset)
    free_response_chats: Any = field(default_factory=frozenset)
    mention_patterns: Any = field(default_factory=tuple)
    allow_all: bool = False
    allow_bots: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)
    compiled_mention_patterns: tuple[re.Pattern[str], ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "approved_users", _string_set(self.approved_users))
        object.__setattr__(self, "allowed_users", _string_set(self.allowed_users))
        object.__setattr__(self, "group_allowed_users", _string_set(self.group_allowed_users))
        object.__setattr__(self, "allowed_chats", _string_set(self.allowed_chats))
        object.__setattr__(self, "group_allowed_chats", _string_set(self.group_allowed_chats))
        object.__setattr__(self, "allowed_topics", _string_set(self.allowed_topics))
        object.__setattr__(self, "free_response_chats", _string_set(self.free_response_chats))
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))
        patterns = _pattern_values(self.mention_patterns)
        object.__setattr__(self, "mention_patterns", patterns)
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise ValueError(f"invalid mention pattern {pattern!r}: {exc}") from exc
        object.__setattr__(self, "compiled_mention_patterns", tuple(compiled))


@dataclass(frozen=True)
class PermissionSnapshot:
    """Immutable snapshot of all loaded gateway permissions."""

    version: int
    loaded_at: datetime
    platforms: Mapping[Platform, PlatformPermissionSnapshot]
    config: GatewayConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "platforms", MappingProxyType(dict(self.platforms)))


@dataclass(frozen=True)
class ReloadResult:
    ok: bool
    reason: str
    snapshot: PermissionSnapshot | None = None


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    reason: str
    platform: Platform | None
    user_id: str | None
    chat_id: str | None


class PermissionManager:
    """Load, validate, and atomically swap gateway permission snapshots."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], GatewayConfig] = load_gateway_config,
        pairing_store: Any = None,
    ) -> None:
        self._config_loader = config_loader
        self._pairing_store = pairing_store
        self._snapshot: PermissionSnapshot | None = None
        self._version = 0

    @property
    def snapshot(self) -> PermissionSnapshot:
        if self._snapshot is None:
            result = self.reload()
            if not result.ok:
                raise RuntimeError(result.reason)
        assert self._snapshot is not None
        return self._snapshot

    def reload(self) -> ReloadResult:
        """Build a candidate snapshot and swap it in only after validation."""
        try:
            candidate = self.load_candidate()
        except Exception as exc:
            return ReloadResult(False, str(exc), self._snapshot)
        self.commit(candidate)
        return ReloadResult(True, "permissions reloaded", candidate)

    def load_candidate(self) -> PermissionSnapshot:
        """Build and validate a snapshot without changing active permissions."""
        return self._build_snapshot()

    def commit(self, snapshot: PermissionSnapshot) -> None:
        """Make a previously validated snapshot active."""
        self._snapshot = snapshot

    def authorize(self, source: SessionSource) -> AuthDecision:
        """Return an authorization decision for source using the active snapshot."""
        platform = source.platform
        user_id = str(source.user_id).strip() if source.user_id else None
        chat_id = str(source.chat_id).strip() if source.chat_id else None

        if platform in {Platform.HOMEASSISTANT, Platform.WEBHOOK}:
            return AuthDecision(True, "system_platform", platform, user_id, chat_id)

        platform_snapshot = self.snapshot.platforms.get(platform) if platform else None
        if platform_snapshot is None:
            return AuthDecision(False, "platform_not_snapshotted", platform, user_id, chat_id)

        if source.chat_type in {"group", "forum", "channel"} and chat_id:
            if "*" in platform_snapshot.group_allowed_chats or chat_id in platform_snapshot.group_allowed_chats:
                return AuthDecision(True, "group_allowed_chat", platform, user_id, chat_id)

        if not user_id:
            return AuthDecision(False, "missing_user_id", platform, user_id, chat_id)

        if platform_snapshot.allow_all:
            return AuthDecision(True, "allow_all", platform, user_id, chat_id)

        if getattr(source, "is_bot", False) and platform_snapshot.allow_bots:
            return AuthDecision(True, "bot_allowed", platform, user_id, chat_id)

        if user_id in platform_snapshot.approved_users:
            return AuthDecision(True, "approved_user", platform, user_id, chat_id)

        if source.chat_type in {"group", "forum"}:
            if user_id in platform_snapshot.group_allowed_users:
                return AuthDecision(True, "group_allowed_user", platform, user_id, chat_id)
            if chat_id and ("*" in platform_snapshot.group_allowed_chats or chat_id in platform_snapshot.group_allowed_chats):
                return AuthDecision(True, "group_allowed_chat", platform, user_id, chat_id)

        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@", 1)[0])
        if "*" in platform_snapshot.allowed_users or check_ids & platform_snapshot.allowed_users:
            return AuthDecision(True, "allowed_user", platform, user_id, chat_id)

        return AuthDecision(False, "not_allowed", platform, user_id, chat_id)

    def _build_snapshot(self) -> PermissionSnapshot:
        config = self._config_loader()
        if not isinstance(config, GatewayConfig):
            raise TypeError("permission reload expected GatewayConfig")
        self._version += 1
        platforms = {
            platform: self._platform_snapshot(platform, platform_config)
            for platform, platform_config in config.platforms.items()
        }
        return PermissionSnapshot(
            version=self._version,
            loaded_at=datetime.now(timezone.utc),
            platforms=platforms,
            config=config,
        )

    def _platform_snapshot(
        self,
        platform: Platform,
        platform_config: PlatformConfig,
    ) -> PlatformPermissionSnapshot:
        extra = dict(platform_config.extra or {})
        approved_users = self._approved_users(platform)
        if platform == Platform.TELEGRAM:
            return self._telegram_snapshot(extra, approved_users)
        return PlatformPermissionSnapshot(platform=platform, approved_users=approved_users, extra=extra)

    def _approved_users(self, platform: Platform) -> frozenset[str]:
        if self._pairing_store is None:
            return frozenset()
        try:
            rows = self._pairing_store.list_approved(platform.value)
        except Exception as exc:
            raise ValueError(f"failed to load approved users for {platform.value}: {exc}") from exc
        return frozenset(str(row.get("user_id", "")).strip() for row in rows if str(row.get("user_id", "")).strip())

    def _telegram_snapshot(
        self,
        extra: dict[str, Any],
        approved_users: frozenset[str],
    ) -> PlatformPermissionSnapshot:
        group_allowed_users = _extra_or_env_set(
            extra, "group_allow_from", "TELEGRAM_GROUP_ALLOWED_USERS"
        )
        return PlatformPermissionSnapshot(
            platform=Platform.TELEGRAM,
            approved_users=approved_users,
            allowed_users=(
                _extra_or_env_set(extra, "allow_from", "TELEGRAM_ALLOWED_USERS")
                | _string_set(os.getenv("GATEWAY_ALLOWED_USERS"))
            ),
            group_allowed_users=group_allowed_users,
            allowed_chats=_extra_or_env_set(extra, "allowed_chats", "TELEGRAM_ALLOWED_CHATS"),
            group_allowed_chats=(
                _extra_or_env_set(extra, "group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS")
                | frozenset(value for value in group_allowed_users if value.startswith("-"))
            ),
            allowed_topics=_extra_or_env_set(extra, "allowed_topics", "TELEGRAM_ALLOWED_TOPICS"),
            free_response_chats=_extra_or_env_set(
                extra, "free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS"
            ),
            mention_patterns=_extra_or_env_value(extra, "mention_patterns", "TELEGRAM_MENTION_PATTERNS"),
            allow_all=(
                _bool_value(_extra_or_env_value(extra, "allow_all_users", "TELEGRAM_ALLOW_ALL_USERS"), False)
                or _bool_value(os.getenv("GATEWAY_ALLOW_ALL_USERS"), False)
            ),
            allow_bots=_bool_value(extra.get("allow_bots"), False),
            extra=extra,
        )

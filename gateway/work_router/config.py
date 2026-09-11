"""Fail-closed configuration and logical bot registry for Work Router."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CHANNEL_ID = "C0BN14PKL9E"
CANONICAL_MEETING_CHANNEL_ID = "C0BDCURRS9W"
DEFAULT_SINCLAIR_USER_ID = "U0BC6GDPRAQ"
DEFAULT_MEETING_CHANNEL_ALLOWLIST = (CANONICAL_MEETING_CHANNEL_ID,)
MAX_MEETING_ROUNDS = 5
MIN_MEETING_PARTICIPANTS = 3
MAX_MEETING_PARTICIPANTS = 4
PARTICIPANT_TIMEOUT_SECONDS = 90.0
_SLACK_CHANNEL_ID_RE = re.compile(r"C[A-Z0-9]+\Z")
_SLACK_USER_ID_RE = re.compile(r"U[A-Z0-9]+\Z")
DEFAULT_PROFILES = (
    "Demian",
    "Hans",
    "Wendy",
    "Tesla",
    "Turing",
    "Mason",
    "Watson",
)


class RouterConfigError(ValueError):
    """Raised when an enabled Router configuration is not exact and safe."""


@dataclass(frozen=True)
class BotProfile:
    name: str
    slack_user_id: str


@dataclass(frozen=True)
class BotRegistry:
    """The seven logical profiles and their injected Slack bot identities."""

    profiles: tuple[BotProfile, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(profile.name for profile in self.profiles)

    @property
    def default(self) -> str:
        return DEFAULT_PROFILES[0]

    def by_name(self, name: str) -> BotProfile | None:
        return next((p for p in self.profiles if p.name == name), None)

    def by_user_id(self, user_id: str) -> BotProfile | None:
        return next((p for p in self.profiles if p.slack_user_id == user_id), None)

    def user_id_for(self, name: str) -> str | None:
        profile = self.by_name(name)
        return profile.slack_user_id if profile else None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BotRegistry":
        if not isinstance(value, Mapping):
            raise RouterConfigError("Router bot registry is missing or not a mapping")
        if tuple(value.keys()) != DEFAULT_PROFILES and set(value) != set(DEFAULT_PROFILES):
            raise RouterConfigError(
                "Router bot registry must contain exactly Demian, Hans, Wendy, "
                "Tesla, Turing, Mason, and Watson"
            )
        profiles: list[BotProfile] = []
        user_ids: list[str] = []
        for name in DEFAULT_PROFILES:
            raw = value.get(name)
            if isinstance(raw, Mapping):
                raw = raw.get("slack_user_id")
            if not isinstance(raw, str) or not raw.strip():
                raise RouterConfigError(f"Slack user ID for {name} is missing")
            user_id = raw.strip()
            if user_id in user_ids:
                raise RouterConfigError("Router bot registry contains duplicate Slack user IDs")
            user_ids.append(user_id)
            profiles.append(BotProfile(name=name, slack_user_id=user_id))
        return cls(tuple(profiles))

    def to_dict(self) -> dict[str, str]:
        return {profile.name: profile.slack_user_id for profile in self.profiles}


@dataclass(frozen=True)
class WorkExecutionConfig:
    """Explicit scope for the reserved Demian directive, disabled by default."""

    enabled: bool = False
    ready: bool = False
    channel_allowlist: tuple[str, ...] = ()
    slack_team_id: str = ""
    max_body_bytes: int = 0

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, router_channels: tuple[str, ...]
    ) -> "WorkExecutionConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise RouterConfigError("work_execution must be a mapping")
        if set(value) - {"enabled", "channel_allowlist", "slack_team_id", "max_body_bytes"}:
            raise RouterConfigError("work_execution contains unknown settings")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise RouterConfigError("work_execution.enabled must be boolean")
        if not enabled:
            return cls()
        raw_channels = value.get("channel_allowlist")
        if not isinstance(raw_channels, (list, tuple)):
            raise RouterConfigError("work_execution.channel_allowlist must be a list")
        try:
            channels = _channel_allowlist(raw_channels)
        except RouterConfigError as exc:
            raise RouterConfigError(f"work_execution: {exc}") from exc
        if not set(channels).issubset(router_channels):
            raise RouterConfigError("work_execution channels must be owned by the Router")
        team_id = value.get("slack_team_id")
        if not isinstance(team_id, str) or re.fullmatch(r"T[A-Z0-9]+", team_id) is None:
            raise RouterConfigError("work_execution.slack_team_id must be explicit")
        max_body_bytes = value.get("max_body_bytes")
        if type(max_body_bytes) is not int or max_body_bytes <= 0:
            raise RouterConfigError("work_execution.max_body_bytes must be a positive integer")
        return cls(
            enabled=True, ready=True, channel_allowlist=channels,
            slack_team_id=team_id, max_body_bytes=max_body_bytes,
        )


@dataclass(frozen=True)
class RouterConfig:
    """Opt-in Work Router settings.

    ``from_mapping`` never turns malformed enabled input into a usable partial
    configuration.  It returns ``ready=False`` so callers can fail closed
    without guessing IDs or falling back to a different channel.
    """

    enabled: bool = False
    mode: str = "work"
    channel_allowlist: tuple[str, ...] = ()
    sinclair_user_id: str = DEFAULT_SINCLAIR_USER_ID
    meeting_channel_allowlist: tuple[str, ...] = DEFAULT_MEETING_CHANNEL_ALLOWLIST
    max_meeting_rounds: int = MAX_MEETING_ROUNDS
    min_meeting_participants: int = MIN_MEETING_PARTICIPANTS
    max_meeting_participants: int = MAX_MEETING_PARTICIPANTS
    participant_timeout_seconds: float = PARTICIPANT_TIMEOUT_SECONDS
    registry: BotRegistry | None = None
    db_path: Path | None = None
    thread_ttl_seconds: float = 24 * 60 * 60
    lease_seconds: float = 30.0
    completion_ttl_seconds: float = 7 * 24 * 60 * 60
    max_processing_attempts: int = 3
    lounge_spontaneous_enabled: bool = False
    work_execution: WorkExecutionConfig = field(default_factory=WorkExecutionConfig)
    ready: bool = False
    error: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RouterConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            return cls(enabled=True, error="Router configuration is not a mapping")
        enabled = value.get("enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
        else:
            enabled = bool(enabled)
        if not enabled:
            return cls(enabled=False)
        try:
            mode = str(value.get("mode", "work")).strip().lower()
            raw_channels = value.get("channel_allowlist")
            if not isinstance(raw_channels, (list, tuple)):
                raise RouterConfigError("Router channel allowlist is missing or not a list")
            channels = _channel_allowlist(raw_channels)
            if mode != "work":
                raise RouterConfigError("Only Work Router mode is supported")
            raw_sinclair_user_id = value.get(
                "sinclair_user_id",
                DEFAULT_SINCLAIR_USER_ID,
            )
            if not isinstance(raw_sinclair_user_id, str):
                raise RouterConfigError("Sinclair Slack user ID is invalid")
            sinclair_user_id = raw_sinclair_user_id.strip()
            if _SLACK_USER_ID_RE.fullmatch(sinclair_user_id) is None:
                raise RouterConfigError("Sinclair Slack user ID is invalid")
            registry = BotRegistry.from_mapping(value.get("bot_registry"))
            db_path_raw = value.get("db_path")
            db_path = Path(db_path_raw).expanduser() if db_path_raw else None
            if db_path is None:
                raise RouterConfigError("Router db_path is required when enabled")
            ttl = _positive_number(value.get("thread_ttl_seconds", 24 * 60 * 60), "thread_ttl_seconds")
            lease = _positive_number(value.get("lease_seconds", 30.0), "lease_seconds")
            completion_ttl = _positive_number(
                value.get("completion_ttl_seconds", 7 * 24 * 60 * 60),
                "completion_ttl_seconds",
            )
            max_attempts = _positive_integer(
                value.get("max_processing_attempts", 3),
                "max_processing_attempts",
            )
            lounge_raw = value.get("lounge_spontaneous", {})
            if lounge_raw is None:
                lounge_raw = {}
            if not isinstance(lounge_raw, dict):
                raise RouterConfigError("lounge_spontaneous must be a mapping")
            lounge_spontaneous_enabled = lounge_raw.get("enabled", False)
            if not isinstance(lounge_spontaneous_enabled, bool):
                raise RouterConfigError("lounge_spontaneous.enabled must be boolean")
            return cls(
                enabled=True,
                mode=mode,
                channel_allowlist=channels,
                sinclair_user_id=sinclair_user_id,
                registry=registry,
                db_path=db_path,
                thread_ttl_seconds=ttl,
                lease_seconds=lease,
                completion_ttl_seconds=completion_ttl,
                max_processing_attempts=max_attempts,
                lounge_spontaneous_enabled=lounge_spontaneous_enabled,
                work_execution=WorkExecutionConfig.from_mapping(
                    value.get("work_execution"), router_channels=channels,
                ),
                ready=True,
            )
        except RouterConfigError as exc:
            return cls(enabled=True, error=str(exc))
        except (TypeError, ValueError, OSError) as exc:
            return cls(enabled=True, error=f"Invalid Router configuration: {exc}")

    def require_ready(self) -> "RouterConfig":
        if not self.enabled:
            raise RouterConfigError("Work Router is disabled")
        if not self.ready or self.registry is None or self.db_path is None:
            raise RouterConfigError(self.error or "Work Router configuration is not ready")
        return self

    def owns_channel(self, channel_id: str, channel_type: str = "channel") -> bool:
        if not self.enabled or not self.ready:
            return False
        if channel_type.strip().lower() in {"dm", "im", "mpim", "app_home", "home", "workspace"}:
            return False
        # Slack channel_type is absent in a few envelope variants.  D/G IDs
        # are still private DM/group-DM identifiers and must never enter the
        # Work Router on that fallback path.
        if channel_id.startswith(("D", "G")):
            return False
        return channel_id in self.channel_allowlist

    def owns_meeting_channel(self, channel_id: str, channel_type: str = "channel") -> bool:
        """Return true only for an approved meeting source channel."""

        return (
            self.owns_channel(channel_id, channel_type)
            and channel_id in self.meeting_channel_allowlist
        )

    def authorizes_meeting_control(
        self,
        channel_id: str,
        user_id: str | None,
        channel_type: str = "channel",
    ) -> bool:
        """Fail closed unless Sinclair controls an approved meeting channel."""

        return self.owns_meeting_channel(channel_id, channel_type) and user_id == self.sinclair_user_id


def _channel_allowlist(value: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
    if not value:
        raise RouterConfigError("Router channel allowlist must not be empty")
    channels: list[str] = []
    for raw_channel in value:
        if not isinstance(raw_channel, str):
            raise RouterConfigError("Router channel allowlist entries must be strings")
        channel = raw_channel.strip()
        if _SLACK_CHANNEL_ID_RE.fullmatch(channel) is None:
            raise RouterConfigError(
                "Router channel allowlist entries must be explicit Slack channel IDs"
            )
        if channel in channels:
            raise RouterConfigError("Router channel allowlist contains duplicate channel IDs")
        channels.append(channel)
    return tuple(channels)


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise RouterConfigError(f"{field_name} must be positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RouterConfigError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise RouterConfigError(f"{field_name} must be positive")
    return parsed


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise RouterConfigError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RouterConfigError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise RouterConfigError(f"{field_name} must be a positive integer")
    return parsed

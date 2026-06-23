"""
Channel context models.

A ``ChannelContext`` captures metadata about the communication channel
(platform, thread, user, etc.) through which an agent request arrives.
It is intentionally kept as a plain dataclass so it can be serialised to /
deserialised from JSON without any heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ChannelType(str, Enum):
    """Supported channel / platform types."""

    SLACK = "slack"
    FEISHU = "feishu"
    EMAIL = "email"
    API = "api"
    CLI = "cli"
    UNKNOWN = "unknown"


@dataclass
class ChannelContext:
    """Metadata describing the channel from which a request originates.

    Attributes
    ----------
    channel_type:
        The kind of platform / transport (e.g. Slack, Feishu, …).
    channel_id:
        A platform-specific identifier for the channel or conversation
        (e.g. a Slack channel ID such as ``C01234567``).
    thread_id:
        Optional identifier for a thread within the channel.  For
        platforms that do not have threads this should be ``None``.
    user_id:
        The platform-specific identifier of the user who sent the
        message.
    user_display_name:
        Human-readable name of the user, if available.
    message_id:
        Platform-specific identifier of the triggering message.
    extra:
        Arbitrary additional key/value pairs that a specific platform
        adapter may need to propagate (e.g. workspace ID, bot token
        alias, …).  Consumers should treat unknown keys as advisory.
    """

    channel_type: ChannelType = ChannelType.UNKNOWN
    channel_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    user_display_name: Optional[str] = None
    message_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "channel_type": self.channel_type.value,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "user_display_name": self.user_display_name,
            "message_id": self.message_id,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelContext":
        """Construct a :class:`ChannelContext` from a plain dictionary.

        Unknown keys inside *data* are silently ignored so that older
        serialised payloads remain forward-compatible.
        """
        raw_type = data.get("channel_type", ChannelType.UNKNOWN.value)
        try:
            channel_type = ChannelType(raw_type)
        except ValueError:
            channel_type = ChannelType.UNKNOWN

        return cls(
            channel_type=channel_type,
            channel_id=data.get("channel_id"),
            thread_id=data.get("thread_id"),
            user_id=data.get("user_id"),
            user_display_name=data.get("user_display_name"),
            message_id=data.get("message_id"),
            extra=dict(data.get("extra") or {}),
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_slack(
        cls,
        *,
        channel_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        user_display_name: Optional[str] = None,
        **extra: Any,
    ) -> "ChannelContext":
        """Shorthand constructor for Slack channels."""
        return cls(
            channel_type=ChannelType.SLACK,
            channel_id=channel_id,
            thread_id=thread_id,
            user_id=user_id,
            user_display_name=user_display_name,
            message_id=message_id,
            extra=extra,
        )

    @classmethod
    def for_feishu(
        cls,
        *,
        channel_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        user_display_name: Optional[str] = None,
        **extra: Any,
    ) -> "ChannelContext":
        """Shorthand constructor for Feishu / Lark channels."""
        return cls(
            channel_type=ChannelType.FEISHU,
            channel_id=channel_id,
            thread_id=thread_id,
            user_id=user_id,
            user_display_name=user_display_name,
            message_id=message_id,
            extra=extra,
        )

    @classmethod
    def for_cli(cls, *, user_id: Optional[str] = None, **extra: Any) -> "ChannelContext":
        """Shorthand constructor for CLI invocations."""
        return cls(
            channel_type=ChannelType.CLI,
            user_id=user_id,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def is_threaded(self) -> bool:
        """Return ``True`` when the context carries a thread identifier."""
        return self.thread_id is not None

    def __repr__(self) -> str:  # pragma: no cover
        parts = [f"channel_type={self.channel_type.value!r}"]
        if self.channel_id:
            parts.append(f"channel_id={self.channel_id!r}")
        if self.thread_id:
            parts.append(f"thread_id={self.thread_id!r}")
        if self.user_id:
            parts.append(f"user_id={self.user_id!r}")
        return f"ChannelContext({', '.join(parts)})"

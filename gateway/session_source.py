"""Message-origin identity shared by gateway routing and persistence."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import Platform


@dataclass
class SessionSource:
    """
    Describes where a message originated from.

    This information is used to:
    1. Route responses back to the right place
    2. Inject context into the system prompt
    3. Track origin for cron job delivery
    """

    platform: Platform
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"  # "dm", "group", "channel", "thread"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_id: Optional[str] = None  # For forum topics, Discord threads, etc.
    chat_topic: Optional[str] = None  # Channel topic/description (Discord, Slack)
    user_id_alt: Optional[str] = (
        None  # Platform-specific stable alt ID (Signal UUID, Feishu union_id)
    )
    chat_id_alt: Optional[str] = None  # Signal group internal ID
    is_bot: bool = False  # True when the message author is a bot/webhook (Discord)
    # Platform-neutral SCOPE discriminator (Discord guild / Slack workspace /
    # Matrix server). Drives server/workspace isolation + the relay delta/epsilon/zeta gate.
    # Wire migration (D-Q2.5): `scope_id` is the canonical name; `guild_id` is a
    # deprecated legacy alias kept during the cross-repo dual-read/dual-write
    # overlap. Both are written by to_dict and read by from_dict (scope_id wins);
    # the `guild_id` alias is dropped in a follow-up once both repos deploy.
    scope_id: Optional[str] = None
    guild_id: Optional[str] = None  # @deprecated legacy alias for scope_id (D-Q2.5)
    parent_chat_id: Optional[str] = (
        None  # Parent channel when chat_id refers to a thread
    )
    message_id: Optional[str] = (
        None  # ID of the triggering message (for pin/reply/react)
    )
    role_authorized: bool = (
        False  # True when adapter granted access via role (not user ID)
    )
    # Profile this inbound message is routed to in a multiplexing gateway
    # (from the /p/<profile>/ URL prefix or per-credential adapter ownership).
    # None => the gateway's active/default profile. Drives both session-key
    # namespacing and the per-turn config/credential scope.
    profile: Optional[str] = None
    # Transport-local fail-closed signal for an explicit profile route whose
    # target is not served. Excluded from repr/equality and wire serialization.
    profile_route_rejected: bool = field(default=False, repr=False, compare=False)
    # Transport-local trust facts; never deserialize these from stored/wire data.
    is_one_to_one: Optional[bool] = field(default=None, repr=False, compare=False)
    message_is_edit: bool = field(default=False, repr=False, compare=False)
    message_had_attachments: bool = field(default=False, repr=False, compare=False)

    # Discord auto-thread metadata. Newly auto-created Discord threads start
    # with a fast placeholder title from the raw message, then the gateway can
    # rename them after the first agent turn using the generated session title.
    # Keep this explicit so pre-existing or human-renamed threads are not
    # mistaken for safe rename targets.
    auto_thread_created: bool = False
    auto_thread_initial_name: Optional[str] = None

    # Discord auto-thread session-continuity signal. Set by the connector on an
    # inbound CHANNEL message (no thread_id yet) that its auto-thread policy WILL
    # deliver into a newly-created thread. A Discord thread created from a message
    # reuses that message's id as the thread id, so the connector knows the id
    # before the thread exists. The gateway keys the session on this so a
    # channel message and its thread follow-ups share ONE session: the channel
    # message INITIATES it (keyed on the prospective thread id), and later
    # messages arriving in that thread (real thread_id == this value) CONTINUE
    # it. Without this, every channel message collapses into one parent-channel
    # session and only the first auto-thread ever gets an auto-title/rename.
    prospective_thread_id: Optional[str] = None

    # Internal, wire-INVISIBLE trust signal: True when this event was delivered
    # to the gateway over the per-instance-authenticated relay WebSocket (the
    # Team Gateway connector). The connector authenticates the gateway's socket
    # with a per-instance secret and resolves owner-only author bindings BEFORE
    # delivering, so a relay-delivered event is already authorized as this
    # instance's bound user. `platform` carries the UNDERLYING platform (for
    # example, `discord`) for session-keying/egress, not `relay`, so authz must
    # key the upstream-trust decision off this flag rather than `platform`.
    # Set locally by the relay transport; deliberately excluded from
    # `to_dict`/`from_dict` so a peer can never forge it across the wire or have
    # it restored from persistence.
    delivered_via_upstream_relay: bool = False

    def __post_init__(self) -> None:
        # D-Q2.5 dual-field reconciliation: `scope_id` is canonical, `guild_id`
        # is the deprecated alias. Mirror whichever was provided onto the other
        # (scope_id wins on conflict) so internal readers of either field see the
        # same value during the cross-repo wire migration overlap.
        if self.scope_id is None and self.guild_id is not None:
            self.scope_id = self.guild_id
        elif self.scope_id is not None:
            self.guild_id = self.scope_id

    @property
    def description(self) -> str:
        """Human-readable description of the source."""
        if self.platform == Platform.LOCAL:
            return "CLI terminal"

        parts = []
        if self.chat_type == "dm":
            parts.append(f"DM with {self.user_name or self.user_id or 'user'}")
        elif self.chat_type == "group":
            parts.append(f"group: {self.chat_name or self.chat_id}")
        elif self.chat_type == "channel":
            parts.append(f"channel: {self.chat_name or self.chat_id}")
        else:
            parts.append(self.chat_name or self.chat_id)

        if self.thread_id:
            parts.append(f"thread: {self.thread_id}")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "thread_id": self.thread_id,
            "chat_topic": self.chat_topic,
            "is_bot": self.is_bot,
        }
        if self.user_id_alt:
            d["user_id_alt"] = self.user_id_alt
        if self.chat_id_alt:
            d["chat_id_alt"] = self.chat_id_alt
        # D-Q2.5 dual-write: emit both the canonical `scope_id` and deprecated
        # `guild_id` alias so a connector on either migration side resolves it.
        scope = self.scope_id if self.scope_id is not None else self.guild_id
        if scope:
            d["scope_id"] = scope
            d["guild_id"] = scope
        if self.parent_chat_id:
            d["parent_chat_id"] = self.parent_chat_id
        if self.message_id:
            d["message_id"] = self.message_id
        if self.profile:
            d["profile"] = self.profile
        if self.auto_thread_created:
            d["auto_thread_created"] = True
        if self.auto_thread_initial_name:
            d["auto_thread_initial_name"] = self.auto_thread_initial_name
        if self.prospective_thread_id:
            d["prospective_thread_id"] = self.prospective_thread_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSource":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            chat_name=data.get("chat_name"),
            chat_type=data.get("chat_type", "dm"),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            thread_id=data.get("thread_id"),
            chat_topic=data.get("chat_topic"),
            user_id_alt=data.get("user_id_alt"),
            chat_id_alt=data.get("chat_id_alt"),
            scope_id=data.get("scope_id", data.get("guild_id")),
            parent_chat_id=data.get("parent_chat_id"),
            message_id=data.get("message_id"),
            is_bot=bool(data.get("is_bot", False)),
            profile=data.get("profile"),
            auto_thread_created=bool(data.get("auto_thread_created", False)),
            auto_thread_initial_name=data.get("auto_thread_initial_name"),
            prospective_thread_id=data.get("prospective_thread_id"),
        )

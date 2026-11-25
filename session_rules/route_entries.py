"""The routing-entry rules the store applies when a branch or a reset publishes a local route: the message source, the routing-index entry and its JSON codec, the session key and the clock.

``Platform`` has no plugin-platform lookup (it reads no plugin registry or manifest), so only a member's exact value converts to a ``Platform``. ``build_session_key`` and ``_canonical_participant`` read no WhatsApp bridge mapping files, so a WhatsApp source keys by its ids as given. No ``from __future__ import annotations``: the dataclass field types are evaluated objects, not strings. Standard library only.
"""

import logging
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any, Optional


logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported messaging platforms."""
    LOCAL = "local"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    WHATSAPP_CLOUD = "whatsapp_cloud"
    SLACK = "slack"
    SIGNAL = "signal"
    MATTERMOST = "mattermost"
    MATRIX = "matrix"
    EMAIL = "email"
    SMS = "sms"
    DINGTALK = "dingtalk"
    API_SERVER = "api_server"
    WEBHOOK = "webhook"
    MSGRAPH_WEBHOOK = "msgraph_webhook"
    FEISHU = "feishu"
    WECOM = "wecom"
    WECOM_CALLBACK = "wecom_callback"
    WEIXIN = "weixin"
    BLUEBUBBLES = "bluebubbles"
    QQBOT = "qqbot"
    YUANBAO = "yuanbao"
    RELAY = "relay"  # generic relay adapter fronted by the connector (EXPERIMENTAL)


def _is_path_unsafe(value: object, *, strict: bool = True) -> bool:
    """True if ``value`` could traverse outside the sessions dir.

    Session ids become filenames, so the strict form rejects ``..``, ANY path separator, and a
    leading Windows drive letter. ``strict=False`` is for *logical* session keys, where interior
    ``/`` is legitimate (Google Chat ``spaces/<id>/threads/<id>``): only a *leading* one is refused.
    """
    if not value:
        return False
    s = str(value)
    if ".." in s or (strict and ("/" in s or "\\" in s)):
        return True
    if not strict and s.startswith(("/", "\\")):
        return True
    return len(s) >= 2 and s[0].isalpha() and s[1] == ":"


_CHAT_TYPE_PREFIX = {"group": "group: ", "channel": "channel: "}


@dataclass
class SessionSource:
    """Where a message originated: routes responses, feeds the system-prompt
    context block, and records origin for cron delivery."""
    platform: Platform
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"  # "dm", "group", "channel", "thread"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_id: Optional[str] = None  # forum topics, Discord threads, etc.
    chat_topic: Optional[str] = None  # channel topic/description (Discord, Slack)
    user_id_alt: Optional[str] = None  # platform-specific stable alt ID (Signal UUID, Feishu union_id)
    chat_id_alt: Optional[str] = None  # Signal group internal ID
    is_bot: bool = False  # message author is a bot/webhook (Discord)
    # Platform-neutral SCOPE discriminator (Discord guild / Slack workspace / Matrix server) driving
    # isolation. ``guild_id`` is a deprecated alias: both written, ``scope_id`` wins on read.
    scope_id: Optional[str] = None
    guild_id: Optional[str] = None
    parent_chat_id: Optional[str] = None  # parent channel when chat_id is a thread
    message_id: Optional[str] = None  # triggering message (pin/reply/react)
    role_authorized: bool = False  # adapter granted access via role, not user ID
    # Multiplex profile this message routes to (None => active/default); namespaces the key.
    profile: Optional[str] = None
    # Transport-local fail-closed signal: explicit profile route whose target is not served.
    profile_route_rejected: bool = field(default=False, repr=False, compare=False)
    # Discord auto-thread metadata: explicit so pre-existing/renamed threads are never renamed.
    auto_thread_created: bool = False
    auto_thread_initial_name: Optional[str] = None
    # Discord auto-thread continuity: the thread id a CHANNEL message WILL be delivered into, so
    # the initiating message and later in-thread follow-ups share ONE session.
    prospective_thread_id: Optional[str] = None
    # The adapter declares this chat's session shared by every participant even though the
    # sender id is kept (Telegram observe mode: commands keep user_id for slash access checks).
    shared_session: bool = False
    # Wire-INVISIBLE trust signal (never in to_dict/from_dict, so a peer cannot forge it): came
    # over the authenticated relay WebSocket. ``platform`` is the UNDERLYING platform, not
    # ``relay``, so authz must key upstream trust off THIS flag.
    delivered_via_upstream_relay: bool = False
    # Wire-INVISIBLE: the key of a conversation its chat moved past, still run from the apps
    # (gateway/session_detached.py). It replaces the chat's key, so nothing keyed by route reaches
    # the chat's current conversation, and delivery goes nowhere.
    detached_route: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Mirror scope_id/guild_id onto each other (scope_id wins) so readers of EITHER agree.
        if self.scope_id is None and self.guild_id is not None:
            self.scope_id = self.guild_id
        elif self.scope_id is not None:
            self.guild_id = self.scope_id

    @staticmethod
    def _describe(chat_type: str, user_label: str, chat_label: str) -> str:
        if chat_type == "dm":
            return f"DM with {user_label}"
        return f"{_CHAT_TYPE_PREFIX.get(chat_type, '')}{chat_label}"

    @property
    def description(self) -> str:
        """Human-readable description of the source."""
        if self.platform == Platform.LOCAL:
            return "CLI terminal"
        user, chat = self.user_name or self.user_id or "user", self.chat_name or self.chat_id
        desc = self._describe(self.chat_type, user, chat)
        return f"{desc}, thread: {self.thread_id}" if self.thread_id else desc

    # Wire layout (order matters for byte-stable JSON): always-present, then truthy-only
    # optionals around the dual-written scope pair.
    _ALWAYS_FIELDS = ("chat_id", "chat_name", "chat_type", "user_id", "user_name", "thread_id", "chat_topic")
    _OPTIONAL_PRE_SCOPE = ("user_id_alt", "chat_id_alt")
    _OPTIONAL_POST_SCOPE = ("parent_chat_id", "message_id", "profile")
    _OPTIONAL_TAIL = ("auto_thread_initial_name", "prospective_thread_id")

    def to_dict(self) -> dict[str, Any]:
        d = {"platform": self.platform.value}
        d.update((name, getattr(self, name)) for name in self._ALWAYS_FIELDS)

        def _optional(names) -> None:
            d.update((name, v) for name in names if (v := getattr(self, name)))

        _optional(self._OPTIONAL_PRE_SCOPE)
        # Dual-write scope_id + deprecated guild_id alias during the migration.
        scope = self.scope_id if self.scope_id is not None else self.guild_id
        if scope:
            d["scope_id"] = d["guild_id"] = scope
        _optional(self._OPTIONAL_POST_SCOPE)
        if self.auto_thread_created:
            d["auto_thread_created"] = True
        _optional(self._OPTIONAL_TAIL)
        if self.shared_session:
            d["shared_session"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSource":
        plain = {
            name: data.get(name)
            for name in cls._ALWAYS_FIELDS[1:] + cls._OPTIONAL_PRE_SCOPE + cls._OPTIONAL_POST_SCOPE + cls._OPTIONAL_TAIL
            if name != "chat_type"
        }
        return cls(
            platform=Platform(data["platform"]), chat_id=str(data["chat_id"]),
            chat_type=data.get("chat_type", "dm"),
            scope_id=data.get("scope_id", data.get("guild_id")),
            auto_thread_created=bool(data.get("auto_thread_created", False)),
            shared_session=bool(data.get("shared_session", False)), **plain,
        )


def _now() -> datetime:
    """Return the current local time."""
    return datetime.now()


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_iso(value) -> Optional[datetime]:
    """``datetime.fromisoformat`` that returns None for empty/malformed input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


PROMPT_PIN_VERSION = 1


def sanitize_prompt_pin(pin: Any) -> Optional[dict[str, Any]]:
    """Validated durable snapshot of the exact ephemeral inputs reused by internal turns.

    ``redact_pii`` is the ``privacy.redact_pii`` the context bytes were rendered under; a pin
    without it cannot prove which privacy policy produced them and is refused."""
    if not isinstance(pin, dict) or pin.get("version") != PROMPT_PIN_VERSION:
        return None
    context_key = pin.get("context_key")
    context_prompt = pin.get("context_prompt")
    redact_pii = pin.get("redact_pii")
    channel_prompt = pin.get("channel_prompt")
    parent_chat_id = pin.get("parent_chat_id")
    if not isinstance(context_key, str) or not context_key or not isinstance(context_prompt, str):
        return None
    if not isinstance(redact_pii, bool):
        return None
    if channel_prompt is not None and not isinstance(channel_prompt, str):
        return None
    if parent_chat_id is not None and not isinstance(parent_chat_id, str):
        return None
    return {
        "version": PROMPT_PIN_VERSION,
        "context_key": context_key,
        "context_prompt": context_prompt,
        "redact_pii": redact_pii,
        "channel_prompt": channel_prompt,
        "parent_chat_id": parent_chat_id,
    }


# /model override keys safe to persist; ``api_key``/``api_mode`` must NEVER reach sessions.json.
PERSISTABLE_MODEL_OVERRIDE_KEYS = ("model", "provider", "base_url")


def sanitize_model_override(override: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
    """Copy of *override* with only persistable, non-secret keys, or ``None`` when empty."""
    if not isinstance(override, dict):
        return None
    cleaned = {
        k: str(v) for k, v in override.items()
        if k in PERSISTABLE_MODEL_OVERRIDE_KEYS and v not in (None, "")
    }
    return cleaned or None


@dataclass
class SessionEntry:
    """Routing-index entry: maps a session key to its current session ID and metadata."""
    session_key: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    origin: Optional[SessionSource] = None  # delivery routing
    display_name: Optional[str] = None
    platform: Optional[Platform] = None
    chat_type: str = "dm"
    # Small, JSON-serializable per-entry state (e.g. Slack thread watermarks).
    metadata: dict[str, Any] = field(default_factory=dict)
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_status: str = "unknown"
    last_prompt_tokens: int = 0  # last API-reported prompt tokens (compression pre-check)
    # Suspension replacement metadata; historical automatic-reset rows retain these fields.
    was_auto_reset: bool = False
    auto_reset_reason: Optional[str] = None
    reset_had_activity: bool = False
    prev_session_id: Optional[str] = None  # feeds the continuity note
    # Explicit /new or /reset triggers topic/channel skill re-injection on the first turn.
    is_fresh_reset: bool = False
    # Historical finalization fence; timers no longer write it.
    expiry_finalized: bool = False
    # Next get_or_create_session() auto-resets; set by /stop to break stuck-resume loops.
    # When True the next call to get_or_create_session() will auto-reset this session (create a new
    # session_id) so the user starts fresh. See #7536.
    suspended: bool = False
    # Interrupted by a restart/drain timeout, recovery expected: unlike ``suspended`` the
    # session_id is kept so the agent auto-continues. Cleared after the next successful turn;
    # escalation to ``suspended`` is the runner's ``.restart_failure_counts`` job.
    # Unlike ``suspended``, ``resume_pending`` preserves the existing session_id on next access — the user
    # stays on the same transcript and the agent auto-continues from where it left off. Escalation to
    # ``suspended`` is handled by the existing ``.restart_failure_counts`` stuck-loop counter (#7536), not
    # by a parallel counter on this entry.
    resume_pending: bool = False
    resume_reason: Optional[str] = None  # e.g. "restart_timeout"
    last_resume_marked_at: Optional[datetime] = None
    # Durable marker of the executing turn; CAS-cleared on normal unwind, left behind by
    # SIGKILL/OOM so unclean startup recovers the exact session instead of guessing.
    active_turn_token: Optional[str] = None
    active_turn_started_at: Optional[datetime] = None
    # Session-scoped /model override (model/provider/base_url ONLY — never credentials, see
    # sanitize_model_override). Persisted so a restart keeps the chosen model.
    model_override: Optional[dict[str, str]] = None
    # Profile owning the bot that received this lane's traffic (``RoutingIdentity.transport_profile``,
    # "default" spelled out). The key namespace only says where the turn RUNS; after a restart this is
    # what says which bot may deliver to it. None = unknown (row predates the field, or standalone).
    transport_profile: Optional[str] = None
    # Exact session-context/channel inputs from the last human turn. Append-only dataclass field so
    # older positional construction of transport_profile keeps its meaning.
    prompt_pin: Optional[dict[str, Any]] = None
    # Gateway ``/yolo`` bypass for this lane, mirrored from ``tools.approval``'s in-memory set so a restart
    # keeps it. Tri-state: True/False is an explicit toggle (an OFF stays off, even over a ``--yolo`` launch);
    # None means nobody toggled it (new, branched, reset, migrated, cron entries, and every conversation
    # boundary, ``_clear_session_boundary_security_state``), so the session's launch policy applies.
    yolo: Optional[bool] = None

    # Fields (de)serialized verbatim, in wire order (``from_dict`` reads them with
    # ``data.get(name, <dataclass default>)``), split around the three ISO-datetime/token keys.
    _PLAIN_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "total_tokens", "last_prompt_tokens", "estimated_cost_usd", "cost_status",
        "expiry_finalized", "suspended", "resume_pending", "resume_reason",
    )
    _RESET_FIELDS = (
        "is_fresh_reset", "was_auto_reset", "auto_reset_reason", "reset_had_activity",
        "prev_session_id",
    )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "session_key": self.session_key, "session_id": self.session_id,
            "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat(),
            "display_name": self.display_name,
            "platform": self.platform.value if self.platform else None,
            "chat_type": self.chat_type, "metadata": self.metadata,
        }
        result.update((name, getattr(self, name)) for name in self._PLAIN_FIELDS)
        # ``yolo`` keeps the pre-tri-state meaning (ON only) for older readers; ``yolo_toggle`` is authoritative.
        result["yolo"], result["yolo_toggle"] = self.yolo is True, self.yolo
        result["last_resume_marked_at"] = _iso(self.last_resume_marked_at)
        result["active_turn_token"] = self.active_turn_token
        result["active_turn_started_at"] = _iso(self.active_turn_started_at)
        result.update((name, getattr(self, name)) for name in self._RESET_FIELDS)
        if self.model_override:
            # Defence-in-depth against an unsanitized dict stored directly.
            result["model_override"] = sanitize_model_override(self.model_override)
        if self.prompt_pin:
            # Same defence-in-depth: routing JSON must never preserve malformed pin state.
            if pin := sanitize_prompt_pin(self.prompt_pin):
                result["prompt_pin"] = pin
        if self.transport_profile:
            result["transport_profile"] = self.transport_profile
        if self.origin:
            result["origin"] = self.origin.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionEntry":
        origin = data.get("origin")
        origin = SessionSource.from_dict(origin) if isinstance(origin, dict) else None
        platform = None
        if data.get("platform"):
            try:
                platform = Platform(data["platform"])
            except ValueError as e:
                logger.debug("Unknown platform value %r: %s", data["platform"], e)
        token = data.get("active_turn_token")
        started_at = _parse_iso(data.get("active_turn_started_at"))
        if not isinstance(token, str) or not token:
            # The pair is written atomically; a partial/malformed pair must not auto-resume.
            token = started_at = None

        session_key, session_id = data["session_key"], data["session_id"]
        # CWE-22: session_id becomes a filename (strict); session_key allows interior ``/``.
        if _is_path_unsafe(session_id):
            raise ValueError("Invalid session_id: potential directory traversal detected")
        if _is_path_unsafe(session_key, strict=False):
            raise ValueError("Invalid session_key: potential directory traversal detected")

        defaults = {f.name: f.default for f in fields(cls)}
        plain = {n: data.get(n, defaults[n]) for n in cls._PLAIN_FIELDS + cls._RESET_FIELDS}
        plain["expiry_finalized"] = data.get("expiry_finalized", data.get("memory_flushed", False))
        # Rows written before the tri-state stored a bare ``yolo`` bool where False also meant "never
        # toggled": only their ON is a real toggle; False follows the launch policy, as it did then.
        plain["yolo"] = data["yolo_toggle"] if "yolo_toggle" in data else (True if data.get("yolo") is True else None)
        transport_profile = data.get("transport_profile")
        return cls(
            session_key=session_key, session_id=session_id,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]), origin=origin,
            display_name=data.get("display_name"), platform=platform,
            chat_type=data.get("chat_type", "dm"), metadata=dict(data.get("metadata") or {}),
            last_resume_marked_at=_parse_iso(data.get("last_resume_marked_at")),
            active_turn_token=token, active_turn_started_at=started_at,
            model_override=sanitize_model_override(data.get("model_override")),
            prompt_pin=sanitize_prompt_pin(data.get("prompt_pin")),
            transport_profile=transport_profile if isinstance(transport_profile, str) and transport_profile else None,
            **plain,
        )


def _session_key_namespace(profile: Optional[str]) -> str:
    """``agent:<ns>`` prefix for a session key: default/None profile → ``agent:main``
    (BYTE-IDENTICAL to every historical key); named profile → ``agent:<name>`` so two
    profiles serving the same chat never collide. A profile literally named ``main`` would
    otherwise produce the default's namespace and share every session (routing index, agent
    cache, store) with it, so it is marked ``main~``: ``~`` is outside the profile-id alphabet,
    so the marked form can never be another profile's id."""
    if not profile or profile == "default":
        return "agent:main"
    return "agent:main~" if profile == "main" else f"agent:{profile}"


def detached_route_of(source) -> str | None:
    """*source*'s pinned detached key; None for any other source (duck-typed test sources too)."""
    route = getattr(source, 'detached_route', None)
    return route if isinstance(route, str) else None


def _canonical_participant(source: SessionSource) -> Optional[str]:
    """Sender id for key isolation."""
    participant_id = source.user_id_alt or source.user_id
    return participant_id


def build_session_key(
    source: SessionSource, group_sessions_per_user: bool = True,
    thread_sessions_per_user: bool = False, profile: Optional[str] = None,
) -> str:
    """Build a deterministic session key from a message source (single source of truth).

    Layout: ``<ns>:<platform>:<chat_type>[:<slack scope_id>][:<chat_id>][:<thread_id>][:<user>]``.
    Slack ``scope_id`` precedes chat ids (Discord guild scope is deliberately NOT added, for key
    compatibility). DMs are isolated per chat_id, falling back to the sender id, then to one
    session per platform. Groups add the participant id only when ``group_sessions_per_user`` and
    not in a thread (threads are shared unless ``thread_sessions_per_user``).
    """
    if detached := detached_route_of(source):
        return detached
    is_dm = source.chat_type == "dm"
    chat_id = source.chat_id
    # Discord auto-thread continuity: key a channel-initiating message on the thread it WILL be
    # delivered into (prospective_thread_id), and normalize the chat_type slot to "thread" so
    # in-thread follow-ups byte-match. A real thread_id always wins. DMs use thread_id only.
    thread_id = source.thread_id or (None if is_dm else source.prospective_thread_id)
    chat_type_slot = "thread" if thread_id and not source.thread_id else source.chat_type
    if is_dm:
        # No chat_id: fall back to the sender id before the bare per-platform sink, or every
        # chat_id-less DM shares one agent.
        isolate_user = not chat_id
    else:
        # Threads are shared by default; per-user isolation only via thread_sessions_per_user or
        # outside a thread.
        isolate_user = (
            group_sessions_per_user and not (thread_id and not thread_sessions_per_user)
            and not getattr(source, "shared_session", False)
        )
    # Duck-typed sources may lack user_id_alt: read the participant only when it matters.
    participant_id = _canonical_participant(source) if (isolate_user or not is_dm) else None

    parts = [_session_key_namespace(profile), source.platform.value, chat_type_slot]
    if source.platform == Platform.SLACK and source.scope_id:
        parts.append(str(source.scope_id))
    if chat_id:
        parts.append(chat_id)
    # DMs put the participant before the thread; groups/threads put it after.
    user_part = [str(participant_id)] if isolate_user and participant_id else []
    thread_part = [thread_id] if thread_id else []
    parts += user_part + thread_part if is_dm else thread_part + user_part
    return ":".join(str(part) for part in parts)

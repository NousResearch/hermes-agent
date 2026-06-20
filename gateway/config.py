"""
Gateway configuration management.

Handles loading and validating configuration for:
- Connected platforms (Telegram, Discord, WhatsApp, Weixin, and more)
- Home channels for each platform
- Session reset policies
- Delivery preferences
"""

import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from hermes_cli.config import get_hermes_home
from gateway.realtime_voice.config import RealtimeVoiceConfig
from utils import env_int, is_truthy_value

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Coerce bool-ish config values, preserving a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return is_truthy_value(value, default=default)


def _coerce_float(value: Any, default: float) -> float:
    """Coerce numeric config values, falling back on malformed input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    """Coerce integer config values, falling back on malformed input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_positive_int(value: Any, key: str) -> Optional[int]:
    """Coerce an optional positive integer config value.

    ``None``/0/negative disable the setting. Malformed values are ignored with
    a warning so a typo never prevents the gateway from starting.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)",
            key,
            value,
        )
        return None
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(value)
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value.strip(), 10)
        else:
            parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)",
            key,
            value,
        )
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_unauthorized_dm_behavior(value: Any, default: str = "pair") -> str:
    """Normalize unauthorized DM behavior to a supported value."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pair", "ignore"}:
            return normalized
    return default


def _normalize_notice_delivery(value: Any, default: str = "public") -> str:
    """Normalize notice delivery mode to a supported value."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"public", "private"}:
            return normalized
    return default


def _ensure_platform_extra_dict(platforms_data: dict, name: str) -> tuple[dict, dict]:
    """Get-or-create ``platforms_data[name]`` and its nested ``extra`` dict.

    Both slots are coerced to ``{}`` if a non-dict value is encountered, so
    callers can safely write keys without type-checking.  Returns
    ``(plat_data, extra)`` for in-place mutation.
    """
    plat_data = platforms_data.setdefault(name, {})
    if not isinstance(plat_data, dict):
        plat_data = {}
        platforms_data[name] = plat_data
    extra = plat_data.setdefault("extra", {})
    if not isinstance(extra, dict):
        extra = {}
        plat_data["extra"] = extra
    return plat_data, extra


# Module-level cache for bundled platform plugin names (lives outside the
# enum so it doesn't become an accidental enum member).
_Platform__bundled_plugin_names: Optional[set] = None


class Platform(Enum):
    """Supported messaging platforms.

    Built-in platforms have explicit members.  Plugin platforms use dynamic
    members created on-demand by ``_missing_()`` so that
    ``Platform("irc")`` works without modifying this enum.  Dynamic members
    are cached in ``_value2member_map_`` for identity-stable comparisons.
    """
    LOCAL = "local"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    WHATSAPP_CLOUD = "whatsapp_cloud"
    SLACK = "slack"
    SIGNAL = "signal"
    MATTERMOST = "mattermost"
    MATRIX = "matrix"
    HOMEASSISTANT = "homeassistant"
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
    @classmethod
    def _missing_(cls, value):
        """Accept unknown platform names only for known plugin adapters.

        Creates a pseudo-member cached in ``_value2member_map_`` so that
        ``Platform("irc") is Platform("irc")`` holds True (identity-stable).
        Arbitrary strings are rejected to prevent enum pollution.
        """
        if not isinstance(value, str) or not value.strip():
            return None
        # Normalise to lowercase to avoid case mismatches in config
        value = value.strip().lower()
        # Check cache first (another call may have created it already)
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]

        # Only create pseudo-members for bundled plugin platforms (discovered
        # via filesystem scan) or runtime-registered plugin platforms.
        global _Platform__bundled_plugin_names
        if _Platform__bundled_plugin_names is None:
            _Platform__bundled_plugin_names = cls._scan_bundled_plugin_platforms()
        if value in _Platform__bundled_plugin_names:
            pseudo = object.__new__(cls)
            pseudo._value_ = value
            pseudo._name_ = value.upper().replace("-", "_").replace(" ", "_")
            cls._value2member_map_[value] = pseudo
            cls._member_map_[pseudo._name_] = pseudo
            return pseudo

        # Runtime-registered plugins (e.g. user-installed, discovered after
        # the enum was defined).
        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(value):
                pseudo = object.__new__(cls)
                pseudo._value_ = value
                pseudo._name_ = value.upper().replace("-", "_").replace(" ", "_")
                cls._value2member_map_[value] = pseudo
                cls._member_map_[pseudo._name_] = pseudo
                return pseudo
        except Exception:
            pass

        return None

    @classmethod
    def _scan_bundled_plugin_platforms(cls) -> set:
        """Return names of bundled platform plugins under ``plugins/platforms/``."""
        names: set = set()
        try:
            platforms_dir = Path(__file__).parent.parent / "plugins" / "platforms"
            if platforms_dir.is_dir():
                for child in platforms_dir.iterdir():
                    if (
                        child.is_dir()
                        and (child / "__init__.py").exists()
                        and (
                            (child / "plugin.yaml").exists()
                            or (child / "plugin.yml").exists()
                        )
                    ):
                        names.add(child.name.lower())
        except Exception:
            pass
        return names


# Snapshot of built-in platform values before any dynamic _missing_ lookups.
# Used to distinguish real platforms from arbitrary strings.
_BUILTIN_PLATFORM_VALUES = frozenset(m.value for m in Platform.__members__.values())


@dataclass
class HomeChannel:
    """
    Default destination for a platform.
    
    When a cron job specifies deliver="telegram" without a specific chat ID,
    messages are sent to this home channel. Thread-aware platforms may also
    store a thread/topic ID so the bare platform target routes to the exact
    conversation where /sethome was run.
    """
    platform: Platform
    chat_id: str
    name: str  # Human-readable name for display
    thread_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "name": self.name,
        }
        if self.thread_id:
            result["thread_id"] = self.thread_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HomeChannel":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            name=data.get("name", "Home"),
            thread_id=str(data["thread_id"]) if data.get("thread_id") else None,
        )


@dataclass
class SessionResetPolicy:
    """
    Controls when sessions reset (lose context).
    
    Modes:
    - "daily": Reset at a specific hour each day
    - "idle": Reset after N minutes of inactivity
    - "both": Whichever triggers first (daily boundary OR idle timeout)
    - "none": Never auto-reset (context managed only by compression)
    """
    mode: str = "both"  # "daily", "idle", "both", or "none"
    at_hour: int = 4  # Hour for daily reset (0-23, local time)
    idle_minutes: int = 1440  # Minutes of inactivity before reset (24 hours)
    notify: bool = True  # Send a notification to the user when auto-reset occurs
    notify_exclude_platforms: tuple = ("api_server", "webhook")  # Platforms that don't get reset notifications
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "at_hour": self.at_hour,
            "idle_minutes": self.idle_minutes,
            "notify": self.notify,
            "notify_exclude_platforms": list(self.notify_exclude_platforms),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionResetPolicy":
        # Handle both missing keys and explicit null values (YAML null → None)
        mode = data.get("mode")
        at_hour = data.get("at_hour")
        idle_minutes = data.get("idle_minutes")
        notify = data.get("notify")
        exclude = data.get("notify_exclude_platforms")
        return cls(
            mode=mode if mode is not None else "both",
            at_hour=at_hour if at_hour is not None else 4,
            idle_minutes=idle_minutes if idle_minutes is not None else 1440,
            notify=_coerce_bool(notify, True),
            notify_exclude_platforms=tuple(exclude) if exclude is not None else ("api_server", "webhook"),
        )


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""
    enabled: bool = False
    token: Optional[str] = None  # Bot token (Telegram, Discord)
    api_key: Optional[str] = None  # API key if different from token
    home_channel: Optional[HomeChannel] = None
    
    # Reply threading mode (Telegram/Slack)
    # - "off": Never thread replies to original message
    # - "first": Only first chunk threads to user's message (default)
    # - "all": All chunks in multi-part replies thread to user's message
    reply_to_mode: str = "first"

    # Whether the gateway is allowed to send "♻️ Gateway online" /
    # "♻ Gateway restarted" lifecycle notifications on this platform.
    # Default True preserves prior behavior. Set False on platforms used
    # by end users (e.g. Slack) where operator-flavored restart pings are
    # noise; keep True for back-channels where the operator wants them.
    gateway_restart_notification: bool = True

    # Platform-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "extra": self.extra,
            "reply_to_mode": self.reply_to_mode,
            "gateway_restart_notification": self.gateway_restart_notification,
        }
        if self.token:
            result["token"] = self.token
        if self.api_key:
            result["api_key"] = self.api_key
        if self.home_channel:
            result["home_channel"] = self.home_channel.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformConfig":
        home_channel = None
        if "home_channel" in data:
            home_channel = HomeChannel.from_dict(data["home_channel"])

        # gateway_restart_notification may be bridged into extra via the
        # shared-key loop in load_gateway_config(); check both top-level
        # and extra so YAML ``discord: gateway_restart_notification: false``
        # works without needing a separate platforms: block.
        _grn = data.get("gateway_restart_notification")
        if _grn is None:
            _grn = data.get("extra", {}).get("gateway_restart_notification")

        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_channel=home_channel,
            reply_to_mode=data.get("reply_to_mode", "first"),
            gateway_restart_notification=_coerce_bool(_grn, True),
            extra=data.get("extra", {}),
        )


# Streaming defaults — single source of truth so both StreamingConfig and
# StreamConsumerConfig agree on the out-of-the-box edit rhythm.  Tuned for
# Telegram's ~1 edit/s flood envelope: a touch under 1s lets the cadence
# breathe without bumping into rate limits, and a smaller buffer threshold
# makes short replies feel near-instant in DMs.
DEFAULT_STREAMING_EDIT_INTERVAL: float = 0.8
DEFAULT_STREAMING_BUFFER_THRESHOLD: int = 24
DEFAULT_STREAMING_CURSOR: str = " ▉"


@dataclass
class StreamingConfig:
    """Configuration for real-time token streaming to messaging platforms."""
    enabled: bool = False
    # Transport selection:
    #   "auto"  — prefer native streaming-draft updates when the platform
    #             supports them (Telegram sendMessageDraft, Bot API 9.5+);
    #             fall back to edit-based when not.
    #   "draft" — explicitly request native drafts; falls back to edit when
    #             the platform/chat doesn't support them.
    #   "edit"  — progressive editMessageText only (legacy behaviour).
    #   "off"   — disable streaming entirely.
    #
    # Default is "auto": prefer native draft streaming on platforms that
    # support it (Telegram DMs via sendMessageDraft, Bot API 9.5+) and fall
    # back to edit-based streaming everywhere else.  This is safe as a global
    # default because adapters without draft support (Discord, Slack, Matrix,
    # …) report supports_draft_streaming() == False and transparently use the
    # edit path — so "auto" never regresses non-Telegram platforms, it only
    # upgrades the chats that can render the smoother native preview.
    transport: str = "auto"
    edit_interval: float = DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = DEFAULT_STREAMING_CURSOR
    # Ported from openclaw/openclaw#72038.  When >0, the final edit for
    # a long-running streamed response is delivered as a fresh message
    # if the original preview has been visible for at least this many
    # seconds, so the platform's visible timestamp reflects completion
    # time instead of the preview creation time.  Currently applied to
    # Telegram only (other platforms ignore the setting).  Default 0 disables
    # the fresh-message replacement path; set >0 to opt in.
    fresh_final_after_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "transport": self.transport,
            "edit_interval": self.edit_interval,
            "buffer_threshold": self.buffer_threshold,
            "cursor": self.cursor,
            "fresh_final_after_seconds": self.fresh_final_after_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingConfig":
        if not data:
            return cls()
        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            transport=data.get("transport", "auto"),
            edit_interval=_coerce_float(
                data.get("edit_interval"), DEFAULT_STREAMING_EDIT_INTERVAL,
            ),
            buffer_threshold=_coerce_int(
                data.get("buffer_threshold"), DEFAULT_STREAMING_BUFFER_THRESHOLD,
            ),
            cursor=data.get("cursor", DEFAULT_STREAMING_CURSOR),
            fresh_final_after_seconds=_coerce_float(
                data.get("fresh_final_after_seconds"), 0.0
            ),
        )


# -----------------------------------------------------------------------------
# Built-in platform connection checkers
# -----------------------------------------------------------------------------
# Each callable receives a ``PlatformConfig`` and returns ``True`` when the
# platform is sufficiently configured to be considered "connected".  Platforms
# that rely on the generic ``token or api_key`` check (Telegram, Discord,
# Slack, Matrix, Mattermost, HomeAssistant) do not need an entry here.
_PLATFORM_CONNECTED_CHECKERS: dict[Platform, Callable[[PlatformConfig], bool]] = {
    Platform.WEIXIN: lambda cfg: bool(
        cfg.extra.get("account_id") and (cfg.token or cfg.extra.get("token"))
    ),
    Platform.WHATSAPP_CLOUD: lambda cfg: bool(
        cfg.extra.get("phone_number_id") and cfg.extra.get("access_token")
    ),
    Platform.SIGNAL: lambda cfg: bool(cfg.extra.get("http_url")),
    Platform.API_SERVER: lambda cfg: True,
    Platform.WEBHOOK: lambda cfg: True,
    Platform.MSGRAPH_WEBHOOK: lambda cfg: bool(
        str(cfg.extra.get("client_state") or "").strip()
    ),
    Platform.BLUEBUBBLES: lambda cfg: bool(
        cfg.extra.get("server_url") and cfg.extra.get("password")
    ),
    Platform.QQBOT: lambda cfg: bool(
        cfg.extra.get("app_id") and cfg.extra.get("client_secret")
    ),
    Platform.YUANBAO: lambda cfg: bool(
        cfg.extra.get("app_id") and cfg.extra.get("app_secret")
    ),
    # Relay dials OUT to a connector; it is "connected" once an endpoint URL is
    # configured (extra["relay_url"] or extra["url"]). The capability descriptor
    # is negotiated at handshake time, so the URL is the only config-level
    # signal in the experimental phase. EXPERIMENTAL — may change.
    Platform.RELAY: lambda cfg: bool(
        cfg.extra.get("relay_url") or cfg.extra.get("url")
    ),
}


@dataclass
class GatewayConfig:
    """
    Main gateway configuration.
    
    Manages all platform connections, session policies, and delivery settings.
    """
    # Platform configurations
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    
    # Session reset policies by type
    default_reset_policy: SessionResetPolicy = field(default_factory=SessionResetPolicy)
    reset_by_type: Dict[str, SessionResetPolicy] = field(default_factory=dict)
    reset_by_platform: Dict[Platform, SessionResetPolicy] = field(default_factory=dict)
    
    # Reset trigger commands
    reset_triggers: List[str] = field(default_factory=lambda: ["/new", "/reset"])

    # User-defined quick commands (slash commands that bypass the agent loop)
    quick_commands: Dict[str, Any] = field(default_factory=dict)
    
    # Storage paths
    sessions_dir: Path = field(default_factory=lambda: get_hermes_home() / "sessions")
    
    # Delivery settings
    always_log_local: bool = True  # Always save cron outputs to local files
    # Drop outbound "silence narration" messages (e.g. *(silent)*, 🔇, a bare
    # ".") pre-send. These are model hallucinations emitted when a persona has
    # nothing actionable to say; in bot-to-bot channels they mirror back and
    # forth, burning tokens and crashing models. Substrate-level guard that
    # survives SOUL.md/prompt drift across providers. Opt out with False for
    # raw passthrough.
    filter_silence_narration: bool = True

    # STT settings
    stt_enabled: bool = True  # Whether to auto-transcribe inbound voice messages

    # Session isolation in shared chats
    group_sessions_per_user: bool = True  # Isolate group/channel sessions per participant when user IDs are available
    thread_sessions_per_user: bool = False  # When False (default), threads are shared across all participants
    max_concurrent_sessions: Optional[int] = None  # Positive int caps simultaneous active chat sessions

    # Multi-profile multiplexing (opt-in; default off preserves one-gateway-per-profile).
    # When True, the default profile's gateway serves inbound messages for every
    # profile on the host: profiles are stamped into session keys and (in later
    # phases) per-profile adapters/credentials are resolved. When False, the
    # gateway behaves exactly as before — single HERMES_HOME, no profile stamping.
    multiplex_profiles: bool = False

    # Unauthorized DM policy
    unauthorized_dm_behavior: str = "pair"  # "pair" or "ignore"

    # Streaming configuration
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # Experimental realtime voice configuration
    realtime_voice: RealtimeVoiceConfig = field(default_factory=RealtimeVoiceConfig)

    # Session store pruning: drop SessionEntry records older than this many
    # days from the in-memory dict and sessions.json.  Keeps the store from
    # growing unbounded in gateways serving many chats/threads/users over
    # months.  Pruning is invisible to users — if they resume, they get a
    # fresh session exactly as if the reset policy had fired.  0 = disabled.
    session_store_max_age_days: int = 90

    def get_connected_platforms(self) -> List[Platform]:
        """Return list of platforms that are enabled and configured."""
        connected = []
        for platform, config in self.platforms.items():
            if not config.enabled:
                continue
            if self._is_platform_connected(platform, config):
                connected.append(platform)
        return connected

    def _is_platform_connected(self, platform: Platform, config: PlatformConfig) -> bool:
        """Check whether a single platform is sufficiently configured."""
        # Weixin requires both a token and an account_id (checked first so
        # the generic token branch doesn't let it through without account_id).
        if platform == Platform.WEIXIN:
            return bool(
                config.extra.get("account_id")
                and (config.token or config.extra.get("token"))
            )

        # Generic token/api_key auth covers Telegram, Discord, Slack, etc.
        if config.token or config.api_key:
            return True

        # Platform-specific check
        checker = _PLATFORM_CONNECTED_CHECKERS.get(platform)
        if checker is not None:
            return checker(config)

        # Plugin-registered platforms.  Force plugin discovery first so this
        # works even when GatewayConfig is constructed directly (e.g. in tests
        # or callers that bypass load_gateway_config(), which is what triggers
        # discovery in the normal path).  discover_plugins() is idempotent.
        try:
            from gateway.platform_registry import platform_registry
            try:
                from hermes_cli.plugins import discover_plugins
                discover_plugins()
            except Exception:
                pass
            entry = platform_registry.get(platform.value)
            if entry:
                if entry.is_connected is not None:
                    return entry.is_connected(config)
                if entry.validate_config is not None:
                    return entry.validate_config(config)
                return True
        except Exception:
            pass  # Registry not yet initialised during early import

        return False
    
    def get_home_channel(self, platform: Platform) -> Optional[HomeChannel]:
        """Get the home channel for a platform."""
        config = self.platforms.get(platform)
        if config:
            return config.home_channel
        return None
    
    def get_reset_policy(
        self, 
        platform: Optional[Platform] = None,
        session_type: Optional[str] = None
    ) -> SessionResetPolicy:
        """
        Get the appropriate reset policy for a session.
        
        Priority: platform override > type override > default
        """
        # Platform-specific override takes precedence
        if platform and platform in self.reset_by_platform:
            return self.reset_by_platform[platform]
        
        # Type-specific override (dm, group, thread)
        if session_type and session_type in self.reset_by_type:
            return self.reset_by_type[session_type]
        
        return self.default_reset_policy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms": {
                p.value: c.to_dict() for p, c in self.platforms.items()
            },
            "default_reset_policy": self.default_reset_policy.to_dict(),
            "reset_by_type": {
                k: v.to_dict() for k, v in self.reset_by_type.items()
            },
            "reset_by_platform": {
                p.value: v.to_dict() for p, v in self.reset_by_platform.items()
            },
            "reset_triggers": self.reset_triggers,
            "quick_commands": self.quick_commands,
            "sessions_dir": str(self.sessions_dir),
            "always_log_local": self.always_log_local,
            "filter_silence_narration": self.filter_silence_narration,
            "stt_enabled": self.stt_enabled,
            "group_sessions_per_user": self.group_sessions_per_user,
            "thread_sessions_per_user": self.thread_sessions_per_user,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "multiplex_profiles": self.multiplex_profiles,
            "unauthorized_dm_behavior": self.unauthorized_dm_behavior,
            "streaming": self.streaming.to_dict(),
            "realtime_voice": self.realtime_voice.to_dict(),
            "session_store_max_age_days": self.session_store_max_age_days,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        platforms = {}
        for platform_name, platform_data in data.get("platforms", {}).items():
            try:
                platform = Platform(platform_name)
                platforms[platform] = PlatformConfig.from_dict(platform_data)
            except ValueError:
                pass  # Skip unknown platforms
        
        reset_by_type = {}
        for type_name, policy_data in data.get("reset_by_type", {}).items():
            reset_by_type[type_name] = SessionResetPolicy.from_dict(policy_data)
        
        reset_by_platform = {}
        for platform_name, policy_data in data.get("reset_by_platform", {}).items():
            try:
                platform = Platform(platform_name)
                reset_by_platform[platform] = SessionResetPolicy.from_dict(policy_data)
            except ValueError:
                pass
        
        default_policy = SessionResetPolicy()
        if "default_reset_policy" in data:
            default_policy = SessionResetPolicy.from_dict(data["default_reset_policy"])
        
        sessions_dir = get_hermes_home() / "sessions"
        if "sessions_dir" in data:
            sessions_dir = Path(data["sessions_dir"])
        
        quick_commands = data.get("quick_commands", {})
        if not isinstance(quick_commands, dict):
            quick_commands = {}

        stt_enabled = data.get("stt_enabled")
        if stt_enabled is None:
            stt_enabled = data.get("stt", {}).get("enabled") if isinstance(data.get("stt"), dict) else None

        group_sessions_per_user = data.get("group_sessions_per_user")
        thread_sessions_per_user = data.get("thread_sessions_per_user")
        multiplex_profiles = data.get("multiplex_profiles")
        nested_gateway = data.get("gateway") if isinstance(data.get("gateway"), dict) else {}
        if multiplex_profiles is None and isinstance(nested_gateway, dict):
            # Also honor gateway.multiplex_profiles written by
            # ``hermes config set gateway.multiplex_profiles true``.
            multiplex_profiles = nested_gateway.get("multiplex_profiles")
        if "max_concurrent_sessions" in data:
            max_concurrent_raw = data.get("max_concurrent_sessions")
            max_concurrent_key = "max_concurrent_sessions"
        else:
            max_concurrent_raw = nested_gateway.get("max_concurrent_sessions")
            max_concurrent_key = "gateway.max_concurrent_sessions"
        max_concurrent_sessions = _coerce_optional_positive_int(
            max_concurrent_raw,
            max_concurrent_key,
        )
        unauthorized_dm_behavior = _normalize_unauthorized_dm_behavior(
            data.get("unauthorized_dm_behavior"),
            "pair",
        )

        try:
            session_store_max_age_days = int(data.get("session_store_max_age_days", 90))
            session_store_max_age_days = max(session_store_max_age_days, 0)
        except (TypeError, ValueError):
            session_store_max_age_days = 90

        return cls(
            platforms=platforms,
            default_reset_policy=default_policy,
            reset_by_type=reset_by_type,
            reset_by_platform=reset_by_platform,
            reset_triggers=data.get("reset_triggers", ["/new", "/reset"]),
            quick_commands=quick_commands,
            sessions_dir=sessions_dir,
            always_log_local=_coerce_bool(data.get("always_log_local"), True),
            filter_silence_narration=_coerce_bool(
                data.get("filter_silence_narration"), True
            ),
            stt_enabled=_coerce_bool(stt_enabled, True),
            group_sessions_per_user=_coerce_bool(group_sessions_per_user, True),
            thread_sessions_per_user=_coerce_bool(thread_sessions_per_user, False),
            multiplex_profiles=_coerce_bool(multiplex_profiles, False),
            max_concurrent_sessions=max_concurrent_sessions,
            unauthorized_dm_behavior=unauthorized_dm_behavior,
            streaming=StreamingConfig.from_dict(data.get("streaming", {})),
            realtime_voice=RealtimeVoiceConfig.from_dict(data.get("realtime_voice", {})),
            session_store_max_age_days=session_store_max_age_days,
        )

    def get_unauthorized_dm_behavior(self, platform: Optional[Platform] = None) -> str:
        """Return the effective unauthorized-DM behavior for a platform."""
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "unauthorized_dm_behavior" in platform_cfg.extra:
                return _normalize_unauthorized_dm_behavior(
                    platform_cfg.extra.get("unauthorized_dm_behavior"),
                    self.unauthorized_dm_behavior,
                )
        return self.unauthorized_dm_behavior

    def get_notice_delivery(self, platform: Optional[Platform] = None) -> str:
        """Return the effective notice-delivery mode for a platform."""
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "notice_delivery" in platform_cfg.extra:
                return _normalize_notice_delivery(
                    platform_cfg.extra.get("notice_delivery"),
                    "public",
                )
        return "public"


def load_gateway_config() -> GatewayConfig:
    """
    Load gateway configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables
    2. ~/.hermes/config.yaml (primary user-facing config)
    3. ~/.hermes/gateway.json (legacy — provides defaults under config.yaml)
    4. Built-in defaults
    """
    _home = get_hermes_home()
    gw_data: dict = {}

    # Legacy fallback: gateway.json provides the base layer.
    # config.yaml keys always win when both specify the same setting.
    gateway_json_path = _home / "gateway.json"
    if gateway_json_path.exists():
        try:
            with open(gateway_json_path, "r", encoding="utf-8") as f:
                gw_data = json.load(f) or {}
            logger.info(
                "Loaded legacy %s — consider moving settings to config.yaml",
                gateway_json_path,
            )
        except Exception as e:
            logger.warning("Failed to load %s: %s", gateway_json_path, e)

    # Primary source: config.yaml
    try:
        import yaml
        config_yaml_path = _home / "config.yaml"
        if config_yaml_path.exists():
            with open(config_yaml_path, encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}

            # Managed scope: overlay administrator-pinned values so the gateway
            # honors them too. This loader builds its own dict instead of going
            # through hermes_cli.config.load_config, so without this a managed
            # session_reset / quick_commands / stt / model would be ignored by
            # the messaging gateway. Fail-open via the shared helper.
            from hermes_cli import managed_scope
            yaml_cfg = managed_scope.apply_managed_overlay(yaml_cfg)

            # Map config.yaml keys → GatewayConfig.from_dict() schema.
            # Each key overwrites whatever gateway.json may have set.
            sr = yaml_cfg.get("session_reset")
            if sr and isinstance(sr, dict):
                gw_data["default_reset_policy"] = sr

            qc = yaml_cfg.get("quick_commands")
            if qc is not None:
                if isinstance(qc, dict):
                    gw_data["quick_commands"] = qc
                else:
                    logger.warning(
                        "Ignoring invalid quick_commands in config.yaml "
                        "(expected mapping, got %s)",
                        type(qc).__name__,
                    )

            stt_cfg = yaml_cfg.get("stt")
            if isinstance(stt_cfg, dict):
                gw_data["stt"] = stt_cfg

            realtime_voice_cfg = yaml_cfg.get("realtime_voice")
            if isinstance(realtime_voice_cfg, dict):
                gw_data["realtime_voice"] = realtime_voice_cfg

            if "group_sessions_per_user" in yaml_cfg:
                gw_data["group_sessions_per_user"] = yaml_cfg["group_sessions_per_user"]

            if "thread_sessions_per_user" in yaml_cfg:
                gw_data["thread_sessions_per_user"] = yaml_cfg["thread_sessions_per_user"]

            # Multiplexing flag: accept both the top-level key and the nested
            # gateway.multiplex_profiles form (from_dict resolves the nested
            # fallback, but surface the top-level key here for parity with the
            # other session-scope flags above).
            if "multiplex_profiles" in yaml_cfg:
                gw_data["multiplex_profiles"] = yaml_cfg["multiplex_profiles"]

            gateway_section = yaml_cfg.get("gateway")
            if isinstance(gateway_section, dict) and "max_concurrent_sessions" in gateway_section:
                gw_data["max_concurrent_sessions"] = gateway_section["max_concurrent_sessions"]

            if "max_concurrent_sessions" in yaml_cfg:
                gw_data["max_concurrent_sessions"] = yaml_cfg["max_concurrent_sessions"]

            streaming_cfg = yaml_cfg.get("streaming")
            if not isinstance(streaming_cfg, dict):
                # Fall back to nested gateway.streaming written by
                # ``hermes config set gateway.streaming.*``
                streaming_cfg = yaml_cfg.get("gateway", {}).get("streaming")
            if isinstance(streaming_cfg, dict):
                gw_data["streaming"] = streaming_cfg

            if "reset_triggers" in yaml_cfg:
                gw_data["reset_triggers"] = yaml_cfg["reset_triggers"]

            if "always_log_local" in yaml_cfg:
                gw_data["always_log_local"] = yaml_cfg["always_log_local"]

            if "filter_silence_narration" in yaml_cfg:
                gw_data["filter_silence_narration"] = yaml_cfg[
                    "filter_silence_narration"
                ]

            if "unauthorized_dm_behavior" in yaml_cfg:
                gw_data["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                    yaml_cfg.get("unauthorized_dm_behavior"),
                    "pair",
                )

            # Merge platform config into gw_data so runtime-only settings under
            # ``gateway.platforms`` are loaded the same way as top-level
            # ``platforms``. Merge nested first so top-level config keeps
            # precedence, matching the existing gateway.streaming fallback.
            gateway_cfg = yaml_cfg.get("gateway")
            gateway_platforms = gateway_cfg.get("platforms") if isinstance(gateway_cfg, dict) else None
            platforms_data = gw_data.setdefault("platforms", {})
            if not isinstance(platforms_data, dict):
                platforms_data = {}
                gw_data["platforms"] = platforms_data

            def _merge_platform_map(source_platforms: Any) -> None:
                if not isinstance(source_platforms, dict):
                    return
                for plat_name, plat_block in source_platforms.items():
                    if not isinstance(plat_block, dict):
                        continue
                    existing = platforms_data.get(plat_name, {})
                    if not isinstance(existing, dict):
                        existing = {}
                    # Deep-merge extra dicts so gateway.json defaults survive
                    merged_extra = {**existing.get("extra", {}), **plat_block.get("extra", {})}
                    if "enabled" in plat_block:
                        merged_extra["_enabled_explicit"] = True
                    merged = {**existing, **plat_block}
                    if merged_extra:
                        merged["extra"] = merged_extra
                    platforms_data[plat_name] = merged

            _merge_platform_map(gateway_platforms)
            _merge_platform_map(yaml_cfg.get("platforms"))
            if platforms_data:
                gw_data["platforms"] = platforms_data
            # Iterate built-in platforms plus any registered plugin platforms
            # so plugin authors get the same shared-key bridging (#24836).
            try:
                from hermes_cli.plugins import discover_plugins
                discover_plugins()  # idempotent
                from gateway.platform_registry import platform_registry as _pr
            except Exception as e:
                logger.debug("plugin discovery skipped: %s", e)
                _pr = None

            _shared_loop_targets: list = list(Platform)
            if _pr is not None:
                for _entry in _pr.plugin_entries():
                    try:
                        _plat = Platform(_entry.name)
                    except (ValueError, KeyError):
                        continue
                    if _plat not in _shared_loop_targets:
                        _shared_loop_targets.append(_plat)

            for plat in _shared_loop_targets:
                if plat == Platform.LOCAL:
                    continue
                platform_cfg = yaml_cfg.get(plat.value)
                _cfg_toplevel = isinstance(platform_cfg, dict)
                # Fall back to the platform's block under ``platforms`` /
                # ``gateway.platforms`` so shared-key bridging (allow_from,
                # require_mention, free_response_channels, …) still runs when
                # the user configured the platform only under those nested paths
                # and not via a top-level block.  Mirrors the identical fallback
                # already applied to the apply_yaml_config_fn dispatch below
                # (#44f3e51).
                # Note: ``enabled`` is only written to plat_data from a
                # top-level block (``_cfg_toplevel``); for nested-only configs
                # ``_merge_platform_map`` already merged it with the correct
                # precedence, so re-applying it here would overwrite that.
                if not _cfg_toplevel:
                    for _src in (gateway_platforms, yaml_cfg.get("platforms")):
                        if isinstance(_src, dict):
                            _candidate = _src.get(plat.value)
                            if isinstance(_candidate, dict):
                                platform_cfg = _candidate
                                break
                if not isinstance(platform_cfg, dict):
                    continue
                # Collect bridgeable keys from this platform section
                bridged = {}
                if "unauthorized_dm_behavior" in platform_cfg:
                    bridged["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                        platform_cfg.get("unauthorized_dm_behavior"),
                        gw_data.get("unauthorized_dm_behavior", "pair"),
                    )
                if "notice_delivery" in platform_cfg:
                    bridged["notice_delivery"] = _normalize_notice_delivery(
                        platform_cfg.get("notice_delivery"),
                        "public",
                    )
                if "reply_prefix" in platform_cfg:
                    bridged["reply_prefix"] = platform_cfg["reply_prefix"]
                if "reply_in_thread" in platform_cfg:
                    bridged["reply_in_thread"] = platform_cfg["reply_in_thread"]
                if "require_mention" in platform_cfg:
                    bridged["require_mention"] = platform_cfg["require_mention"]
                if plat == Platform.TELEGRAM and "allowed_chats" in platform_cfg:
                    bridged["allowed_chats"] = platform_cfg["allowed_chats"]
                if plat == Platform.TELEGRAM and "group_allowed_chats" in platform_cfg:
                    bridged["group_allowed_chats"] = platform_cfg["group_allowed_chats"]
                if plat == Platform.TELEGRAM and "allowed_topics" in platform_cfg:
                    bridged["allowed_topics"] = platform_cfg["allowed_topics"]
                if "free_response_channels" in platform_cfg:
                    bridged["free_response_channels"] = platform_cfg["free_response_channels"]
                if "mention_patterns" in platform_cfg:
                    bridged["mention_patterns"] = platform_cfg["mention_patterns"]
                if "exclusive_bot_mentions" in platform_cfg:
                    bridged["exclusive_bot_mentions"] = platform_cfg["exclusive_bot_mentions"]
                if plat == Platform.TELEGRAM and "observe_unmentioned_group_messages" in platform_cfg:
                    bridged["observe_unmentioned_group_messages"] = platform_cfg["observe_unmentioned_group_messages"]
                if "dm_policy" in platform_cfg:
                    bridged["dm_policy"] = platform_cfg["dm_policy"]
                if "allow_from" in platform_cfg:
                    bridged["allow_from"] = platform_cfg["allow_from"]
                if "allow_admin_from" in platform_cfg:
                    bridged["allow_admin_from"] = platform_cfg["allow_admin_from"]
                if "user_allowed_commands" in platform_cfg:
                    bridged["user_allowed_commands"] = platform_cfg["user_allowed_commands"]
                if "group_policy" in platform_cfg:
                    bridged["group_policy"] = platform_cfg["group_policy"]
                if "group_allow_from" in platform_cfg:
                    bridged["group_allow_from"] = platform_cfg["group_allow_from"]
                if "group_allow_admin_from" in platform_cfg:
                    bridged["group_allow_admin_from"] = platform_cfg["group_allow_admin_from"]
                if "group_user_allowed_commands" in platform_cfg:
                    bridged["group_user_allowed_commands"] = platform_cfg["group_user_allowed_commands"]
                if plat in {Platform.DISCORD, Platform.SLACK} and "channel_skill_bindings" in platform_cfg:
                    bridged["channel_skill_bindings"] = platform_cfg["channel_skill_bindings"]
                if "channel_prompts" in platform_cfg:
                    channel_prompts = platform_cfg["channel_prompts"]
                    if isinstance(channel_prompts, dict):
                        bridged["channel_prompts"] = {str(k): v for k, v in channel_prompts.items()}
                    else:
                        bridged["channel_prompts"] = channel_prompts
                if "gateway_restart_notification" in platform_cfg:
                    bridged["gateway_restart_notification"] = platform_cfg["gateway_restart_notification"]
                if plat == Platform.DISCORD and "voice_backend" in platform_cfg:
                    bridged["voice_backend"] = platform_cfg["voice_backend"]
                if plat == Platform.DISCORD and "voice_timeout_seconds" in platform_cfg:
                    bridged["voice_timeout_seconds"] = platform_cfg["voice_timeout_seconds"]
                enabled_was_explicit = _cfg_toplevel and "enabled" in platform_cfg
                if not bridged and not enabled_was_explicit:
                    continue
                plat_data, extra = _ensure_platform_extra_dict(platforms_data, plat.value)
                if enabled_was_explicit:
                    plat_data["enabled"] = platform_cfg["enabled"]
                    # Mark the explicit enable/disable so the registry-driven
                    # plugin-enable pass in _apply_env_overrides honors an
                    # explicit ``enabled: false`` for migrated plugin platforms
                    # (slack, telegram, matrix, dingtalk, whatsapp, feishu …)
                    # instead of re-enabling them on token/SDK presence. #41112.
                    extra["_enabled_explicit"] = True
                extra.update(bridged)

            # Plugin-owned YAML→env config bridges (#24836).  See
            # ``PlatformEntry.apply_yaml_config_fn`` for the hook contract.
            # Order: shared-key loop (above) → this dispatch → legacy hardcoded
            # blocks (below; no-op when a hook already set their env var) →
            # ``_apply_env_overrides()`` after ``GatewayConfig.from_dict``.
            if _pr is not None:
                for entry in _pr.all_entries():
                    if entry.apply_yaml_config_fn is None:
                        continue
                    platform_cfg = yaml_cfg.get(entry.name)
                    # Fall back to the platform's block under ``platforms`` /
                    # ``gateway.platforms`` so adapter hooks still run when the
                    # user configured the platform only under those nested paths
                    # (e.g. ``platforms.discord.extra.allow_from``) and not via a
                    # top-level ``discord:`` block.
                    if not isinstance(platform_cfg, dict):
                        for _src in (gateway_platforms, yaml_cfg.get("platforms")):
                            if isinstance(_src, dict):
                                _candidate = _src.get(entry.name)
                                if isinstance(_candidate, dict):
                                    platform_cfg = _candidate
                                    break
                    if not isinstance(platform_cfg, dict):
                        continue
                    try:
                        seeded = entry.apply_yaml_config_fn(yaml_cfg, platform_cfg)
                    except Exception as e:
                        logger.debug(
                            "apply_yaml_config_fn for %s raised: %s",
                            entry.name, e,
                        )
                        continue
                    if not isinstance(seeded, dict) or not seeded:
                        continue
                    _, extra = _ensure_platform_extra_dict(platforms_data, entry.name)
                    extra.update(seeded)

            # Slack settings → env vars: migrated to the slack plugin's
            # ``apply_yaml_config_fn`` hook (see plugins/platforms/slack/
            # adapter.py::_apply_yaml_config), dispatched in the
            # ``apply_yaml_config_fn`` loop above. #41112 / #3823.

            # Bridge top-level require_mention to Telegram when the telegram: section
            # does not already provide one.  Users often write "require_mention: true"
            # at the top level alongside group_sessions_per_user, expecting it to work
            # the same way (#3979).
            _tl_require_mention = yaml_cfg.get("require_mention")
            if _tl_require_mention is not None:
                _tg_section = yaml_cfg.get("telegram") or {}
                if "require_mention" not in _tg_section:
                    _tg_plat = platforms_data.setdefault(Platform.TELEGRAM.value, {})
                    _tg_extra = _tg_plat.setdefault("extra", {})
                    _tg_extra.setdefault("require_mention", _tl_require_mention)
                    # Also bridge to the TELEGRAM_REQUIRE_MENTION env var that the
                    # adapter reads at runtime.  This used to live in the telegram_cfg
                    # block in core; it stays in core because it keys off the TOP-LEVEL
                    # require_mention (not a telegram: block), so the telegram plugin's
                    # apply_yaml_config_fn hook — which only runs when a telegram config
                    # block exists — can't cover the no-telegram-block case (#3979).
                    if not os.getenv("TELEGRAM_REQUIRE_MENTION"):
                        os.environ["TELEGRAM_REQUIRE_MENTION"] = str(_tl_require_mention).lower()

            # Telegram settings → env vars / extra: migrated to the telegram
            # plugin's apply_yaml_config_fn hook
            # (plugins/platforms/telegram/adapter.py). #41112 / #3823.

            # WhatsApp settings → env vars: migrated to the whatsapp plugin's
            # apply_yaml_config_fn hook (plugins/platforms/whatsapp/adapter.py).
            # #41112 / #3823.

            # Signal settings → env vars (env vars take precedence)
            signal_cfg = yaml_cfg.get("signal", {})
            if isinstance(signal_cfg, dict):
                if "require_mention" in signal_cfg and not os.getenv("SIGNAL_REQUIRE_MENTION"):
                    os.environ["SIGNAL_REQUIRE_MENTION"] = str(signal_cfg["require_mention"]).lower()

            # DingTalk settings → env vars: migrated to the dingtalk plugin's
            # apply_yaml_config_fn hook (plugins/platforms/dingtalk/adapter.py).
            # #41112 / #3823.

            # Mattermost config bridge moved into plugins/platforms/mattermost/
            # adapter.py::_apply_yaml_config — see #25443 (apply_yaml_config_fn).

            # Matrix settings → env vars: migrated to the matrix plugin's
            # apply_yaml_config_fn hook (plugins/platforms/matrix/adapter.py).
            # #41112 / #3823.

            # Feishu settings → env vars: migrated to the feishu plugin's
            # apply_yaml_config_fn hook (plugins/platforms/feishu/adapter.py).
            # #41112 / #3823.

    except Exception as e:
        logger.warning(
            "Failed to process config.yaml — falling back to .env / gateway.json values. "
            "Check %s for syntax errors. Error: %s",
            _home / "config.yaml",
            e,
        )

    config = GatewayConfig.from_dict(gw_data)

    # Override with environment variables
    _apply_env_overrides(config)
    
    # --- Validate loaded values ---
    _validate_gateway_config(config)

    return config


def _validate_gateway_config(config: "GatewayConfig") -> None:
    """Validate and sanitize a loaded GatewayConfig in place.

    Called by ``load_gateway_config()`` after all config sources are merged.
    Extracted as a separate function for testability.
    """
    policy = config.default_reset_policy

    if not (0 <= policy.at_hour <= 23):
        logger.warning(
            "Invalid at_hour=%s (must be 0-23). Using default 4.", policy.at_hour
        )
        policy.at_hour = 4

    if policy.idle_minutes is None or policy.idle_minutes <= 0:
        logger.warning(
            "Invalid idle_minutes=%s (must be positive). Using default 1440.",
            policy.idle_minutes,
        )
        policy.idle_minutes = 1440

    # Warn about empty bot tokens — platforms that loaded an empty string
    # won't connect and the cause can be confusing without a log line.
    _token_env_names = {
        Platform.TELEGRAM: "TELEGRAM_BOT_TOKEN",
        Platform.DISCORD: "DISCORD_BOT_TOKEN",
        Platform.SLACK: "SLACK_BOT_TOKEN",
        Platform.MATTERMOST: "MATTERMOST_TOKEN",
        Platform.MATRIX: "MATRIX_ACCESS_TOKEN",
        Platform.WEIXIN: "WEIXIN_TOKEN",
    }
    for platform, pconfig in config.platforms.items():
        if not pconfig.enabled:
            continue
        env_name = _token_env_names.get(platform)
        if env_name and pconfig.token is not None and not pconfig.token.strip():
            logger.warning(
                "%s is enabled but %s is empty. "
                "The adapter will likely fail to connect.",
                platform.value, env_name,
            )

    # Reject known-weak placeholder tokens.
    # Ported from openclaw/openclaw#64586: users who copy .env.example
    # without changing placeholder values get a clear startup error instead
    # of a confusing "auth failed" from the platform API.
    try:
        from hermes_cli.auth import has_usable_secret
    except ImportError:
        has_usable_secret = None  # type: ignore[assignment]

    if has_usable_secret is not None:
        for platform, pconfig in config.platforms.items():
            if not pconfig.enabled:
                continue
            env_name = _token_env_names.get(platform)
            if not env_name:
                continue
            token = pconfig.token
            if token and token.strip() and not has_usable_secret(token, min_length=4):
                logger.error(
                    "%s is enabled but %s is set to a placeholder value ('%s'). "
                    "Set a real bot token before starting the gateway. "
                    "The adapter will NOT be started.",
                    platform.value, env_name, token.strip()[:6] + "...",
                )
                pconfig.enabled = False


def _apply_env_overrides(config: GatewayConfig) -> None:
    """Apply environment variable overrides to config."""

    def _enable_from_env(platform: Platform) -> PlatformConfig:
        if platform not in config.platforms:
            config.platforms[platform] = PlatformConfig(enabled=True)
            return config.platforms[platform]

        platform_config = config.platforms[platform]
        # Read (don't pop) the explicit-enable marker: the registry-driven
        # plugin-enable pass later in this function also needs it to avoid
        # re-enabling a platform the user explicitly disabled (migrated plugin
        # platforms — telegram, matrix — flow through here too, #41112). The
        # flag is cleared once for all platforms in the final cleanup at the
        # end of _apply_env_overrides.
        enabled_was_explicit = bool(platform_config.extra.get("_enabled_explicit", False))
        if not platform_config.enabled and not enabled_was_explicit:
            platform_config.enabled = True
        return platform_config
    
    # Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        telegram_config = _enable_from_env(Platform.TELEGRAM)
        telegram_config.token = telegram_token
    
    # Reply threading mode for Telegram (off/first/all)
    telegram_reply_mode = os.getenv("TELEGRAM_REPLY_TO_MODE", "").lower()
    if telegram_reply_mode in {"off", "first", "all"}:
        if Platform.TELEGRAM not in config.platforms:
            config.platforms[Platform.TELEGRAM] = PlatformConfig()
        config.platforms[Platform.TELEGRAM].reply_to_mode = telegram_reply_mode
    
    telegram_fallback_ips = os.getenv("TELEGRAM_FALLBACK_IPS", "")
    if telegram_fallback_ips:
        if Platform.TELEGRAM not in config.platforms:
            config.platforms[Platform.TELEGRAM] = PlatformConfig()
        config.platforms[Platform.TELEGRAM].extra["fallback_ips"] = [
            ip.strip() for ip in telegram_fallback_ips.split(",") if ip.strip()
        ]

    telegram_home = os.getenv("TELEGRAM_HOME_CHANNEL")
    if telegram_home and Platform.TELEGRAM in config.platforms:
        config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id=telegram_home,
            name=os.getenv("TELEGRAM_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("TELEGRAM_HOME_CHANNEL_THREAD_ID") or None,
        )
    
    # Discord
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    if discord_token:
        discord_config = _enable_from_env(Platform.DISCORD)
        discord_config.token = discord_token
    
    discord_home = os.getenv("DISCORD_HOME_CHANNEL")
    if discord_home and Platform.DISCORD in config.platforms:
        config.platforms[Platform.DISCORD].home_channel = HomeChannel(
            platform=Platform.DISCORD,
            chat_id=discord_home,
            name=os.getenv("DISCORD_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("DISCORD_HOME_CHANNEL_THREAD_ID") or None,
        )
    
    # Reply threading mode for Discord (off/first/all)
    discord_reply_mode = os.getenv("DISCORD_REPLY_TO_MODE", "").lower()
    if discord_reply_mode in {"off", "first", "all"}:
        if Platform.DISCORD not in config.platforms:
            config.platforms[Platform.DISCORD] = PlatformConfig()
        config.platforms[Platform.DISCORD].reply_to_mode = discord_reply_mode
    
    # WhatsApp (typically uses different auth mechanism)
    whatsapp_enabled = os.getenv("WHATSAPP_ENABLED", "").lower() in {"true", "1", "yes"}
    whatsapp_disabled_explicitly = os.getenv("WHATSAPP_ENABLED", "").lower() in {"false", "0", "no"}
    if Platform.WHATSAPP in config.platforms:
        # YAML config exists — respect explicit disable
        wa_cfg = config.platforms[Platform.WHATSAPP]
        if whatsapp_disabled_explicitly:
            wa_cfg.enabled = False
        elif whatsapp_enabled:
            wa_cfg.enabled = True
        # else: keep whatever the YAML set
    elif whatsapp_enabled:
        config.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True)
    whatsapp_home = os.getenv("WHATSAPP_HOME_CHANNEL")
    if whatsapp_home and Platform.WHATSAPP in config.platforms:
        config.platforms[Platform.WHATSAPP].home_channel = HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id=whatsapp_home,
            name=os.getenv("WHATSAPP_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("WHATSAPP_HOME_CHANNEL_THREAD_ID") or None,
        )

    # WhatsApp Cloud API (official Business Platform via Meta).
    # Distinct from the Baileys bridge: pure HTTP graph.facebook.com calls
    # outbound, public webhook inbound. Both adapters can run in parallel
    # against different phone numbers.
    whatsapp_cloud_phone_id = os.getenv("WHATSAPP_CLOUD_PHONE_NUMBER_ID")
    whatsapp_cloud_token = os.getenv("WHATSAPP_CLOUD_ACCESS_TOKEN")
    if whatsapp_cloud_phone_id and whatsapp_cloud_token:
        if Platform.WHATSAPP_CLOUD not in config.platforms:
            config.platforms[Platform.WHATSAPP_CLOUD] = PlatformConfig()
        config.platforms[Platform.WHATSAPP_CLOUD].enabled = True
        config.platforms[Platform.WHATSAPP_CLOUD].extra.update({
            "phone_number_id": whatsapp_cloud_phone_id,
            "access_token": whatsapp_cloud_token,
        })
        # Optional: app_id / app_secret (signature verification)
        wa_cloud_app_id = os.getenv("WHATSAPP_CLOUD_APP_ID")
        if wa_cloud_app_id:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["app_id"] = wa_cloud_app_id
        wa_cloud_app_secret = os.getenv("WHATSAPP_CLOUD_APP_SECRET")
        if wa_cloud_app_secret:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["app_secret"] = wa_cloud_app_secret
        # Optional: WABA id (analytics, future use)
        wa_cloud_waba_id = os.getenv("WHATSAPP_CLOUD_WABA_ID")
        if wa_cloud_waba_id:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["waba_id"] = wa_cloud_waba_id
        # Webhook verify token — Meta hub.verify_token shared secret
        wa_cloud_verify_token = os.getenv("WHATSAPP_CLOUD_VERIFY_TOKEN")
        if wa_cloud_verify_token:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["verify_token"] = wa_cloud_verify_token
        # Webhook server bind config (defaults baked into the adapter)
        wa_cloud_host = os.getenv("WHATSAPP_CLOUD_WEBHOOK_HOST")
        if wa_cloud_host:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["webhook_host"] = wa_cloud_host
        wa_cloud_port = os.getenv("WHATSAPP_CLOUD_WEBHOOK_PORT")
        if wa_cloud_port:
            try:
                config.platforms[Platform.WHATSAPP_CLOUD].extra["webhook_port"] = int(wa_cloud_port)
            except ValueError:
                pass
        wa_cloud_path = os.getenv("WHATSAPP_CLOUD_WEBHOOK_PATH")
        if wa_cloud_path:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["webhook_path"] = wa_cloud_path
        # Graph API version override (rarely needed)
        wa_cloud_api_version = os.getenv("WHATSAPP_CLOUD_API_VERSION")
        if wa_cloud_api_version:
            config.platforms[Platform.WHATSAPP_CLOUD].extra["api_version"] = wa_cloud_api_version
    whatsapp_cloud_home = os.getenv("WHATSAPP_CLOUD_HOME_CHANNEL")
    if whatsapp_cloud_home and Platform.WHATSAPP_CLOUD in config.platforms:
        config.platforms[Platform.WHATSAPP_CLOUD].home_channel = HomeChannel(
            platform=Platform.WHATSAPP_CLOUD,
            chat_id=whatsapp_cloud_home,
            name=os.getenv("WHATSAPP_CLOUD_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("WHATSAPP_CLOUD_HOME_CHANNEL_THREAD_ID") or None,
        )

    # Slack
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if slack_token:
        if Platform.SLACK not in config.platforms:
            # No yaml config for Slack — env-only setup, enable it
            config.platforms[Platform.SLACK] = PlatformConfig()
            config.platforms[Platform.SLACK].enabled = True
        else:
            slack_config = config.platforms[Platform.SLACK]
            # Read (don't pop) the explicit-enable marker: the registry-driven
            # plugin-enable pass below also needs it to avoid re-enabling a
            # platform the user explicitly disabled (Slack is now a plugin
            # entry — #41112). The flag is cleared once for all platforms in
            # the final cleanup at the end of _apply_env_overrides.
            enabled_was_explicit = bool(slack_config.extra.get("_enabled_explicit", False))
            if not slack_config.enabled and not enabled_was_explicit:
                # Top-level Slack settings such as channel prompts should not
                # turn an env-token setup into a disabled platform. Only an
                # explicit slack.enabled/platforms.slack.enabled false should.
                slack_config.enabled = True
        # If yaml config exists, respect its enabled flag (don't override
        # explicit enabled: false). Token is still stored so skills that
        # send Slack messages can use it without activating the gateway adapter.
        config.platforms[Platform.SLACK].token = slack_token
    slack_home = os.getenv("SLACK_HOME_CHANNEL")
    if slack_home and Platform.SLACK in config.platforms:
        config.platforms[Platform.SLACK].home_channel = HomeChannel(
            platform=Platform.SLACK,
            chat_id=slack_home,
            name=os.getenv("SLACK_HOME_CHANNEL_NAME", ""),
            thread_id=os.getenv("SLACK_HOME_CHANNEL_THREAD_ID") or None,
        )
    
    # Signal
    signal_url = os.getenv("SIGNAL_HTTP_URL")
    signal_account = os.getenv("SIGNAL_ACCOUNT")
    if signal_url and signal_account:
        signal_config = _enable_from_env(Platform.SIGNAL)
        signal_config.extra.update({
            "http_url": signal_url,
            "account": signal_account,
            "ignore_stories": os.getenv("SIGNAL_IGNORE_STORIES", "true").lower() in {"true", "1", "yes"},
        })
    signal_home = os.getenv("SIGNAL_HOME_CHANNEL")
    if signal_home and Platform.SIGNAL in config.platforms:
        config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
            platform=Platform.SIGNAL,
            chat_id=signal_home,
            name=os.getenv("SIGNAL_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("SIGNAL_HOME_CHANNEL_THREAD_ID") or None,
        )

    # Mattermost
    mattermost_token = os.getenv("MATTERMOST_TOKEN")
    if mattermost_token:
        mattermost_url = os.getenv("MATTERMOST_URL", "")
        if not mattermost_url:
            logger.warning("MATTERMOST_TOKEN set but MATTERMOST_URL is missing")
        mattermost_config = _enable_from_env(Platform.MATTERMOST)
        mattermost_config.token = mattermost_token
        mattermost_config.extra["url"] = mattermost_url
    mattermost_home = os.getenv("MATTERMOST_HOME_CHANNEL")
    if mattermost_home and Platform.MATTERMOST in config.platforms:
        config.platforms[Platform.MATTERMOST].home_channel = HomeChannel(
            platform=Platform.MATTERMOST,
            chat_id=mattermost_home,
            name=os.getenv("MATTERMOST_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("MATTERMOST_HOME_CHANNEL_THREAD_ID") or None,
        )

    # Matrix
    matrix_token = os.getenv("MATRIX_ACCESS_TOKEN")
    matrix_homeserver = os.getenv("MATRIX_HOMESERVER", "")
    if matrix_token or os.getenv("MATRIX_PASSWORD"):
        if not matrix_homeserver:
            logger.warning("MATRIX_ACCESS_TOKEN/MATRIX_PASSWORD set but MATRIX_HOMESERVER is missing")
        matrix_config = _enable_from_env(Platform.MATRIX)
        if matrix_token:
            matrix_config.token = matrix_token
        matrix_config.extra["homeserver"] = matrix_homeserver
        matrix_user = os.getenv("MATRIX_USER_ID", "")
        if matrix_user:
            matrix_config.extra["user_id"] = matrix_user
        matrix_password = os.getenv("MATRIX_PASSWORD", "")
        if matrix_password:
            matrix_config.extra["password"] = matrix_password
        matrix_e2ee_mode = os.getenv("MATRIX_E2EE_MODE", "").strip().lower()
        matrix_e2ee = (
            matrix_e2ee_mode in ("required", "require", "optional", "prefer", "preferred")
            or os.getenv("MATRIX_ENCRYPTION", "").lower() in ("true", "1", "yes")
        )
        matrix_config.extra["encryption"] = matrix_e2ee
        if matrix_e2ee_mode:
            matrix_config.extra["e2ee_mode"] = matrix_e2ee_mode
        matrix_device_id = os.getenv("MATRIX_DEVICE_ID", "")
        if matrix_device_id:
            matrix_config.extra["device_id"] = matrix_device_id
    matrix_home = os.getenv("MATRIX_HOME_ROOM")
    if matrix_home and Platform.MATRIX in config.platforms:
        config.platforms[Platform.MATRIX].home_channel = HomeChannel(
            platform=Platform.MATRIX,
            chat_id=matrix_home,
            name=os.getenv("MATRIX_HOME_ROOM_NAME", "Home"),
            thread_id=os.getenv("MATRIX_HOME_ROOM_THREAD_ID") or None,
        )

    # Home Assistant
    hass_token = os.getenv("HASS_TOKEN")
    if hass_token:
        if Platform.HOMEASSISTANT not in config.platforms:
            config.platforms[Platform.HOMEASSISTANT] = PlatformConfig()
        config.platforms[Platform.HOMEASSISTANT].enabled = True
        config.platforms[Platform.HOMEASSISTANT].token = hass_token
        hass_url = os.getenv("HASS_URL")
        if hass_url:
            config.platforms[Platform.HOMEASSISTANT].extra["url"] = hass_url

    # Email
    email_addr = os.getenv("EMAIL_ADDRESS")
    email_pwd = os.getenv("EMAIL_PASSWORD")
    email_imap = os.getenv("EMAIL_IMAP_HOST")
    email_smtp = os.getenv("EMAIL_SMTP_HOST")
    if all([email_addr, email_pwd, email_imap, email_smtp]):
        if Platform.EMAIL not in config.platforms:
            config.platforms[Platform.EMAIL] = PlatformConfig()
        config.platforms[Platform.EMAIL].enabled = True
        config.platforms[Platform.EMAIL].extra.update({
            "address": email_addr,
            "imap_host": email_imap,
            "smtp_host": email_smtp,
        })
    email_home = os.getenv("EMAIL_HOME_ADDRESS")
    if email_home and Platform.EMAIL in config.platforms:
        config.platforms[Platform.EMAIL].home_channel = HomeChannel(
            platform=Platform.EMAIL,
            chat_id=email_home,
            name=os.getenv("EMAIL_HOME_ADDRESS_NAME", "Home"),
            thread_id=os.getenv("EMAIL_HOME_ADDRESS_THREAD_ID") or None,
        )

    # SMS (Twilio)
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    if twilio_sid:
        if Platform.SMS not in config.platforms:
            config.platforms[Platform.SMS] = PlatformConfig()
        config.platforms[Platform.SMS].enabled = True
        config.platforms[Platform.SMS].api_key = os.getenv("TWILIO_AUTH_TOKEN", "")
    sms_home = os.getenv("SMS_HOME_CHANNEL")
    if sms_home and Platform.SMS in config.platforms:
        config.platforms[Platform.SMS].home_channel = HomeChannel(
            platform=Platform.SMS,
            chat_id=sms_home,
            name=os.getenv("SMS_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("SMS_HOME_CHANNEL_THREAD_ID") or None,
        )

    # API Server
    api_server_enabled = os.getenv("API_SERVER_ENABLED", "").lower() in {"true", "1", "yes"}
    api_server_key = os.getenv("API_SERVER_KEY", "")
    api_server_cors_origins = os.getenv("API_SERVER_CORS_ORIGINS", "")
    api_server_port = os.getenv("API_SERVER_PORT")
    api_server_host = os.getenv("API_SERVER_HOST")
    if api_server_enabled or api_server_key:
        if Platform.API_SERVER not in config.platforms:
            config.platforms[Platform.API_SERVER] = PlatformConfig()
        config.platforms[Platform.API_SERVER].enabled = True
        if api_server_key:
            config.platforms[Platform.API_SERVER].extra["key"] = api_server_key
        if api_server_cors_origins:
            origins = [origin.strip() for origin in api_server_cors_origins.split(",") if origin.strip()]
            if origins:
                config.platforms[Platform.API_SERVER].extra["cors_origins"] = origins
        if api_server_port:
            try:
                config.platforms[Platform.API_SERVER].extra["port"] = int(api_server_port)
            except ValueError:
                pass
        if api_server_host:
            config.platforms[Platform.API_SERVER].extra["host"] = api_server_host
        api_server_model_name = os.getenv("API_SERVER_MODEL_NAME", "")
        if api_server_model_name:
            config.platforms[Platform.API_SERVER].extra["model_name"] = api_server_model_name

    # Webhook platform
    webhook_enabled = os.getenv("WEBHOOK_ENABLED", "").lower() in {"true", "1", "yes"}
    webhook_port = os.getenv("WEBHOOK_PORT")
    webhook_secret = os.getenv("WEBHOOK_SECRET", "")
    if webhook_enabled:
        if Platform.WEBHOOK not in config.platforms:
            config.platforms[Platform.WEBHOOK] = PlatformConfig()
        config.platforms[Platform.WEBHOOK].enabled = True
        if webhook_port:
            try:
                config.platforms[Platform.WEBHOOK].extra["port"] = int(webhook_port)
            except ValueError:
                pass
        if webhook_secret:
            config.platforms[Platform.WEBHOOK].extra["secret"] = webhook_secret

    # Microsoft Graph webhook platform
    msgraph_webhook_enabled = os.getenv("MSGRAPH_WEBHOOK_ENABLED", "").lower() in {
        "true",
        "1",
        "yes",
    }
    msgraph_webhook_port = os.getenv("MSGRAPH_WEBHOOK_PORT")
    msgraph_webhook_client_state = os.getenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "")
    msgraph_webhook_resources = os.getenv("MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES", "")
    msgraph_webhook_allowed_cidrs = os.getenv(
        "MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS", ""
    )
    if (
        msgraph_webhook_enabled
        or Platform.MSGRAPH_WEBHOOK in config.platforms
        or msgraph_webhook_port
        or msgraph_webhook_client_state
        or msgraph_webhook_resources
        or msgraph_webhook_allowed_cidrs
    ):
        if Platform.MSGRAPH_WEBHOOK not in config.platforms:
            config.platforms[Platform.MSGRAPH_WEBHOOK] = PlatformConfig()
        if msgraph_webhook_enabled:
            config.platforms[Platform.MSGRAPH_WEBHOOK].enabled = True
        if msgraph_webhook_port:
            try:
                config.platforms[Platform.MSGRAPH_WEBHOOK].extra["port"] = int(
                    msgraph_webhook_port
                )
            except ValueError:
                pass
        if msgraph_webhook_client_state:
            config.platforms[Platform.MSGRAPH_WEBHOOK].extra["client_state"] = (
                msgraph_webhook_client_state
            )
        if msgraph_webhook_resources:
            resources = [
                resource.strip()
                for resource in msgraph_webhook_resources.split(",")
                if resource.strip()
            ]
            if resources:
                config.platforms[Platform.MSGRAPH_WEBHOOK].extra[
                    "accepted_resources"
                ] = resources
        if msgraph_webhook_allowed_cidrs:
            cidrs = [
                cidr.strip()
                for cidr in msgraph_webhook_allowed_cidrs.split(",")
                if cidr.strip()
            ]
            if cidrs:
                config.platforms[Platform.MSGRAPH_WEBHOOK].extra[
                    "allowed_source_cidrs"
                ] = cidrs

    # DingTalk
    dingtalk_client_id = os.getenv("DINGTALK_CLIENT_ID")
    dingtalk_client_secret = os.getenv("DINGTALK_CLIENT_SECRET")
    if dingtalk_client_id and dingtalk_client_secret:
        if Platform.DINGTALK not in config.platforms:
            config.platforms[Platform.DINGTALK] = PlatformConfig()
        config.platforms[Platform.DINGTALK].enabled = True
        config.platforms[Platform.DINGTALK].extra.update({
            "client_id": dingtalk_client_id,
            "client_secret": dingtalk_client_secret,
        })
        dingtalk_home = os.getenv("DINGTALK_HOME_CHANNEL")
        if dingtalk_home:
            config.platforms[Platform.DINGTALK].home_channel = HomeChannel(
                platform=Platform.DINGTALK,
                chat_id=dingtalk_home,
                name=os.getenv("DINGTALK_HOME_CHANNEL_NAME", "Home"),
                thread_id=os.getenv("DINGTALK_HOME_CHANNEL_THREAD_ID") or None,
            )

    # Feishu / Lark
    feishu_app_id = os.getenv("FEISHU_APP_ID")
    feishu_app_secret = os.getenv("FEISHU_APP_SECRET")
    if feishu_app_id and feishu_app_secret:
        if Platform.FEISHU not in config.platforms:
            config.platforms[Platform.FEISHU] = PlatformConfig()
        config.platforms[Platform.FEISHU].enabled = True
        config.platforms[Platform.FEISHU].extra.update({
            "app_id": feishu_app_id,
            "app_secret": feishu_app_secret,
            "domain": os.getenv("FEISHU_DOMAIN", "feishu"),
            "connection_mode": os.getenv("FEISHU_CONNECTION_MODE", "websocket"),
        })
        feishu_encrypt_key = os.getenv("FEISHU_ENCRYPT_KEY", "")
        if feishu_encrypt_key:
            config.platforms[Platform.FEISHU].extra["encrypt_key"] = feishu_encrypt_key
        feishu_verification_token = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
        if feishu_verification_token:
            config.platforms[Platform.FEISHU].extra["verification_token"] = feishu_verification_token
        feishu_home = os.getenv("FEISHU_HOME_CHANNEL")
        if feishu_home:
            config.platforms[Platform.FEISHU].home_channel = HomeChannel(
                platform=Platform.FEISHU,
                chat_id=feishu_home,
                name=os.getenv("FEISHU_HOME_CHANNEL_NAME", "Home"),
                thread_id=os.getenv("FEISHU_HOME_CHANNEL_THREAD_ID") or None,
            )

    # WeCom (Enterprise WeChat)
    wecom_bot_id = os.getenv("WECOM_BOT_ID")
    wecom_secret = os.getenv("WECOM_SECRET")
    if wecom_bot_id and wecom_secret:
        if Platform.WECOM not in config.platforms:
            config.platforms[Platform.WECOM] = PlatformConfig()
        config.platforms[Platform.WECOM].enabled = True
        config.platforms[Platform.WECOM].extra.update({
            "bot_id": wecom_bot_id,
            "secret": wecom_secret,
        })
        wecom_ws_url = os.getenv("WECOM_WEBSOCKET_URL", "")
        if wecom_ws_url:
            config.platforms[Platform.WECOM].extra["websocket_url"] = wecom_ws_url
        wecom_home = os.getenv("WECOM_HOME_CHANNEL")
        if wecom_home:
            config.platforms[Platform.WECOM].home_channel = HomeChannel(
                platform=Platform.WECOM,
                chat_id=wecom_home,
                name=os.getenv("WECOM_HOME_CHANNEL_NAME", "Home"),
                thread_id=os.getenv("WECOM_HOME_CHANNEL_THREAD_ID") or None,
            )

    # WeCom callback mode (self-built apps)
    wecom_callback_corp_id = os.getenv("WECOM_CALLBACK_CORP_ID")
    wecom_callback_corp_secret = os.getenv("WECOM_CALLBACK_CORP_SECRET")
    if wecom_callback_corp_id and wecom_callback_corp_secret:
        if Platform.WECOM_CALLBACK not in config.platforms:
            config.platforms[Platform.WECOM_CALLBACK] = PlatformConfig()
        config.platforms[Platform.WECOM_CALLBACK].enabled = True
        config.platforms[Platform.WECOM_CALLBACK].extra.update({
            "corp_id": wecom_callback_corp_id,
            "corp_secret": wecom_callback_corp_secret,
            "agent_id": os.getenv("WECOM_CALLBACK_AGENT_ID", ""),
            "token": os.getenv("WECOM_CALLBACK_TOKEN", ""),
            "encoding_aes_key": os.getenv("WECOM_CALLBACK_ENCODING_AES_KEY", ""),
            "host": os.getenv("WECOM_CALLBACK_HOST", "0.0.0.0"),
            "port": env_int("WECOM_CALLBACK_PORT", 8645),
        })

    # Weixin (personal WeChat via iLink Bot API)
    weixin_token = os.getenv("WEIXIN_TOKEN")
    weixin_account_id = os.getenv("WEIXIN_ACCOUNT_ID")
    if weixin_token or weixin_account_id:
        if Platform.WEIXIN not in config.platforms:
            config.platforms[Platform.WEIXIN] = PlatformConfig()
        config.platforms[Platform.WEIXIN].enabled = True
        if weixin_token:
            config.platforms[Platform.WEIXIN].token = weixin_token
        extra = config.platforms[Platform.WEIXIN].extra
        if weixin_account_id:
            extra["account_id"] = weixin_account_id
        weixin_base_url = os.getenv("WEIXIN_BASE_URL", "").strip()
        if weixin_base_url:
            extra["base_url"] = weixin_base_url.rstrip("/")
        weixin_cdn_base_url = os.getenv("WEIXIN_CDN_BASE_URL", "").strip()
        if weixin_cdn_base_url:
            extra["cdn_base_url"] = weixin_cdn_base_url.rstrip("/")
        weixin_dm_policy = os.getenv("WEIXIN_DM_POLICY", "").strip().lower()
        if weixin_dm_policy:
            extra["dm_policy"] = weixin_dm_policy
        weixin_group_policy = os.getenv("WEIXIN_GROUP_POLICY", "").strip().lower()
        if weixin_group_policy:
            extra["group_policy"] = weixin_group_policy
        weixin_allowed_users = os.getenv("WEIXIN_ALLOWED_USERS", "").strip()
        if weixin_allowed_users:
            extra["allow_from"] = weixin_allowed_users
        weixin_group_allowed_users = os.getenv("WEIXIN_GROUP_ALLOWED_USERS", "").strip()
        if weixin_group_allowed_users:
            extra["group_allow_from"] = weixin_group_allowed_users
        weixin_split_multiline = os.getenv("WEIXIN_SPLIT_MULTILINE_MESSAGES", "").strip()
        if weixin_split_multiline:
            extra["split_multiline_messages"] = weixin_split_multiline
        weixin_home = os.getenv("WEIXIN_HOME_CHANNEL", "").strip()
        if weixin_home:
            config.platforms[Platform.WEIXIN].home_channel = HomeChannel(
                platform=Platform.WEIXIN,
                chat_id=weixin_home,
                name=os.getenv("WEIXIN_HOME_CHANNEL_NAME", "Home"),
                thread_id=os.getenv("WEIXIN_HOME_CHANNEL_THREAD_ID") or None,
            )

    # BlueBubbles (iMessage)
    bluebubbles_server_url = os.getenv("BLUEBUBBLES_SERVER_URL")
    bluebubbles_password = os.getenv("BLUEBUBBLES_PASSWORD")
    if bluebubbles_server_url and bluebubbles_password:
        if Platform.BLUEBUBBLES not in config.platforms:
            config.platforms[Platform.BLUEBUBBLES] = PlatformConfig()
        config.platforms[Platform.BLUEBUBBLES].enabled = True
        config.platforms[Platform.BLUEBUBBLES].extra.update({
            "server_url": bluebubbles_server_url.rstrip("/"),
            "password": bluebubbles_password,
            "webhook_host": os.getenv("BLUEBUBBLES_WEBHOOK_HOST", "127.0.0.1"),
            "webhook_port": env_int("BLUEBUBBLES_WEBHOOK_PORT", 8645),
            "webhook_path": os.getenv("BLUEBUBBLES_WEBHOOK_PATH", "/bluebubbles-webhook"),
            "send_read_receipts": os.getenv("BLUEBUBBLES_SEND_READ_RECEIPTS", "true").lower() in {"true", "1", "yes"},
        })
        bluebubbles_require_mention = os.getenv("BLUEBUBBLES_REQUIRE_MENTION")
        if bluebubbles_require_mention is not None:
            config.platforms[Platform.BLUEBUBBLES].extra["require_mention"] = (
                bluebubbles_require_mention.lower() in {"true", "1", "yes", "on"}
            )
        bluebubbles_mention_patterns = os.getenv("BLUEBUBBLES_MENTION_PATTERNS")
        if bluebubbles_mention_patterns:
            try:
                parsed_patterns = json.loads(bluebubbles_mention_patterns)
            except Exception:
                parsed_patterns = [
                    part.strip()
                    for part in bluebubbles_mention_patterns.replace("\n", ",").split(",")
                    if part.strip()
                ]
            config.platforms[Platform.BLUEBUBBLES].extra["mention_patterns"] = parsed_patterns
    bluebubbles_home = os.getenv("BLUEBUBBLES_HOME_CHANNEL")
    if bluebubbles_home and Platform.BLUEBUBBLES in config.platforms:
        config.platforms[Platform.BLUEBUBBLES].home_channel = HomeChannel(
            platform=Platform.BLUEBUBBLES,
            chat_id=bluebubbles_home,
            name=os.getenv("BLUEBUBBLES_HOME_CHANNEL_NAME", "Home"),
            thread_id=os.getenv("BLUEBUBBLES_HOME_CHANNEL_THREAD_ID") or None,
        )

    # QQ (Official Bot API v2)
    qq_app_id = os.getenv("QQ_APP_ID")
    qq_client_secret = os.getenv("QQ_CLIENT_SECRET")
    if qq_app_id or qq_client_secret:
        if Platform.QQBOT not in config.platforms:
            config.platforms[Platform.QQBOT] = PlatformConfig()
        config.platforms[Platform.QQBOT].enabled = True
        extra = config.platforms[Platform.QQBOT].extra
        if qq_app_id:
            extra["app_id"] = qq_app_id
        if qq_client_secret:
            extra["client_secret"] = qq_client_secret
        qq_allowed_users = os.getenv("QQ_ALLOWED_USERS", "").strip()
        if qq_allowed_users:
            extra["allow_from"] = qq_allowed_users
        qq_group_allowed = os.getenv("QQ_GROUP_ALLOWED_USERS", "").strip()
        if qq_group_allowed:
            extra["group_allow_from"] = qq_group_allowed
        qq_home = os.getenv("QQBOT_HOME_CHANNEL", "").strip()
        qq_home_name_env = "QQBOT_HOME_CHANNEL_NAME"
        if not qq_home:
            # Back-compat: accept the pre-rename name and log a one-time warning.
            legacy_home = os.getenv("QQ_HOME_CHANNEL", "").strip()
            if legacy_home:
                qq_home = legacy_home
                qq_home_name_env = "QQ_HOME_CHANNEL_NAME"
                logging.getLogger(__name__).warning(
                    "QQ_HOME_CHANNEL is deprecated; rename to QQBOT_HOME_CHANNEL "
                    "in your .env for consistency with the platform key."
                )
        if qq_home:
            config.platforms[Platform.QQBOT].home_channel = HomeChannel(
                platform=Platform.QQBOT,
                chat_id=qq_home,
                name=os.getenv("QQBOT_HOME_CHANNEL_NAME") or os.getenv(qq_home_name_env, "Home"),
                thread_id=(
                    os.getenv("QQBOT_HOME_CHANNEL_THREAD_ID")
                    or os.getenv("QQ_HOME_CHANNEL_THREAD_ID")
                    or None
                ),
            )

    # Yuanbao — YUANBAO_APP_ID preferred
    yuanbao_app_id = os.getenv("YUANBAO_APP_ID") or os.getenv("YUANBAO_APP_KEY")
    yuanbao_app_secret = os.getenv("YUANBAO_APP_SECRET")
    if yuanbao_app_id and yuanbao_app_secret:
        if Platform.YUANBAO not in config.platforms:
            config.platforms[Platform.YUANBAO] = PlatformConfig()
        config.platforms[Platform.YUANBAO].enabled = True
        extra = config.platforms[Platform.YUANBAO].extra
        extra["app_id"] = yuanbao_app_id
        extra["app_secret"] = yuanbao_app_secret
        yuanbao_bot_id = os.getenv("YUANBAO_BOT_ID")
        if yuanbao_bot_id:
            extra["bot_id"] = yuanbao_bot_id
        yuanbao_ws_url = os.getenv("YUANBAO_WS_URL")
        if yuanbao_ws_url:
            extra["ws_url"] = yuanbao_ws_url
        yuanbao_api_domain = os.getenv("YUANBAO_API_DOMAIN")
        if yuanbao_api_domain:
            extra["api_domain"] = yuanbao_api_domain
        yuanbao_route_env = os.getenv("YUANBAO_ROUTE_ENV")
        if yuanbao_route_env:
            extra["route_env"] = yuanbao_route_env
        yuanbao_home = os.getenv("YUANBAO_HOME_CHANNEL")
        if yuanbao_home:
            config.platforms[Platform.YUANBAO].home_channel = HomeChannel(
                platform=Platform.YUANBAO,
                chat_id=yuanbao_home,
                name=os.getenv("YUANBAO_HOME_CHANNEL_NAME", "Home"),
                thread_id=os.getenv("YUANBAO_HOME_CHANNEL_THREAD_ID") or None,
            )
        yuanbao_dm_policy = os.getenv("YUANBAO_DM_POLICY")
        if yuanbao_dm_policy:
            extra["dm_policy"] = yuanbao_dm_policy.strip().lower()
        yuanbao_dm_allow_from = os.getenv("YUANBAO_DM_ALLOW_FROM")
        if yuanbao_dm_allow_from:
            extra["dm_allow_from"] = yuanbao_dm_allow_from
        yuanbao_group_policy = os.getenv("YUANBAO_GROUP_POLICY")
        if yuanbao_group_policy:
            extra["group_policy"] = yuanbao_group_policy.strip().lower()
        yuanbao_group_allow_from = os.getenv("YUANBAO_GROUP_ALLOW_FROM")
        if yuanbao_group_allow_from:
            extra["group_allow_from"] = yuanbao_group_allow_from

    # Session settings
    idle_minutes = os.getenv("SESSION_IDLE_MINUTES")
    if idle_minutes:
        try:
            config.default_reset_policy.idle_minutes = int(idle_minutes)
        except ValueError:
            pass
    
    reset_hour = os.getenv("SESSION_RESET_HOUR")
    if reset_hour:
        try:
            config.default_reset_policy.at_hour = int(reset_hour)
        except ValueError:
            pass

    # Registry-driven enable for plugin platforms.  Built-ins have explicit
    # blocks above; plugins expose check_fn() which is the single source of
    # truth for "are my env vars set?".  When it returns True, ensure the
    # platform is enabled so start() will create its adapter.  Plugins that
    # need to seed ``PlatformConfig.extra`` from env vars (e.g. Google Chat's
    # project_id / subscription_name) can supply ``env_enablement_fn`` on
    # their PlatformEntry — called here BEFORE adapter construction.
    #
    # Enablement gate (#31116): when a plugin registers ``is_connected``
    # (the "has the user actually configured credentials for this?" check),
    # we MUST consult it before flipping ``enabled = True``.  Otherwise
    # ``check_fn`` alone — which for adapter plugins typically just
    # verifies the SDK is importable / lazy-installs it — silently enables
    # platforms the user never opted into, and the gateway then tries to
    # connect to Discord / Teams / Google Chat with no token and emits
    # noisy retry-forever errors.  ``_platform_status`` was already fixed
    # for the same bug class in commit 7849a3d73; this is the runtime
    # counterpart.
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            try:
                if not entry.check_fn():
                    continue
            except Exception as e:
                logger.debug("check_fn for %s raised: %s", entry.name, e)
                continue
            platform = Platform(entry.name)
            existing_cfg = config.platforms.get(platform)
            # Respect an explicit ``enabled: false`` (YAML / gateway.json /
            # dashboard PUT).  ``_enabled_explicit`` is set in
            # load_gateway_config() (via _merge_platform_map / the shared-key
            # loop) when the user wrote ``enabled`` for this platform; if they
            # explicitly disabled it, never re-enable here just because
            # check_fn() / is_connected() pass (e.g. a token is present but the
            # user set telegram.enabled: false). #41112.
            if (
                existing_cfg is not None
                and not existing_cfg.enabled
                and bool((existing_cfg.extra or {}).get("_enabled_explicit", False))
            ):
                continue
            # Seed candidate extras from ``env_enablement_fn`` so plugins
            # whose ``is_connected`` reads ``config.extra`` (e.g. Google
            # Chat's ``_is_connected`` checks ``config.extra["project_id"]``)
            # see the same state they will after enablement. Without this,
            # Google-Chat-on-env-vars-only setups silently fail the gate
            # below even though the user is configured.  Plugins whose
            # ``is_connected`` reads env vars directly (Discord, IRC,
            # Teams, LINE, ntfy, Simplex) are unaffected; this only
            # restores Google Chat.
            seed_for_probe = None
            if entry.env_enablement_fn is not None:
                try:
                    seed_for_probe = entry.env_enablement_fn()
                except Exception as e:
                    logger.debug(
                        "env_enablement_fn for %s raised: %s", entry.name, e
                    )
                    seed_for_probe = None

            # Only consult is_connected for platforms that are NOT already
            # explicitly configured in YAML / env (existing_cfg with
            # enabled=True means the user wrote it themselves or another
            # env-var bridge enabled it — keep that decision).
            if existing_cfg is None or not existing_cfg.enabled:
                if entry.is_connected is not None:
                    try:
                        # Probe with ``enabled=True`` since we're asking
                        # "would this plugin BE configured if we enabled
                        # it?" not "is it currently enabled?". Google
                        # Chat's ``_is_connected`` short-circuits on
                        # ``config.enabled`` being False, which on the
                        # default ``PlatformConfig()`` would fail the
                        # gate even with proper env vars set.
                        if existing_cfg is not None:
                            probe_cfg = existing_cfg
                            if not probe_cfg.enabled:
                                probe_cfg = PlatformConfig(
                                    enabled=True,
                                    extra=dict(probe_cfg.extra or {}),
                                )
                        else:
                            probe_cfg = PlatformConfig(enabled=True)
                        if isinstance(seed_for_probe, dict) and seed_for_probe:
                            # Don't mutate ``existing_cfg``; the probe gets
                            # a transient view with env-seeded extras layered
                            # on top of whatever's already there.
                            probe_extra = dict(getattr(probe_cfg, "extra", {}) or {})
                            for k, v in seed_for_probe.items():
                                if k == "home_channel":
                                    continue
                                probe_extra.setdefault(k, v)
                            probe_cfg = PlatformConfig(
                                enabled=True,
                                extra=probe_extra,
                            )
                        configured = bool(entry.is_connected(probe_cfg))
                    except Exception as exc:
                        logger.debug(
                            "is_connected for %s raised: %s — skipping enablement",
                            entry.name, exc,
                        )
                        configured = False
                    if not configured:
                        logger.debug(
                            "Plugin platform '%s' available but not configured "
                            "(is_connected returned False) — skipping enable",
                            entry.name,
                        )
                        continue
            if platform not in config.platforms:
                config.platforms[platform] = PlatformConfig()
            config.platforms[platform].enabled = True
            # Commit env-seeded extras onto the now-enabled platform.
            # We've already called ``env_enablement_fn`` above (for the
            # probe); reuse that result instead of calling it twice.
            if isinstance(seed_for_probe, dict) and seed_for_probe:
                seed = dict(seed_for_probe)
                # Extract the home_channel dict (if provided) so we wire it
                # up as a proper HomeChannel dataclass.  Everything else is
                # merged into ``extra``.
                home = seed.pop("home_channel", None)
                config.platforms[platform].extra.update(seed)
                if isinstance(home, dict) and home.get("chat_id"):
                    config.platforms[platform].home_channel = HomeChannel(
                        platform=platform,
                        chat_id=str(home["chat_id"]),
                        name=str(home.get("name") or "Home"),
                        thread_id=(
                            str(home["thread_id"])
                            if home.get("thread_id")
                            else None
                        ),
                    )
    except Exception as e:
        logger.debug("Plugin platform enable pass failed: %s", e)

    # Relay (generic connector-fronted platform, EXPERIMENTAL). Enabled when a
    # connector relay URL is configured via GATEWAY_RELAY_URL (env) or
    # gateway.relay_url (config.yaml). The adapter is registered into the
    # platform_registry at gateway startup (gateway.relay.register_relay_adapter)
    # and dials OUT to the connector — so, like Telegram/Matrix, it has no public
    # inbound port and just needs Platform.RELAY present+enabled in
    # config.platforms for start_gateway()'s connect loop to bring it up. The
    # connected-checker (Platform.RELAY in _PLATFORM_CONNECTED_CHECKERS) keys on
    # extra["relay_url"], so mirror the URL into extra here.
    relay_url_env = os.getenv("GATEWAY_RELAY_URL", "").strip()
    relay_url_yaml = ""
    existing_relay = config.platforms.get(Platform.RELAY)
    if existing_relay is not None:
        relay_url_yaml = str(existing_relay.extra.get("relay_url") or "").strip()
    relay_url_val = relay_url_env or relay_url_yaml
    if relay_url_val:
        relay_config = _enable_from_env(Platform.RELAY)
        relay_config.extra["relay_url"] = relay_url_val.rstrip("/")

    for platform_config in config.platforms.values():
        platform_config.extra.pop("_enabled_explicit", None)

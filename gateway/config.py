"""
Gateway configuration management.

Handles loading and validating configuration for:
- Connected platforms (Telegram, Discord, WhatsApp)
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
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Coerce bool-ish config values, preserving a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        return default
    return is_truthy_value(value, default=default)


def _normalize_unauthorized_dm_behavior(value: Any, default: str = "pair") -> str:
    """Normalize unauthorized DM behavior to a supported value."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pair", "ignore"}:
            return normalized
    return default


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
    FEISHU = "feishu"
    WECOM = "wecom"
    WECOM_CALLBACK = "wecom_callback"
    WEIXIN = "weixin"
    BLUEBUBBLES = "bluebubbles"
    QQBOT = "qqbot"
    YUANBAO = "yuanbao"
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
    messages are sent to this home channel.
    """
    platform: Platform
    chat_id: str
    name: str  # Human-readable name for display
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HomeChannel":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            name=data.get("name", "Home"),
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
    
    # Platform-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "extra": self.extra,
            "reply_to_mode": self.reply_to_mode,
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
        
        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_channel=home_channel,
            reply_to_mode=data.get("reply_to_mode", "first"),
            extra=data.get("extra", {}),
        )


@dataclass
class StreamingConfig:
    """Configuration for real-time token streaming to messaging platforms."""
    enabled: bool = False
    transport: str = "edit"       # "edit" (progressive editMessageText) or "off"
    edit_interval: float = 1.0    # Seconds between message edits (Telegram rate-limits at ~1/s)
    buffer_threshold: int = 40    # Chars before forcing an edit
    cursor: str = " ▉"           # Cursor shown during streaming
    # Ported from openclaw/openclaw#72038.  When >0, the final edit for
    # a long-running streamed response is delivered as a fresh message
    # if the original preview has been visible for at least this many
    # seconds, so the platform's visible timestamp reflects completion
    # time instead of the preview creation time.  Currently applied to
    # Telegram only (other platforms ignore the setting).  Default 60s
    # matches the OpenClaw rollout.  Set to 0 to disable.
    fresh_final_after_seconds: float = 60.0

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
            enabled=data.get("enabled", False),
            transport=data.get("transport", "edit"),
            edit_interval=float(data.get("edit_interval", 1.0)),
            buffer_threshold=int(data.get("buffer_threshold", 40)),
            cursor=data.get("cursor", " ▉"),
            fresh_final_after_seconds=float(
                data.get("fresh_final_after_seconds", 60.0)
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
    Platform.WHATSAPP: lambda cfg: True,  # bridge handles auth
    Platform.SIGNAL: lambda cfg: bool(cfg.extra.get("http_url")),
    Platform.EMAIL: lambda cfg: bool(cfg.extra.get("address")),
    Platform.SMS: lambda cfg: bool(os.getenv("TWILIO_ACCOUNT_SID")),
    Platform.API_SERVER: lambda cfg: True,
    Platform.WEBHOOK: lambda cfg: True,
    Platform.FEISHU: lambda cfg: bool(cfg.extra.get("app_id")),
    Platform.WECOM: lambda cfg: bool(cfg.extra.get("bot_id")),
    Platform.WECOM_CALLBACK: lambda cfg: bool(
        cfg.extra.get("corp_id") or cfg.extra.get("apps")
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
    Platform.DINGTALK: lambda cfg: bool(
        (cfg.extra.get("client_id") or os.getenv("DINGTALK_CLIENT_ID"))
        and (cfg.extra.get("client_secret") or os.getenv("DINGTALK_CLIENT_SECRET"))
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

    # STT settings
    stt_enabled: bool = True  # Whether to auto-transcribe inbound voice messages

    # Session isolation in shared chats
    group_sessions_per_user: bool = True  # Isolate group/channel sessions per participant when user IDs are available
    thread_sessions_per_user: bool = False  # When False (default), threads are shared across all participants

    # Unauthorized DM policy
    unauthorized_dm_behavior: str = "pair"  # "pair" or "ignore"

    # Streaming configuration
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

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

        # Plugin-registered platforms
        try:
            from gateway.platform_registry import platform_registry
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
            "stt_enabled": self.stt_enabled,
            "group_sessions_per_user": self.group_sessions_per_user,
            "thread_sessions_per_user": self.thread_sessions_per_user,
            "unauthorized_dm_behavior": self.unauthorized_dm_behavior,
            "streaming": self.streaming.to_dict(),
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
        unauthorized_dm_behavior = _normalize_unauthorized_dm_behavior(
            data.get("unauthorized_dm_behavior"),
            "pair",
        )

        try:
            session_store_max_age_days = int(data.get("session_store_max_age_days", 90))
            if session_store_max_age_days < 0:
                session_store_max_age_days = 0
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
            stt_enabled=_coerce_bool(stt_enabled, True),
            group_sessions_per_user=_coerce_bool(group_sessions_per_user, True),
            thread_sessions_per_user=_coerce_bool(thread_sessions_per_user, False),
            unauthorized_dm_behavior=unauthorized_dm_behavior,
            streaming=StreamingConfig.from_dict(data.get("streaming", {})),
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

            if "group_sessions_per_user" in yaml_cfg:
                gw_data["group_sessions_per_user"] = yaml_cfg["group_sessions_per_user"]

            if "thread_sessions_per_user" in yaml_cfg:
                gw_data["thread_sessions_per_user"] = yaml_cfg["thread_sessions_per_user"]

            streaming_cfg = yaml_cfg.get("streaming")
            if isinstance(streaming_cfg, dict):
                gw_data["streaming"] = streaming_cfg

            if "reset_triggers" in yaml_cfg:
                gw_data["reset_triggers"] = yaml_cfg["reset_triggers"]

            if "always_log_local" in yaml_cfg:
                gw_data["always_log_local"] = yaml_cfg["always_log_local"]

            if "unauthorized_dm_behavior" in yaml_cfg:
                gw_data["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                    yaml_cfg.get("unauthorized_dm_behavior"),
                    "pair",
                )

            # Merge platforms section from config.yaml into gw_data so that
            # nested keys like platforms.webhook.extra.routes are loaded.
            yaml_platforms = yaml_cfg.get("platforms")
            platforms_data = gw_data.setdefault("platforms", {})
            if not isinstance(platforms_data, dict):
                platforms_data = {}
                gw_data["platforms"] = platforms_data
            if isinstance(yaml_platforms, dict):
                for plat_name, plat_block in yaml_platforms.items():
                    if not isinstance(plat_block, dict):
                        continue
                    existing = platforms_data.get(plat_name, {})
                    if not isinstance(existing, dict):
                        existing = {}
                    # Deep-merge extra dicts so gateway.json defaults survive
                    merged_extra = {**existing.get("extra", {}), **plat_block.get("extra", {})}
                    if plat_name == Platform.SLACK.value and "enabled" in plat_block:
                        merged_extra["_enabled_explicit"] = True
                    merged = {**existing, **plat_block}
                    if merged_extra:
                        merged["extra"] = merged_extra
                    platforms_data[plat_name] = merged
                gw_data["platforms"] = platforms_data
            for plat in Platform:
                if plat == Platform.LOCAL:
                    continue
                platform_cfg = yaml_cfg.get(plat.value)
                if not isinstance(platform_cfg, dict):
                    continue
                # Collect bridgeable keys from this platform section
                bridged = {}
                if "unauthorized_dm_behavior" in platform_cfg:
                    bridged["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                        platform_cfg.get("unauthorized_dm_behavior"),
                        gw_data.get("unauthorized_dm_behavior", "pair"),
                    )
                if "reply_prefix" in platform_cfg:
                    bridged["reply_prefix"] = platform_cfg["reply_prefix"]
                if "reply_in_thread" in platform_cfg:
                    bridged["reply_in_thread"] = platform_cfg["reply_in_thread"]
                if "require_mention" in platform_cfg:
                    bridged["require_mention"] = platform_cfg["require_mention"]
                if "free_response_channels" in platform_cfg:
                    bridged["free_response_channels"] = platform_cfg["free_response_channels"]
                if "mention_patterns" in platform_cfg:
                    bridged["mention_patterns"] = platform_cfg["mention_patterns"]
                if "dm_policy" in platform_cfg:
                    bridged["dm_policy"] = platform_cfg["dm_policy"]
                if "allow_from" in platform_cfg:
                    bridged["allow_from"] = platform_cfg["allow_from"]
                if "group_policy" in platform_cfg:
                    bridged["group_policy"] = platform_cfg["group_policy"]
                if "group_allow_from" in platform_cfg:
                    bridged["group_allow_from"] = platform_cfg["group_allow_from"]
                if plat in (Platform.DISCORD, Platform.SLACK) and "channel_skill_bindings" in platform_cfg:
                    bridged["channel_skill_bindings"] = platform_cfg["channel_skill_bindings"]
                if "channel_prompts" in platform_cfg:
                    channel_prompts = platform_cfg["channel_prompts"]
                    if isinstance(channel_prompts, dict):
                        bridged["channel_prompts"] = {str(k): v for k, v in channel_prompts.items()}
                    else:
                        bridged["channel_prompts"] = channel_prompts
                enabled_was_explicit = "enabled" in platform_cfg
                if not bridged and not enabled_was_explicit:
                    continue
                plat_data = platforms_data.setdefault(plat.value, {})
                if not isinstance(plat_data, dict):
                    plat_data = {}
                    platforms_data[plat.value] = plat_data
                if enabled_was_explicit:
                    plat_data["enabled"] = platform_cfg["enabled"]
                extra = plat_data.setdefault("extra", {})
                if not isinstance(extra, dict):
                    extra = {}
                    plat_data["extra"] = extra
                if plat == Platform.SLACK and enabled_was_explicit:
                    extra["_enabled_explicit"] = True
                extra.update(bridged)

            # Slack settings → env vars (env vars take precedence)
            slack_cfg = yaml_cfg.get("slack", {})
            if isinstance(slack_cfg, dict):
                if "require_mention" in slack_cfg and not os.getenv("SLACK_REQUIRE_MENTION"):
                    os.environ["SLACK_REQUIRE_MENTION"] = str(slack_cfg["require_mention"]).lower()
                if "strict_mention" in slack_cfg and not os.getenv("SLACK_STRICT_MENTION"):
                    os.environ["SLACK_STRICT_MENTION"] = str(slack_cfg["strict_mention"]).lower()
                if "allow_bots" in slack_cfg and not os.getenv("SLACK_ALLOW_BOTS"):
                    os.environ["SLACK_ALLOW_BOTS"] = str(slack_cfg["allow_bots"]).lower()
                frc = slack_cfg.get("free_response_channels")
                if frc is not None and not os.getenv("SLACK_FREE_RESPONSE_CHANNELS"):
                    if isinstance(frc, list):
                        frc = ",".join(str(v) for v in frc)
                    os.environ["SLACK_FREE_RESPONSE_CHANNELS"] = str(frc)
                if "reactions" in slack_cfg and not os.getenv("SLACK_REACTIONS"):
                    os.environ["SLACK_REACTIONS"] = str(slack_cfg["reactions"]).lower()

            # Discord settings → env vars (env vars take precedence)
            discord_cfg = yaml_cfg.get("discord", {})
            if isinstance(discord_cfg, dict):
                if "require_mention" in discord_cfg and not os.getenv("DISCORD_REQUIRE_MENTION"):
                    os.environ["DISCORD_REQUIRE_MENTION"] = str(discord_cfg["require_mention"]).lower()
                frc = discord_cfg.get("free_response_channels")
                if frc is not None and not os.getenv("DISCORD_FREE_RESPONSE_CHANNELS"):
                    if isinstance(frc, list):
                        frc = ",".join(str(v) for v in frc)
                    os.environ["DISCORD_FREE_RESPONSE_CHANNELS"] = str(frc)
                if "auto_thread" in discord_cfg and not os.getenv("DISCORD_AUTO_THREAD"):
                    os.environ["DISCORD_AUTO_THREAD"] = str(discord_cfg["auto_thread"]).lower()
                if "reactions" in discord_cfg and not os.getenv("DISCORD_REACTIONS"):
                    os.environ["DISCORD_REACTIONS"] = str(discord_cfg["reactions"]).lower()
                # ignored_channels: channels where bot never responds (even when mentioned)
                ic = discord_cfg.get("ignored_channels")
                if ic is not None and not os.getenv("DISCORD_IGNORED_CHANNELS"):
                    if isinstance(ic, list):
                        ic = ",".join(str(v) for v in ic)
                    os.environ["DISCORD_IGNORED_CHANNELS"] = str(ic)
                # allowed_channels: if set, bot ONLY responds in these channels (whitelist)
                ac = discord_cfg.get("allowed_channels")
                if ac is not None and not os.getenv("DISCORD_ALLOWED_CHANNELS"):
                    if isinstance(ac, list):
                        ac = ",".join(str(v) for v in ac)
                    os.environ["DISCORD_ALLOWED_CHANNELS"] = str(ac)
                # no_thread_channels: channels where bot responds directly without creating thread
                ntc = discord_cfg.get("no_thread_channels")
                if ntc is not None and not os.getenv("DISCORD_NO_THREAD_CHANNELS"):
                    if isinstance(ntc, list):
                        ntc = ",".join(str(v) for v in ntc)
                    os.environ["DISCORD_NO_THREAD_CHANNELS"] = str(ntc)
                # allow_mentions: granular control over what the bot can ping.
                # Safe defaults (no @everyone/roles) are applied in the adapter;
                # these YAML keys only override when set and let users opt back
                # into unsafe modes (e.g. roles=true) if they actually want it.
                allow_mentions_cfg = discord_cfg.get("allow_mentions")
                if isinstance(allow_mentions_cfg, dict):
                    for yaml_key, env_key in (
                        ("everyone", "DISCORD_ALLOW_MENTION_EVERYONE"),
                        ("roles", "DISCORD_ALLOW_MENTION_ROLES"),
                        ("users", "DISCORD_ALLOW_MENTION_USERS"),
                        ("replied_user", "DISCORD_ALLOW_MENTION_REPLIED_USER"),
                    ):
                        if yaml_key in allow_mentions_cfg and not os.getenv(env_key):
                            os.environ[env_key] = str(allow_mentions_cfg[yaml_key]).lower()

            # Telegram settings → env vars (env vars take precedence)
            telegram_cfg = yaml_cfg.get("telegram", {})
            if isinstance(telegram_cfg, dict):
                if "require_mention" in telegram_cfg and not os.getenv("TELEGRAM_REQUIRE_MENTION"):
                    os.environ["TELEGRAM_REQUIRE_MENTION"] = str(telegram_cfg["require_mention"]).lower()
                if "mention_patterns" in telegram_cfg and not os.getenv("TELEGRAM_MENTION_PATTERNS"):
                    os.environ["TELEGRAM_MENTION_PATTERNS"] = json.dumps(telegram_cfg["mention_patterns"])
                frc = telegram_cfg.get("free_response_chats")
                if frc is not None and not os.getenv("TELEGRAM_FREE_RESPONSE_CHATS"):
                    if isinstance(frc, list):
                        frc = ",".join(str(v) for v in frc)
                    os.environ["TELEGRAM_FREE_RESPONSE_CHATS"] = str(frc)
                ignored_threads = telegram_cfg.get("ignored_threads")
                if ignored_threads is not None and not os.getenv("TELEGRAM_IGNORED_THREADS"):
                    if isinstance(ignored_threads, list):
                        ignored_threads = ",".join(str(v) for v in ignored_threads)
                    os.environ["TELEGRAM_IGNORED_THREADS"] = str(ignored_threads)
                if "reactions" in telegram_cfg and not os.getenv("TELEGRAM_REACTIONS"):
                    os.environ["TELEGRAM_REACTIONS"] = str(telegram_cfg["reactions"]).lower()
                if "proxy_url" in telegram_cfg and not os.getenv("TELEGRAM_PROXY"):
                    os.environ["TELEGRAM_PROXY"] = str(telegram_cfg["proxy_url"]).strip()
                allowed_users = telegram_cfg.get("allow_from")
                if allowed_users is not None and not os.getenv("TELEGRAM_ALLOWED_USERS"):
                    if isinstance(allowed_users, list):
                        allowed_users = ",".join(str(v) for v in allowed_users)
                    os.environ["TELEGRAM_ALLOWED_USERS"] = str(allowed_users)
                group_allowed_users = telegram_cfg.get("group_allow_from")
                if group_allowed_users is not None and not os.getenv("TELEGRAM_GROUP_ALLOWED_USERS"):
                    if isinstance(group_allowed_users, list):
                        group_allowed_users = ",".join(str(v) for v in group_allowed_users)
                    os.environ["TELEGRAM_GROUP_ALLOWED_USERS"] = str(group_allowed_users)
                group_allowed_chats = telegram_cfg.get("group_allowed_chats")
                if group_allowed_chats is not None and not os.getenv("TELEGRAM_GROUP_ALLOWED_CHATS"):
                    if isinstance(group_allowed_chats, list):
                        group_allowed_chats = ",".join(str(v) for v in group_allowed_chats)
                    os.environ["TELEGRAM_GROUP_ALLOWED_CHATS"] = str(group_allowed_chats)
                if "disable_link_previews" in telegram_cfg:
                    plat_data = platforms_data.setdefault(Platform.TELEGRAM.value, {})
                    if not isinstance(plat_data, dict):
                        plat_data = {}
                        platforms_data[Platform.TELEGRAM.value] = plat_data
                    extra = plat_data.setdefault("extra", {})
                    if not isinstance(extra, dict):
                        extra = {}
                        plat_data["extra"] = extra
                    extra["disable_link_previews"] = telegram_cfg["disable_link_previews"]

            whatsapp_cfg = yaml_cfg.get("whatsapp", {})
            if isinstance(whatsapp_cfg, dict):
                if "require_mention" in whatsapp_cfg and not os.getenv("WHATSAPP_REQUIRE_MENTION"):
                    os.environ["WHATSAPP_REQUIRE_MENTION"] = str(whatsapp_cfg["require_mention"]).lower()
                if "mention_patterns" in whatsapp_cfg and not os.getenv("WHATSAPP_MENTION_PATTERNS"):
                    os.environ["WHATSAPP_MENTION_PATTERNS"] = json.dumps(whatsapp_cfg["mention_patterns"])
                frc = whatsapp_cfg.get("free_response_chats")
                if frc is not None and not os.getenv("WHATSAPP_FREE_RESPONSE_CHATS"):
                    if isinstance(frc, list):
                        frc = ",".join(str(v) for v in frc)
                    os.environ["WHATSAPP_FREE_RESPONSE_CHATS"] = str(frc)
                if "dm_policy" in whatsapp_cfg and not os.getenv("WHATSAPP_DM_POLICY"):
                    os.environ["WHATSAPP_DM_POLICY"] = str(whatsapp_cfg["dm_policy"]).lower()
                af = whatsapp_cfg.get("allow_from")
                if af is not None and not os.getenv("WHATSAPP_ALLOWED_USERS"):
                    if isinstance(af, list):
                        af = ",".join(str(v) for v in af)
                    os.environ["WHATSAPP_ALLOWED_USERS"] = str(af)
                if "group_policy" in whatsapp_cfg and not os.getenv("WHATSAPP_GROUP_POLICY"):
                    os.environ["WHATSAPP_GROUP_POLICY"] = str(whatsapp_cfg["group_policy"]).lower()
                gaf = whatsapp_cfg.get("group_allow_from")
                if gaf is not None and not os.getenv("WHATSAPP_GROUP_ALLOWED_USERS"):
                    if isinstance(gaf, list):
                        gaf = ",".join(str(v) for v in gaf)
                    os.environ["WHATSAPP_GROUP_ALLOWED_USERS"] = str(gaf)

            # DingTalk settings → env vars (env vars take precedence)
            dingtalk_cfg = yaml_cfg.get("dingtalk", {})
            if isinstance(dingtalk_cfg, dict):
                if "require_mention" in dingtalk_cfg and not os.getenv("DINGTALK_REQUIRE_MENTION"):
                    os.environ["DINGTALK_REQUIRE_MENTION"] = str(dingtalk_cfg["require_mention"]).lower()
                if "mention_patterns" in dingtalk_cfg and not os.getenv("DINGTALK_MENTION_PATTERNS"):
                    os.environ["DINGTALK_MENTION_PATTERNS"] = json.dumps(dingtalk_cfg["mention_patterns"])
                frc = dingtalk_cfg.get("free_response_chats")
                if frc is not None and not os.getenv("DINGTALK_FREE_RESPONSE_CHATS"):
                    if isinstance(frc, list):
                        frc = ",".join(str(v) for v in frc)
                    os.environ["DINGTALK_FREE_RESPONSE_CHATS"] = str(frc)
                allowed = dingtalk_cfg.get("allowed_users")
                if allowed is not None and not os.getenv("DINGTALK_ALLOWED_USERS"):
                    if isinstance(allowed, list):
                        allowed = ",".join(str(v) for v in allowed)
                    os.environ["DINGTALK_ALLOWED_USERS"] = str(allowed)

            # Matrix settings → env vars (env vars take precedence)
            matrix_cfg = yaml_cfg.get("matrix", {})
            if isinstance(matrix_cfg, dict):
                if "require_mention" in matrix_cfg and not os.getenv("MATRIX_REQUIRE_MENTION"):
                    os.environ["MATRIX_REQUIRE_MENTION"] = str(matrix_cfg["require_mention"]).lower()
                frc = matrix_cfg.get("free_response_rooms")
                if frc is not None and not os.getenv("MATRIX_FREE_RESPONSE_ROOMS"):
                    if isinstance(frc, list):
                        frc = ",".join(str(v) for v in frc)
                    os.environ["MATRIX_FREE_RESPONSE_ROOMS"] = str(frc)
                if "auto_thread" in matrix_cfg and not os.getenv("MATRIX_AUTO_THREAD"):
                    os.environ["MATRIX_AUTO_THREAD"] = str(matrix_cfg["auto_thread"]).lower()
                if "dm_mention_threads" in matrix_cfg and not os.getenv("MATRIX_DM_MENTION_THREADS"):
                    os.environ["MATRIX_DM_MENTION_THREADS"] = str(matrix_cfg["dm_mention_threads"]).lower()

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
    
    # Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        if Platform.TELEGRAM not in config.platforms:
            config.platforms[Platform.TELEGRAM] = PlatformConfig()
        config.platforms[Platform.TELEGRAM].enabled = True
        config.platforms[Platform.TELEGRAM].token = telegram_token
    
    # Reply threading mode for Telegram (off/first/all)
    telegram_reply_mode = os.getenv("TELEGRAM_REPLY_TO_MODE", "").lower()
    if telegram_reply_mode in ("off", "first", "all"):
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
        )
    
    # Discord
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    if discord_token:
        if Platform.DISCORD not in config.platforms:
            config.platforms[Platform.DISCORD] = PlatformConfig()
        config.platforms[Platform.DISCORD].enabled = True
        config.platforms[Platform.DISCORD].token = discord_token
    
    discord_home = os.getenv("DISCORD_HOME_CHANNEL")
    if discord_home and Platform.DISCORD in config.platforms:
        config.platforms[Platform.DISCORD].home_channel = HomeChannel(
            platform=Platform.DISCORD,
            chat_id=discord_home,
            name=os.getenv("DISCORD_HOME_CHANNEL_NAME", "Home"),
        )
    
    # Reply threading mode for Discord (off/first/all)
    discord_reply_mode = os.getenv("DISCORD_REPLY_TO_MODE", "").lower()
    if discord_reply_mode in ("off", "first", "all"):
        if Platform.DISCORD not in config.platforms:
            config.platforms[Platform.DISCORD] = PlatformConfig()
        config.platforms[Platform.DISCORD].reply_to_mode = discord_reply_mode
    
    # WhatsApp (typically uses different auth mechanism)
    whatsapp_enabled = os.getenv("WHATSAPP_ENABLED", "").lower() in ("true", "1", "yes")
    if whatsapp_enabled:
        if Platform.WHATSAPP not in config.platforms:
            config.platforms[Platform.WHATSAPP] = PlatformConfig()
        config.platforms[Platform.WHATSAPP].enabled = True
    
    # Slack
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if slack_token:
        if Platform.SLACK not in config.platforms:
            # No yaml config for Slack — env-only setup, enable it
            config.platforms[Platform.SLACK] = PlatformConfig()
            config.platforms[Platform.SLACK].enabled = True
        else:
            slack_config = config.platforms[Platform.SLACK]
            enabled_was_explicit = bool(slack_config.extra.pop("_enabled_explicit", False))
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
        )
    
    # Signal
    signal_url = os.getenv("SIGNAL_HTTP_URL")
    signal_account = os.getenv("SIGNAL_ACCOUNT")
    if signal_url and signal_account:
        if Platform.SIGNAL not in config.platforms:
            config.platforms[Platform.SIGNAL] = PlatformConfig()
        config.platforms[Platform.SIGNAL].enabled = True
        config.platforms[Platform.SIGNAL].extra.update({
            "http_url": signal_url,
            "account": signal_account,
            "ignore_stories": os.getenv("SIGNAL_IGNORE_STORIES", "true").lower() in ("true", "1", "yes"),
        })
    signal_home = os.getenv("SIGNAL_HOME_CHANNEL")
    if signal_home and Platform.SIGNAL in config.platforms:
        config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
            platform=Platform.SIGNAL,
            chat_id=signal_home,
            name=os.getenv("SIGNAL_HOME_CHANNEL_NAME", "Home"),
        )

    # Mattermost
    mattermost_token = os.getenv("MATTERMOST_TOKEN")
    if mattermost_token:
        mattermost_url = os.getenv("MATTERMOST_URL", "")
        if not mattermost_url:
            logger.warning("MATTERMOST_TOKEN set but MATTERMOST_URL is missing")
        if Platform.MATTERMOST not in config.platforms:
            config.platforms[Platform.MATTERMOST] = PlatformConfig()
        config.platforms[Platform.MATTERMOST].enabled = True
        config.platforms[Platform.MATTERMOST].token = mattermost_token
        config.platforms[Platform.MATTERMOST].extra["url"] = mattermost_url
    mattermost_home = os.getenv("MATTERMOST_HOME_CHANNEL")
    if mattermost_home and Platform.MATTERMOST in config.platforms:
        config.platforms[Platform.MATTERMOST].home_channel = HomeChannel(
            platform=Platform.MATTERMOST,
            chat_id=mattermost_home,
            name=os.getenv("MATTERMOST_HOME_CHANNEL_NAME", "Home"),
        )

    # Matrix
    matrix_token = os.getenv("MATRIX_ACCESS_TOKEN")
    matrix_homeserver = os.getenv("MATRIX_HOMESERVER", "")
    if matrix_token or os.getenv("MATRIX_PASSWORD"):
        if not matrix_homeserver:
            logger.warning("MATRIX_ACCESS_TOKEN/MATRIX_PASSWORD set but MATRIX_HOMESERVER is missing")
        if Platform.MATRIX not in config.platforms:
            config.platforms[Platform.MATRIX] = PlatformConfig()
        config.platforms[Platform.MATRIX].enabled = True
        if matrix_token:
            config.platforms[Platform.MATRIX].token = matrix_token
        config.platforms[Platform.MATRIX].extra["homeserver"] = matrix_homeserver
        matrix_user = os.getenv("MATRIX_USER_ID", "")
        if matrix_user:
            config.platforms[Platform.MATRIX].extra["user_id"] = matrix_user
        matrix_password = os.getenv("MATRIX_PASSWORD", "")
        if matrix_password:
            config.platforms[Platform.MATRIX].extra["password"] = matrix_password
        matrix_e2ee = os.getenv("MATRIX_ENCRYPTION", "").lower() in ("true", "1", "yes")
        config.platforms[Platform.MATRIX].extra["encryption"] = matrix_e2ee
        matrix_device_id = os.getenv("MATRIX_DEVICE_ID", "")
        if matrix_device_id:
            config.platforms[Platform.MATRIX].extra["device_id"] = matrix_device_id
    matrix_home = os.getenv("MATRIX_HOME_ROOM")
    if matrix_home and Platform.MATRIX in config.platforms:
        config.platforms[Platform.MATRIX].home_channel = HomeChannel(
            platform=Platform.MATRIX,
            chat_id=matrix_home,
            name=os.getenv("MATRIX_HOME_ROOM_NAME", "Home"),
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
        )

    # API Server
    api_server_enabled = os.getenv("API_SERVER_ENABLED", "").lower() in ("true", "1", "yes")
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
    webhook_enabled = os.getenv("WEBHOOK_ENABLED", "").lower() in ("true", "1", "yes")
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
            "port": int(os.getenv("WECOM_CALLBACK_PORT", "8645")),
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
            "webhook_port": int(os.getenv("BLUEBUBBLES_WEBHOOK_PORT", "8645")),
            "webhook_path": os.getenv("BLUEBUBBLES_WEBHOOK_PATH", "/bluebubbles-webhook"),
            "send_read_receipts": os.getenv("BLUEBUBBLES_SEND_READ_RECEIPTS", "true").lower() in ("true", "1", "yes"),
        })
    bluebubbles_home = os.getenv("BLUEBUBBLES_HOME_CHANNEL")
    if bluebubbles_home and Platform.BLUEBUBBLES in config.platforms:
        config.platforms[Platform.BLUEBUBBLES].home_channel = HomeChannel(
            platform=Platform.BLUEBUBBLES,
            chat_id=bluebubbles_home,
            name=os.getenv("BLUEBUBBLES_HOME_CHANNEL_NAME", "Home"),
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
    # platform is enabled so start() will create its adapter.
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
            if platform not in config.platforms:
                config.platforms[platform] = PlatformConfig()
            config.platforms[platform].enabled = True
    except Exception as e:
        logger.debug("Plugin platform enable pass failed: %s", e)

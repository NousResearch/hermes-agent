"""Gateway configuration: connected platforms, home channels, session reset
policies and delivery preferences, loaded from config.yaml / gateway.json / env.
"""

import contextlib
import logging
import math
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from hermes_cli.config import get_hermes_home
from agent.secret_scope import current_secret_scope, get_secret as _get_secret
from gateway.shutdown_watchdog import (
    DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
)
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "no", "off"})


def _bool_token(value: Any) -> Optional[bool]:
    """True/False for a recognized truthy/falsy token, else None."""
    token = str(value).strip().lower()
    return True if token in _TRUTHY_STRINGS else False if token in _FALSY_STRINGS else None


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Coerce bool-ish config values, preserving a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        parsed = _bool_token(value)
        return default if parsed is None else parsed
    return is_truthy_value(value, default=default)


def _normalize_multiplex_profile_allowlist(value: Any) -> Optional[List[str]]:
    """Normalize the optional named-profile allowlist: ``None`` = serve all; a malformed
    outer value fails safe to ``[]`` (default profile only); bad entries are skipped."""
    if value is None:
        return None
    if not isinstance(value, list):
        logger.warning(
            "Invalid gateway.multiplex_profile_allowlist (expected a list, got %s); "
            "serving only the default profile",
            type(value).__name__,
        )
        return []

    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    normalized: List[str] = []
    for entry in value:
        if not isinstance(entry, str):
            logger.warning("Skipping invalid gateway.multiplex_profile_allowlist entry %r (expected a profile name)", entry)
            continue
        try:
            name = normalize_profile_name(entry)
            validate_profile_name(name)
        except ValueError:
            logger.warning("Skipping invalid gateway.multiplex_profile_allowlist entry %r", entry)
            continue
        if name != "default" and name not in normalized:
            normalized.append(name)
    return normalized


def _env_multiplex_profiles_override() -> "bool | None":
    """GATEWAY_MULTIPLEX_PROFILES operator override: True/False for a recognized token.

    ``None`` when unset, blank, or unrecognized so the caller keeps the config.yaml
    value (env > config > default). Blank is deliberately ``None``, not ``False``:
    a provisioned-but-unpopulated Fly secret arrives as ``""`` and must NOT shadow
    a config.yaml opt-in.
    """
    raw = os.getenv("GATEWAY_MULTIPLEX_PROFILES")
    if not (raw or "").strip():
        return None
    parsed = _bool_token(raw)
    if parsed is None:
        logger.warning(
            "Ignoring unrecognized GATEWAY_MULTIPLEX_PROFILES=%r "
            "(expected one of %s or %s); falling back to config.yaml.",
            raw, sorted(_TRUTHY_STRINGS), sorted(_FALSY_STRINGS),
        )
    return parsed


def _normalize_transport_token(value: Any) -> str:
    """Canonical streaming transport token. YAML 1.1 parses bare ``on``/``off`` as
    booleans (``mode: off`` → ``False`` → ``"false"`` would ENABLE streaming), so
    booleans map to ``"auto"``/``"off"``; anything else lower-cases, default ``"auto"``."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "auto" if value else "off"
    return str(value).strip().lower() or "auto"


def _coerce_num(cast, value: Any, default):
    # OverflowError: ``int(float("inf"))`` — non-finite YAML must degrade, not abort loading.
    try:
        return default if value is None else cast(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    return _coerce_num(float, value, default)


def _coerce_int(value: Any, default: int) -> int:
    return _coerce_num(int, value, default)


def _coerce_optional_positive_int(value: Any, key: str) -> Optional[int]:
    """``None``/0/negative disable; malformed values are ignored with a warning so a typo never blocks startup."""
    if value is None:
        return None
    try:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(value)
        parsed = int(value.strip(), 10) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r (expected a positive integer; 0/null disables)", key, value)
        return None
    return parsed if parsed > 0 else None


_SYSTEMD_WATCHDOG_MAX_SECONDS = 2_147_483_647


def coerce_systemd_watchdog_seconds(
    value: Any, key: str = "gateway.systemd_watchdog_seconds"
) -> int:
    """Bounded positive watchdog interval, or zero when disabled/invalid. Shared by runtime
    and service generation so a value can never enable ``Type=notify`` without heartbeats."""
    if value is None:
        return 0
    parsed: Optional[int] = None
    if isinstance(value, int) and not isinstance(value, bool):
        parsed = value
    elif isinstance(value, str) and value.strip().isascii() and value.strip().isdecimal():
        with contextlib.suppress(TypeError, ValueError, OverflowError):  # int() digit limit
            parsed = int(value.strip(), 10)
    if parsed is None:
        logger.warning("Ignoring invalid %s (expected a positive integer)", key)
        return 0
    if parsed and not 0 < parsed <= _SYSTEMD_WATCHDOG_MAX_SECONDS:
        logger.warning("Ignoring invalid %s (expected an integer from 1 to %d)", key, _SYSTEMD_WATCHDOG_MAX_SECONDS)
        return 0
    return parsed


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_choice(value: Any, choices: set, default: str) -> str:
    """Lower-cased *value* when it is one of *choices*, else *default*."""
    normalized = value.strip().lower() if isinstance(value, str) else None
    return normalized if normalized in choices else default


def _dict_slot(container: dict, key: str) -> dict:
    """Get-or-create ``container[key]`` as a dict, replacing a non-dict value with ``{}``."""
    value = container.setdefault(key, {})
    if not isinstance(value, dict):
        value = {}
        container[key] = value
    return value


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """Env read through the active profile secret scope when present (multiplexed
    per-profile secrets must win); otherwise legacy ``os.getenv``."""
    if current_secret_scope() is not None:
        scope_val = _get_secret(name, None)
        return scope_val if scope_val is not None else default
    return os.environ.get(name, default)


def _getenv_str(name: str, default: str = "") -> str:
    return val if (val := _getenv(name, default)) is not None else default


_Platform__bundled_plugin_names: Optional[set] = None  # cached outside the enum: never a member


class Platform(Enum):
    """Supported messaging platforms. Plugin platforms are dynamic members created on
    demand by ``_missing_`` and cached so ``Platform("irc") is Platform("irc")`` holds."""
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
        """Accept unknown names only for bundled or runtime-registered plugin adapters (no enum pollution)."""
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip().lower()
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]
        global _Platform__bundled_plugin_names
        if _Platform__bundled_plugin_names is None:
            _Platform__bundled_plugin_names = cls._scan_bundled_plugin_platforms()
        registered = value in _Platform__bundled_plugin_names
        if not registered:
            with contextlib.suppress(Exception):
                from gateway.platform_registry import platform_registry
                registered = platform_registry.is_registered(value)
        return cls._add_pseudo_member(value) if registered else None

    @classmethod
    def _add_pseudo_member(cls, value: str) -> "Platform":
        pseudo = object.__new__(cls)
        pseudo._value_ = value
        pseudo._name_ = value.upper().replace("-", "_").replace(" ", "_")
        cls._value2member_map_[value] = pseudo
        cls._member_map_[pseudo._name_] = pseudo
        return pseudo

    @classmethod
    def _scan_bundled_plugin_platforms(cls) -> set:
        """Names of bundled platform plugins under ``plugins/platforms/``."""
        try:
            platforms_dir = Path(__file__).parent.parent / "plugins" / "platforms"
            return {
                child.name.lower()
                for child in (platforms_dir.iterdir() if platforms_dir.is_dir() else ())
                if child.is_dir() and (child / "__init__.py").exists()
                and ((child / "plugin.yaml").exists() or (child / "plugin.yml").exists())
            }
        except Exception:
            return set()


# Built-in values snapshotted before any dynamic _missing_ lookup.
_BUILTIN_PLATFORM_VALUES = frozenset(m.value for m in Platform.__members__.values())

# Platforms that bind a host TCP port. In a multiplexer only the default profile owns the
# shared listener, so a SECONDARY profile enabling one is a misconfiguration (single source
# of truth for gateway/run.py and hermes_cli/web_server.py validation).
PORT_BINDING_PLATFORM_VALUES = frozenset({
    "webhook", "api_server", "msgraph_webhook", "feishu", "wecom_callback",
    "bluebubbles", "sms", "whatsapp_cloud", "line", "teams",
})
# Platforms that only bind in one connection mode (Feishu's default websocket mode is outbound).
PORT_BINDING_CONDITIONAL_MODES: dict[str, str] = {"feishu": "webhook"}


def platform_binds_port(platform_value: str, extra: Optional[dict] = None) -> bool:
    """True when *platform_value* actually binds a port for *extra* config."""
    if platform_value not in PORT_BINDING_PLATFORM_VALUES:
        return False
    expected_mode = PORT_BINDING_CONDITIONAL_MODES.get(platform_value)
    return expected_mode is None or str((extra or {}).get("connection_mode", "websocket")).strip().lower() == expected_mode


@dataclass
class HomeChannel:
    """Default destination for a platform (``deliver="telegram"`` without a chat ID);
    ``thread_id`` routes the bare target to the topic where /sethome was run."""
    platform: Platform
    chat_id: str
    name: str
    thread_id: Optional[str] = None
    # Authenticated logical-target provenance (relay egress re-attaches; connector stays the authz boundary).
    user_id: Optional[str] = None
    scope_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        optional = {k: v for k in ("thread_id", "user_id", "scope_id") if (v := getattr(self, k))}
        return {"platform": self.platform.value, "chat_id": self.chat_id, "name": self.name, **optional}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HomeChannel":
        optional = {k: str(data[k]) if data.get(k) else None for k in ("thread_id", "user_id", "scope_id")}
        return cls(platform=Platform(data["platform"]), chat_id=str(data["chat_id"]), name=data.get("name", "Home"), **optional)


def persist_home_channel(home: HomeChannel, *, enabled_if_new: bool = False) -> None:
    """Persist a logical home without falsely enabling a Relay-fronted adapter."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    platform_config = _dict_slot(_dict_slot(config, "platforms"), home.platform.value)
    if enabled_if_new:
        platform_config.setdefault("enabled", True)
    platform_config["home_channel"] = home.to_dict()
    save_config(config)


@dataclass
class SessionResetPolicy:
    """When sessions reset: "daily" (at ``at_hour``), "idle" (after ``idle_minutes``),
    "both" (whichever first), "none" (default: only compression manages context)."""
    mode: str = "none"
    at_hour: int = 4  # 0-23, local time
    idle_minutes: int = 1440
    notify: bool = True  # Notify the user when auto-reset occurs
    notify_exclude_platforms: tuple = ("api_server", "webhook")
    # A background process this old no longer blocks reset (not killed, only ignored by the guard).
    # A forgotten preview server should not keep a session alive forever (#29177). Raise this if you run
    # legitimate multi-day jobs whose liveness should pin the conversation open.
    bg_process_max_age_hours: int = 24

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "notify_exclude_platforms": list(self.notify_exclude_platforms)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionResetPolicy":
        data = _coerce_dict(data)
        exclude = data.get("notify_exclude_platforms")
        # Missing keys and explicit YAML nulls both take the field default.
        plain = {
            f.name: f.default if data.get(f.name) is None else data[f.name]
            for f in fields(cls) if f.name not in ("notify", "notify_exclude_platforms")
        }
        return cls(
            notify=_coerce_bool(data.get("notify"), True),
            notify_exclude_platforms=tuple(exclude) if exclude is not None else ("api_server", "webhook"),
            **plain,
        )


@dataclass
class ChannelOverride:
    """Per-channel model/provider/system_prompt override (``platforms.<name>.channel_overrides[channel_id]``)."""
    model: Optional[str] = None
    provider: Optional[str] = None
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelOverride":
        return cls(**{f.name: data.get(f.name) for f in fields(cls)}) if data else cls()


# Platforms whose primary credential is ``PlatformConfig.token`` → its env var (empty-token
# warnings; multiplex primary-startup gate in ``gateway.run``). Platforms absent here
# authenticate another way and must never be skipped for a missing token.
# Platforms absent from this map authenticate some other way (session files, port-bound webhooks,
# api_key-only) and must never be skipped for a missing token. See #64674.
PLATFORM_TOKEN_ENV_NAMES: dict["Platform", str] = {
    Platform.TELEGRAM: "TELEGRAM_BOT_TOKEN",
    Platform.DISCORD: "DISCORD_BOT_TOKEN",
    Platform.SLACK: "SLACK_BOT_TOKEN",
    Platform.MATTERMOST: "MATTERMOST_TOKEN",
    Platform.MATRIX: "MATRIX_ACCESS_TOKEN",
    Platform.WEIXIN: "WEIXIN_TOKEN",
}


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""
    enabled: bool = False
    token: Optional[str] = None
    api_key: Optional[str] = None  # API key if different from token
    home_channel: Optional[HomeChannel] = None
    reply_to_mode: str = "first"  # "off" never threads, "first" only the first chunk, "all" every chunk
    gateway_restart_notification: bool = True  # "♻️ Gateway online/restarted" pings; noise on end-user platforms
    typing_indicator: bool = True  # drives _keep_typing; False where unwanted (Slack setStatus blocks compose)
    # Working-state text for text-rendering indicators (Slack status, Google Chat marker); None = platform default.
    typing_status_text: Optional[str] = None
    channel_overrides: Dict[str, ChannelOverride] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)  # Platform-specific settings

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled, "extra": self.extra, "reply_to_mode": self.reply_to_mode,
            "gateway_restart_notification": self.gateway_restart_notification,
            "typing_indicator": self.typing_indicator,
            **({"typing_status_text": self.typing_status_text} if self.typing_status_text is not None else {}),
            **{k: v for k in ("token", "api_key") if (v := getattr(self, k))},
        }
        if self.home_channel:
            result["home_channel"] = self.home_channel.to_dict()
        if self.channel_overrides:
            result["channel_overrides"] = {cid: ov.to_dict() for cid, ov in self.channel_overrides.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformConfig":
        data = _coerce_dict(data)
        home = data.get("home_channel")
        # The typing/restart-notification keys may be top-level or bridged into ``extra``; top-level wins.
        extra = _coerce_dict(data.get("extra", {}))

        def toplevel_or_extra(key: str) -> Any:
            value = data.get(key)
            return extra.get(key) if value is None else value

        raw_overrides = data.get("channel_overrides") or {}
        channel_overrides = {
            str(cid): ChannelOverride.from_dict(ov_data)
            for cid, ov_data in raw_overrides.items()
            if isinstance(ov_data, dict)
        } if isinstance(raw_overrides, dict) else {}

        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_channel=HomeChannel.from_dict(home) if isinstance(home, dict) else None,
            reply_to_mode=data.get("reply_to_mode", "first"),
            gateway_restart_notification=_coerce_bool(toplevel_or_extra("gateway_restart_notification"), True),
            typing_indicator=_coerce_bool(toplevel_or_extra("typing_indicator"), True),
            typing_status_text=toplevel_or_extra("typing_status_text"),  # string passthrough, no coercion
            channel_overrides=channel_overrides,
            extra=extra,
        )


# Shared by StreamingConfig and StreamConsumerConfig. Tuned for Telegram's ~1 edit/s
# flood envelope; the small buffer threshold makes short DM replies feel instant.
DEFAULT_STREAMING_EDIT_INTERVAL: float = 0.8
DEFAULT_STREAMING_BUFFER_THRESHOLD: int = 24
DEFAULT_STREAMING_CURSOR: str = " ▉"


@dataclass
class StreamingConfig:
    """Real-time token streaming to messaging platforms."""
    enabled: bool = False
    # "auto" prefers native drafts (Telegram sendMessageDraft) with edit fallback (adapters without
    # draft support use the edit path unchanged); "draft" / "edit" force one; "off" disables.
    transport: str = "auto"
    edit_interval: float = DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = DEFAULT_STREAMING_CURSOR
    # >0: final edit becomes a fresh message once the preview was visible this long (Telegram only; 0 = off).
    # Ported from openclaw/openclaw#72038. When >0, the final edit for a long-running streamed response is
    # delivered as a fresh message if the original preview has been visible for at least this many seconds,
    # so the platform's visible timestamp reflects completion time instead of the preview creation time.
    # Currently applied to Telegram only (other platforms ignore the setting). Default 0 disables the
    # fresh-message replacement path; set >0 to opt in.
    fresh_final_after_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingConfig":
        if not isinstance(data, dict) or not data:
            return cls()

        # ``mode`` is a transport alias that ALSO implies ``enabled`` (``mode: off`` disables;
        # explicit ``enabled`` wins). A bare ``transport`` does NOT imply enabled:
        # ``streaming.enabled`` is the documented master switch.
        raw_transport = data.get("transport")
        raw_mode = data.get("mode")
        if "enabled" in data:
            enabled = _coerce_bool(data.get("enabled"), False)
        else:
            enabled = raw_mode is not None and _normalize_transport_token(raw_mode) != "off"
        return cls(
            enabled=enabled,
            transport=_normalize_transport_token(raw_transport if raw_transport is not None else raw_mode),
            edit_interval=_coerce_float(data.get("edit_interval"), DEFAULT_STREAMING_EDIT_INTERVAL),
            buffer_threshold=_coerce_int(data.get("buffer_threshold"), DEFAULT_STREAMING_BUFFER_THRESHOLD),
            cursor=data.get("cursor", DEFAULT_STREAMING_CURSOR),
            fresh_final_after_seconds=_coerce_float(data.get("fresh_final_after_seconds"), 0.0),
        )


def _has_usable_api_server_key(key: object) -> bool:
    """True when API_SERVER_KEY is strong enough for the adapter to start (mirrors the
    ``has_usable_secret(min_length=16)`` guard in ``gateway/platforms/api_server.py``)."""
    if not key:
        return False
    try:
        from hermes_cli.auth import has_usable_secret
        return has_usable_secret(key, min_length=16)
    except ImportError:
        return len(str(key).strip()) >= 16


def _needs_extra(*keys: str) -> Callable[[PlatformConfig], bool]:
    return lambda cfg: all(cfg.extra.get(k) for k in keys)


# Built-in "sufficiently configured?" checks; platforms covered by the generic
# ``token or api_key`` check (Telegram, Discord, Slack, Matrix, ...) need no entry.
_PLATFORM_CONNECTED_CHECKERS: dict[Platform, Callable[[PlatformConfig], bool]] = {
    Platform.WEIXIN: lambda cfg: bool(cfg.extra.get("account_id") and (cfg.token or cfg.extra.get("token"))),
    Platform.WHATSAPP_CLOUD: _needs_extra("phone_number_id", "access_token"),
    Platform.SIGNAL: _needs_extra("http_url"),
    Platform.API_SERVER: lambda cfg: _has_usable_api_server_key(cfg.extra.get("key") if cfg else None),
    Platform.WEBHOOK: lambda cfg: True,
    Platform.MSGRAPH_WEBHOOK: lambda cfg: bool(str(cfg.extra.get("client_state") or "").strip()),
    Platform.BLUEBUBBLES: _needs_extra("server_url", "password"),
    Platform.QQBOT: _needs_extra("app_id", "client_secret"),
    Platform.YUANBAO: _needs_extra("app_id", "app_secret"),
    # Relay dials OUT: "connected" once an endpoint URL is configured. EXPERIMENTAL.
    Platform.RELAY: lambda cfg: bool(cfg.extra.get("relay_url") or cfg.extra.get("url")),
}


# Top-level bool-ish keys read verbatim (no nested ``gateway.`` fallback) with their defaults.
_TOPLEVEL_BOOL_DEFAULTS = {
    "write_sessions_json": True, "always_log_local": True, "filter_silence_narration": True,
    "group_sessions_per_user": True, "thread_sessions_per_user": False,
    "hosted_rooms_enabled": True,
}


@dataclass
class GatewayConfig:
    """Main gateway configuration: platform connections, session policies, delivery settings."""
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    default_reset_policy: SessionResetPolicy = field(default_factory=SessionResetPolicy)
    reset_by_type: Dict[str, SessionResetPolicy] = field(default_factory=dict)
    reset_by_platform: Dict[Platform, SessionResetPolicy] = field(default_factory=dict)
    reset_triggers: List[str] = field(default_factory=lambda: ["/new", "/reset"])
    quick_commands: Dict[str, Any] = field(default_factory=dict)  # slash commands that bypass the agent loop
    sessions_dir: Path = field(default_factory=lambda: get_hermes_home() / "sessions")
    # Legacy sessions.json mirror of the routing index (primary: state.db) for external tooling / downgrades.
    # The primary copy lives in state.db (gateway_routing table, #9006). Default True for backward
    # compatibility with external tooling and downgrade safety; set gateway.write_sessions_json: false in
    # config.yaml to stop producing the file.
    write_sessions_json: bool = True
    # Whether the gateway starts the Group Chat (hosted rooms) worker. The
    # worker operates on the install-wide shared state.db and runs in every
    # profile gateway by default. On multi-profile installs, a fleet restart
    # (e.g. `hermes update`) fires all workers back-to-back on the same
    # shared state.db, which has corrupted the store in the field (#102120).
    # Set to false to disable the worker entirely — the gateway will still
    # serve real messages, just without Group Chat rooms.
    hosted_rooms_enabled: bool = True
    always_log_local: bool = True  # Always save cron outputs to local files
    # Drop outbound "silence narration" (*(silent)*, 🔇, a bare ".") that ping-pongs in bot-to-bot
    # channels; a substrate guard that survives prompt drift.
    filter_silence_narration: bool = True
    stt_enabled: bool = True  # Auto-transcribe inbound voice messages
    stt_echo_transcripts: bool = True  # Echo raw STT transcripts back to the user
    group_sessions_per_user: bool = True  # Isolate group sessions per participant when user IDs exist
    thread_sessions_per_user: bool = False  # False = threads shared across participants
    max_concurrent_sessions: Optional[int] = None  # Positive int caps simultaneous active sessions
    # Opt-in: the default profile's gateway serves every profile on the host (profiles stamped into
    # session keys, per-profile adapters/credentials). Allowlist None = serve all; [] = default only.
    multiplex_profiles: bool = False
    multiplex_profile_allowlist: Optional[List[str]] = None
    # Public HTTPS endpoint for scoped RoomLink calls (an API key alone must never advertise a
    # route); HERMES_ROOM_LINK_URL overrides.
    room_link_url: Optional[str] = None
    systemd_watchdog_seconds: int = 0  # opt-in; zero keeps Type=simple and disables sd_notify
    # In-process loop liveness watchdog: after consecutive missed probes it dumps all-thread stacks
    # and hard-exits with the service-restart code. The knobs tolerate transient self-recovering
    # stalls (adapter reconnect doing sync socket I/O) so a short block does not cause restart churn.
    # max_strikes ~= 90-120s sustained block; the heartbeat-fsync false positive is fixed at the root
    # (off-loop write + two-witness probe), so raising it would only delay recovery.
    # On by default; set gateway.loop_watchdog: false in config.yaml to disable. Telegram/Discord reconnect
    # doing synchronous socket I/O during a network blip — so a short block does not force exit code 75 and
    # trigger a restart churn that stalls cron dispatch (recurring fleet incidents on 2026-08-17, kanban
    # t_0f76430f/t_70483f23). A genuine wedge (event loop frozen for the full tolerance window) still
    # escalates to a supervised restart. See #69089.
    loop_watchdog: bool = True
    loop_watchdog_probe_interval_s: float = DEFAULT_LOOP_WATCHDOG_INTERVAL_S
    loop_watchdog_probe_timeout_s: float = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S
    loop_watchdog_max_strikes: int = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES
    unauthorized_dm_behavior: str = "pair"  # "pair" or "ignore"
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    # Prune SessionEntry records older than this (a resumed chat gets a fresh session). 0 = off.
    session_store_max_age_days: int = 90
    profile_routes: list = field(default_factory=list)  # gateway/profile_routing.py

    # Scalar fields serialized verbatim by ``to_dict`` (in output order).
    _SCALAR_DICT_FIELDS = (
        "write_sessions_json", "hosted_rooms_enabled", "always_log_local", "filter_silence_narration", "stt_enabled",
        "stt_echo_transcripts", "group_sessions_per_user", "thread_sessions_per_user",
        "max_concurrent_sessions", "multiplex_profiles", "multiplex_profile_allowlist",
        "room_link_url", "systemd_watchdog_seconds", "loop_watchdog",
        "loop_watchdog_probe_interval_s", "loop_watchdog_probe_timeout_s",
        "loop_watchdog_max_strikes", "unauthorized_dm_behavior",
    )

    def __post_init__(self) -> None:
        self.multiplex_profile_allowlist = _normalize_multiplex_profile_allowlist(self.multiplex_profile_allowlist)
        self.systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(self.systemd_watchdog_seconds)

    def get_connected_platforms(self) -> List[Platform]:
        """Enabled + configured platforms, sorted by value so the rendered "Connected
        Platforms" prompt block is byte-stable (a reorder busts the prompt cache)."""
        connected = [p for p, c in self.platforms.items() if c.enabled and self._is_platform_connected(p, c)]
        return sorted(connected, key=lambda p: str(p.value))

    def _is_platform_connected(self, platform: Platform, config: PlatformConfig) -> bool:
        checker = _PLATFORM_CONNECTED_CHECKERS.get(platform)
        # Weixin needs token AND account_id, so it must bypass the generic token branch.
        if platform == Platform.WEIXIN:
            return checker(config)
        if config.token or config.api_key:
            return True
        if checker is not None:
            return checker(config)

        # Plugin platforms; force (idempotent) discovery for directly-constructed configs.
        try:
            from gateway.platform_registry import platform_registry
            with contextlib.suppress(Exception):
                # Iterate built-in platforms plus any registered plugin platforms so plugin authors get the
                # same shared-key bridging (#24836).
                # Registry-driven enable for plugin platforms. Built-ins have explicit blocks above. A
                # plugin platform is enabled when its credentials are configured (``is_connected``) and its
                # dependencies are either present (passive ``check_fn``) or installable on demand
                # (``ensure_deps_fn``, run later by ``create_adapter()`` — never here). Plugins that need to
                # seed ``PlatformConfig.extra`` from env vars (e.g. Google Chat's project_id /
                # subscription_name) can supply ``env_enablement_fn`` on their PlatformEntry — called here
                # BEFORE adapter construction. Enablement gate (#31116): when a plugin registers
                # ``is_connected`` (the "has the user actually configured credentials for this?" check), we
                # MUST consult it before flipping ``enabled = True``. Otherwise ``check_fn`` alone — a
                # passive "is the SDK importable?" probe — silently enables platforms the user never opted
                # into, and the gateway then tries to connect to Discord / Teams / Google Chat with no token
                # and emits noisy retry-forever errors. ``_platform_status`` was already fixed for the same
                # bug class in commit 7849a3d73; this is the runtime counterpart.
                from hermes_cli.plugins import discover_plugins
                discover_plugins()
            entry = platform_registry.get(platform.value)
            if entry:
                check = entry.is_connected if entry.is_connected is not None else entry.validate_config
                return True if check is None else check(config)
        except Exception:
            pass  # Registry not yet initialised during early import
        return False

    def get_home_channel(self, platform: Platform) -> Optional[HomeChannel]:
        return self.platforms[platform].home_channel if self.platforms.get(platform) else None

    def get_reset_policy(self, platform: Optional[Platform] = None, session_type: Optional[str] = None) -> SessionResetPolicy:
        """Priority: platform override > type override > default."""
        if platform and platform in self.reset_by_platform:
            return self.reset_by_platform[platform]
        if session_type and session_type in self.reset_by_type:
            return self.reset_by_type[session_type]
        return self.default_reset_policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms": {p.value: c.to_dict() for p, c in self.platforms.items()},
            "default_reset_policy": self.default_reset_policy.to_dict(),
            "reset_by_type": {k: v.to_dict() for k, v in self.reset_by_type.items()},
            "reset_by_platform": {p.value: v.to_dict() for p, v in self.reset_by_platform.items()},
            "reset_triggers": self.reset_triggers,
            "quick_commands": self.quick_commands,
            "sessions_dir": str(self.sessions_dir),
            **{name: getattr(self, name) for name in self._SCALAR_DICT_FIELDS},
            "streaming": self.streaming.to_dict(),
            "session_store_max_age_days": self.session_store_max_age_days,
            "profile_routes": [
                asdict(r) if is_dataclass(r) and not isinstance(r, type) else r for r in self.profile_routes
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        data = _coerce_dict(data)
        nested_gateway = _coerce_dict(data.get("gateway"))

        def pick(key: str) -> Any:
            """Top-level key wins by presence; else the nested ``gateway.<key>`` form."""
            return data[key] if key in data else nested_gateway.get(key)

        def key_label(key: str) -> str:
            """Warning key prefix: "gateway." when the nested form was the one consulted."""
            return key if key in data else f"gateway.{key}"

        def by_platform(key: str, parse, *, dicts_only: bool = False) -> dict:
            """``{Platform(name): parse(block)}`` for a platform-keyed mapping; unknown platforms skipped."""
            out = {}
            for platform_name, block in _coerce_dict(data.get(key, {})).items():
                if dicts_only and not isinstance(block, dict):
                    continue
                try:
                    out[Platform(platform_name)] = parse(block)
                except ValueError:
                    pass
            return out

        def stt_setting(flat_key: str, nested_key: str) -> Any:
            value = data.get(flat_key)
            return _coerce_dict(data.get("stt")).get(nested_key) if value is None else value

        def bounded_float(key: str, default: float, lo: float, hi: float) -> float:
            # Out-of-range / non-finite watchdog knobs fall back to the shipped defaults.
            value = _coerce_float(pick(key), default)
            return value if math.isfinite(value) and lo <= value <= hi else default

        room_link_url = data.get("room_link_url")
        max_strikes = _coerce_int(pick("loop_watchdog_max_strikes"), DEFAULT_LOOP_WATCHDOG_MAX_STRIKES)
        if not 1 <= max_strikes <= 1000:
            max_strikes = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES

        systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(
            pick("systemd_watchdog_seconds"), key_label("systemd_watchdog_seconds")
        )
        # env > config.yaml > False: a recognized GATEWAY_MULTIPLEX_PROFILES wins (hosted deployments
        # stamp it on the container); blank/unrecognized falls through to the top-level VALUE when
        # not None, else ``gateway.multiplex_profiles``.
        multiplex_profiles = data.get("multiplex_profiles")
        if multiplex_profiles is None:
            multiplex_profiles = nested_gateway.get("multiplex_profiles")
        env_multiplex = _env_multiplex_profiles_override()
        if env_multiplex is not None:
            multiplex_profiles = env_multiplex
        max_concurrent_sessions = _coerce_optional_positive_int(
            pick("max_concurrent_sessions"), key_label("max_concurrent_sessions")
        )

        try:
            session_store_max_age_days = max(int(data.get("session_store_max_age_days", 90)), 0)
        except (TypeError, ValueError):
            session_store_max_age_days = 90

        from gateway.profile_routing import parse_profile_routes

        return cls(
            platforms=by_platform("platforms", PlatformConfig.from_dict, dicts_only=True),
            default_reset_policy=SessionResetPolicy.from_dict(data["default_reset_policy"])
            if "default_reset_policy" in data
            else SessionResetPolicy(),
            reset_by_type={
                type_name: SessionResetPolicy.from_dict(policy_data)
                for type_name, policy_data in _coerce_dict(data.get("reset_by_type", {})).items()
            },
            reset_by_platform=by_platform("reset_by_platform", SessionResetPolicy.from_dict),
            reset_triggers=data.get("reset_triggers", ["/new", "/reset"]),
            quick_commands=_coerce_dict(data.get("quick_commands", {})),
            sessions_dir=Path(data["sessions_dir"]) if "sessions_dir" in data else get_hermes_home() / "sessions",
            **{name: _coerce_bool(data.get(name), default) for name, default in _TOPLEVEL_BOOL_DEFAULTS.items()},
            stt_enabled=_coerce_bool(stt_setting("stt_enabled", "enabled"), True),
            stt_echo_transcripts=_coerce_bool(stt_setting("stt_echo_transcripts", "echo_transcripts"), True),
            multiplex_profiles=_coerce_bool(multiplex_profiles, False),
            multiplex_profile_allowlist=pick("multiplex_profile_allowlist"),
            room_link_url=room_link_url if isinstance(room_link_url, str) else None,
            systemd_watchdog_seconds=systemd_watchdog_seconds,
            loop_watchdog=_coerce_bool(pick("loop_watchdog"), True),
            loop_watchdog_probe_interval_s=bounded_float("loop_watchdog_probe_interval_s", DEFAULT_LOOP_WATCHDOG_INTERVAL_S, 1.0, 3600.0),
            loop_watchdog_probe_timeout_s=bounded_float("loop_watchdog_probe_timeout_s", DEFAULT_LOOP_WATCHDOG_TIMEOUT_S, 1.0, 600.0),
            loop_watchdog_max_strikes=max_strikes,
            max_concurrent_sessions=max_concurrent_sessions,
            unauthorized_dm_behavior=_normalize_choice(data.get("unauthorized_dm_behavior"), {"pair", "ignore"}, "pair"),
            streaming=StreamingConfig.from_dict(data.get("streaming", {})),
            session_store_max_age_days=session_store_max_age_days,
            profile_routes=parse_profile_routes(data.get("profile_routes") or []),
        )

    def _extra_choice(self, platform: Optional[Platform], key: str, choices: set, default: str) -> Optional[str]:
        """Normalized ``platforms[platform].extra[key]`` when the key is present, else None."""
        platform_cfg = self.platforms.get(platform) if platform else None
        if platform_cfg and key in platform_cfg.extra:
            return _normalize_choice(platform_cfg.extra.get(key), choices, default)
        return None

    def get_unauthorized_dm_behavior(self, platform: Optional[Platform] = None) -> str:
        """Effective unauthorized-DM behavior. Email is inbox-shaped so it defaults to ``"ignore"``
        unless its own ``unauthorized_dm_behavior`` opts in (a global default does not)."""
        choice = self._extra_choice(platform, "unauthorized_dm_behavior", {"pair", "ignore"}, self.unauthorized_dm_behavior)
        if choice is not None:
            return choice
        return "ignore" if platform == Platform.EMAIL else self.unauthorized_dm_behavior

    def get_notice_delivery(self, platform: Optional[Platform] = None) -> str:
        """Effective notice-delivery mode ("public"/"private") for a platform."""
        choice = self._extra_choice(platform, "notice_delivery", {"public", "private"}, "public")
        return "public" if choice is None else choice


def load_gateway_config() -> GatewayConfig:
    """Load gateway configuration. Priority: env > ~/.hermes/config.yaml > legacy gateway.json > defaults."""
    from gateway import config_loader

    _home = get_hermes_home()
    gw_data = config_loader.load_legacy_gateway_json(_home)
    try:
        config_loader.load_yaml_layer(_home, gw_data)
    except Exception as e:
        logger.warning(
            # DingTalk settings → env vars: migrated to the dingtalk plugin's apply_yaml_config_fn hook
            # (plugins/platforms/dingtalk/adapter.py). #41112 / #3823.
            # Mattermost config bridge moved into plugins/platforms/mattermost/
            # adapter.py::_apply_yaml_config — see #25443 (apply_yaml_config_fn).
            # Matrix settings → env vars: migrated to the matrix plugin's apply_yaml_config_fn hook
            # (plugins/platforms/matrix/adapter.py). #41112 / #3823.
            # Feishu settings → env vars: migrated to the feishu plugin's apply_yaml_config_fn hook
            # (plugins/platforms/feishu/adapter.py). #41112 / #3823.
            "Failed to process config.yaml — falling back to .env / gateway.json values. "
            "Check %s for syntax errors. Error: %s",
            _home / "config.yaml", e,
        )

    config = GatewayConfig.from_dict(gw_data)
    _apply_env_overrides(config)
    _validate_gateway_config(config)
    return config


def _validate_gateway_config(config: "GatewayConfig") -> None:
    """Validate and sanitize a loaded GatewayConfig in place (after all sources are merged)."""
    policy = config.default_reset_policy

    if not (0 <= policy.at_hour <= 23):
        logger.warning("Invalid at_hour=%s (must be 0-23). Using default 4.", policy.at_hour)
        policy.at_hour = 4

    if policy.idle_minutes is None or policy.idle_minutes <= 0:
        logger.warning("Invalid idle_minutes=%s (must be positive). Using default 1440.", policy.idle_minutes)
        policy.idle_minutes = 1440

    try:
        # Reject known-weak placeholder tokens. Ported from openclaw/openclaw#64586: users who copy
        # .env.example without changing placeholder values get a clear startup error instead of a confusing
        # "auth failed" from the platform API.
        from hermes_cli.auth import has_usable_secret
    except ImportError:
        has_usable_secret = None

    token_platforms = [
        (p, c, PLATFORM_TOKEN_ENV_NAMES[p]) for p, c in config.platforms.items()
        if c.enabled and p in PLATFORM_TOKEN_ENV_NAMES and c.token is not None
    ]
    for platform, pconfig, env_name in token_platforms:  # an empty token won't connect; say so
        if not pconfig.token.strip():
            logger.warning("%s is enabled but %s is empty. The adapter will likely fail to connect.", platform.value, env_name)
    if has_usable_secret is None:
        return
    for platform, pconfig, env_name in token_platforms:  # reject placeholder tokens (copied .env.example)
        token = pconfig.token
        if token.strip() and not has_usable_secret(token, min_length=4):
            logger.error(
                "%s is enabled but %s is set to a placeholder value ('%s'). "
                "Set a real bot token before starting the gateway. "
                "The adapter will NOT be started.",
                platform.value, env_name, token.strip()[:6] + "...",
            )
            pconfig.enabled = False


def _apply_env_overrides(config: GatewayConfig) -> None:
    """Apply environment variable overrides to config (see ``gateway.config_env``)."""
    from gateway.config_env import _apply_env_overrides as _impl
    _impl(config)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

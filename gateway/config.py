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
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported messaging platforms."""
    LOCAL = "local"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    SLACK = "slack"


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "at_hour": self.at_hour,
            "idle_minutes": self.idle_minutes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionResetPolicy":
        return cls(
            mode=data.get("mode", "both"),
            at_hour=data.get("at_hour", 4),
            idle_minutes=data.get("idle_minutes", 1440),
        )


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""
    enabled: bool = False
    token: Optional[str] = None  # Bot token (Telegram, Discord)
    api_key: Optional[str] = None  # API key if different from token
    home_channel: Optional[HomeChannel] = None
    
    # Platform-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "extra": self.extra,
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
            enabled=data.get("enabled", False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_channel=home_channel,
            extra=data.get("extra", {}),
        )


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
    
    # Storage paths
    sessions_dir: Path = field(default_factory=lambda: Path.home() / ".hermes" / "sessions")
    
    # Delivery settings
    always_log_local: bool = True  # Always save cron outputs to local files
    
    def get_connected_platforms(self) -> List[Platform]:
        """Return list of platforms that are enabled and configured."""
        connected = []
        for platform, config in self.platforms.items():
            if config.enabled and (config.token or config.api_key):
                connected.append(platform)
        return connected
    
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
            "sessions_dir": str(self.sessions_dir),
            "always_log_local": self.always_log_local,
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
        
        sessions_dir = Path.home() / ".hermes" / "sessions"
        if "sessions_dir" in data:
            sessions_dir = Path(data["sessions_dir"])
        
        return cls(
            platforms=platforms,
            default_reset_policy=default_policy,
            reset_by_type=reset_by_type,
            reset_by_platform=reset_by_platform,
            reset_triggers=data.get("reset_triggers", ["/new", "/reset"]),
            sessions_dir=sessions_dir,
            always_log_local=data.get("always_log_local", True),
        )


def load_gateway_config() -> GatewayConfig:
    """
    Load gateway configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Environment variables
    2. ~/.hermes/gateway.json
    3. cli-config.yaml gateway section
    4. Defaults
    """
    config = GatewayConfig()
    
    # Try loading from ~/.hermes/gateway.json
    gateway_config_path = Path.home() / ".hermes" / "gateway.json"
    if gateway_config_path.exists():
        try:
            with open(gateway_config_path, "r") as f:
                data = json.load(f)
                config = GatewayConfig.from_dict(data)
        except Exception as e:
            print(f"[gateway] Warning: Failed to load {gateway_config_path}: {e}")
    
    # Bridge session_reset from config.yaml (the user-facing config file)
    # into the gateway config. config.yaml takes precedence over gateway.json
    # for session reset policy since that's where hermes setup writes it.
    try:
        import yaml
        config_yaml_path = Path.home() / ".hermes" / "config.yaml"
        if config_yaml_path.exists():
            with open(config_yaml_path) as f:
                yaml_cfg = yaml.safe_load(f) or {}
            sr = yaml_cfg.get("session_reset")
            if sr and isinstance(sr, dict):
                config.default_reset_policy = SessionResetPolicy.from_dict(sr)
    except Exception:
        pass

    # Override with environment variables
    _apply_env_overrides(config)
    
    # --- Validate loaded values ---
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

    return config


def _apply_env_overrides(config: GatewayConfig) -> None:
    """Apply environment variable overrides to config."""
    
    # Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        if Platform.TELEGRAM not in config.platforms:
            config.platforms[Platform.TELEGRAM] = PlatformConfig()
        config.platforms[Platform.TELEGRAM].enabled = True
        config.platforms[Platform.TELEGRAM].token = telegram_token
    
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
            config.platforms[Platform.SLACK] = PlatformConfig()
        config.platforms[Platform.SLACK].enabled = True
        config.platforms[Platform.SLACK].token = slack_token
        # Home channel
        slack_home = os.getenv("SLACK_HOME_CHANNEL")
        if slack_home:
            config.platforms[Platform.SLACK].home_channel = HomeChannel(
                platform=Platform.SLACK,
                chat_id=slack_home,
                name=os.getenv("SLACK_HOME_CHANNEL_NAME", ""),
            )
    
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


def save_gateway_config(config: GatewayConfig) -> None:
    """Save gateway configuration to ~/.hermes/gateway.json."""
    gateway_config_path = Path.home() / ".hermes" / "gateway.json"
    gateway_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(gateway_config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

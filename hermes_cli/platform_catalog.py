"""Shared platform metadata for CLI, tools, skills, and gateway setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


EnvGetter = Callable[[str], str]


def _has_value(value: str) -> bool:
    return bool(str(value or "").strip())


def _is_truthy_enabled(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return False
    return lowered not in {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    label: str
    emoji: str
    default_toolset: str
    configured_any_of: tuple[str, ...] = ()
    configured_all_of: tuple[str, ...] = ()
    configured_truthy_any_of: tuple[str, ...] = ()
    home_channel_env: str = ""
    include_in_tools: bool = True
    include_in_skills: bool = True
    include_in_setup: bool = False
    setup_label: str = ""
    warn_missing_home: bool = False

    @property
    def label_with_emoji(self) -> str:
        return f"{self.emoji} {self.label}" if self.emoji else self.label

    @property
    def setup_display_label(self) -> str:
        return self.setup_label or self.label

    @property
    def primary_config_env(self) -> str:
        if self.configured_truthy_any_of:
            return self.configured_truthy_any_of[0]
        if self.configured_any_of:
            return self.configured_any_of[0]
        if self.configured_all_of:
            return self.configured_all_of[0]
        return ""

    def is_configured(self, get_env_value: EnvGetter) -> bool:
        if self.key == "cli":
            return True
        if self.configured_all_of and not all(_has_value(get_env_value(var)) for var in self.configured_all_of):
            return False
        if self.configured_truthy_any_of:
            return any(_is_truthy_enabled(get_env_value(var)) for var in self.configured_truthy_any_of)
        if self.configured_any_of:
            return any(_has_value(get_env_value(var)) for var in self.configured_any_of)
        return bool(self.configured_all_of)


_PLATFORM_SPECS: tuple[PlatformSpec, ...] = (
    PlatformSpec("cli", "CLI", "🖥️", "hermes-cli"),
    PlatformSpec("telegram", "Telegram", "📱", "hermes-telegram", configured_any_of=("TELEGRAM_BOT_TOKEN",), home_channel_env="TELEGRAM_HOME_CHANNEL", include_in_setup=True, warn_missing_home=True),
    PlatformSpec("discord", "Discord", "💬", "hermes-discord", configured_any_of=("DISCORD_BOT_TOKEN",), home_channel_env="DISCORD_HOME_CHANNEL", include_in_setup=True, warn_missing_home=True),
    PlatformSpec("slack", "Slack", "💼", "hermes-slack", configured_any_of=("SLACK_BOT_TOKEN",), home_channel_env="SLACK_HOME_CHANNEL", include_in_setup=True, warn_missing_home=True),
    PlatformSpec("whatsapp", "WhatsApp", "📱", "hermes-whatsapp", configured_truthy_any_of=("WHATSAPP_ENABLED",), include_in_setup=True),
    PlatformSpec("signal", "Signal", "📡", "hermes-signal", configured_all_of=("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"), home_channel_env="SIGNAL_HOME_CHANNEL"),
    PlatformSpec("bluebubbles", "BlueBubbles", "💙", "hermes-bluebubbles", configured_any_of=("BLUEBUBBLES_SERVER_URL",), home_channel_env="BLUEBUBBLES_HOME_CHANNEL", include_in_setup=True, setup_label="BlueBubbles (iMessage)", warn_missing_home=True),
    PlatformSpec("homeassistant", "Home Assistant", "🏠", "hermes-homeassistant", configured_all_of=("HASS_TOKEN", "HASS_URL")),
    PlatformSpec("email", "Email", "📧", "hermes-email", configured_any_of=("EMAIL_ADDRESS",)),
    PlatformSpec("matrix", "Matrix", "💬", "hermes-matrix", configured_any_of=("MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD"), home_channel_env="MATRIX_HOME_ROOM", include_in_setup=True),
    PlatformSpec("dingtalk", "DingTalk", "💬", "hermes-dingtalk", configured_any_of=("DINGTALK_CLIENT_ID",)),
    PlatformSpec("feishu", "Feishu", "🪽", "hermes-feishu", configured_any_of=("FEISHU_APP_ID",), home_channel_env="FEISHU_HOME_CHANNEL"),
    PlatformSpec("wecom", "WeCom", "💬", "hermes-wecom", configured_any_of=("WECOM_BOT_ID",), home_channel_env="WECOM_HOME_CHANNEL"),
    PlatformSpec("weixin", "Weixin", "💬", "hermes-weixin", configured_any_of=("WEIXIN_ACCOUNT_ID",), home_channel_env="WEIXIN_HOME_CHANNEL", include_in_setup=True, setup_label="Weixin (WeChat)"),
    PlatformSpec("api_server", "API Server", "🌐", "hermes-api-server", configured_truthy_any_of=("API_SERVER_ENABLED",), configured_any_of=("API_SERVER_KEY",), include_in_skills=False),
    PlatformSpec("mattermost", "Mattermost", "💬", "hermes-mattermost", configured_any_of=("MATTERMOST_TOKEN",), home_channel_env="MATTERMOST_HOME_CHANNEL", include_in_setup=True),
    PlatformSpec("sms", "SMS", "📱", "hermes-sms", configured_all_of=("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"), home_channel_env="SMS_HOME_CHANNEL", include_in_setup=False, include_in_skills=False),
    PlatformSpec("webhook", "Webhook", "🔗", "hermes-webhook", configured_truthy_any_of=("WEBHOOK_ENABLED",), include_in_setup=True, setup_label="Webhooks (GitHub, GitLab, etc.)"),
)

_PLATFORM_BY_KEY: Dict[str, PlatformSpec] = {spec.key: spec for spec in _PLATFORM_SPECS}


def get_platform_spec(key: str) -> Optional[PlatformSpec]:
    return _PLATFORM_BY_KEY.get(key)


def iter_platform_specs() -> Iterable[PlatformSpec]:
    return _PLATFORM_SPECS


def iter_tool_platform_specs() -> List[PlatformSpec]:
    return [spec for spec in _PLATFORM_SPECS if spec.include_in_tools]


def iter_skills_platform_specs() -> List[PlatformSpec]:
    return [spec for spec in _PLATFORM_SPECS if spec.include_in_skills]


def iter_setup_platform_specs() -> List[PlatformSpec]:
    return [spec for spec in _PLATFORM_SPECS if spec.include_in_setup]


def configured_platform_keys(get_env_value: EnvGetter, *, include_cli: bool = True) -> List[str]:
    keys: List[str] = []
    for spec in _PLATFORM_SPECS:
        if spec.key == "cli" and not include_cli:
            continue
        if spec.is_configured(get_env_value):
            keys.append(spec.key)
    return keys

"""
Shared platform registry for Hermes Agent.

Single source of truth for platform metadata consumed by both
skills_config (label display) and tools_config (default toolset
resolution).  Import ``PLATFORMS`` from here instead of maintaining
duplicate dicts in each module.
"""

from collections import OrderedDict
from typing import NamedTuple


class PlatformInfo(NamedTuple):
    """Metadata for a single platform entry."""
    label: str
    default_toolset: str


# Ordered so that TUI menus are deterministic.
PLATFORMS: OrderedDict[str, PlatformInfo] = OrderedDict([
    ("cli",            PlatformInfo(label="🖥️  CLI",            default_toolset="hermes-cli")),
    ("telegram",       PlatformInfo(label="📱 Telegram",        default_toolset="hermes-telegram")),
    ("discord",        PlatformInfo(label="💬 Discord",         default_toolset="hermes-discord")),
    ("slack",          PlatformInfo(label="💼 Slack",           default_toolset="hermes-slack")),
    ("whatsapp",       PlatformInfo(label="📱 WhatsApp",        default_toolset="hermes-whatsapp")),
    ("signal",         PlatformInfo(label="📡 Signal",          default_toolset="hermes-signal")),
    ("bluebubbles",    PlatformInfo(label="💙 BlueBubbles",     default_toolset="hermes-bluebubbles")),
    ("email",          PlatformInfo(label="📧 Email",           default_toolset="hermes-email")),
    ("homeassistant",  PlatformInfo(label="🏠 Home Assistant",  default_toolset="hermes-homeassistant")),
    ("mattermost",     PlatformInfo(label="💬 Mattermost",      default_toolset="hermes-mattermost")),
    ("matrix",         PlatformInfo(label="💬 Matrix",          default_toolset="hermes-matrix")),
    ("dingtalk",       PlatformInfo(label="💬 DingTalk",        default_toolset="hermes-dingtalk")),
    ("feishu",         PlatformInfo(label="🪽 Feishu",          default_toolset="hermes-feishu")),
    ("wecom",          PlatformInfo(label="💬 WeCom",           default_toolset="hermes-wecom")),
    ("wecom_callback", PlatformInfo(label="💬 WeCom Callback",  default_toolset="hermes-wecom-callback")),
    ("weixin",         PlatformInfo(label="💬 Weixin",          default_toolset="hermes-weixin")),
    ("qqbot",          PlatformInfo(label="💬 QQBot",           default_toolset="hermes-qqbot")),
    ("webhook",        PlatformInfo(label="🔗 Webhook",         default_toolset="hermes-webhook")),
    ("api_server",     PlatformInfo(label="🌐 API Server",      default_toolset="hermes-api-server")),
])


def platform_label(key: str, default: str = "") -> str:
    """Return the display label for a platform key, or *default*."""
    info = PLATFORMS.get(key)
    return info.label if info is not None else default

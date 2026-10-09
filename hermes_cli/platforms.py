"""Shared platform registry for Hermes Agent."""

from collections import OrderedDict
from typing import NamedTuple
from tools.platform_policy import PLATFORM_DEFAULT_TOOLSETS


class PlatformInfo(NamedTuple):
    """Metadata for a single platform entry."""
    label: str
    default_toolset: str


# Ordered so that TUI menus are deterministic.
PLATFORMS: OrderedDict[str, PlatformInfo] = OrderedDict([
    ("cli",            PlatformInfo(label="🖥️  CLI",            default_toolset=PLATFORM_DEFAULT_TOOLSETS["cli"])),
    ("telegram",       PlatformInfo(label="📱 Telegram",        default_toolset=PLATFORM_DEFAULT_TOOLSETS["telegram"])),
    ("discord",        PlatformInfo(label="💬 Discord",         default_toolset=PLATFORM_DEFAULT_TOOLSETS["discord"])),
    ("slack",          PlatformInfo(label="💼 Slack",           default_toolset=PLATFORM_DEFAULT_TOOLSETS["slack"])),
    ("whatsapp",       PlatformInfo(label="📱 WhatsApp",        default_toolset=PLATFORM_DEFAULT_TOOLSETS["whatsapp"])),
    ("whatsapp_cloud", PlatformInfo(label="📱 WhatsApp Business (Cloud)", default_toolset=PLATFORM_DEFAULT_TOOLSETS["whatsapp_cloud"])),
    ("signal",         PlatformInfo(label="📡 Signal",          default_toolset=PLATFORM_DEFAULT_TOOLSETS["signal"])),
    ("bluebubbles",    PlatformInfo(label="💙 BlueBubbles",     default_toolset=PLATFORM_DEFAULT_TOOLSETS["bluebubbles"])),
    ("email",          PlatformInfo(label="📧 Email",           default_toolset=PLATFORM_DEFAULT_TOOLSETS["email"])),
    ("homeassistant",  PlatformInfo(label="🏠 Home Assistant",  default_toolset=PLATFORM_DEFAULT_TOOLSETS["homeassistant"])),
    ("mattermost",     PlatformInfo(label="💬 Mattermost",      default_toolset=PLATFORM_DEFAULT_TOOLSETS["mattermost"])),
    ("matrix",         PlatformInfo(label="💬 Matrix",          default_toolset=PLATFORM_DEFAULT_TOOLSETS["matrix"])),
    ("dingtalk",       PlatformInfo(label="💬 DingTalk",        default_toolset=PLATFORM_DEFAULT_TOOLSETS["dingtalk"])),
    ("feishu",         PlatformInfo(label="🪽 Feishu",          default_toolset=PLATFORM_DEFAULT_TOOLSETS["feishu"])),
    ("wecom",          PlatformInfo(label="💬 WeCom",           default_toolset=PLATFORM_DEFAULT_TOOLSETS["wecom"])),
    ("wecom_callback", PlatformInfo(label="💬 WeCom Callback",  default_toolset=PLATFORM_DEFAULT_TOOLSETS["wecom_callback"])),
    ("weixin",         PlatformInfo(label="💬 Weixin",          default_toolset=PLATFORM_DEFAULT_TOOLSETS["weixin"])),
    ("qqbot",          PlatformInfo(label="💬 QQBot",           default_toolset=PLATFORM_DEFAULT_TOOLSETS["qqbot"])),
    ("yuanbao",        PlatformInfo(label="🤖 Yuanbao",         default_toolset=PLATFORM_DEFAULT_TOOLSETS["yuanbao"])),
    ("webhook",        PlatformInfo(label="🔗 Webhook",         default_toolset=PLATFORM_DEFAULT_TOOLSETS["webhook"])),
    ("api_server",     PlatformInfo(label="🌐 API Server",      default_toolset=PLATFORM_DEFAULT_TOOLSETS["api_server"])),
    ("cron",           PlatformInfo(label="⏰ Cron",            default_toolset=PLATFORM_DEFAULT_TOOLSETS["cron"])),
])


def _plugin_label(entry) -> str:
    return f"{entry.emoji}  {entry.label}" if entry.emoji else entry.label


def platform_label(key: str, default: str = "") -> str:
    """Return the display label for a platform key (builtin, then plugin registry), or *default*."""
    info = PLATFORMS.get(key)
    if info is not None:
        return info.label
    try:
        from plugin_runtime.platform_registry import platform_registry
        entry = platform_registry.get(key)
        if entry:
            return _plugin_label(entry)
    except Exception:
        pass
    return default


def get_all_platforms() -> "OrderedDict[str, PlatformInfo]":
    """PLATFORMS plus plugin-registered platforms (appended after builtins) — use for menus."""
    merged = OrderedDict(PLATFORMS)
    try:
        from plugin_runtime.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.name not in merged:
                merged[entry.name] = PlatformInfo(_plugin_label(entry), f"hermes-{entry.name}")
    except Exception:
        pass
    return merged

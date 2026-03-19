"""
Cosmos Multi-Channel Messaging Hub
=======================================

Universal messaging integration supporting 10+ platforms.

Channels Supported:
- WhatsApp (via Baileys/whatsapp-web.js)
- Telegram (Bot API)
- Discord (Bot API)
- Slack (Socket Mode / Bolt)
- Signal (signal-cli)
- Matrix (matrix-nio)
- iMessage (macOS native)
- Microsoft Teams (Graph API) - TODO
- Google Chat (API) - TODO
- WebChat (built-in)

Architecture:
    Inbound Message → Channel Adapter → Normalizer → Nexus → Swarm
                                                          ↓
    Outbound Response ← Channel Adapter ← Router ← Deliberation

Setup Requirements:
    pip install python-telegram-bot discord.py slack-bolt aiohttp matrix-nio[e2e]

"One swarm, every channel." - The Collective
"""

from .channel_hub import (
    ChannelHub,
    get_channel_hub,
    ChannelMessage,
    ChannelConfig,
    ChannelType,
    BaseChannel,
    AccessPolicy,
    ActivationMode,
    PairingCode,
)

# Import channel adapters with graceful fallbacks
try:
    from .whatsapp import WhatsAppChannel
except ImportError:
    WhatsAppChannel = None

try:
    from .telegram import TelegramChannel
except ImportError:
    TelegramChannel = None

try:
    from .discord_channel import DiscordChannel
except ImportError:
    DiscordChannel = None

try:
    from .slack_channel import SlackChannel
except ImportError:
    SlackChannel = None

try:
    from .signal_channel import SignalChannel
except ImportError:
    SignalChannel = None

try:
    from .matrix_channel import MatrixChannel
except ImportError:
    MatrixChannel = None

try:
    from .imessage import iMessageChannel
except ImportError:
    iMessageChannel = None

try:
    from .webchat import WebChatChannel
except ImportError:
    WebChatChannel = None

__all__ = [
    # Hub
    "ChannelHub",
    "get_channel_hub",
    # Types
    "ChannelMessage",
    "ChannelConfig",
    "ChannelType",
    "BaseChannel",
    "AccessPolicy",
    "ActivationMode",
    "PairingCode",
    # Channels
    "WhatsAppChannel",
    "TelegramChannel",
    "DiscordChannel",
    "SlackChannel",
    "SignalChannel",
    "MatrixChannel",
    "iMessageChannel",
    "WebChatChannel",
]


def get_available_channels() -> dict:
    """
    Get dict of available channel adapters.

    Returns:
        Dict mapping ChannelType to channel class (or None if unavailable)
    """
    return {
        ChannelType.WHATSAPP: WhatsAppChannel,
        ChannelType.TELEGRAM: TelegramChannel,
        ChannelType.DISCORD: DiscordChannel,
        ChannelType.SLACK: SlackChannel,
        ChannelType.SIGNAL: SignalChannel,
        ChannelType.MATRIX: MatrixChannel,
        ChannelType.IMESSAGE: iMessageChannel,
        ChannelType.WEBCHAT: WebChatChannel,
    }


async def setup_channel_hub(
    enabled_channels: list = None,
    configs: dict = None
) -> ChannelHub:
    """
    Convenience function to set up the channel hub with specified channels.

    Args:
        enabled_channels: List of ChannelType to enable
        configs: Dict of ChannelType -> config dict

    Returns:
        Initialized ChannelHub
    """
    hub = get_channel_hub()
    await hub.initialize()

    available = get_available_channels()
    enabled = enabled_channels or []
    configs = configs or {}

    for channel_type in enabled:
        channel_class = available.get(channel_type)
        if channel_class:
            config = ChannelConfig(
                channel_type=channel_type,
                **(configs.get(channel_type, {}))
            )
            channel = channel_class(config)
            await hub.register_channel(channel_type, channel, config)

    return hub

"""
Cosmos Channel Hub - Universal Multi-Channel Manager
=========================================================

Central hub for managing all messaging channel integrations.

Responsibilities:
- Channel lifecycle management (connect/disconnect)
- Message normalization across platforms
- Routing inbound messages to the swarm
- Routing outbound responses to correct channels
- Access control (allowlists, pairing codes)
- Media handling (upload/download)
- Rate limiting and chunking

Based on HermesAgent's channel architecture with Cosmos enhancements.
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger

# Nexus integration
try:
    from Cosmos.core.nexus import get_nexus, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False


class ChannelType(Enum):
    """Supported channel types."""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    SIGNAL = "signal"
    MATRIX = "matrix"
    IMESSAGE = "imessage"
    TEAMS = "teams"
    GOOGLE_CHAT = "google_chat"
    WEBCHAT = "webchat"
    EMAIL = "email"
    X_TWITTER = "x"


class AccessPolicy(Enum):
    """Channel access control policies."""
    OPEN = "open"              # Anyone can message
    ALLOWLIST = "allowlist"    # Only allowlisted users
    PAIRING = "pairing"        # Require pairing code
    DISABLED = "disabled"      # Channel disabled


class ActivationMode(Enum):
    """Group activation modes."""
    ALWAYS = "always"          # Respond to all messages
    MENTION = "mention"        # Require @mention
    KEYWORD = "keyword"        # Require keyword trigger


@dataclass
class ChannelMessage:
    """
    Normalized message format across all channels.

    All channel adapters convert platform-specific messages to this format.
    """
    # Identity
    message_id: str
    channel_type: ChannelType
    channel_id: str  # Platform-specific chat/channel ID

    # Sender
    sender_id: str
    sender_name: str
    sender_phone: Optional[str] = None  # For WhatsApp/Signal

    # Content
    text: str = ""
    media_type: Optional[str] = None  # image, video, audio, document, sticker
    media_url: Optional[str] = None
    media_path: Optional[str] = None

    # Context
    is_group: bool = False
    group_name: Optional[str] = None
    is_mention: bool = False
    reply_to_id: Optional[str] = None
    reply_to_text: Optional[str] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Format message for inclusion in agent prompt."""
        parts = []

        # Add reply context if present
        if self.reply_to_text:
            parts.append(f"[Replying to: {self.reply_to_text[:100]}...]")

        # Add sender context
        sender = self.sender_name or self.sender_id
        if self.is_group:
            parts.append(f"[{self.group_name} - {sender}]")
        else:
            parts.append(f"[{sender}]")

        # Add content
        if self.text:
            parts.append(self.text)
        if self.media_type:
            parts.append(f"<{self.media_type}: {self.media_url or self.media_path}>")

        return "\n".join(parts)


@dataclass
class ChannelConfig:
    """Configuration for a channel."""
    channel_type: ChannelType
    enabled: bool = True

    # Authentication
    token: Optional[str] = None
    api_key: Optional[str] = None
    credentials_path: Optional[str] = None

    # Access control
    dm_policy: AccessPolicy = AccessPolicy.PAIRING
    group_policy: AccessPolicy = AccessPolicy.ALLOWLIST
    activation_mode: ActivationMode = ActivationMode.MENTION

    # Allowlists
    dm_allowlist: List[str] = field(default_factory=list)
    group_allowlist: List[str] = field(default_factory=list)
    owner_ids: List[str] = field(default_factory=list)

    # Limits
    text_chunk_limit: int = 4000
    media_max_mb: int = 5
    rate_limit_per_minute: int = 30

    # Behavior
    auto_react: bool = True
    auto_react_emoji: str = "👀"
    send_read_receipts: bool = True
    typing_indicator: bool = True

    # Keywords
    mention_patterns: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)


@dataclass
class PairingCode:
    """Pairing code for user authentication."""
    code: str
    channel_type: ChannelType
    user_id: str
    created_at: datetime
    expires_at: datetime
    used: bool = False


class BaseChannel(ABC):
    """
    Abstract base class for all channel adapters.

    Each channel implementation must provide:
    - connect(): Establish connection to platform
    - disconnect(): Clean shutdown
    - send_message(): Send text/media to a chat
    - on_message(): Callback for inbound messages
    """

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.channel_type = config.channel_type
        self._connected = False
        self._message_callback: Optional[Callable] = None
        self._rate_limiter: Dict[str, List[datetime]] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the channel platform."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the channel platform."""
        pass

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send a message to a chat."""
        pass

    def set_message_callback(self, callback: Callable[[ChannelMessage], Any]):
        """Set callback for inbound messages."""
        self._message_callback = callback

    async def _handle_inbound(self, message: ChannelMessage):
        """Process inbound message through callback."""
        if self._message_callback:
            try:
                if asyncio.iscoroutinefunction(self._message_callback):
                    await self._message_callback(message)
                else:
                    self._message_callback(message)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

    def _check_rate_limit(self, chat_id: str) -> bool:
        """Check if rate limit exceeded for chat."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        if chat_id not in self._rate_limiter:
            self._rate_limiter[chat_id] = []

        # Clean old entries
        self._rate_limiter[chat_id] = [
            t for t in self._rate_limiter[chat_id] if t > minute_ago
        ]

        if len(self._rate_limiter[chat_id]) >= self.config.rate_limit_per_minute:
            return False

        self._rate_limiter[chat_id].append(now)
        return True

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks respecting limit."""
        limit = self.config.text_chunk_limit
        if len(text) <= limit:
            return [text]

        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break

            # Try to break at paragraph
            split_idx = text.rfind("\n\n", 0, limit)
            if split_idx == -1:
                # Try sentence
                split_idx = text.rfind(". ", 0, limit)
            if split_idx == -1:
                # Try word
                split_idx = text.rfind(" ", 0, limit)
            if split_idx == -1:
                split_idx = limit

            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].strip()

        return chunks

    def _check_access(self, message: ChannelMessage) -> bool:
        """Check if sender has access."""
        # Owner always has access
        if message.sender_id in self.config.owner_ids:
            return True

        if message.is_group:
            policy = self.config.group_policy
            allowlist = self.config.group_allowlist
        else:
            policy = self.config.dm_policy
            allowlist = self.config.dm_allowlist

        if policy == AccessPolicy.DISABLED:
            return False
        if policy == AccessPolicy.OPEN:
            return True
        if policy == AccessPolicy.ALLOWLIST:
            return message.sender_id in allowlist or "*" in allowlist

        # Pairing mode - handled by hub
        return False

    def _should_activate(self, message: ChannelMessage) -> bool:
        """Check if message should activate the agent."""
        if not message.is_group:
            return True  # Always activate for DMs

        mode = self.config.activation_mode

        if mode == ActivationMode.ALWAYS:
            return True

        if mode == ActivationMode.MENTION:
            if message.is_mention:
                return True
            # Check mention patterns
            for pattern in self.config.mention_patterns:
                if pattern.lower() in message.text.lower():
                    return True
            return False

        if mode == ActivationMode.KEYWORD:
            for keyword in self.config.trigger_keywords:
                if keyword.lower() in message.text.lower():
                    return True
            return False

        return False


class ChannelHub:
    """
    Central hub managing all channel connections.

    Provides:
    - Unified channel management
    - Message routing
    - Access control
    - Pairing code system
    - Statistics and monitoring
    """

    def __init__(self, config_path: str = None):
        """
        Initialize channel hub.

        Args:
            config_path: Path to channels config (default: ~/.cosmos/channels.json)
        """
        self.config_path = Path(config_path or os.path.expanduser("~/.cosmos/channels.json"))
        self.credentials_path = Path(os.path.expanduser("~/.cosmos/credentials"))

        self.channels: Dict[ChannelType, BaseChannel] = {}
        self.configs: Dict[ChannelType, ChannelConfig] = {}
        self.pairing_codes: Dict[str, PairingCode] = {}

        self._message_handlers: List[Callable] = []
        self._nexus = None

        # Statistics
        self.stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
            "start_time": datetime.now()
        }

        # Ensure directories exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.credentials_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        """Initialize the channel hub."""
        try:
            # Load configs
            self._load_configs()

            # Initialize Nexus
            if NEXUS_AVAILABLE:
                self._nexus = get_nexus()

            logger.info("ChannelHub initialized")
            return True

        except Exception as e:
            logger.error(f"ChannelHub initialization failed: {e}")
            return False

    def _load_configs(self):
        """Load channel configurations from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)

                for channel_name, config_data in data.get("channels", {}).items():
                    try:
                        channel_type = ChannelType(channel_name)
                        self.configs[channel_type] = ChannelConfig(
                            channel_type=channel_type,
                            **config_data
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid channel config for {channel_name}: {e}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid channels.json: {e}")

    def _save_configs(self):
        """Save channel configurations to file."""
        data = {"channels": {}}
        for channel_type, config in self.configs.items():
            data["channels"][channel_type.value] = {
                "enabled": config.enabled,
                "dm_policy": config.dm_policy.value,
                "group_policy": config.group_policy.value,
                "activation_mode": config.activation_mode.value,
                "dm_allowlist": config.dm_allowlist,
                "group_allowlist": config.group_allowlist,
                "owner_ids": config.owner_ids,
            }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    async def register_channel(
        self,
        channel_type: ChannelType,
        channel: BaseChannel,
        config: ChannelConfig = None
    ):
        """
        Register a channel with the hub.

        Args:
            channel_type: Type of channel
            channel: Channel adapter instance
            config: Optional config override
        """
        if config:
            self.configs[channel_type] = config

        self.channels[channel_type] = channel
        channel.set_message_callback(self._on_message)

        logger.info(f"Registered channel: {channel_type.value}")

    async def connect_channel(self, channel_type: ChannelType) -> bool:
        """Connect a specific channel."""
        if channel_type not in self.channels:
            logger.error(f"Channel not registered: {channel_type.value}")
            return False

        channel = self.channels[channel_type]
        try:
            success = await channel.connect()
            if success:
                logger.info(f"Connected channel: {channel_type.value}")

                # Emit to Nexus
                if self._nexus:
                    await self._nexus.emit("CHANNEL_CONNECTED", {
                        "channel": channel_type.value,
                        "timestamp": datetime.now().isoformat()
                    })

            return success

        except Exception as e:
            logger.error(f"Channel connection failed: {channel_type.value}: {e}")
            self.stats["errors"] += 1
            return False

    async def disconnect_channel(self, channel_type: ChannelType):
        """Disconnect a specific channel."""
        if channel_type in self.channels:
            await self.channels[channel_type].disconnect()
            logger.info(f"Disconnected channel: {channel_type.value}")

    async def connect_all(self):
        """Connect all registered and enabled channels."""
        for channel_type, channel in self.channels.items():
            config = self.configs.get(channel_type)
            if config and config.enabled:
                await self.connect_channel(channel_type)

    async def disconnect_all(self):
        """Disconnect all channels."""
        for channel_type in list(self.channels.keys()):
            await self.disconnect_channel(channel_type)

    async def _on_message(self, message: ChannelMessage):
        """Handle inbound message from any channel."""
        self.stats["messages_received"] += 1

        try:
            channel = self.channels.get(message.channel_type)
            if not channel:
                return

            # Check access
            if not channel._check_access(message):
                # Check if pairing mode
                config = self.configs.get(message.channel_type)
                if config and config.dm_policy == AccessPolicy.PAIRING:
                    await self._handle_pairing(message)
                return

            # Check activation
            if not channel._should_activate(message):
                return

            # Emit to Nexus
            if self._nexus:
                await self._nexus.emit("CHANNEL_MESSAGE_RECEIVED", {
                    "channel": message.channel_type.value,
                    "sender": message.sender_id,
                    "is_group": message.is_group,
                    "text_length": len(message.text),
                    "has_media": message.media_type is not None
                })

            # Call handlers
            for handler in self._message_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.stats["errors"] += 1

    async def _handle_pairing(self, message: ChannelMessage):
        """Handle pairing code flow for new users."""
        # Check if message is a pairing code
        text = message.text.strip().upper()

        for code_str, pairing in list(self.pairing_codes.items()):
            if (pairing.channel_type == message.channel_type and
                pairing.user_id == message.sender_id and
                code_str == text and
                not pairing.used and
                datetime.now() < pairing.expires_at):

                # Valid code - add to allowlist
                config = self.configs.get(message.channel_type)
                if config:
                    if message.is_group:
                        config.group_allowlist.append(message.sender_id)
                    else:
                        config.dm_allowlist.append(message.sender_id)

                    self._save_configs()

                pairing.used = True
                del self.pairing_codes[code_str]

                # Send confirmation
                await self.send_message(
                    message.channel_type,
                    message.channel_id,
                    "✓ Paired successfully! You can now chat with me."
                )

                logger.info(f"User paired: {message.sender_id} on {message.channel_type.value}")
                return

        # Generate new pairing code
        code = self._generate_pairing_code(message.channel_type, message.sender_id)
        await self.send_message(
            message.channel_type,
            message.channel_id,
            f"To chat with me, please enter this pairing code: **{code}**\n\nCode expires in 1 hour."
        )

    def _generate_pairing_code(self, channel_type: ChannelType, user_id: str) -> str:
        """Generate a new pairing code."""
        import random
        import string

        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        self.pairing_codes[code] = PairingCode(
            code=code,
            channel_type=channel_type,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )

        return code

    def add_message_handler(self, handler: Callable[[ChannelMessage], Any]):
        """Add a handler for inbound messages."""
        self._message_handlers.append(handler)

    async def send_message(
        self,
        channel_type: ChannelType,
        chat_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Send a message through a channel.

        Args:
            channel_type: Target channel
            chat_id: Platform-specific chat ID
            text: Message text
            media_path: Optional media file path
            reply_to: Optional message ID to reply to

        Returns:
            True if sent successfully
        """
        if channel_type not in self.channels:
            logger.error(f"Channel not available: {channel_type.value}")
            return False

        channel = self.channels[channel_type]

        if not channel.is_connected:
            logger.error(f"Channel not connected: {channel_type.value}")
            return False

        try:
            # Chunk text if needed
            chunks = channel._chunk_text(text)

            for i, chunk in enumerate(chunks):
                # Only attach media to first chunk
                media = media_path if i == 0 else None

                success = await channel.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    media_path=media,
                    reply_to=reply_to if i == 0 else None,
                    **kwargs
                )

                if not success:
                    return False

                # Small delay between chunks
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)

            self.stats["messages_sent"] += 1

            # Emit to Nexus
            if self._nexus:
                await self._nexus.emit("CHANNEL_MESSAGE_SENT", {
                    "channel": channel_type.value,
                    "chat_id": chat_id,
                    "text_length": len(text),
                    "has_media": media_path is not None
                })

            return True

        except Exception as e:
            logger.error(f"Send message failed: {e}")
            self.stats["errors"] += 1
            return False

    async def broadcast(
        self,
        text: str,
        channels: List[ChannelType] = None,
        exclude_channels: List[ChannelType] = None
    ) -> Dict[ChannelType, bool]:
        """
        Broadcast a message to multiple channels.

        Args:
            text: Message to broadcast
            channels: Specific channels (default: all connected)
            exclude_channels: Channels to exclude

        Returns:
            Dict of channel -> success status
        """
        results = {}
        target_channels = channels or list(self.channels.keys())
        exclude = exclude_channels or []

        for channel_type in target_channels:
            if channel_type in exclude:
                continue

            channel = self.channels.get(channel_type)
            if channel and channel.is_connected:
                # Would need chat IDs - this is a simplified version
                results[channel_type] = True  # Placeholder

        return results

    def get_status(self) -> Dict:
        """Get hub status and statistics."""
        channel_status = {}
        for channel_type, channel in self.channels.items():
            channel_status[channel_type.value] = {
                "connected": channel.is_connected,
                "enabled": self.configs.get(channel_type, ChannelConfig(channel_type)).enabled
            }

        return {
            "channels": channel_status,
            "stats": {
                **self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds()
            },
            "active_pairing_codes": len(self.pairing_codes)
        }

    def add_to_allowlist(
        self,
        channel_type: ChannelType,
        user_id: str,
        is_group: bool = False
    ):
        """Add a user to allowlist."""
        config = self.configs.get(channel_type)
        if not config:
            config = ChannelConfig(channel_type=channel_type)
            self.configs[channel_type] = config

        if is_group:
            if user_id not in config.group_allowlist:
                config.group_allowlist.append(user_id)
        else:
            if user_id not in config.dm_allowlist:
                config.dm_allowlist.append(user_id)

        self._save_configs()


# =============================================================================
# SINGLETON
# =============================================================================

_channel_hub: Optional[ChannelHub] = None


def get_channel_hub() -> ChannelHub:
    """Get or create the global channel hub."""
    global _channel_hub
    if _channel_hub is None:
        _channel_hub = ChannelHub()
    return _channel_hub

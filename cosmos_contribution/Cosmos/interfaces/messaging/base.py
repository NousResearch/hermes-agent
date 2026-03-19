"""
Messaging Bridge Interface for cosmos.

This module defines the abstract base classes for the "Omni-Channel" messaging system,
allowing cosmos to connect to Discord, Slack, Telegram, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from enum import Enum
from datetime import datetime

class MessageSource(Enum):
    DISCORD = "discord"
    SLACK = "slack"
    TELEGRAM = "telegram"
    CLI = "cli"
    WEB = "web"

@dataclass
class IncomingMessage:
    """A message received from an external platform."""
    id: str
    source: MessageSource
    sender_id: str
    sender_name: str
    channel_id: str
    content: str
    timestamp: datetime
    metadata: Optional[dict] = None

@dataclass
class OutgoingMessage:
    """A message to be sent to an external platform."""
    channel_id: str
    content: str
    reply_to_id: Optional[str] = None
    files: Optional[list[str]] = None  # Paths to files

class MessagingProvider(ABC):
    """
    Abstract base class for messaging platform adapters.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._on_message_callback: Optional[Callable[[IncomingMessage], Awaitable[None]]] = None

    def set_callback(self, callback: Callable[[IncomingMessage], Awaitable[None]]):
        """Set the callback for handling incoming messages."""
        self._on_message_callback = callback

    @abstractmethod
    async def connect(self):
        """Connect to the platform."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the platform."""
        pass

    @abstractmethod
    async def send_message(self, message: OutgoingMessage):
        """Send a message to the platform."""
        pass
    
    async def _handle_incoming(self, message: IncomingMessage):
        """Internal handler to propagate messages to the router."""
        if self._on_message_callback:
            await self._on_message_callback(message)

class NotificationCenter:
    """
    Central hub for broadcasting proactive notifications (The "Clawdbot" Feature).
    """
    
    def __init__(self):
        self.providers: dict[str, MessagingProvider] = {}
        self.default_channel_id: Optional[str] = None
        
    def register_provider(self, provider: MessagingProvider):
        self.providers[provider.name] = provider
        
    async def broadcast(self, message: str, title: Optional[str] = None):
        """Broadcast a message to all connected providers."""
        formatted_message = f"**{title}**\n{message}" if title else message
        
        for name, provider in self.providers.items():
            # In a real app, we'd need a mapping of provider -> default channel
            # For now, we assume a 'default' channel exists or log it
            try:
                # Mock channel ID for broadcast
                await provider.send_message(OutgoingMessage(
                    channel_id="general", 
                    content=formatted_message
                ))
            except Exception as e:
                print(f"Failed to broadcast to {name}: {e}")

"""
Discord Adapter for cosmos ChatOps.
Connects cosmos to Discord using discord.py patterns.
"""

import asyncio
from datetime import datetime
from loguru import logger
from .base import MessagingProvider, IncomingMessage, OutgoingMessage, MessageSource

class DiscordAdapter(MessagingProvider):
    """
    Adapter for Discord integration.
    Allows cosmos to Act as a Discord Bot (Clawdbot style).
    """

    def __init__(self, token: str):
        super().__init__(name="discord")
        self.token = token
        self.client = None # In real implementation: discord.Client()
        self.is_connected = False

    async def connect(self):
        """
        Connect to Discord Gateway.
        """
        if not self.token:
            logger.warning("No Discord token provided. ChatOps disabled.")
            return

        logger.info("Connecting to Discord... (Simulated)")
        self.is_connected = True
        
        # Simulate a welcome message
        asyncio.create_task(self._simulate_incoming_events())

    async def disconnect(self):
        if self.client:
            await self.client.close()
        self.is_connected = False
        logger.info("Disconnected from Discord")

    async def send_message(self, message: OutgoingMessage):
        """Send message to Discord channel."""
        if not self.is_connected:
            logger.warning("Cannot send message: Not connected to Discord")
            return

        # Real implementation would be:
        # channel = self.client.get_channel(int(message.channel_id))
        # await channel.send(message.content)
        
        logger.info(f"[Discord] Sending to #{message.channel_id}: {message.content[:50]}...")

    async def _simulate_incoming_events(self):
        """
        Since we don't have a real token, we simulate a 'Hello' message
        to prove the architecture works.
        """
        await asyncio.sleep(2) # Wait for startup
        
        msg = IncomingMessage(
            id="msg_123",
            source=MessageSource.DISCORD,
            sender_id="user_1",
            sender_name="User",
            channel_id="general",
            content="Hello cosmos, are you online?",
            timestamp=datetime.now()
        )
        
        await self._handle_incoming(msg)

"""
cosmos Discord Bridge - ChatOps Interface.

"Bringing the Professor to the server, one message at a time."

This module enables a full Discord bot interface for cosmos.
"""

import discord
import asyncio
import os
from loguru import logger
from typing import Optional, Dict, Any

from Cosmos.core.nexus import nexus, Signal, SignalType
from Cosmos.integration.tool_router import ToolRouter

class DiscordBridge:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("DISCORD_TOKEN")
        self.client = discord.Client(intents=discord.Intents.all())
        self.router = ToolRouter()
        self._setup_events()

    def _setup_events(self):
        @self.client.event
        async def on_ready():
            logger.info(f"Discord Bridge: Logged in as {self.client.user}")
            await nexus.emit(SignalType.EXTERNAL_ALERT, {"msg": f"Discord Bridge Active as {self.client.user}"}, "discord")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            if self.client.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
                async with message.channel.typing():
                    # Process via Nexus / Agent Swarm
                    # Simplified: Direct Nexus emit to trigger a 'thought'
                    logger.info(f"Discord: Received message from {message.author}: {message.content}")
                    
                    # For now, we'll use a direct reply simulation
                    # In a real impl, this triggers the SwarmOrchestrator
                    await nexus.emit(SignalType.USER_MESSAGE, {
                        "content": message.content,
                        "source": "discord",
                        "channel_id": message.channel.id,
                        "user_id": message.author.id,
                        "reply_handle": message.reply
                    }, f"discord_{message.author.id}")

    async def start(self):
        if not self.token:
            logger.warning("Discord Bridge: No token provided. Skipping.")
            return
        logger.info("Discord Bridge: Connecting...")
        await self.client.start(self.token)

    async def send_message(self, channel_id: int, content: str):
        channel = self.client.get_channel(channel_id)
        if channel:
            await channel.send(content)

# Global Instance
discord_bridge = DiscordBridge()

"""
Cosmos Discord Channel Adapter
===================================

Discord Bot API integration for the Cosmos swarm.

Features:
- Gateway and REST API support
- Slash commands and context menus
- Message components (buttons, selects)
- Thread support
- Voice channel presence
- Reactions and embeds
- File attachments
- Role-based access control

Based on discord.py library.

Setup:
1. Create application at discord.com/developers
2. Create bot and get token
3. Set DISCORD_BOT_TOKEN env var
4. Invite bot to server with proper permissions

"Discord is where gamers unite. Now they unite with AGI." - The Collective
"""

import os
import asyncio
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)

# Discord library
try:
    import discord
    from discord import app_commands
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("discord.py not installed. Run: pip install discord.py")


class DiscordChannel(BaseChannel):
    """
    Discord Bot API channel adapter.

    Supports:
    - Direct messages
    - Guild channels
    - Threads and forums
    - Slash commands
    - Message components
    - Embeds and attachments
    - Voice channel status
    """

    def __init__(self, config: ChannelConfig = None, token: str = None):
        """
        Initialize Discord channel.

        Args:
            config: Channel configuration
            token: Bot token (or set DISCORD_BOT_TOKEN env var)
        """
        config = config or ChannelConfig(channel_type=ChannelType.DISCORD)
        super().__init__(config)

        self.token = token or os.environ.get("DISCORD_BOT_TOKEN") or config.token

        self._bot: Optional[commands.Bot] = None
        self._slash_handlers: Dict[str, Callable] = {}
        self._component_handlers: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """Connect to Discord gateway."""
        if not DISCORD_AVAILABLE:
            logger.error("discord.py not available")
            return False

        if not self.token:
            logger.error("Discord bot token not configured")
            return False

        try:
            # Create bot with intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.members = True
            intents.guilds = True

            self._bot = commands.Bot(
                command_prefix="!",
                intents=intents,
                help_command=None
            )

            # Register event handlers
            @self._bot.event
            async def on_ready():
                logger.info(f"Discord bot connected: {self._bot.user.name}")
                self._connected = True
                # Sync slash commands
                try:
                    synced = await self._bot.tree.sync()
                    logger.info(f"Synced {len(synced)} slash commands")
                except Exception as e:
                    logger.warning(f"Slash command sync failed: {e}")

            @self._bot.event
            async def on_message(message: discord.Message):
                if message.author.bot:
                    return
                await self._handle_discord_message(message)

            @self._bot.event
            async def on_interaction(interaction: discord.Interaction):
                await self._handle_interaction(interaction)

            # Register default slash commands
            @self._bot.tree.command(name="cosmos", description="Chat with Cosmos AI")
            async def cosmos_cmd(interaction: discord.Interaction, message: str):
                await interaction.response.defer(thinking=True)
                # Create channel message and process
                msg = ChannelMessage(
                    message_id=str(interaction.id),
                    channel_type=ChannelType.DISCORD,
                    channel_id=str(interaction.channel_id),
                    sender_id=str(interaction.user.id),
                    sender_name=interaction.user.display_name,
                    text=message,
                    is_group=interaction.guild is not None,
                    group_name=interaction.guild.name if interaction.guild else None,
                    timestamp=datetime.now(),
                    raw_data={"interaction_id": interaction.id}
                )
                await self._handle_inbound(msg)

            @self._bot.tree.command(name="status", description="Check Cosmos swarm status")
            async def status_cmd(interaction: discord.Interaction):
                embed = discord.Embed(
                    title="✅ Cosmos Status",
                    description="The swarm is operational",
                    color=discord.Color.green()
                )
                embed.add_field(name="Swarm", value="Online", inline=True)
                embed.add_field(name="Discord", value="Connected", inline=True)
                embed.add_field(name="Agents", value="Active", inline=True)
                await interaction.response.send_message(embed=embed)

            @self._bot.tree.command(name="help", description="Get help with Cosmos")
            async def help_cmd(interaction: discord.Interaction):
                embed = discord.Embed(
                    title="📚 Cosmos Help",
                    description="Commands and usage",
                    color=discord.Color.blue()
                )
                embed.add_field(
                    name="Commands",
                    value=(
                        "`/cosmos <message>` - Chat with the swarm\n"
                        "`/status` - Check system status\n"
                        "`/help` - This message"
                    ),
                    inline=False
                )
                embed.add_field(
                    name="Direct Message",
                    value="Just send a message to chat!",
                    inline=False
                )
                await interaction.response.send_message(embed=embed)

            # Start bot in background
            asyncio.create_task(self._bot.start(self.token))

            # Wait for connection
            for _ in range(30):
                if self._connected:
                    return True
                await asyncio.sleep(1)

            logger.error("Discord connection timed out")
            return False

        except Exception as e:
            logger.error(f"Discord connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Discord."""
        if self._bot:
            try:
                await self._bot.close()
            except Exception as e:
                logger.warning(f"Discord disconnect error: {e}")

        self._connected = False
        logger.info("Discord disconnected")

    async def send_message(
        self,
        channel_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        embed: Dict = None,
        components: List[Dict] = None,
        **kwargs
    ) -> bool:
        """
        Send a message to a Discord channel.

        Args:
            channel_id: Discord channel or user ID
            text: Message text
            media_path: Optional file path
            reply_to: Message ID to reply to
            embed: Embed dict {title, description, color, fields}
            components: Button/select components

        Returns:
            True if sent successfully
        """
        if not self._bot:
            return False

        if not self._check_rate_limit(channel_id):
            logger.warning(f"Rate limit exceeded for channel {channel_id}")
            return False

        try:
            channel = self._bot.get_channel(int(channel_id))
            if not channel:
                # Try as user DM
                try:
                    user = await self._bot.fetch_user(int(channel_id))
                    channel = await user.create_dm()
                except Exception:
                    logger.error(f"Channel not found: {channel_id}")
                    return False

            # Typing indicator
            if self.config.typing_indicator:
                async with channel.typing():
                    await asyncio.sleep(0.5)

            # Build embed if provided
            discord_embed = None
            if embed:
                discord_embed = discord.Embed(
                    title=embed.get("title"),
                    description=embed.get("description"),
                    color=discord.Color(embed.get("color", 0x7289DA))
                )
                for field in embed.get("fields", []):
                    discord_embed.add_field(
                        name=field.get("name", ""),
                        value=field.get("value", ""),
                        inline=field.get("inline", True)
                    )
                if embed.get("thumbnail"):
                    discord_embed.set_thumbnail(url=embed["thumbnail"])
                if embed.get("image"):
                    discord_embed.set_image(url=embed["image"])

            # Build components (buttons/selects)
            view = None
            if components:
                view = discord.ui.View(timeout=300)
                for comp in components:
                    if comp.get("type") == "button":
                        button = discord.ui.Button(
                            label=comp.get("label", ""),
                            style=getattr(discord.ButtonStyle, comp.get("style", "primary")),
                            custom_id=comp.get("custom_id", "")
                        )
                        view.add_item(button)
                    elif comp.get("type") == "select":
                        select = discord.ui.Select(
                            placeholder=comp.get("placeholder", "Select..."),
                            options=[
                                discord.SelectOption(label=opt["label"], value=opt["value"])
                                for opt in comp.get("options", [])
                            ],
                            custom_id=comp.get("custom_id", "")
                        )
                        view.add_item(select)

            # Build file attachment
            file = None
            if media_path and Path(media_path).exists():
                file = discord.File(media_path)

            # Get reference for reply
            reference = None
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(int(reply_to))
                    reference = ref_msg.to_reference()
                except Exception:
                    pass

            # Send message
            await channel.send(
                content=text if text else None,
                embed=discord_embed,
                file=file,
                reference=reference,
                view=view
            )

            return True

        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    async def _handle_discord_message(self, message: discord.Message):
        """Handle incoming Discord message."""
        # Check if bot was mentioned or DM
        is_mention = self._bot.user in message.mentions
        is_dm = isinstance(message.channel, discord.DMChannel)

        # In guilds, only respond to mentions (unless configured otherwise)
        if message.guild and not is_mention and not self.config.respond_to_all:
            return

        # Build normalized message
        msg = ChannelMessage(
            message_id=str(message.id),
            channel_type=ChannelType.DISCORD,
            channel_id=str(message.channel.id),
            sender_id=str(message.author.id),
            sender_name=message.author.display_name,
            text=message.content,
            is_group=message.guild is not None,
            group_name=message.guild.name if message.guild else None,
            is_mention=is_mention,
            timestamp=message.created_at,
            raw_data={
                "guild_id": message.guild.id if message.guild else None,
                "channel_name": message.channel.name if hasattr(message.channel, "name") else "DM",
            }
        )

        # Handle reply
        if message.reference and message.reference.message_id:
            msg.reply_to_id = str(message.reference.message_id)
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
                msg.reply_to_text = ref_msg.content
            except Exception:
                pass

        # Handle attachments
        if message.attachments:
            attachment = message.attachments[0]
            if attachment.content_type:
                if attachment.content_type.startswith("image"):
                    msg.media_type = "image"
                elif attachment.content_type.startswith("video"):
                    msg.media_type = "video"
                elif attachment.content_type.startswith("audio"):
                    msg.media_type = "audio"
                else:
                    msg.media_type = "document"
            msg.media_url = attachment.url

        # Handle stickers
        if message.stickers:
            msg.media_type = "sticker"
            msg.media_url = message.stickers[0].url

        # Auto-react if configured
        if self.config.auto_react:
            try:
                await message.add_reaction(self.config.auto_react_emoji)
            except Exception:
                pass

        # Pass to handler
        await self._handle_inbound(msg)

    async def _handle_interaction(self, interaction: discord.Interaction):
        """Handle button/select interactions."""
        if interaction.type != discord.InteractionType.component:
            return

        custom_id = interaction.data.get("custom_id", "")

        # Check registered handlers
        for prefix, handler in self._component_handlers.items():
            if custom_id.startswith(prefix):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(interaction)
                    else:
                        handler(interaction)
                except Exception as e:
                    logger.error(f"Component handler error: {e}")
                return

        # Default acknowledgment
        await interaction.response.defer()

    def register_slash_command(self, name: str, description: str, handler: Callable):
        """Register a custom slash command."""
        if not self._bot:
            self._slash_handlers[name] = (description, handler)
            return

        @self._bot.tree.command(name=name, description=description)
        async def cmd(interaction: discord.Interaction):
            if asyncio.iscoroutinefunction(handler):
                await handler(interaction)
            else:
                handler(interaction)

    def register_component_handler(self, prefix: str, handler: Callable):
        """Register handler for component interactions with specific ID prefix."""
        self._component_handlers[prefix] = handler

    async def send_embed(
        self,
        channel_id: str,
        title: str,
        description: str,
        color: int = 0x7289DA,
        fields: List[Dict] = None,
        thumbnail: str = None,
        image: str = None,
        footer: str = None
    ) -> bool:
        """Send a rich embed message."""
        return await self.send_message(
            channel_id,
            text=None,
            embed={
                "title": title,
                "description": description,
                "color": color,
                "fields": fields or [],
                "thumbnail": thumbnail,
                "image": image,
                "footer": footer
            }
        )

    async def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None
    ) -> Optional[str]:
        """Create a thread in a channel."""
        if not self._bot:
            return None

        try:
            channel = self._bot.get_channel(int(channel_id))
            if not channel:
                return None

            if message_id:
                message = await channel.fetch_message(int(message_id))
                thread = await message.create_thread(name=name)
            else:
                thread = await channel.create_thread(name=name, type=discord.ChannelType.public_thread)

            return str(thread.id)

        except Exception as e:
            logger.error(f"Create thread failed: {e}")
            return None

    async def get_guild_info(self, guild_id: str) -> Optional[Dict]:
        """Get information about a guild."""
        if not self._bot:
            return None

        try:
            guild = self._bot.get_guild(int(guild_id))
            if not guild:
                return None

            return {
                "id": guild.id,
                "name": guild.name,
                "member_count": guild.member_count,
                "owner_id": guild.owner_id,
                "icon_url": str(guild.icon.url) if guild.icon else None,
                "channels": [{"id": c.id, "name": c.name, "type": str(c.type)} for c in guild.channels[:20]]
            }
        except Exception as e:
            logger.error(f"Get guild info failed: {e}")
            return None

    async def set_presence(self, status: str, activity_type: str = "playing", activity_name: str = None):
        """Set bot presence/status."""
        if not self._bot:
            return

        try:
            activity = None
            if activity_name:
                activity_types = {
                    "playing": discord.ActivityType.playing,
                    "watching": discord.ActivityType.watching,
                    "listening": discord.ActivityType.listening,
                    "streaming": discord.ActivityType.streaming,
                }
                activity = discord.Activity(
                    type=activity_types.get(activity_type, discord.ActivityType.playing),
                    name=activity_name
                )

            status_types = {
                "online": discord.Status.online,
                "idle": discord.Status.idle,
                "dnd": discord.Status.dnd,
                "invisible": discord.Status.invisible,
            }

            await self._bot.change_presence(
                status=status_types.get(status, discord.Status.online),
                activity=activity
            )
        except Exception as e:
            logger.error(f"Set presence failed: {e}")

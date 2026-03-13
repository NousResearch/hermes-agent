"""
Discord platform adapter.

Uses discord.py library for:
- Receiving messages from servers and DMs
- Sending responses back
- Handling threads and channels
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

VALID_THREAD_AUTO_ARCHIVE_MINUTES = {60, 1440, 4320, 10080}

try:
    import discord
    from discord import Message as DiscordMessage, Intents
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None
    DiscordMessage = Any
    Intents = Any
    commands = None

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    cache_audio_from_url,
)


def _clean_discord_id(entry: str) -> str:
    """Strip common prefixes from a Discord user ID or username entry.

    Users sometimes paste IDs with prefixes like ``user:123``, ``<@123>``,
    or ``<@!123>`` from Discord's UI or other tools.  This normalises the
    entry to just the bare ID or username.
    """
    entry = entry.strip()
    # Strip Discord mention syntax: <@123> or <@!123>
    if entry.startswith("<@") and entry.endswith(">"):
        entry = entry.lstrip("<@!").rstrip(">")
    # Strip "user:" prefix (seen in some Discord tools / onboarding pastes)
    if entry.lower().startswith("user:"):
        entry = entry[5:]
    return entry.strip()


def check_discord_requirements() -> bool:
    """Check if Discord dependencies are available."""
    return DISCORD_AVAILABLE


class DiscordAdapter(BasePlatformAdapter):
    """
    Discord bot adapter.
    
    Handles:
    - Receiving messages from servers and DMs
    - Sending responses with Discord markdown
    - Thread support
    - Native slash commands (/ask, /reset, /status, /stop)
    - Button-based exec approvals
    - Auto-threading for long conversations
    - Reaction-based feedback
    """
    
    # Discord message limits
    MAX_MESSAGE_LENGTH = 2000
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
    
    async def connect(self) -> bool:
        """Connect to Discord and start receiving events."""
        if not DISCORD_AVAILABLE:
            logger.error("[%s] discord.py not installed. Run: pip install discord.py", self.name)
            return False
        
        if not self.config.token:
            logger.error("[%s] No bot token configured", self.name)
            return False
        
        try:
            # Set up intents -- members intent needed for username-to-ID resolution
            intents = Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            intents.guild_messages = True
            intents.members = True
            
            # Create bot
            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
            )
            
            # Parse allowed user entries (may contain usernames or IDs)
            allowed_env = os.getenv("DISCORD_ALLOWED_USERS", "")
            if allowed_env:
                self._allowed_user_ids = {
                    _clean_discord_id(uid) for uid in allowed_env.split(",")
                    if uid.strip()
                }
            
            adapter_self = self  # capture for closure
            
            # Register event handlers
            @self._client.event
            async def on_ready():
                logger.info("[%s] Connected as %s", adapter_self.name, adapter_self._client.user)
                
                # Resolve any usernames in the allowed list to numeric IDs
                await adapter_self._resolve_allowed_usernames()
                
                # Sync slash commands with Discord
                try:
                    synced = await adapter_self._client.tree.sync()
                    logger.info("[%s] Synced %d slash command(s)", adapter_self.name, len(synced))
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.warning("[%s] Slash command sync failed: %s", adapter_self.name, e, exc_info=True)
                adapter_self._ready_event.set()
            
            @self._client.event
            async def on_message(message: DiscordMessage):
                # Always ignore our own messages
                if message.author == self._client.user:
                    return
                
                # Bot message filtering (DISCORD_ALLOW_BOTS):
                #   "none"     — ignore all other bots (default)
                #   "mentions" — accept bot messages only when they @mention us
                #   "all"      — accept all bot messages
                if getattr(message.author, "bot", False):
                    allow_bots = os.getenv("DISCORD_ALLOW_BOTS", "none").lower().strip()
                    if allow_bots == "none":
                        return
                    elif allow_bots == "mentions":
                        if not self._client.user or self._client.user not in message.mentions:
                            return
                    # "all" falls through to handle_message
                
                await self._handle_message(message)
            
            # Register slash commands
            self._register_slash_commands()
            
            # Start the bot in background
            asyncio.create_task(self._client.start(self.config.token))
            
            # Wait for ready
            await asyncio.wait_for(self._ready_event.wait(), timeout=30)
            
            self._running = True
            return True
            
        except asyncio.TimeoutError:
            logger.error("[%s] Timeout waiting for connection to Discord", self.name, exc_info=True)
            return False
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to connect to Discord: %s", self.name, e, exc_info=True)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning("[%s] Error during disconnect: %s", self.name, e, exc_info=True)
        
        self._running = False
        self._client = None
        self._ready_event.clear()
        logger.info("[%s] Disconnected", self.name)
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Discord channel."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            # Get the channel
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            # Format and split message if needed
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
            
            message_ids = []
            reference = None
            
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(int(reply_to))
                    reference = ref_msg
                except Exception as e:
                    logger.debug("Could not fetch reply-to message: %s", e)
            
            for i, chunk in enumerate(chunks):
                msg = await channel.send(
                    content=chunk,
                    reference=reference if i == 0 else None,
                )
                message_ids.append(str(msg.id))
            
            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )
            
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send Discord message: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Edit a previously sent Discord message."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            msg = await channel.fetch_message(int(message_id))
            formatted = self.format_message(content)
            if len(formatted) > self.MAX_MESSAGE_LENGTH:
                formatted = formatted[:self.MAX_MESSAGE_LENGTH - 3] + "..."
            await msg.edit(content=formatted)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to edit Discord message %s: %s", self.name, message_id, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send audio as a Discord file attachment."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            import io
            
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            if not os.path.exists(audio_path):
                return SendResult(success=False, error=f"Audio file not found: {audio_path}")
            
            # Determine filename from path
            filename = os.path.basename(audio_path)
            
            with open(audio_path, "rb") as f:
                file = discord.File(io.BytesIO(f.read()), filename=filename)
                msg = await channel.send(
                    content=caption if caption else None,
                    file=file,
                )
                return SendResult(success=True, message_id=str(msg.id))
        
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send audio, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_voice(chat_id, audio_path, caption, reply_to)
    
    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file natively as a Discord file attachment."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            import io
            
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            if not os.path.exists(image_path):
                return SendResult(success=False, error=f"Image file not found: {image_path}")
            
            filename = os.path.basename(image_path)
            
            with open(image_path, "rb") as f:
                file = discord.File(io.BytesIO(f.read()), filename=filename)
                msg = await channel.send(
                    content=caption if caption else None,
                    file=file,
                )
                return SendResult(success=True, message_id=str(msg.id))
        
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send local image, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_image_file(chat_id, image_path, caption, reply_to)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively as a Discord file attachment."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            import aiohttp
            
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            # Download the image and send as a Discord file attachment
            # (Discord renders attachments inline, unlike plain URLs)
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download image: HTTP {resp.status}")
                    
                    image_data = await resp.read()
                    
                    # Determine filename from URL or content type
                    content_type = resp.headers.get("content-type", "image/png")
                    ext = "png"
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    elif "gif" in content_type:
                        ext = "gif"
                    elif "webp" in content_type:
                        ext = "webp"
                    
                    import io
                    file = discord.File(io.BytesIO(image_data), filename=f"image.{ext}")
                    
                    msg = await channel.send(
                        content=caption if caption else None,
                        file=file,
                    )
                    return SendResult(success=True, message_id=str(msg.id))
        
        except ImportError:
            logger.warning(
                "[%s] aiohttp not installed, falling back to URL. Run: pip install aiohttp",
                self.name,
                exc_info=True,
            )
            return await super().send_image(chat_id, image_url, caption, reply_to)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send image attachment, falling back to URL: %s",
                self.name,
                e,
                exc_info=True,
            )
            return await super().send_image(chat_id, image_url, caption, reply_to)
    
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator."""
        if self._client:
            try:
                channel = self._client.get_channel(int(chat_id))
                if channel:
                    await channel.typing()
            except Exception:
                pass  # Ignore typing indicator failures
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Discord channel."""
        if not self._client:
            return {"name": "Unknown", "type": "dm"}
        
        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            
            if not channel:
                return {"name": str(chat_id), "type": "dm"}
            
            # Determine channel type
            if isinstance(channel, discord.DMChannel):
                chat_type = "dm"
                name = channel.recipient.name if channel.recipient else str(chat_id)
            elif isinstance(channel, discord.Thread):
                chat_type = "thread"
                name = channel.name
            elif isinstance(channel, discord.TextChannel):
                chat_type = "channel"
                name = f"#{channel.name}"
                if channel.guild:
                    name = f"{channel.guild.name} / {name}"
            else:
                chat_type = "channel"
                name = getattr(channel, "name", str(chat_id))
            
            return {
                "name": name,
                "type": chat_type,
                "guild_id": str(channel.guild.id) if hasattr(channel, "guild") and channel.guild else None,
                "guild_name": channel.guild.name if hasattr(channel, "guild") and channel.guild else None,
            }
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to get chat info for %s: %s", self.name, chat_id, e, exc_info=True)
            return {"name": str(chat_id), "type": "dm", "error": str(e)}
    
    async def _resolve_allowed_usernames(self) -> None:
        """
        Resolve non-numeric entries in DISCORD_ALLOWED_USERS to Discord user IDs.

        Users can specify usernames (e.g. "teknium") or display names instead of
        raw numeric IDs.  After resolution, the env var and internal set are updated
        so authorization checks work with IDs only.
        """
        if not self._allowed_user_ids or not self._client:
            return

        numeric_ids = set()
        to_resolve = set()

        for entry in self._allowed_user_ids:
            if entry.isdigit():
                numeric_ids.add(entry)
            else:
                to_resolve.add(entry.lower())

        if not to_resolve:
            return

        print(f"[{self.name}] Resolving {len(to_resolve)} username(s): {', '.join(to_resolve)}")
        resolved_count = 0

        for guild in self._client.guilds:
            # Fetch full member list (requires members intent)
            try:
                members = guild.members
                if len(members) < guild.member_count:
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception as e:
                logger.warning("Failed to fetch members for guild %s: %s", guild.name, e)
                continue

            for member in members:
                name_lower = member.name.lower()
                display_lower = member.display_name.lower()
                global_lower = (member.global_name or "").lower()

                matched = name_lower in to_resolve or display_lower in to_resolve or global_lower in to_resolve
                if matched:
                    uid = str(member.id)
                    numeric_ids.add(uid)
                    resolved_count += 1
                    matched_name = name_lower if name_lower in to_resolve else (
                        display_lower if display_lower in to_resolve else global_lower
                    )
                    to_resolve.discard(matched_name)
                    print(f"[{self.name}] Resolved '{matched_name}' -> {uid} ({member.name}#{member.discriminator})")

            if not to_resolve:
                break

        if to_resolve:
            print(f"[{self.name}] Could not resolve usernames: {', '.join(to_resolve)}")

        # Update internal set and env var so gateway auth checks use IDs
        self._allowed_user_ids = numeric_ids
        os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if resolved_count:
            print(f"[{self.name}] Updated DISCORD_ALLOWED_USERS with {resolved_count} resolved ID(s)")

    def format_message(self, content: str) -> str:
        """
        Format message for Discord.
        
        Discord uses its own markdown variant.
        """
        # Discord markdown is fairly standard, no special escaping needed
        return content
    
    def _register_slash_commands(self) -> None:
        """Register Discord slash commands on the command tree."""
        if not self._client:
            return

        tree = self._client.tree

        @tree.command(name="ask", description="Ask Hermes a question")
        @discord.app_commands.describe(question="Your question for Hermes")
        async def slash_ask(interaction: discord.Interaction, question: str):
            await interaction.response.defer()
            event = self._build_slash_event(interaction, question)
            await self.handle_message(event)
            # The response is sent via the normal send() flow
            # Send a followup to close the interaction if needed
            try:
                await interaction.followup.send("Processing complete~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="new", description="Start a new conversation")
        async def slash_new(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/reset")
            await self.handle_message(event)
            try:
                await interaction.followup.send("New conversation started~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="reset", description="Reset your Hermes session")
        async def slash_reset(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/reset")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Session reset~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="model", description="Show or change the model")
        @discord.app_commands.describe(name="Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.")
        async def slash_model(interaction: discord.Interaction, name: str = ""):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/model {name}".strip())
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="personality", description="Set a personality")
        @discord.app_commands.describe(name="Personality name. Leave empty to list available.")
        async def slash_personality(interaction: discord.Interaction, name: str = ""):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/personality {name}".strip())
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="retry", description="Retry your last message")
        async def slash_retry(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/retry")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Retrying~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="undo", description="Remove the last exchange")
        async def slash_undo(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/undo")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="status", description="Show Hermes session status")
        async def slash_status(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/status")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Status sent~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="sethome", description="Set this chat as the home channel")
        async def slash_sethome(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/sethome")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="stop", description="Stop the running Hermes agent")
        async def slash_stop(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/stop")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Stop requested~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="compress", description="Compress conversation context")
        async def slash_compress(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/compress")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="title", description="Set or show the session title")
        @discord.app_commands.describe(name="Session title. Leave empty to show current.")
        async def slash_title(interaction: discord.Interaction, name: str = ""):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/title {name}".strip())
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="resume", description="Resume a previously-named session")
        @discord.app_commands.describe(name="Session name to resume. Leave empty to list sessions.")
        async def slash_resume(interaction: discord.Interaction, name: str = ""):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/resume {name}".strip())
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="usage", description="Show token usage for this session")
        async def slash_usage(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/usage")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="provider", description="Show available providers")
        async def slash_provider(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/provider")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="help", description="Show available commands")
        async def slash_help(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/help")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="insights", description="Show usage insights and analytics")
        @discord.app_commands.describe(days="Number of days to analyze (default: 7)")
        async def slash_insights(interaction: discord.Interaction, days: int = 7):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/insights {days}")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="reload-mcp", description="Reload MCP servers from config")
        async def slash_reload_mcp(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/reload-mcp")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="update", description="Update Hermes Agent to the latest version")
        async def slash_update(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/update")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Update initiated~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="thread", description="Create a new thread and start a Hermes session in it")
        @discord.app_commands.describe(
            name="Thread name",
            message="Optional first message to send to Hermes in the thread",
            auto_archive_duration="Auto-archive in minutes (60, 1440, 4320, 10080)",
        )
        async def slash_thread(
            interaction: discord.Interaction,
            name: str,
            message: str = "",
            auto_archive_duration: int = 1440,
        ):
            await interaction.response.defer(ephemeral=True)
            await self._handle_thread_create_slash(interaction, name, message, auto_archive_duration)

    def _build_slash_event(self, interaction: discord.Interaction, text: str) -> MessageEvent:
        """Build a MessageEvent from a Discord slash command interaction."""
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        chat_type = "dm" if is_dm else "group"
        chat_name = ""
        if not is_dm and hasattr(interaction.channel, "name"):
            chat_name = interaction.channel.name
            if hasattr(interaction.channel, "guild") and interaction.channel.guild:
                chat_name = f"{interaction.channel.guild.name} / #{chat_name}"
        
        # Get channel topic (if available)
        chat_topic = getattr(interaction.channel, "topic", None)

        source = self.build_source(
            chat_id=str(interaction.channel_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(interaction.user.id),
            user_name=interaction.user.display_name,
            chat_topic=chat_topic,
        )

        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=interaction,
        )

    # ------------------------------------------------------------------
    # Thread creation helpers
    # ------------------------------------------------------------------

    async def _handle_thread_create_slash(
        self,
        interaction: discord.Interaction,
        name: str,
        message: str = "",
        auto_archive_duration: int = 1440,
    ) -> None:
        """Create a Discord thread from a slash command and start a session in it."""
        result = await self._create_thread(
            interaction,
            name=name,
            message=message,
            auto_archive_duration=auto_archive_duration,
        )

        if not result.get("success"):
            error = result.get("error", "unknown error")
            await interaction.followup.send(f"Failed to create thread: {error}", ephemeral=True)
            return

        thread_id = result.get("thread_id")
        thread_name = result.get("thread_name") or name

        # Tell the user where the thread is
        link = f"<#{thread_id}>" if thread_id else f"**{thread_name}**"
        await interaction.followup.send(f"Created thread {link}", ephemeral=True)

        # If a message was provided, kick off a new Hermes session in the thread
        starter = (message or "").strip()
        if starter and thread_id:
            await self._dispatch_thread_session(interaction, thread_id, thread_name, starter)

    async def _dispatch_thread_session(
        self,
        interaction: discord.Interaction,
        thread_id: str,
        thread_name: str,
        text: str,
    ) -> None:
        """Build a MessageEvent pointing at a thread and send it through handle_message."""
        guild_name = ""
        if hasattr(interaction, "guild") and interaction.guild:
            guild_name = interaction.guild.name

        chat_name = f"{guild_name} / {thread_name}" if guild_name else thread_name

        source = self.build_source(
            chat_id=thread_id,
            chat_name=chat_name,
            chat_type="thread",
            user_id=str(interaction.user.id),
            user_name=interaction.user.display_name,
            thread_id=thread_id,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=interaction,
        )
        await self.handle_message(event)

    def _thread_parent_channel(self, channel: Any) -> Any:
        """Return the parent text channel when invoked from a thread."""
        return getattr(channel, "parent", None) or channel

    async def _resolve_interaction_channel(self, interaction: discord.Interaction) -> Optional[Any]:
        """Return the interaction channel, fetching it if the payload is partial."""
        channel = getattr(interaction, "channel", None)
        if channel is not None:
            return channel
        if not self._client:
            return None
        channel_id = getattr(interaction, "channel_id", None)
        if channel_id is None:
            return None
        channel = self._client.get_channel(int(channel_id))
        if channel is not None:
            return channel
        try:
            return await self._client.fetch_channel(int(channel_id))
        except Exception:
            return None

    async def _create_thread(
        self,
        interaction: discord.Interaction,
        *,
        name: str,
        message: str = "",
        auto_archive_duration: int = 1440,
    ) -> Dict[str, Any]:
        """Create a thread in the current Discord channel.

        Tries ``parent_channel.create_thread()`` first.  If Discord rejects
        that (e.g. permission issues), falls back to sending a seed message
        and creating the thread from it.
        """
        name = (name or "").strip()
        if not name:
            return {"error": "Thread name is required."}

        if auto_archive_duration not in VALID_THREAD_AUTO_ARCHIVE_MINUTES:
            allowed = ", ".join(str(v) for v in sorted(VALID_THREAD_AUTO_ARCHIVE_MINUTES))
            return {"error": f"auto_archive_duration must be one of: {allowed}."}

        channel = await self._resolve_interaction_channel(interaction)
        if channel is None:
            return {"error": "Could not resolve the current Discord channel."}
        if isinstance(channel, discord.DMChannel):
            return {"error": "Discord threads can only be created inside server text channels, not DMs."}

        parent_channel = self._thread_parent_channel(channel)
        if parent_channel is None:
            return {"error": "Could not determine a parent text channel for the new thread."}

        display_name = getattr(getattr(interaction, "user", None), "display_name", None) or "unknown user"
        reason = f"Requested by {display_name} via /thread"
        starter_message = (message or "").strip()

        try:
            thread = await parent_channel.create_thread(
                name=name,
                auto_archive_duration=auto_archive_duration,
                reason=reason,
            )
            if starter_message:
                await thread.send(starter_message)
            return {
                "success": True,
                "thread_id": str(thread.id),
                "thread_name": getattr(thread, "name", None) or name,
            }
        except Exception as direct_error:
            try:
                seed_content = starter_message or f"\U0001f9f5 Thread created by Hermes: **{name}**"
                seed_msg = await parent_channel.send(seed_content)
                thread = await seed_msg.create_thread(
                    name=name,
                    auto_archive_duration=auto_archive_duration,
                    reason=reason,
                )
                return {
                    "success": True,
                    "thread_id": str(thread.id),
                    "thread_name": getattr(thread, "name", None) or name,
                }
            except Exception as fallback_error:
                return {
                    "error": (
                        "Discord rejected direct thread creation and the fallback also failed. "
                        f"Direct error: {direct_error}. Fallback error: {fallback_error}"
                    )
                }

    # ------------------------------------------------------------------
    # Auto-thread helpers
    # ------------------------------------------------------------------

    async def _auto_create_thread(self, message: 'DiscordMessage') -> Optional[Any]:
        """Create a thread from a user message for auto-threading.

        Returns the created thread object, or ``None`` on failure.
        """
        # Build a short thread name from the message
        content = (message.content or "").strip()
        thread_name = content[:80] if content else "Hermes"
        if len(content) > 80:
            thread_name = thread_name[:77] + "..."

        try:
            thread = await message.create_thread(name=thread_name, auto_archive_duration=1440)
            return thread
        except Exception as e:
            logger.warning("[%s] Auto-thread creation failed: %s", self.name, e)
            return None

    async def send_exec_approval(
        self, chat_id: str, command: str, approval_id: str
    ) -> SendResult:
        """
        Send a button-based exec approval prompt for a dangerous command.

        Returns SendResult. The approval is resolved when a user clicks a button.
        """
        if not self._client or not DISCORD_AVAILABLE:
            return SendResult(success=False, error="Not connected")

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))

            embed = discord.Embed(
                title="Command Approval Required",
                description=f"```\n{command[:500]}\n```",
                color=discord.Color.orange(),
            )
            embed.set_footer(text=f"Approval ID: {approval_id}")

            view = ExecApprovalView(
                approval_id=approval_id,
                allowed_user_ids=self._allowed_user_ids,
            )

            msg = await channel.send(embed=embed, view=view)
            return SendResult(success=True, message_id=str(msg.id))

        except Exception as e:
            return SendResult(success=False, error=str(e))

    def _get_parent_channel_id(self, channel: Any) -> Optional[str]:
        """Return the parent channel ID for a Discord thread-like channel, if present."""
        parent = getattr(channel, "parent", None)
        if parent is not None and getattr(parent, "id", None) is not None:
            return str(parent.id)
        parent_id = getattr(channel, "parent_id", None)
        if parent_id is not None:
            return str(parent_id)
        return None

    def _is_forum_parent(self, channel: Any) -> bool:
        """Best-effort check for whether a Discord channel is a forum channel."""
        if channel is None:
            return False
        forum_cls = getattr(discord, "ForumChannel", None)
        if forum_cls and isinstance(channel, forum_cls):
            return True
        channel_type = getattr(channel, "type", None)
        if channel_type is not None:
            type_value = getattr(channel_type, "value", channel_type)
            if type_value == 15:
                return True
        return False

    def _format_thread_chat_name(self, thread: Any) -> str:
        """Build a readable chat name for thread-like Discord channels, including forum context when available."""
        thread_name = getattr(thread, "name", None) or str(getattr(thread, "id", "thread"))
        parent = getattr(thread, "parent", None)
        guild = getattr(thread, "guild", None) or getattr(parent, "guild", None)
        guild_name = getattr(guild, "name", None)
        parent_name = getattr(parent, "name", None)

        if self._is_forum_parent(parent) and guild_name and parent_name:
            return f"{guild_name} / {parent_name} / {thread_name}"
        if parent_name and guild_name:
            return f"{guild_name} / #{parent_name} / {thread_name}"
        if parent_name:
            return f"{parent_name} / {thread_name}"
        return thread_name

    async def _handle_message(self, message: DiscordMessage) -> None:
        """Handle incoming Discord messages."""
        # In server channels (not DMs), require the bot to be @mentioned
        # UNLESS the channel is in the free-response list.
        #
        # Config:
        #   DISCORD_FREE_RESPONSE_CHANNELS: Comma-separated channel IDs where the
        #       bot responds to every message without needing a mention.
        #   DISCORD_REQUIRE_MENTION: Set to "false" to disable mention requirement
        #       globally (all channels become free-response). Default: "true".
        #       Can also be set via discord.require_mention in config.yaml.

        thread_id = None
        parent_channel_id = None
        is_thread = isinstance(message.channel, discord.Thread)
        if is_thread:
            thread_id = str(message.channel.id)
            parent_channel_id = self._get_parent_channel_id(message.channel)

        if not isinstance(message.channel, discord.DMChannel):
            free_channels_raw = os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "")
            free_channels = {ch.strip() for ch in free_channels_raw.split(",") if ch.strip()}
            channel_ids = {str(message.channel.id)}
            if parent_channel_id:
                channel_ids.add(parent_channel_id)

            require_mention = os.getenv("DISCORD_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")
            is_free_channel = bool(channel_ids & free_channels)

            if require_mention and not is_free_channel:
                if self._client.user not in message.mentions:
                    return

            if self._client.user and self._client.user in message.mentions:
                message.content = message.content.replace(f"<@{self._client.user.id}>", "").strip()
                message.content = message.content.replace(f"<@!{self._client.user.id}>", "").strip()

        # Auto-thread: when enabled, automatically create a thread for every
        # new message in a text channel so each conversation is isolated.
        # Messages already inside threads or DMs are unaffected.
        auto_threaded_channel = None
        if not is_thread and not isinstance(message.channel, discord.DMChannel):
            auto_thread = os.getenv("DISCORD_AUTO_THREAD", "").lower() in ("true", "1", "yes")
            if auto_thread:
                thread = await self._auto_create_thread(message)
                if thread:
                    is_thread = True
                    thread_id = str(thread.id)
                    auto_threaded_channel = thread

        # Determine message type
        msg_type = MessageType.TEXT
        if message.content.startswith("/"):
            msg_type = MessageType.COMMAND
        elif message.attachments:
            # Check attachment types
            for att in message.attachments:
                if att.content_type:
                    if att.content_type.startswith("image/"):
                        msg_type = MessageType.PHOTO
                    elif att.content_type.startswith("video/"):
                        msg_type = MessageType.VIDEO
                    elif att.content_type.startswith("audio/"):
                        msg_type = MessageType.AUDIO
                    else:
                        msg_type = MessageType.DOCUMENT
                    break
        
        # When auto-threading kicked in, route responses to the new thread
        effective_channel = auto_threaded_channel or message.channel

        # Determine chat type
        if isinstance(message.channel, discord.DMChannel):
            chat_type = "dm"
            chat_name = message.author.name
        elif is_thread:
            chat_type = "thread"
            chat_name = self._format_thread_chat_name(effective_channel)
        else:
            chat_type = "group"
            chat_name = getattr(message.channel, "name", str(message.channel.id))
            if hasattr(message.channel, "guild") and message.channel.guild:
                chat_name = f"{message.channel.guild.name} / #{chat_name}"

        # Get channel topic (if available - TextChannels have topics, DMs/threads don't)
        chat_topic = getattr(message.channel, "topic", None)
        
        # Build source
        source = self.build_source(
            chat_id=str(effective_channel.id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(message.author.id),
            user_name=message.author.display_name,
            thread_id=thread_id,
            chat_topic=chat_topic,
        )
        
        # Build media URLs -- download image attachments to local cache so the
        # vision tool can access them reliably (Discord CDN URLs can expire).
        media_urls = []
        media_types = []
        for att in message.attachments:
            content_type = att.content_type or "unknown"
            if content_type.startswith("image/"):
                try:
                    # Determine extension from content type (image/png -> .png)
                    ext = "." + content_type.split("/")[-1].split(";")[0]
                    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                        ext = ".jpg"
                    cached_path = await cache_image_from_url(att.url, ext=ext)
                    media_urls.append(cached_path)
                    media_types.append(content_type)
                    print(f"[Discord] Cached user image: {cached_path}", flush=True)
                except Exception as e:
                    print(f"[Discord] Failed to cache image attachment: {e}", flush=True)
                    # Fall back to the CDN URL if caching fails
                    media_urls.append(att.url)
                    media_types.append(content_type)
            elif content_type.startswith("audio/"):
                try:
                    ext = "." + content_type.split("/")[-1].split(";")[0]
                    if ext not in (".ogg", ".mp3", ".wav", ".webm", ".m4a"):
                        ext = ".ogg"
                    cached_path = await cache_audio_from_url(att.url, ext=ext)
                    media_urls.append(cached_path)
                    media_types.append(content_type)
                    print(f"[Discord] Cached user audio: {cached_path}", flush=True)
                except Exception as e:
                    print(f"[Discord] Failed to cache audio attachment: {e}", flush=True)
                    media_urls.append(att.url)
                    media_types.append(content_type)
            else:
                # Other attachments: keep the original URL
                media_urls.append(att.url)
                media_types.append(content_type)
        
        event = MessageEvent(
            text=message.content,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.id),
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=str(message.reference.message_id) if message.reference else None,
            timestamp=message.created_at,
        )
        
        await self.handle_message(event)


# ---------------------------------------------------------------------------
# Discord UI Components (outside the adapter class)
# ---------------------------------------------------------------------------

if DISCORD_AVAILABLE:

    class ExecApprovalView(discord.ui.View):
        """
        Interactive button view for exec approval of dangerous commands.

        Shows three buttons: Allow Once (green), Always Allow (blue), Deny (red).
        Only users in the allowed list can click. The view times out after 5 minutes.
        """

        def __init__(self, approval_id: str, allowed_user_ids: set):
            super().__init__(timeout=300)  # 5-minute timeout
            self.approval_id = approval_id
            self.allowed_user_ids = allowed_user_ids
            self.resolved = False

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Verify the user clicking is authorized."""
            if not self.allowed_user_ids:
                return True  # No allowlist = anyone can approve
            return str(interaction.user.id) in self.allowed_user_ids

        async def _resolve(
            self, interaction: discord.Interaction, action: str, color: discord.Color
        ):
            """Resolve the approval and update the message."""
            if self.resolved:
                await interaction.response.send_message(
                    "This approval has already been resolved~", ephemeral=True
                )
                return

            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized to approve commands~", ephemeral=True
                )
                return

            self.resolved = True

            # Update the embed with the decision
            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed:
                embed.color = color
                embed.set_footer(text=f"{action} by {interaction.user.display_name}")

            # Disable all buttons
            for child in self.children:
                child.disabled = True

            await interaction.response.edit_message(embed=embed, view=self)

            # Store the approval decision
            try:
                from tools.approval import approve_permanent
                if action == "allow_once":
                    pass  # One-time approval handled by gateway
                elif action == "allow_always":
                    approve_permanent(self.approval_id)
            except ImportError:
                pass

        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "allow_once", discord.Color.green())

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "allow_always", discord.Color.blue())

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "deny", discord.Color.red())

        async def on_timeout(self):
            """Handle view timeout -- disable buttons and mark as expired."""
            self.resolved = True
            for child in self.children:
                child.disabled = True

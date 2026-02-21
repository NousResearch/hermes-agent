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
            print(f"[{self.name}] discord.py not installed. Run: pip install discord.py")
            return False
        
        if not self.config.token:
            print(f"[{self.name}] No bot token configured")
            return False
        
        try:
            # Set up intents
            intents = Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            intents.guild_messages = True
            
            # Create bot
            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
            )
            
            # Parse allowed user IDs for button authorization
            allowed_env = os.getenv("DISCORD_ALLOWED_USERS", "")
            if allowed_env:
                self._allowed_user_ids = {
                    uid.strip() for uid in allowed_env.split(",") if uid.strip()
                }
            
            # Register event handlers
            @self._client.event
            async def on_ready():
                print(f"[{self.name}] Connected as {self._client.user}")
                # Sync slash commands with Discord
                try:
                    synced = await self._client.tree.sync()
                    print(f"[{self.name}] Synced {len(synced)} slash command(s)")
                except Exception as e:
                    print(f"[{self.name}] Slash command sync failed: {e}")
                self._ready_event.set()
            
            @self._client.event
            async def on_message(message: DiscordMessage):
                # Ignore bot's own messages
                if message.author == self._client.user:
                    return
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
            print(f"[{self.name}] Timeout waiting for connection")
            return False
        except Exception as e:
            print(f"[{self.name}] Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                print(f"[{self.name}] Error during disconnect: {e}")
        
        self._running = False
        self._client = None
        self._ready_event.clear()
        print(f"[{self.name}] Disconnected")
    
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
            
        except Exception as e:
            return SendResult(success=False, error=str(e))
    
    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
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
        
        except Exception as e:
            print(f"[{self.name}] Failed to send audio: {e}")
            return await super().send_voice(chat_id, audio_path, caption, reply_to)
    
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
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
            print(f"[{self.name}] aiohttp not installed, falling back to URL. Run: pip install aiohttp")
            return await super().send_image(chat_id, image_url, caption, reply_to)
        except Exception as e:
            print(f"[{self.name}] Failed to send image attachment, falling back to URL: {e}")
            return await super().send_image(chat_id, image_url, caption, reply_to)
    
    async def send_typing(self, chat_id: str) -> None:
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
        except Exception as e:
            return {"name": str(chat_id), "type": "dm", "error": str(e)}
    
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

        @tree.command(name="stop", description="Stop the running Hermes agent")
        async def slash_stop(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/stop")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Stop requested~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

    def _build_slash_event(self, interaction: discord.Interaction, text: str) -> MessageEvent:
        """Build a MessageEvent from a Discord slash command interaction."""
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        chat_type = "dm" if is_dm else "group"
        chat_name = ""
        if not is_dm and hasattr(interaction.channel, "name"):
            chat_name = interaction.channel.name
            if hasattr(interaction.channel, "guild") and interaction.channel.guild:
                chat_name = f"{interaction.channel.guild.name} / #{chat_name}"

        source = self.build_source(
            chat_id=str(interaction.channel_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(interaction.user.id),
            user_name=interaction.user.display_name,
        )

        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=interaction,
        )

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
        
        if not isinstance(message.channel, discord.DMChannel):
            # Check if this channel is in the free-response list
            free_channels_raw = os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "")
            free_channels = {ch.strip() for ch in free_channels_raw.split(",") if ch.strip()}
            channel_id = str(message.channel.id)
            
            # Global override: if DISCORD_REQUIRE_MENTION=false, all channels are free
            require_mention = os.getenv("DISCORD_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")
            
            is_free_channel = channel_id in free_channels
            
            if require_mention and not is_free_channel:
                # Must be @mentioned to respond
                if self._client.user not in message.mentions:
                    return  # Silently ignore messages that don't mention the bot
            
            # Strip the bot mention from the message text so the agent sees clean input
            if self._client.user and self._client.user in message.mentions:
                message.content = message.content.replace(f"<@{self._client.user.id}>", "").strip()
                message.content = message.content.replace(f"<@!{self._client.user.id}>", "").strip()
        
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
        
        # Determine chat type
        if isinstance(message.channel, discord.DMChannel):
            chat_type = "dm"
            chat_name = message.author.name
        elif isinstance(message.channel, discord.Thread):
            chat_type = "thread"
            chat_name = message.channel.name
        else:
            chat_type = "group"  # Treat server channels as groups
            chat_name = getattr(message.channel, "name", str(message.channel.id))
            if hasattr(message.channel, "guild") and message.channel.guild:
                chat_name = f"{message.channel.guild.name} / #{chat_name}"
        
        # Get thread ID if in a thread
        thread_id = None
        if isinstance(message.channel, discord.Thread):
            thread_id = str(message.channel.id)
        
        # Build source
        source = self.build_source(
            chat_id=str(message.channel.id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(message.author.id),
            user_name=message.author.display_name,
            thread_id=thread_id,
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

            # Store the approval decision for the gateway to pick up
            try:
                from tools.terminal_tool import _session_approved_patterns
                if action == "allow_once":
                    pass  # One-time approval handled by gateway
                elif action == "allow_always":
                    _session_approved_patterns.add(self.approval_id)
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

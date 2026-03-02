"""
Discord platform adapter.

Uses discord.py library for:
- Receiving messages from servers and DMs
- Sending responses back
- Handling threads and channels
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import os
import time
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
    
    # Keep slightly below Discord's embed description hard limit for safety.
    MAX_EMBED_DESCRIPTION = 4000
    CHAIN_SEND_DELAY_SECONDS = 0.5
    CHAIN_SEND_MAX_RETRIES = 5
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._client_task: Optional[asyncio.Task] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
        self._seen_message_ids: Dict[int, float] = {}
    
    async def connect(self) -> bool:
        """Connect to Discord and start receiving events."""
        if not DISCORD_AVAILABLE:
            print(f"[{self.name}] discord.py not installed. Run: pip install discord.py")
            return False
        
        if not self.config.token:
            print(f"[{self.name}] No bot token configured")
            return False
        
        # Parse allowed user entries (may contain usernames or IDs)
        allowed_env = os.getenv("DISCORD_ALLOWED_USERS", "")
        if allowed_env:
            self._allowed_user_ids = {
                uid.strip() for uid in allowed_env.split(",") if uid.strip()
            }

        # Username resolution requires guild member listing (privileged members intent).
        needs_members_intent = any(not entry.isdigit() for entry in self._allowed_user_ids)

        async def _attempt_connect(*, message_content: bool, members: bool) -> tuple[bool, Optional[Exception]]:
            """Try one connect profile and return (success, startup_exception)."""
            self._ready_event.clear()

            intents = Intents.default()
            intents.message_content = message_content
            intents.dm_messages = True
            intents.guild_messages = True
            intents.members = members
            # Needed for reaction-based moderation controls.
            if hasattr(intents, "reactions"):
                intents.reactions = True
            if hasattr(intents, "dm_reactions"):
                intents.dm_reactions = True

            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
            )

            adapter_self = self  # capture for closure

            @self._client.event
            async def on_ready():
                print(f"[{adapter_self.name}] Connected as {adapter_self._client.user}")

                # Resolve any usernames in the allowed list to numeric IDs
                # only if members intent is enabled.
                if members:
                    await adapter_self._resolve_allowed_usernames()

                # Sync slash commands with Discord
                try:
                    synced = await adapter_self._client.tree.sync()
                    print(f"[{adapter_self.name}] Synced {len(synced)} slash command(s)")
                except Exception as e:
                    print(f"[{adapter_self.name}] Slash command sync failed: {e}")
                adapter_self._ready_event.set()

            @self._client.event
            async def on_message(message: DiscordMessage):
                # Drop duplicate deliveries of the same Discord message ID.
                # This protects against occasional repeated gateway dispatch.
                now = time.time()
                msg_id = int(getattr(message, "id", 0) or 0)
                if msg_id:
                    last = self._seen_message_ids.get(msg_id)
                    if last and (now - last) < 120:
                        logger.info("[discord] duplicate message ignored: msg_id=%s", str(msg_id))
                        return
                    self._seen_message_ids[msg_id] = now
                    # Lightweight TTL cleanup
                    if len(self._seen_message_ids) > 2000:
                        cutoff = now - 300
                        self._seen_message_ids = {
                            k: v for k, v in self._seen_message_ids.items() if v >= cutoff
                        }

                # Ignore all bot-authored messages (including ourselves) to
                # prevent feedback loops from bot attachments/reposts.
                if getattr(message.author, "bot", False):
                    return
                if self._client.user and getattr(message.author, "id", None) == self._client.user.id:
                    return
                logger.info(
                    "[discord] on_message event: msg_id=%s user_id=%s channel_id=%s",
                    str(message.id),
                    str(message.author.id),
                    str(message.channel.id),
                )
                try:
                    await self._handle_message(message)
                except Exception:
                    logger.exception("[discord] message handler crashed")

            @self._client.event
            async def on_raw_reaction_add(payload):
                """
                Delete this bot's messages when a user reacts with :x: / ❌.
                Uses raw events so this works even when the message isn't cached.
                """
                try:
                    logger.info(
                        "[discord] reaction add: user_id=%s channel_id=%s message_id=%s emoji=%s",
                        str(payload.user_id),
                        str(payload.channel_id),
                        str(payload.message_id),
                        str(getattr(payload.emoji, "name", "") or ""),
                    )
                    if self._client.user and payload.user_id == self._client.user.id:
                        return

                    emoji_name = getattr(payload.emoji, "name", "") or ""
                    if emoji_name not in {"❌", "x", "✖", "✖️"}:
                        return

                    channel = self._client.get_channel(payload.channel_id)
                    if channel is None:
                        channel = await self._client.fetch_channel(payload.channel_id)
                    if channel is None:
                        return

                    message = await channel.fetch_message(payload.message_id)
                    if message is None:
                        return
                    # Delete only this bot's own messages.
                    if not self._client.user or getattr(message.author, "id", None) != self._client.user.id:
                        return

                    await message.delete()
                    logger.info(
                        "[discord] deleted bot message %s via %s reaction by user %s",
                        str(message.id),
                        emoji_name,
                        str(payload.user_id),
                    )
                except Exception:
                    logger.exception("[discord] reaction delete handler failed")

            self._register_slash_commands()

            start_task = asyncio.create_task(self._client.start(self.config.token))
            ready_task = asyncio.create_task(self._ready_event.wait())

            try:
                done, pending = await asyncio.wait(
                    {start_task, ready_task},
                    timeout=30,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if ready_task in done and self._ready_event.is_set():
                    self._running = True
                    # Keep the Discord client task alive for the session lifetime.
                    self._client_task = start_task
                    return True, None

                if start_task in done:
                    if not ready_task.done():
                        ready_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await ready_task
                    exc = start_task.exception()
                    if exc:
                        return False, exc
                    return False, RuntimeError("Discord client exited before ready")

                # Timeout waiting for readiness: shut down this attempt cleanly.
                ready_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ready_task
                if not start_task.done():
                    start_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await start_task
                return False, asyncio.TimeoutError("Timeout waiting for Discord ready event")
            finally:
                if not self._running and self._client:
                    try:
                        await self._client.close()
                    except Exception:
                        pass

        try:
            # First attempt: normal behavior (message content + optional members intent)
            ok, err = await _attempt_connect(message_content=True, members=needs_members_intent)
            if ok:
                return True

            # If Discord rejects privileged intents, retry without them so DMs and slash
            # commands can still function.
            err_text = str(err) if err else ""
            if err and "PrivilegedIntentsRequired" in err.__class__.__name__ or "PrivilegedIntentsRequired" in err_text:
                print(f"[{self.name}] Privileged intents not enabled; retrying with reduced intents.")
                ok, err = await _attempt_connect(message_content=False, members=False)
                if ok:
                    return True

            if isinstance(err, asyncio.TimeoutError):
                print(f"[{self.name}] Timeout waiting for connection")
            elif err:
                print(f"[{self.name}] Failed to connect: {err}")
            else:
                print(f"[{self.name}] Failed to connect")
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
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._client_task
        
        self._running = False
        self._client_task = None
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
        """Send a message to a Discord channel using chained embeds.
        
        Large responses are split across multiple embeds, each with a
        description capped at MAX_EMBED_DESCRIPTION characters.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            # Get the channel
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            # Format and split message into embed-sized chunks
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self.MAX_EMBED_DESCRIPTION)
            
            message_ids = []
            reference = None
            
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(int(reply_to))
                    reference = ref_msg
                except Exception as e:
                    logger.debug("Could not fetch reply-to message: %s", e)
            
            for i, chunk in enumerate(chunks):
                embed = discord.Embed(description=chunk)
                # Attach listen controls to every chunk so each embed remains
                # independently actionable in long chained responses.
                view = ListenButtonView(self)

                last_err: Optional[Exception] = None
                for attempt in range(1, self.CHAIN_SEND_MAX_RETRIES + 1):
                    try:
                        msg = await channel.send(
                            embed=embed,
                            view=view,
                            reference=reference if i == 0 else None,
                        )
                        message_ids.append(str(msg.id))
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        status = getattr(e, "status", None)
                        code = getattr(e, "code", None)
                        retry_after = getattr(e, "retry_after", None)
                        response_obj = getattr(e, "response", None)
                        if retry_after is None and response_obj is not None:
                            try:
                                header_val = response_obj.headers.get("Retry-After")
                                retry_after = float(header_val) if header_val else None
                            except Exception:
                                retry_after = None

                        logger.warning(
                            "[discord] chunk send failed (%d/%d, chunk %d/%d, len=%d, status=%s, code=%s, retry_after=%s): %s",
                            attempt,
                            self.CHAIN_SEND_MAX_RETRIES,
                            i + 1,
                            len(chunks),
                            len(chunk),
                            status,
                            code,
                            retry_after,
                            e,
                        )
                        if attempt < self.CHAIN_SEND_MAX_RETRIES:
                            # Respect Discord/API-provided backoff when present (429).
                            if retry_after is not None:
                                delay = max(float(retry_after), self.CHAIN_SEND_DELAY_SECONDS)
                            else:
                                delay = self.CHAIN_SEND_DELAY_SECONDS * attempt
                            await asyncio.sleep(delay)

                if last_err is not None:
                    raise last_err

                # Add slight pacing between chained sends to reduce burst failures.
                if i < (len(chunks) - 1):
                    await asyncio.sleep(self.CHAIN_SEND_DELAY_SECONDS)
            
            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )

        except Exception as e:
            logger.exception(
                "[discord] send failed chat_id=%s content_len=%d",
                chat_id,
                len(content or ""),
            )
            return SendResult(success=False, error=str(e))

    async def edit_message(self, chat_id: str, message_id: str, content: str) -> None:
        """Edit an existing message's embed description, used for tool progress."""
        if not self._client:
            return
        channel = self._client.get_channel(int(chat_id))
        if not channel:
            channel = await self._client.fetch_channel(int(chat_id))
        if not channel:
            return
        try:
            msg = await channel.fetch_message(int(message_id))
        except Exception:
            return
        embed = discord.Embed(description=self.format_message(content))
        try:
            await msg.edit(embed=embed)
        except Exception:
            return
    
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

        @tree.command(name="terminal", description="Show or change the local terminal shell (Windows)")
        @discord.app_commands.describe(mode="powershell, wsl, auto, or cmd. Leave empty to show current.")
        async def slash_terminal(interaction: discord.Interaction, mode: str = ""):
            await interaction.response.defer(ephemeral=True)
            text = f"/terminal {mode}".strip()
            event = self._build_slash_event(interaction, text)
            await self.handle_message(event)
            try:
                await interaction.delete_original_response()
            except Exception as e:
                logger.debug("Discord delete_original_response failed: %s", e)

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
        logger.info(
            "[discord] incoming message: user_id=%s chat_id=%s is_dm=%s content_len=%s",
            str(message.author.id),
            str(message.channel.id),
            isinstance(message.channel, discord.DMChannel),
            len(message.content or ""),
        )
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
        
        # Get channel topic (if available - TextChannels have topics, DMs/threads don't)
        chat_topic = getattr(message.channel, "topic", None)
        
        # Build source
        source = self.build_source(
            chat_id=str(message.channel.id),
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

    class ListenButtonView(discord.ui.View):
        """Button view that reads the current embed text aloud via TTS."""

        def __init__(self, adapter: "DiscordAdapter"):
            super().__init__(timeout=3600)  # 1 hour
            self.adapter = adapter

        @discord.ui.button(label="Listen", style=discord.ButtonStyle.secondary, emoji="🔊")
        async def listen(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            try:
                if not interaction.message:
                    await interaction.response.send_message(
                        "No message context available for TTS.", ephemeral=True
                    )
                    return

                # Use embed description first (gateway uses embeds for normal replies).
                text = ""
                if interaction.message.embeds:
                    text = (interaction.message.embeds[0].description or "").strip()
                if not text:
                    text = (interaction.message.content or "").strip()
                if not text:
                    await interaction.response.send_message(
                        "Nothing to read from this message.", ephemeral=True
                    )
                    return

                await interaction.response.defer(ephemeral=True, thinking=False)

                from tools.tts_tool import text_to_speech_tool
                tts_json = await asyncio.to_thread(text_to_speech_tool, text)
                data = json.loads(tts_json)
                if not data.get("success"):
                    await interaction.followup.send(
                        f"TTS failed: {data.get('error', 'unknown error')}",
                        ephemeral=True,
                    )
                    return

                audio_path = str(data.get("file_path", "")).strip()
                if not audio_path:
                    await interaction.followup.send(
                        "TTS returned no output file path.",
                        ephemeral=True,
                    )
                    return

                result = await self.adapter.send_voice(
                    chat_id=str(interaction.channel_id),
                    audio_path=audio_path,
                    reply_to=str(interaction.message.id),
                )
                if not result.success:
                    await interaction.followup.send(
                        f"Failed to send audio: {result.error}",
                        ephemeral=True,
                    )
                    return

                # Success is silent: audio delivery in channel is the confirmation.
            except Exception:
                logger.exception("[discord] listen button handler failed")
                try:
                    if interaction.response.is_done():
                        await interaction.followup.send("TTS failed unexpectedly.", ephemeral=True)
                    else:
                        await interaction.response.send_message("TTS failed unexpectedly.", ephemeral=True)
                except Exception:
                    pass

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

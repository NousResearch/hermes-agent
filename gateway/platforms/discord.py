from __future__ import annotations

"""
Discord platform adapter.

Uses discord.py library for:
- Receiving messages from servers and DMs
- Sending responses back
- Handling threads and channels
"""

import asyncio
import json
import logging
import os
import struct
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Any

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
from gateway.platforms.discord_impl import config as discord_config
from gateway.platforms.discord_impl import delivery as discord_delivery
from gateway.platforms.discord_impl import intake as discord_intake
from gateway.platforms.discord_impl import interactions as discord_interactions
from gateway.platforms.discord_impl import state as discord_state
from gateway.platforms.discord_impl import threads as discord_threads


def check_discord_requirements() -> bool:
    """Check if Discord dependencies are available."""
    return DISCORD_AVAILABLE


class VoiceReceiver:
    """Captures and decodes voice audio from a Discord voice channel.

    Attaches to a VoiceClient's socket listener, decrypts RTP packets
    (NaCl transport + DAVE E2EE), decodes Opus to PCM, and buffers
    per-user audio.  A polling loop detects silence and delivers
    completed utterances via a callback.
    """

    SILENCE_THRESHOLD = 1.5    # seconds of silence → end of utterance
    MIN_SPEECH_DURATION = 0.5  # minimum seconds to process (skip noise)
    SAMPLE_RATE = 48000        # Discord native rate
    CHANNELS = 2               # Discord sends stereo

    def __init__(self, voice_client, allowed_user_ids: set = None):
        self._vc = voice_client
        self._allowed_user_ids = allowed_user_ids or set()
        self._running = False

        # Decryption
        self._secret_key: Optional[bytes] = None
        self._dave_session = None
        self._bot_ssrc: int = 0

        # SSRC -> user_id mapping (populated from SPEAKING events)
        self._ssrc_to_user: Dict[int, int] = {}
        self._lock = threading.Lock()

        # Per-user audio buffers
        self._buffers: Dict[int, bytearray] = defaultdict(bytearray)
        self._last_packet_time: Dict[int, float] = {}

        # Opus decoder per SSRC (each user needs own decoder state)
        self._decoders: Dict[int, object] = {}

        # Pause flag: don't capture while bot is playing TTS
        self._paused = False

        # Debug logging counter (instance-level to avoid cross-instance races)
        self._packet_debug_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start listening for voice packets."""
        conn = self._vc._connection
        self._secret_key = bytes(conn.secret_key)
        self._dave_session = conn.dave_session
        self._bot_ssrc = conn.ssrc

        self._install_speaking_hook(conn)
        conn.add_socket_listener(self._on_packet)
        self._running = True
        logger.info("VoiceReceiver started (bot_ssrc=%d)", self._bot_ssrc)

    def stop(self):
        """Stop listening and clean up."""
        self._running = False
        try:
            self._vc._connection.remove_socket_listener(self._on_packet)
        except Exception:
            pass
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()
            self._decoders.clear()
            self._ssrc_to_user.clear()
        logger.info("VoiceReceiver stopped")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # ------------------------------------------------------------------
    # SSRC -> user_id mapping via SPEAKING opcode hook
    # ------------------------------------------------------------------

    def map_ssrc(self, ssrc: int, user_id: int):
        with self._lock:
            self._ssrc_to_user[ssrc] = user_id

    def _install_speaking_hook(self, conn):
        """Wrap the voice websocket hook to capture SPEAKING events (op 5).

        VoiceConnectionState stores the hook as ``conn.hook`` (public attr).
        It is passed to DiscordVoiceWebSocket on each (re)connect, so we
        must wrap it on the VoiceConnectionState level AND on the current
        live websocket instance.
        """
        original_hook = conn.hook
        receiver_self = self

        async def wrapped_hook(ws, msg):
            if isinstance(msg, dict) and msg.get("op") == 5:
                data = msg.get("d", {})
                ssrc = data.get("ssrc")
                user_id = data.get("user_id")
                if ssrc and user_id:
                    logger.info("SPEAKING event: ssrc=%d -> user=%s", ssrc, user_id)
                    receiver_self.map_ssrc(int(ssrc), int(user_id))
            if original_hook:
                await original_hook(ws, msg)

        # Set on connection state (for future reconnects)
        conn.hook = wrapped_hook
        # Set on the current live websocket (for immediate effect)
        try:
            from discord.utils import MISSING
            if hasattr(conn, 'ws') and conn.ws is not MISSING:
                conn.ws._hook = wrapped_hook
                logger.info("Speaking hook installed on live websocket")
        except Exception as e:
            logger.warning("Could not install hook on live ws: %s", e)

    # ------------------------------------------------------------------
    # Packet handler (called from SocketReader thread)
    # ------------------------------------------------------------------

    def _on_packet(self, data: bytes):
        if not self._running or self._paused:
            return

        # Log first few raw packets for debugging
        self._packet_debug_count += 1
        if self._packet_debug_count <= 5:
            logger.debug(
                "Raw UDP packet: len=%d, first_bytes=%s",
                len(data), data[:4].hex() if len(data) >= 4 else "short",
            )

        if len(data) < 16:
            return

        # RTP version check: top 2 bits must be 10 (version 2).
        # Lower bits may vary (padding, extension, CSRC count).
        # Payload type (byte 1 lower 7 bits) = 0x78 (120) for voice.
        if (data[0] >> 6) != 2 or (data[1] & 0x7F) != 0x78:
            if self._packet_debug_count <= 5:
                logger.debug("Skipped non-RTP: byte0=0x%02x byte1=0x%02x", data[0], data[1])
            return

        first_byte = data[0]
        _, _, seq, timestamp, ssrc = struct.unpack_from(">BBHII", data, 0)

        # Skip bot's own audio
        if ssrc == self._bot_ssrc:
            return

        # Calculate dynamic RTP header size (RFC 9335 / rtpsize mode)
        cc = first_byte & 0x0F  # CSRC count
        has_extension = bool(first_byte & 0x10)  # extension bit
        header_size = 12 + (4 * cc) + (4 if has_extension else 0)

        if len(data) < header_size + 4:  # need at least header + nonce
            return

        # Read extension length from preamble (for skipping after decrypt)
        ext_data_len = 0
        if has_extension:
            ext_preamble_offset = 12 + (4 * cc)
            ext_words = struct.unpack_from(">H", data, ext_preamble_offset + 2)[0]
            ext_data_len = ext_words * 4

        if self._packet_debug_count <= 10:
            with self._lock:
                known_user = self._ssrc_to_user.get(ssrc, "unknown")
            logger.debug(
                "RTP packet: ssrc=%d, seq=%d, user=%s, hdr=%d, ext_data=%d",
                ssrc, seq, known_user, header_size, ext_data_len,
            )

        header = bytes(data[:header_size])
        payload_with_nonce = data[header_size:]

        # --- NaCl transport decrypt (aead_xchacha20_poly1305_rtpsize) ---
        if len(payload_with_nonce) < 4:
            return
        nonce = bytearray(24)
        nonce[:4] = payload_with_nonce[-4:]
        encrypted = bytes(payload_with_nonce[:-4])

        try:
            import nacl.secret  # noqa: delayed import – only in voice path
            box = nacl.secret.Aead(self._secret_key)
            decrypted = box.decrypt(encrypted, header, bytes(nonce))
        except Exception as e:
            if self._packet_debug_count <= 10:
                logger.warning("NaCl decrypt failed: %s (hdr=%d, enc=%d)", e, header_size, len(encrypted))
            return

        # Skip encrypted extension data to get the actual opus payload
        if ext_data_len and len(decrypted) > ext_data_len:
            decrypted = decrypted[ext_data_len:]

        # --- DAVE E2EE decrypt ---
        if self._dave_session:
            with self._lock:
                user_id = self._ssrc_to_user.get(ssrc, 0)
            if user_id:
                try:
                    import davey
                    decrypted = self._dave_session.decrypt(
                        user_id, davey.MediaType.audio, decrypted
                    )
                except Exception as e:
                    # Unencrypted passthrough — use NaCl-decrypted data as-is
                    if "Unencrypted" not in str(e):
                        if self._packet_debug_count <= 10:
                            logger.warning("DAVE decrypt failed for ssrc=%d: %s", ssrc, e)
                        return
            # If SSRC unknown (no SPEAKING event yet), skip DAVE and try
            # Opus decode directly — audio may be in passthrough mode.
            # Buffer will get a user_id when SPEAKING event arrives later.

        # --- Opus decode -> PCM ---
        try:
            if ssrc not in self._decoders:
                self._decoders[ssrc] = discord.opus.Decoder()
            pcm = self._decoders[ssrc].decode(decrypted)
            with self._lock:
                self._buffers[ssrc].extend(pcm)
                self._last_packet_time[ssrc] = time.monotonic()
        except Exception as e:
            logger.debug("Opus decode error for SSRC %s: %s", ssrc, e)
            return

    # ------------------------------------------------------------------
    # Silence detection
    # ------------------------------------------------------------------

    def _infer_user_for_ssrc(self, ssrc: int) -> int:
        """Try to infer user_id for an unmapped SSRC.

        When the bot rejoins a voice channel, Discord may not resend
        SPEAKING events for users already speaking.  If exactly one
        allowed user is in the channel, map the SSRC to them.
        """
        try:
            channel = self._vc.channel
            if not channel:
                return 0
            bot_id = self._vc.user.id if self._vc.user else 0
            allowed = self._allowed_user_ids
            candidates = [
                m.id for m in channel.members
                if m.id != bot_id and (not allowed or str(m.id) in allowed)
            ]
            if len(candidates) == 1:
                uid = candidates[0]
                self._ssrc_to_user[ssrc] = uid
                logger.info("Auto-mapped ssrc=%d -> user=%d (sole allowed member)", ssrc, uid)
                return uid
        except Exception:
            pass
        return 0

    def check_silence(self) -> list:
        """Return list of (user_id, pcm_bytes) for completed utterances."""
        now = time.monotonic()
        completed = []

        with self._lock:
            ssrc_user_map = dict(self._ssrc_to_user)
            ssrc_list = list(self._buffers.keys())

            for ssrc in ssrc_list:
                last_time = self._last_packet_time.get(ssrc, now)
                silence_duration = now - last_time
                buf = self._buffers[ssrc]
                # 48kHz, 16-bit, stereo = 192000 bytes/sec
                buf_duration = len(buf) / (self.SAMPLE_RATE * self.CHANNELS * 2)

                if silence_duration >= self.SILENCE_THRESHOLD and buf_duration >= self.MIN_SPEECH_DURATION:
                    user_id = ssrc_user_map.get(ssrc, 0)
                    if not user_id:
                        # SSRC not mapped (SPEAKING event missing after bot rejoin).
                        # Infer from allowed users in the voice channel.
                        user_id = self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buf)))
                    self._buffers[ssrc] = bytearray()
                    self._last_packet_time.pop(ssrc, None)
                elif silence_duration >= self.SILENCE_THRESHOLD * 2:
                    # Stale buffer with no valid user — discard
                    self._buffers.pop(ssrc, None)
                    self._last_packet_time.pop(ssrc, None)

        return completed

    # ------------------------------------------------------------------
    # PCM -> WAV conversion (for Whisper STT)
    # ------------------------------------------------------------------

    @staticmethod
    def pcm_to_wav(pcm_data: bytes, output_path: str,
                   src_rate: int = 48000, src_channels: int = 2):
        """Convert raw PCM to 16kHz mono WAV via ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            f.write(pcm_data)
            pcm_path = f.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "s16le",
                    "-ar", str(src_rate),
                    "-ac", str(src_channels),
                    "-i", pcm_path,
                    "-ar", "16000",
                    "-ac", "1",
                    output_path,
                ],
                check=True,
                timeout=10,
            )
        finally:
            try:
                os.unlink(pcm_path)
            except OSError:
                pass


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
    
    # Auto-disconnect from voice channel after this many seconds of inactivity
    VOICE_TIMEOUT = 300

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
        # Voice channel state (per-guild)
        self._voice_clients: Dict[int, Any] = {}  # guild_id -> VoiceClient
        self._voice_text_channels: Dict[int, int] = {}  # guild_id -> text_channel_id
        self._voice_timeout_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> timeout task
        # Phase 2: voice listening
        self._voice_receivers: Dict[int, VoiceReceiver] = {}  # guild_id -> VoiceReceiver
        self._voice_listen_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> listen loop
        self._voice_input_callback: Optional[Callable] = None  # set by run.py
        self._on_voice_disconnect: Optional[Callable] = None  # set by run.py
        # Track threads where the bot has participated so follow-up messages
        # in those threads don't require @mention.  Persisted to disk so the
        # set survives gateway restarts.
        self._bot_participated_threads: set = self._load_participated_threads()
        # Cap to prevent unbounded growth (Discord threads get archived).
        self._MAX_TRACKED_THREADS = 500
    
    async def connect(self) -> bool:
        """Connect to Discord and start receiving events."""
        if not DISCORD_AVAILABLE:
            logger.error("[%s] discord.py not installed. Run: pip install discord.py", self.name)
            return False

        # Load opus codec for voice channel support
        if not discord.opus.is_loaded():
            import ctypes.util
            opus_path = ctypes.util.find_library("opus")
            # ctypes.util.find_library fails on macOS with Homebrew-installed libs,
            # so fall back to known Homebrew paths if needed.
            if not opus_path:
                import sys
                _homebrew_paths = (
                    "/opt/homebrew/lib/libopus.dylib",  # Apple Silicon
                    "/usr/local/lib/libopus.dylib",     # Intel Mac
                )
                if sys.platform == "darwin":
                    for _hp in _homebrew_paths:
                        if os.path.isfile(_hp):
                            opus_path = _hp
                            break
            if opus_path:
                try:
                    discord.opus.load_opus(opus_path)
                except Exception:
                    logger.warning("Opus codec found at %s but failed to load", opus_path)
            if not discord.opus.is_loaded():
                logger.warning("Opus codec not found — voice channel playback disabled")
        
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
            intents.voice_states = True
            
            # Create bot
            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
            )
            
            # Parse allowed user entries (may contain usernames or IDs)
            self._allowed_user_ids = discord_config.parse_allowed_users(
                os.getenv("DISCORD_ALLOWED_USERS", "")
            )
            
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
                is_mentioned = bool(self._client.user and self._client.user in message.mentions)
                if discord_intake.should_filter_bot_message(
                    is_bot=getattr(message.author, "bot", False),
                    policy=discord_config.get_bot_filter_policy(),
                    is_mentioned=is_mentioned,
                ):
                    return
                
                await self._handle_message(message)

            @self._client.event
            async def on_voice_state_update(member, before, after):
                """Track voice channel join/leave events."""
                # Only track channels where the bot is connected
                bot_guild_ids = set(adapter_self._voice_clients.keys())
                if not bot_guild_ids:
                    return
                guild_id = member.guild.id
                if guild_id not in bot_guild_ids:
                    return
                # Ignore the bot itself
                if member == adapter_self._client.user:
                    return

                joined = before.channel is None and after.channel is not None
                left = before.channel is not None and after.channel is None
                switched = (
                    before.channel is not None
                    and after.channel is not None
                    and before.channel != after.channel
                )

                if joined or left or switched:
                    logger.info(
                        "Voice state: %s (%d) %s (guild %d)",
                        member.display_name,
                        member.id,
                        "joined " + after.channel.name if joined
                        else "left " + before.channel.name if left
                        else f"moved {before.channel.name} -> {after.channel.name}",
                        guild_id,
                    )

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
        # Clean up all active voice connections before closing the client
        for guild_id in list(self._voice_clients.keys()):
            try:
                await self.leave_voice_channel(guild_id)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.debug("[%s] Error leaving voice channel %s: %s", self.name, guild_id, e)

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
            channel = await discord_delivery.resolve_channel(self._client, chat_id)
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

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
                chunk_reference = reference if i == 0 else None
                msg = await discord_delivery.send_text_message(
                    channel,
                    chunk,
                    reference=chunk_reference,
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
            channel = await discord_delivery.resolve_channel(self._client, chat_id)
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            msg = await channel.fetch_message(int(message_id))
            formatted = self.format_message(content)
            if len(formatted) > self.MAX_MESSAGE_LENGTH:
                formatted = formatted[:self.MAX_MESSAGE_LENGTH - 3] + "..."
            await msg.edit(content=formatted)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to edit Discord message %s: %s", self.name, message_id, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def _send_file_attachment(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Send a local file as a Discord attachment."""
        return await discord_delivery.send_file_attachment(
            self._client,
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
        )

    async def play_tts(
        self,
        chat_id: str,
        audio_path: str,
        **kwargs,
    ) -> SendResult:
        """Play auto-TTS audio.

        When the bot is in a voice channel for this chat's guild, play
        directly in the VC instead of sending as a file attachment.
        """
        for gid, text_ch_id in self._voice_text_channels.items():
            if str(text_ch_id) == str(chat_id) and self.is_in_voice_channel(gid):
                logger.info("[%s] Playing TTS in voice channel (guild=%d)", self.name, gid)
                success = await self.play_in_voice_channel(gid, audio_path)
                return SendResult(success=success)
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send audio as a Discord file attachment."""
        try:
            import io

            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

            if not os.path.exists(audio_path):
                return SendResult(success=False, error=f"Audio file not found: {audio_path}")

            filename = os.path.basename(audio_path)

            with open(audio_path, "rb") as f:
                file_data = f.read()

            # Try sending as a native voice message via raw API (flags=8192).
            try:
                import base64

                duration_secs = 5.0
                try:
                    from mutagen.oggopus import OggOpus
                    info = OggOpus(audio_path)
                    duration_secs = info.info.length
                except Exception:
                    duration_secs = max(1.0, len(file_data) / 2000.0)

                waveform_bytes = bytes([128] * 256)
                waveform_b64 = base64.b64encode(waveform_bytes).decode()

                import json as _json
                payload = _json.dumps({
                    "flags": 8192,
                    "attachments": [{
                        "id": "0",
                        "filename": "voice-message.ogg",
                        "duration_secs": round(duration_secs, 2),
                        "waveform": waveform_b64,
                    }],
                })
                form = [
                    {"name": "payload_json", "value": payload},
                    {
                        "name": "files[0]",
                        "value": file_data,
                        "filename": "voice-message.ogg",
                        "content_type": "audio/ogg",
                    },
                ]
                msg_data = await self._client.http.request(
                    discord.http.Route("POST", "/channels/{channel_id}/messages", channel_id=channel.id),
                    form=form,
                )
                return SendResult(success=True, message_id=str(msg_data["id"]))
            except Exception as voice_err:
                logger.debug("Voice message flag failed, falling back to file: %s", voice_err)
                file = discord.File(io.BytesIO(file_data), filename=filename)
                msg = await channel.send(file=file)
                return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send audio, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_voice(chat_id, audio_path, caption, reply_to, metadata=metadata)

    # ------------------------------------------------------------------
    # Voice channel methods (join / leave / play)
    # ------------------------------------------------------------------

    async def join_voice_channel(self, channel) -> bool:
        """Join a Discord voice channel. Returns True on success."""
        if not self._client or not DISCORD_AVAILABLE:
            return False
        guild_id = channel.guild.id

        # Already connected in this guild?
        existing = self._voice_clients.get(guild_id)
        if existing and existing.is_connected():
            if existing.channel.id == channel.id:
                self._reset_voice_timeout(guild_id)
                return True
            await existing.move_to(channel)
            self._reset_voice_timeout(guild_id)
            return True

        vc = await channel.connect()
        self._voice_clients[guild_id] = vc
        self._reset_voice_timeout(guild_id)

        # Start voice receiver (Phase 2: listen to users)
        try:
            receiver = VoiceReceiver(vc, allowed_user_ids=self._allowed_user_ids)
            receiver.start()
            self._voice_receivers[guild_id] = receiver
            self._voice_listen_tasks[guild_id] = asyncio.ensure_future(
                self._voice_listen_loop(guild_id)
            )
        except Exception as e:
            logger.warning("Voice receiver failed to start: %s", e)

        return True

    async def leave_voice_channel(self, guild_id: int) -> None:
        """Disconnect from the voice channel in a guild."""
        # Stop voice receiver first
        receiver = self._voice_receivers.pop(guild_id, None)
        if receiver:
            receiver.stop()
        listen_task = self._voice_listen_tasks.pop(guild_id, None)
        if listen_task:
            listen_task.cancel()

        vc = self._voice_clients.pop(guild_id, None)
        if vc and vc.is_connected():
            await vc.disconnect()
        task = self._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()
        self._voice_text_channels.pop(guild_id, None)

    # Maximum seconds to wait for voice playback before giving up
    PLAYBACK_TIMEOUT = 120

    async def play_in_voice_channel(self, guild_id: int, audio_path: str) -> bool:
        """Play an audio file in the connected voice channel."""
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return False

        # Pause voice receiver while playing (echo prevention)
        receiver = self._voice_receivers.get(guild_id)
        if receiver:
            receiver.pause()

        try:
            # Wait for current playback to finish (with timeout)
            wait_start = time.monotonic()
            while vc.is_playing():
                if time.monotonic() - wait_start > self.PLAYBACK_TIMEOUT:
                    logger.warning("Timed out waiting for previous playback to finish")
                    vc.stop()
                    break
                await asyncio.sleep(0.1)

            done = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _after(error):
                if error:
                    logger.error("Voice playback error: %s", error)
                loop.call_soon_threadsafe(done.set)

            source = discord.FFmpegPCMAudio(audio_path)
            source = discord.PCMVolumeTransformer(source, volume=1.0)
            vc.play(source, after=_after)
            try:
                await asyncio.wait_for(done.wait(), timeout=self.PLAYBACK_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Voice playback timed out after %ds", self.PLAYBACK_TIMEOUT)
                vc.stop()
            self._reset_voice_timeout(guild_id)
            return True
        finally:
            if receiver:
                receiver.resume()

    async def get_user_voice_channel(self, guild_id: int, user_id: str):
        """Return the voice channel the user is currently in, or None."""
        if not self._client:
            return None
        guild = self._client.get_guild(guild_id)
        if not guild:
            return None
        member = guild.get_member(int(user_id))
        if not member or not member.voice:
            return None
        return member.voice.channel

    def _reset_voice_timeout(self, guild_id: int) -> None:
        """Reset the auto-disconnect inactivity timer."""
        task = self._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()
        self._voice_timeout_tasks[guild_id] = asyncio.ensure_future(
            self._voice_timeout_handler(guild_id)
        )

    async def _voice_timeout_handler(self, guild_id: int) -> None:
        """Auto-disconnect after VOICE_TIMEOUT seconds of inactivity."""
        try:
            await asyncio.sleep(self.VOICE_TIMEOUT)
        except asyncio.CancelledError:
            return
        text_ch_id = self._voice_text_channels.get(guild_id)
        await self.leave_voice_channel(guild_id)
        # Notify the runner so it can clean up voice_mode state
        if self._on_voice_disconnect and text_ch_id:
            try:
                self._on_voice_disconnect(str(text_ch_id))
            except Exception:
                pass
        if text_ch_id and self._client:
            ch = self._client.get_channel(text_ch_id)
            if ch:
                try:
                    await ch.send("Left voice channel (inactivity timeout).")
                except Exception:
                    pass

    def is_in_voice_channel(self, guild_id: int) -> bool:
        """Check if the bot is connected to a voice channel in this guild."""
        vc = self._voice_clients.get(guild_id)
        return vc is not None and vc.is_connected()

    def get_voice_channel_info(self, guild_id: int) -> Optional[Dict[str, Any]]:
        """Return voice channel awareness info for the given guild.

        Returns None if the bot is not in a voice channel.  Otherwise
        returns a dict with channel name, member list, count, and
        currently-speaking user IDs (from SSRC mapping).
        """
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return None

        channel = vc.channel
        if not channel:
            return None

        # Members currently in the voice channel (includes bot)
        members_info = []
        bot_user = self._client.user if self._client else None
        for m in channel.members:
            if bot_user and m.id == bot_user.id:
                continue  # skip the bot itself
            members_info.append({
                "user_id": m.id,
                "display_name": m.display_name,
                "is_bot": m.bot,
            })

        # Currently speaking users (from SSRC mapping + active buffers)
        speaking_user_ids: set = set()
        receiver = self._voice_receivers.get(guild_id)
        if receiver:
            import time as _time
            now = _time.monotonic()
            with receiver._lock:
                for ssrc, last_t in receiver._last_packet_time.items():
                    # Consider "speaking" if audio received within last 2 seconds
                    if now - last_t < 2.0:
                        uid = receiver._ssrc_to_user.get(ssrc)
                        if uid:
                            speaking_user_ids.add(uid)

        # Tag speaking status on members
        for info in members_info:
            info["is_speaking"] = info["user_id"] in speaking_user_ids

        return {
            "channel_name": channel.name,
            "member_count": len(members_info),
            "members": members_info,
            "speaking_count": len(speaking_user_ids),
        }

    def get_voice_channel_context(self, guild_id: int) -> str:
        """Return a human-readable voice channel context string.

        Suitable for injection into the system/ephemeral prompt so the
        agent is always aware of voice channel state.
        """
        info = self.get_voice_channel_info(guild_id)
        if not info:
            return ""

        parts = [f"[Voice channel: #{info['channel_name']} — {info['member_count']} participant(s)]"]
        for m in info["members"]:
            status = " (speaking)" if m["is_speaking"] else ""
            parts.append(f"  - {m['display_name']}{status}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Voice listening (Phase 2)
    # ------------------------------------------------------------------

    # UDP keepalive interval in seconds — prevents Discord from dropping
    # the UDP route after ~60s of silence.
    _KEEPALIVE_INTERVAL = 15

    async def _voice_listen_loop(self, guild_id: int):
        """Periodically check for completed utterances and process them."""
        receiver = self._voice_receivers.get(guild_id)
        if not receiver:
            return
        last_keepalive = time.monotonic()
        try:
            while receiver._running:
                await asyncio.sleep(0.2)

                # Send periodic UDP keepalive to prevent Discord from
                # dropping the UDP session after ~60s of silence.
                now = time.monotonic()
                if now - last_keepalive >= self._KEEPALIVE_INTERVAL:
                    last_keepalive = now
                    try:
                        vc = self._voice_clients.get(guild_id)
                        if vc and vc.is_connected():
                            vc._connection.send_packet(b'\xf8\xff\xfe')
                    except Exception:
                        pass

                completed = receiver.check_silence()
                for user_id, pcm_data in completed:
                    if not self._is_allowed_user(str(user_id)):
                        continue
                    await self._process_voice_input(guild_id, user_id, pcm_data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Voice listen loop error: %s", e, exc_info=True)

    async def _process_voice_input(self, guild_id: int, user_id: int, pcm_data: bytes):
        """Convert PCM -> WAV -> STT -> callback."""
        from tools.voice_mode import is_whisper_hallucination

        tmp_f = tempfile.NamedTemporaryFile(suffix=".wav", prefix="vc_listen_", delete=False)
        wav_path = tmp_f.name
        tmp_f.close()
        try:
            await asyncio.to_thread(VoiceReceiver.pcm_to_wav, pcm_data, wav_path)

            from tools.transcription_tools import transcribe_audio, get_stt_model_from_config
            stt_model = get_stt_model_from_config()
            result = await asyncio.to_thread(transcribe_audio, wav_path, model=stt_model)

            if not result.get("success"):
                return
            transcript = result.get("transcript", "").strip()
            if not transcript or is_whisper_hallucination(transcript):
                return

            logger.info("Voice input from user %d: %s", user_id, transcript[:100])

            if self._voice_input_callback:
                await self._voice_input_callback(
                    guild_id=guild_id,
                    user_id=user_id,
                    transcript=transcript,
                )
        except Exception as e:
            logger.warning("Voice input processing failed: %s", e, exc_info=True)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _is_allowed_user(self, user_id: str) -> bool:
        """Check if user is in DISCORD_ALLOWED_USERS."""
        if not self._allowed_user_ids:
            return True
        return user_id in self._allowed_user_ids

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file natively as a Discord file attachment."""
        try:
            return await self._send_file_attachment(chat_id, image_path, caption)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Image file not found: {image_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send local image, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata)

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

            channel = await discord_delivery.resolve_channel(self._client, chat_id)
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

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local video file natively as a Discord attachment."""
        try:
            return await self._send_file_attachment(chat_id, video_path, caption)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Video file not found: {video_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send local video, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_video(chat_id, video_path, caption, reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an arbitrary file natively as a Discord attachment."""
        try:
            return await self._send_file_attachment(chat_id, file_path, caption, file_name=file_name)
        except FileNotFoundError:
            return SendResult(success=False, error=f"File not found: {file_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send document, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_document(chat_id, file_path, caption, file_name, reply_to, metadata=metadata)
    
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator."""
        if self._client:
            try:
                channel = await discord_delivery.resolve_channel(self._client, chat_id)
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

    async def _run_simple_slash(
        self,
        interaction: discord.Interaction,
        command_text: str,
        followup_msg: str | None = None,
    ) -> None:
        """Common handler for simple slash commands that dispatch a command string."""
        await interaction.response.defer(ephemeral=True)
        event = self._build_slash_event(interaction, command_text)
        await self.handle_message(event)
        if followup_msg:
            try:
                await interaction.followup.send(followup_msg, ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

    def _register_slash_commands(self) -> None:
        """Register Discord slash commands on the command tree."""
        if not self._client:
            return
        discord_interactions.register_slash_commands(self._client.tree, self)

    def _build_slash_event(self, interaction: discord.Interaction, text: str) -> MessageEvent:
        """Build a MessageEvent from a Discord slash command interaction."""
        return discord_interactions.build_slash_event(self, interaction, text)

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
        await discord_threads.handle_thread_create_slash(
            self,
            interaction,
            name,
            message,
            auto_archive_duration,
        )

    async def _dispatch_thread_session(
        self,
        interaction: discord.Interaction,
        thread_id: str,
        thread_name: str,
        text: str,
    ) -> None:
        """Build a MessageEvent pointing at a thread and send it through handle_message."""
        await discord_threads.dispatch_thread_session(
            self,
            interaction,
            thread_id,
            thread_name,
            text,
        )

    def _thread_parent_channel(self, channel: Any) -> Any:
        """Return the parent text channel when invoked from a thread."""
        return discord_threads.thread_parent_channel(channel)

    async def _resolve_interaction_channel(self, interaction: discord.Interaction) -> Optional[Any]:
        """Return the interaction channel, fetching it if the payload is partial."""
        return await discord_threads.resolve_interaction_channel(self._client, interaction)

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
        return await discord_threads.create_thread(
            self._client,
            interaction,
            name,
            message,
            auto_archive_duration,
            discord_threads.resolve_interaction_channel,
        )

    # ------------------------------------------------------------------
    # Auto-thread helpers
    # ------------------------------------------------------------------

    async def _auto_create_thread(self, message: 'DiscordMessage') -> Optional[Any]:
        """Create a thread from a user message for auto-threading.

        Returns the created thread object, or ``None`` on failure.
        """
        return await discord_threads.auto_create_thread(message)

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
            channel = await discord_delivery.resolve_channel(self._client, chat_id)
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

            # Discord embed description limit is 4096; show full command up to that
            max_desc = 4088
            cmd_display = command if len(command) <= max_desc else command[: max_desc - 3] + "..."
            embed = discord.Embed(
                title="Command Approval Required",
                description=f"```\n{cmd_display}\n```",
                color=discord.Color.orange(),
            )
            embed.set_footer(text=f"Approval ID: {approval_id}")

            view = discord_interactions.ExecApprovalView(
                approval_id=approval_id,
                allowed_user_ids=self._allowed_user_ids,
            )

            msg = await channel.send(embed=embed, view=view)
            return SendResult(success=True, message_id=str(msg.id))

        except Exception as e:
            return SendResult(success=False, error=str(e))

    def _get_parent_channel_id(self, channel: Any) -> Optional[str]:
        """Return the parent channel ID for a Discord thread-like channel, if present."""
        return discord_intake.get_parent_channel_id(channel)

    def _is_forum_parent(self, channel: Any) -> bool:
        """Best-effort check for whether a Discord channel is a forum channel."""
        return discord_intake.is_forum_parent(channel)

    def _format_thread_chat_name(self, thread: Any) -> str:
        """Build a readable chat name for thread-like Discord channels, including forum context when available."""
        return discord_intake.format_thread_chat_name(thread, self._is_forum_parent)

    # ------------------------------------------------------------------
    # Thread participation persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _thread_state_path():
        """Path to the persisted thread participation set."""
        return discord_state.thread_state_path()

    @classmethod
    def _load_participated_threads(cls) -> set:
        """Load persisted thread IDs from disk."""
        return discord_state.load_participated_threads()

    def _save_participated_threads(self) -> None:
        """Persist the current thread set to disk (best-effort)."""
        self._bot_participated_threads = discord_state.save_participated_threads(
            self._bot_participated_threads,
            max_threads=self._MAX_TRACKED_THREADS,
        )

    def _track_thread(self, thread_id: str) -> None:
        """Add a thread to the participation set and persist."""
        self._bot_participated_threads = discord_state.track_thread(
            self._bot_participated_threads,
            thread_id,
            max_threads=self._MAX_TRACKED_THREADS,
        )

    async def _handle_message(self, message: DiscordMessage) -> None:
        """Handle incoming Discord messages."""
        # In server channels (not DMs), require the bot to be @mentioned
        # UNLESS the channel is in the free-response list or the message is
        # in a thread where the bot has already participated.
        #
        # Config (all settable via discord.* in config.yaml):
        #   discord.require_mention: Require @mention in server channels (default: true)
        #   discord.free_response_channels: Channel IDs where bot responds without mention
        #   discord.auto_thread: Auto-create thread on @mention in channels (default: true)

        thread_id = None
        parent_channel_id = None
        is_thread = isinstance(message.channel, discord.Thread)
        if is_thread:
            thread_id = str(message.channel.id)
            parent_channel_id = self._get_parent_channel_id(message.channel)

        if not isinstance(message.channel, discord.DMChannel):
            free_channels = discord_config.get_free_response_channels()
            channel_ids = {str(message.channel.id)}
            if parent_channel_id:
                channel_ids.add(parent_channel_id)

            require_mention = discord_config.is_mention_required()
            is_free_channel = bool(channel_ids & free_channels)

            # Skip the mention check if the message is in a thread where
            # the bot has previously participated (auto-created or replied in).
            in_bot_thread = is_thread and thread_id in self._bot_participated_threads
            is_mentioned = bool(self._client.user and self._client.user in message.mentions)

            if discord_intake.should_skip_for_mention(
                require_mention=require_mention,
                is_free_channel=is_free_channel,
                in_bot_thread=in_bot_thread,
                is_mentioned=is_mentioned,
            ):
                return

            if is_mentioned and self._client.user:
                message.content = discord_intake.strip_mention(message.content, self._client.user.id)

        # Auto-thread: when enabled, automatically create a thread for every
        # @mention in a text channel so each conversation is isolated (like Slack).
        # Messages already inside threads or DMs are unaffected.
        auto_threaded_channel = None
        if not is_thread and not isinstance(message.channel, discord.DMChannel):
            if discord_config.is_auto_thread_enabled():
                thread = await self._auto_create_thread(message)
                if thread:
                    is_thread = True
                    thread_id = str(thread.id)
                    auto_threaded_channel = thread
                    self._track_thread(thread_id)

        # Determine message type
        msg_type = MessageType(
            discord_intake.classify_message_type(message.content, message.attachments)
        )
        
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

        # Track thread participation so the bot won't require @mention for
        # follow-up messages in threads it has already engaged in.
        if thread_id:
            self._track_thread(thread_id)

        await self.handle_message(event)

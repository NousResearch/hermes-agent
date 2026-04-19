from __future__ import annotations

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
import struct
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Any

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
import re

from gateway.platforms.helpers import MessageDeduplicator, ThreadParticipationTracker
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_image_from_url,
    cache_image_from_bytes,
    cache_audio_from_url,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    SUPPORTED_DOCUMENT_TYPES,
)
from tools.url_safety import is_safe_url


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


def _build_allowed_mentions():
    """Build Discord ``AllowedMentions`` with safe defaults, overridable via env.

    Discord bots default to parsing ``@everyone``, ``@here``, role pings, and
    user pings when ``allowed_mentions`` is unset on the client — any LLM
    output or echoed user content that contains ``@everyone`` would therefore
    ping the whole server. We explicitly deny ``@everyone`` and role pings
    by default and keep user / replied-user pings enabled so normal
    conversation still works.

    Override via environment variables (or ``discord.allow_mentions.*`` in
    config.yaml):

        DISCORD_ALLOW_MENTION_EVERYONE      default false  — @everyone + @here
        DISCORD_ALLOW_MENTION_ROLES         default false  — @role pings
        DISCORD_ALLOW_MENTION_USERS         default true   — @user pings
        DISCORD_ALLOW_MENTION_REPLIED_USER  default true   — reply-ping author
    """
    if not DISCORD_AVAILABLE:
        return None

    def _b(name: str, default: bool) -> bool:
        raw = os.getenv(name, "").strip().lower()
        if not raw:
            return default
        return raw in ("true", "1", "yes", "on")

    return discord.AllowedMentions(
        everyone=_b("DISCORD_ALLOW_MENTION_EVERYONE", False),
        roles=_b("DISCORD_ALLOW_MENTION_ROLES", False),
        users=_b("DISCORD_ALLOW_MENTION_USERS", True),
        replied_user=_b("DISCORD_ALLOW_MENTION_REPLIED_USER", True),
    )


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
        has_padding = bool(first_byte & 0x20)  # padding bit (RFC 3550 §5.1)
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

        # --- Strip RTP padding (RFC 3550 §5.1) ---
        # When the P bit is set, the last payload byte holds the count of
        # trailing padding bytes (including itself) that must be removed
        # before further processing. Skipping this passes padding-contaminated
        # bytes into DAVE/Opus and corrupts inbound audio.
        if has_padding:
            if not decrypted:
                if self._packet_debug_count <= 10:
                    logger.warning(
                        "RTP padding bit set but no payload (ssrc=%d)", ssrc,
                    )
                return
            pad_len = decrypted[-1]
            if pad_len == 0 or pad_len > len(decrypted):
                if self._packet_debug_count <= 10:
                    logger.warning(
                        "Invalid RTP padding length %d for payload size %d (ssrc=%d)",
                        pad_len, len(decrypted), ssrc,
                    )
                return
            decrypted = decrypted[:-pad_len]
            if not decrypted:
                # Padding consumed entire payload — nothing to decode
                return

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
    _SPLIT_THRESHOLD = 1900  # near the 2000-char split point

    # Auto-disconnect from voice channel after this many seconds of inactivity
    VOICE_TIMEOUT = 300

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
        self._allowed_role_ids: set = set()  # For DISCORD_ALLOWED_ROLES filtering
        # Voice channel state (per-guild)
        self._voice_clients: Dict[int, Any] = {}  # guild_id -> VoiceClient
        self._voice_locks: Dict[int, asyncio.Lock] = {}  # guild_id -> serialize join/leave
        # Text batching: merge rapid successive messages (Telegram-style)
        self._text_batch_delay_seconds = float(os.getenv("HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS", "0.6"))
        self._text_batch_split_delay_seconds = float(os.getenv("HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS", "2.0"))
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._voice_text_channels: Dict[int, int] = {}  # guild_id -> text_channel_id
        self._voice_sources: Dict[int, Dict[str, Any]] = {}  # guild_id -> linked text channel source metadata
        self._voice_timeout_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> timeout task
        # Phase 2: voice listening
        self._voice_receivers: Dict[int, VoiceReceiver] = {}  # guild_id -> VoiceReceiver
        self._voice_listen_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> listen loop
        self._voice_input_callback: Optional[Callable] = None  # set by run.py
        self._on_voice_disconnect: Optional[Callable] = None  # set by run.py
        # Track threads where the bot has participated so follow-up messages
        # in those threads don't require @mention.  Persisted to disk so the
        # set survives gateway restarts.
        self._threads = ThreadParticipationTracker("discord")
        # Persistent typing indicator loops per channel (DMs don't reliably
        # show the standard typing gateway event for bots)
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._bot_task: Optional[asyncio.Task] = None
        self._post_connect_task: Optional[asyncio.Task] = None
        # Dedup cache: prevents duplicate bot responses when Discord
        # RESUME replays events after reconnects.
        self._dedup = MessageDeduplicator()
        # Reply threading mode: "off" (no replies), "first" (reply on first
        # chunk only, default), "all" (reply-reference on every chunk).
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'

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
            if not self._acquire_platform_lock('discord-bot-token', self.config.token, 'Discord bot token'):
                return False

            # Parse allowed user entries (may contain usernames or IDs)
            allowed_env = os.getenv("DISCORD_ALLOWED_USERS", "")
            if allowed_env:
                self._allowed_user_ids = {
                    _clean_discord_id(uid) for uid in allowed_env.split(",")
                    if uid.strip()
                }

            # Parse DISCORD_ALLOWED_ROLES — comma-separated role IDs.
            # Users with ANY of these roles can interact with the bot.
            roles_env = os.getenv("DISCORD_ALLOWED_ROLES", "")
            if roles_env:
                self._allowed_role_ids = {
                    int(rid.strip()) for rid in roles_env.split(",")
                    if rid.strip().isdigit()
                }

            # Set up intents.
            # Message Content is required for normal text replies.
            # Server Members is only needed when the allowlist contains usernames
            # that must be resolved to numeric IDs. Requesting privileged intents
            # that aren't enabled in the Discord Developer Portal can prevent the
            # bot from coming online at all, so avoid requesting members intent
            # unless it is actually necessary.
            intents = Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            intents.guild_messages = True
            intents.members = (
                any(not entry.isdigit() for entry in self._allowed_user_ids)
                or bool(self._allowed_role_ids)  # Need members intent for role lookup
            )
            intents.voice_states = True

            # Resolve proxy (DISCORD_PROXY > generic env vars > macOS system proxy)
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_bot
            proxy_url = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            if proxy_url:
                logger.info("[%s] Using proxy for Discord: %s", self.name, proxy_url)

            # Create bot — proxy= for HTTP, connector= for SOCKS.
            # allowed_mentions is set with safe defaults (no @everyone/roles)
            # so LLM output or echoed user content can't ping the whole
            # server; override per DISCORD_ALLOW_MENTION_* env vars or the
            # discord.allow_mentions.* block in config.yaml.
            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
                allowed_mentions=_build_allowed_mentions(),
                **proxy_kwargs_for_bot(proxy_url),
            )
            adapter_self = self  # capture for closure

            # Register event handlers
            @self._client.event
            async def on_ready():
                logger.info("[%s] Connected as %s", adapter_self.name, adapter_self._client.user)

                # Resolve any usernames in the allowed list to numeric IDs
                await adapter_self._resolve_allowed_usernames()
                adapter_self._ready_event.set()

                if adapter_self._post_connect_task and not adapter_self._post_connect_task.done():
                    adapter_self._post_connect_task.cancel()
                adapter_self._post_connect_task = asyncio.create_task(
                    adapter_self._run_post_connect_initialization()
                )

            @self._client.event
            async def on_message(message: DiscordMessage):
                # Wait for on_ready to finish resolving username-based
                # allowlist entries.  Without this block, messages
                # arriving between Discord's READY event and the end
                # of _resolve_allowed_usernames compare author IDs
                # (numeric) against a set that may still contain raw
                # usernames (strings) from DISCORD_ALLOWED_USERS —
                # legitimate users get silently rejected for the first
                # few seconds after every reconnect.  The wait is a
                # near-instant no-op in steady state (_ready_event is
                # already set); only the startup / reconnect window
                # ever blocks.
                if not adapter_self._ready_event.is_set():
                    try:
                        await asyncio.wait_for(
                            adapter_self._ready_event.wait(),
                            timeout=30.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[%s] on_message timed out waiting for _ready_event; "
                            "allowlist check may use pre-resolved entries",
                            adapter_self.name,
                        )

                # Dedup: Discord RESUME replays events after reconnects (#4777)
                if adapter_self._dedup.is_duplicate(str(message.id)):
                    return

                # Always ignore our own messages
                if message.author == self._client.user:
                    return

                # Ignore Discord system messages (thread renames, pins, member joins, etc.)
                # Allow both default and reply types — replies have a distinct MessageType.
                if message.type not in (discord.MessageType.default, discord.MessageType.reply):
                    return

                # Bot message filtering (DISCORD_ALLOW_BOTS):
                #   "none"     — ignore all other bots (default)
                #   "mentions" — accept bot messages only when they @mention us
                #   "all"      — accept all bot messages
                # Must run BEFORE the user allowlist check so that bots
                # permitted by DISCORD_ALLOW_BOTS are not rejected for
                # not being in DISCORD_ALLOWED_USERS (fixes #4466).
                if getattr(message.author, "bot", False):
                    allow_bots = os.getenv("DISCORD_ALLOW_BOTS", "none").lower().strip()
                    if allow_bots == "none":
                        return
                    elif allow_bots == "mentions":
                        if not self._client.user or self._client.user not in message.mentions:
                            return
                    # "all" falls through; bot is permitted — skip the
                    # human-user allowlist below (bots aren't in it).
                else:
                    # Non-bot: enforce the configured user/role allowlists.
                    if not self._is_allowed_user(str(message.author.id), message.author):
                        return
                
                # Multi-agent filtering: if the message mentions specific bots
                # but NOT this bot, the sender is talking to another agent —
                # stay silent.  Messages with no bot mentions (general chat)
                # still fall through to _handle_message for the existing
                # DISCORD_REQUIRE_MENTION check.
                #
                # This replaces the older DISCORD_IGNORE_NO_MENTION logic
                # with bot-aware filtering that works correctly when multiple
                # agents share a channel.
                if not isinstance(message.channel, discord.DMChannel) and message.mentions:
                    _self_mentioned = (
                        self._client.user is not None
                        and self._client.user in message.mentions
                    )
                    _other_bots_mentioned = any(
                        m.bot and m != self._client.user
                        for m in message.mentions
                    )
                    # If other bots are mentioned but we're not → not for us
                    if _other_bots_mentioned and not _self_mentioned:
                        return
                    # If humans are mentioned but we're not → not for us
                    # (preserves old DISCORD_IGNORE_NO_MENTION=true behavior)
                    _ignore_no_mention = os.getenv(
                        "DISCORD_IGNORE_NO_MENTION", "true"
                    ).lower() in ("true", "1", "yes")
                    if _ignore_no_mention and not _self_mentioned and not _other_bots_mentioned:
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
            self._bot_task = asyncio.create_task(self._client.start(self.config.token))

            # Wait for ready
            await asyncio.wait_for(self._ready_event.wait(), timeout=30)

            self._running = True
            return True

        except asyncio.TimeoutError:
            logger.error("[%s] Timeout waiting for connection to Discord", self.name, exc_info=True)
            self._release_platform_lock()
            return False
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to connect to Discord: %s", self.name, e, exc_info=True)
            self._release_platform_lock()
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

        if self._post_connect_task and not self._post_connect_task.done():
            self._post_connect_task.cancel()
            try:
                await self._post_connect_task
            except asyncio.CancelledError:
                pass

        self._running = False
        self._client = None
        self._ready_event.clear()
        self._post_connect_task = None

        self._release_platform_lock()

        logger.info("[%s] Disconnected", self.name)

    async def _run_post_connect_initialization(self) -> None:
        """Finish non-critical startup work after Discord is connected."""
        if not self._client:
            return
        try:
            synced = await asyncio.wait_for(self._client.tree.sync(), timeout=30)
            logger.info("[%s] Synced %d slash command(s)", self.name, len(synced))
        except asyncio.TimeoutError:
            logger.warning("[%s] Slash command sync timed out after 30s", self.name)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning("[%s] Slash command sync failed: %s", self.name, e, exc_info=True)

    async def _add_reaction(self, message: Any, emoji: str) -> bool:
        """Add an emoji reaction to a Discord message."""
        if not message or not hasattr(message, "add_reaction"):
            return False
        try:
            await message.add_reaction(emoji)
            return True
        except Exception as e:
            logger.debug("[%s] add_reaction failed (%s): %s", self.name, emoji, e)
            return False

    async def _remove_reaction(self, message: Any, emoji: str) -> bool:
        """Remove the bot's own emoji reaction from a Discord message."""
        if not message or not hasattr(message, "remove_reaction") or not self._client or not self._client.user:
            return False
        try:
            await message.remove_reaction(emoji, self._client.user)
            return True
        except Exception as e:
            logger.debug("[%s] remove_reaction failed (%s): %s", self.name, emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        return os.getenv("DISCORD_REACTIONS", "true").lower() not in ("false", "0", "no")

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction for normal Discord message events."""
        if not self._reactions_enabled():
            return
        message = event.raw_message
        if hasattr(message, "add_reaction"):
            await self._add_reaction(message, "👀")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for a final success/failure reaction."""
        if not self._reactions_enabled():
            return
        message = event.raw_message
        if hasattr(message, "add_reaction"):
            await self._remove_reaction(message, "👀")
            if outcome == ProcessingOutcome.SUCCESS:
                await self._add_reaction(message, "✅")
            elif outcome == ProcessingOutcome.FAILURE:
                await self._add_reaction(message, "❌")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Discord channel or thread.

        When metadata contains a thread_id, the message is sent to that
        thread instead of the parent channel identified by chat_id.

        Forum channels (type 15) reject direct messages — a thread post is
        created automatically.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")

        try:
            # Determine target channel: thread_id in metadata takes precedence.
            thread_id = None
            if metadata and metadata.get("thread_id"):
                thread_id = metadata["thread_id"]

            if thread_id:
                # Fetch the thread directly — threads are addressed by their own ID.
                channel = self._client.get_channel(int(thread_id))
                if not channel:
                    channel = await self._client.fetch_channel(int(thread_id))
                if not channel:
                    return SendResult(success=False, error=f"Thread {thread_id} not found")
            else:
                # Get the parent channel
                channel = self._client.get_channel(int(chat_id))
                if not channel:
                    channel = await self._client.fetch_channel(int(chat_id))
                if not channel:
                    return SendResult(success=False, error=f"Channel {chat_id} not found")

            # Forum channels reject channel.send() — create a thread post instead.
            if self._is_forum_parent(channel):
                return await self._send_to_forum(channel, content)

            # Format and split message if needed
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

            message_ids = []
            reference = None

            if reply_to and self._reply_to_mode != "off":
                try:
                    ref_msg = await channel.fetch_message(int(reply_to))
                    if hasattr(ref_msg, "to_reference"):
                        reference = ref_msg.to_reference(fail_if_not_exists=False)
                    else:
                        reference = ref_msg
                except Exception as e:
                    logger.debug("Could not fetch reply-to message: %s", e)

            for i, chunk in enumerate(chunks):
                if self._reply_to_mode == "all":
                    chunk_reference = reference
                else:  # "first" (default) or "off"
                    chunk_reference = reference if i == 0 else None
                try:
                    msg = await channel.send(
                        content=chunk,
                        reference=chunk_reference,
                    )
                except Exception as e:
                    err_text = str(e)
                    if (
                        chunk_reference is not None
                        and (
                            (
                                "error code: 50035" in err_text
                                and "Cannot reply to a system message" in err_text
                            )
                            or "error code: 10008" in err_text
                        )
                    ):
                        logger.warning(
                            "[%s] Reply target %s rejected the reply reference; retrying send without reply reference",
                            self.name,
                            reply_to,
                        )
                        reference = None
                        msg = await channel.send(
                            content=chunk,
                            reference=None,
                        )
                    else:
                        raise
                message_ids.append(str(msg.id))

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send Discord message: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def _send_to_forum(self, forum_channel: Any, content: str) -> SendResult:
        """Create a thread post in a forum channel with the message as starter content.

        Forum channels (type 15) don't support direct messages.  Instead we
        POST to /channels/{forum_id}/threads with a thread name derived from
        the first line of the message.  Any follow-up chunk failures are
        reported in ``raw_response['warnings']`` so the caller can surface
        partial-send issues.
        """
        from tools.send_message_tool import _derive_forum_thread_name

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

        thread_name = _derive_forum_thread_name(content)

        starter_content = chunks[0] if chunks else thread_name

        try:
            thread = await forum_channel.create_thread(
                name=thread_name,
                content=starter_content,
            )
        except Exception as e:
            logger.error("[%s] Failed to create forum thread in %s: %s", self.name, forum_channel.id, e)
            return SendResult(success=False, error=f"Forum thread creation failed: {e}")

        thread_channel = thread if hasattr(thread, "send") else getattr(thread, "thread", None)
        thread_id = str(getattr(thread_channel, "id", getattr(thread, "id", "")))
        starter_msg = getattr(thread, "message", None)
        message_id = str(getattr(starter_msg, "id", thread_id)) if starter_msg else thread_id

        # Send remaining chunks into the newly created thread.  Track any
        # per-chunk failures so the caller sees partial-send outcomes.
        message_ids = [message_id]
        warnings: list[str] = []
        for chunk in chunks[1:]:
            try:
                msg = await thread_channel.send(content=chunk)
                message_ids.append(str(msg.id))
            except Exception as e:
                warning = f"Failed to send follow-up chunk to forum thread {thread_id}: {e}"
                logger.warning("[%s] %s", self.name, warning)
                warnings.append(warning)

        raw_response: Dict[str, Any] = {"message_ids": message_ids, "thread_id": thread_id}
        if warnings:
            raw_response["warnings"] = warnings

        return SendResult(
            success=True,
            message_id=message_ids[0],
            raw_response=raw_response,
        )

    async def _forum_post_file(
        self,
        forum_channel: Any,
        *,
        thread_name: Optional[str] = None,
        content: str = "",
        file: Any = None,
        files: Optional[list] = None,
    ) -> SendResult:
        """Create a forum thread whose starter message carries file attachments.

        Used by the send_voice / send_image_file / send_document paths when
        the target channel is a forum (type 15).  ``create_thread`` on a
        ForumChannel accepts the same file/files/content kwargs as
        ``channel.send``, creating the thread and starter message atomically.
        """
        from tools.send_message_tool import _derive_forum_thread_name

        if not thread_name:
            # Prefer the text content, fall back to the first attached
            # filename, fall back to the generic default.
            hint = content or ""
            if not hint.strip():
                if file is not None:
                    hint = getattr(file, "filename", "") or ""
                elif files:
                    hint = getattr(files[0], "filename", "") or ""
            thread_name = _derive_forum_thread_name(hint) if hint.strip() else "New Post"

        kwargs: Dict[str, Any] = {"name": thread_name}
        if content:
            kwargs["content"] = content
        if file is not None:
            kwargs["file"] = file
        if files:
            kwargs["files"] = files

        try:
            thread = await forum_channel.create_thread(**kwargs)
        except Exception as e:
            logger.error(
                "[%s] Failed to create forum thread with file in %s: %s",
                self.name,
                getattr(forum_channel, "id", "?"),
                e,
            )
            return SendResult(success=False, error=f"Forum thread creation failed: {e}")

        thread_channel = thread if hasattr(thread, "send") else getattr(thread, "thread", None)
        thread_id = str(getattr(thread_channel, "id", getattr(thread, "id", "")))
        starter_msg = getattr(thread, "message", None)
        message_id = str(getattr(starter_msg, "id", thread_id)) if starter_msg else thread_id

        return SendResult(
            success=True,
            message_id=message_id,
            raw_response={"thread_id": thread_id},
        )

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

    async def _send_file_attachment(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Send a local file as a Discord attachment.

        Forum channels (type 15) get a new thread whose starter message
        carries the file — they reject direct POST /messages.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")

        channel = self._client.get_channel(int(chat_id))
        if not channel:
            channel = await self._client.fetch_channel(int(chat_id))
        if not channel:
            return SendResult(success=False, error=f"Channel {chat_id} not found")

        filename = file_name or os.path.basename(file_path)
        with open(file_path, "rb") as fh:
            file = discord.File(fh, filename=filename)
            if self._is_forum_parent(channel):
                return await self._forum_post_file(
                    channel,
                    content=(caption or "").strip(),
                    file=file,
                )
            msg = await channel.send(content=caption if caption else None, file=file)
        return SendResult(success=True, message_id=str(msg.id))

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

            # Forum channels (type 15) reject direct POST /messages — the
            # native voice flag path also targets /messages so it would fail
            # too.  Create a thread post with the audio as the starter
            # attachment instead.
            if self._is_forum_parent(channel):
                forum_file = discord.File(io.BytesIO(file_data), filename=filename)
                return await self._forum_post_file(
                    channel,
                    content=(caption or "").strip(),
                    file=forum_file,
                )

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

    def _voice_lock_for(self, guild_id: int) -> "asyncio.Lock":
        """Return the per-guild lock, creating it on first use.

        Voice join/leave/move must be serialized per guild — without
        this, two concurrent /voice channel invocations both see
        _voice_clients.get(guild_id) return None, both call
        channel.connect(), and discord.py raises ClientException
        ('Already connected') on the loser.
        """
        lock = self._voice_locks.get(guild_id)
        if lock is None:
            lock = asyncio.Lock()
            self._voice_locks[guild_id] = lock
        return lock

    async def join_voice_channel(self, channel) -> bool:
        """Join a Discord voice channel. Returns True on success."""
        if not self._client or not DISCORD_AVAILABLE:
            return False
        guild_id = channel.guild.id

        async with self._voice_lock_for(guild_id):
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
        async with self._voice_lock_for(guild_id):
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
            self._voice_sources.pop(guild_id, None)

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

            from tools.transcription_tools import transcribe_audio
            result = await asyncio.to_thread(transcribe_audio, wav_path)

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

    def _is_allowed_user(self, user_id: str, author=None) -> bool:
        """Check if user is allowed via DISCORD_ALLOWED_USERS or DISCORD_ALLOWED_ROLES.

        Uses OR semantics: if the user matches EITHER allowlist, they're allowed.
        If both allowlists are empty, everyone is allowed (backwards compatible).
        When author is a Member, checks .roles directly; otherwise falls back
        to scanning the bot's mutual guilds for a Member record.
        """
        # ``getattr`` fallbacks here guard against test fixtures that build
        # an adapter via ``object.__new__(DiscordAdapter)`` and skip __init__
        # (see AGENTS.md pitfall #17 — same pattern as gateway.run).
        allowed_users = getattr(self, "_allowed_user_ids", set())
        allowed_roles = getattr(self, "_allowed_role_ids", set())
        has_users = bool(allowed_users)
        has_roles = bool(allowed_roles)
        if not has_users and not has_roles:
            return True
        # Check user ID allowlist
        if has_users and user_id in allowed_users:
            return True
        # Check role allowlist
        if has_roles:
            # Try direct role check from Member object
            direct_roles = getattr(author, "roles", None) if author is not None else None
            if direct_roles:
                if any(getattr(r, "id", None) in allowed_roles for r in direct_roles):
                    return True
            # Fallback: scan mutual guilds for member's roles
            if self._client is not None:
                try:
                    uid_int = int(user_id)
                except (TypeError, ValueError):
                    uid_int = None
                if uid_int is not None:
                    for guild in self._client.guilds:
                        m = guild.get_member(uid_int)
                        if m is None:
                            continue
                        m_roles = getattr(m, "roles", None) or []
                        if any(getattr(r, "id", None) in allowed_roles for r in m_roles):
                            return True
        return False

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

        if not is_safe_url(image_url):
            logger.warning("[%s] Blocked unsafe image URL during Discord send_image", self.name)
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

        try:
            import aiohttp

            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

            # Download the image and send as a Discord file attachment
            # (Discord renders attachments inline, unlike plain URLs)
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
            _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
            async with aiohttp.ClientSession(**_sess_kw) as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30), **_req_kw) as resp:
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

                    if self._is_forum_parent(channel):
                        return await self._forum_post_file(
                            channel,
                            content=(caption or "").strip(),
                            file=file,
                        )

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

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Discord file attachment."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        if not is_safe_url(animation_url):
            logger.warning("[%s] Blocked unsafe animation URL during Discord send_animation", self.name)
            return await super().send_animation(chat_id, animation_url, caption, reply_to, metadata=metadata)

        try:
            import aiohttp

            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

            # Download the GIF and send as a Discord file attachment
            # (Discord renders .gif attachments as auto-playing animations inline)
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
            _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
            async with aiohttp.ClientSession(**_sess_kw) as session:
                async with session.get(animation_url, timeout=aiohttp.ClientTimeout(total=30), **_req_kw) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download animation: HTTP {resp.status}")

                    animation_data = await resp.read()

                    import io
                    file = discord.File(io.BytesIO(animation_data), filename="animation.gif")

                    if self._is_forum_parent(channel):
                        return await self._forum_post_file(
                            channel,
                            content=(caption or "").strip(),
                            file=file,
                        )

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
            return await super().send_animation(chat_id, animation_url, caption, reply_to, metadata=metadata)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send animation attachment, falling back to URL: %s",
                self.name,
                e,
                exc_info=True,
            )
            return await super().send_animation(chat_id, animation_url, caption, reply_to, metadata=metadata)

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
        """Start a persistent typing indicator for a channel.

        Discord's TYPING_START gateway event is unreliable in DMs for bots.
        Instead, start a background loop that hits the typing endpoint every
        8 seconds (typing indicator lasts ~10s).  The loop is cancelled when
        stop_typing() is called (after the response is sent).
        """
        if not self._client:
            return
        # Don't start a duplicate loop
        if chat_id in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            try:
                while True:
                    try:
                        route = discord.http.Route(
                            "POST", "/channels/{channel_id}/typing",
                            channel_id=chat_id,
                        )
                        await self._client.http.request(route)
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        logger.debug("Discord typing indicator failed for %s: %s", chat_id, e)
                        return
                    await asyncio.sleep(8)
            except asyncio.CancelledError:
                pass

        self._typing_tasks[chat_id] = asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the persistent typing indicator for a channel."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

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
        """Common handler for simple slash commands that dispatch a command string.

        Defers the interaction (shows "thinking..."), dispatches the command,
        then cleans up the deferred response.  If *followup_msg* is provided
        the "thinking..." indicator is replaced with that text; otherwise it
        is deleted so the channel isn't cluttered.
        """
        # Log the invoker so ghost-command reports can be triaged.  Discord
        # native slash invocations are always user-initiated (no bot can fire
        # them), but mobile autocomplete / keyboard shortcuts / other users
        # in the same channel are easy to miss in post-mortems.
        try:
            _user = interaction.user
            _chan_id = getattr(interaction.channel, "id", None) or getattr(interaction, "channel_id", None)
            logger.info(
                "[Discord] slash '%s' invoked by user=%s id=%s channel=%s guild=%s",
                command_text,
                getattr(_user, "name", "?"),
                getattr(_user, "id", "?"),
                _chan_id,
                getattr(interaction, "guild_id", None),
            )
        except Exception:
            pass  # logging must never block command dispatch

        await interaction.response.defer(ephemeral=True)
        event = self._build_slash_event(interaction, command_text)
        await self.handle_message(event)
        try:
            if followup_msg:
                await interaction.edit_original_response(content=followup_msg)
            else:
                await interaction.delete_original_response()
        except Exception as e:
            logger.debug("Discord interaction cleanup failed: %s", e)

    def _register_slash_commands(self) -> None:
        """Register Discord slash commands on the command tree."""
        if not self._client:
            return

        tree = self._client.tree

        @tree.command(name="new", description="Start a new conversation")
        async def slash_new(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/reset", "New conversation started~")

        @tree.command(name="reset", description="Reset your Hermes session")
        async def slash_reset(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/reset", "Session reset~")

        @tree.command(name="model", description="Show or change the model")
        @discord.app_commands.describe(name="Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.")
        async def slash_model(interaction: discord.Interaction, name: str = ""):
            await self._run_simple_slash(interaction, f"/model {name}".strip())

        @tree.command(name="reasoning", description="Show or change reasoning effort")
        @discord.app_commands.describe(effort="Reasoning effort: none, minimal, low, medium, high, or xhigh.")
        async def slash_reasoning(interaction: discord.Interaction, effort: str = ""):
            await self._run_simple_slash(interaction, f"/reasoning {effort}".strip())

        @tree.command(name="personality", description="Set a personality")
        @discord.app_commands.describe(name="Personality name. Leave empty to list available.")
        async def slash_personality(interaction: discord.Interaction, name: str = ""):
            await self._run_simple_slash(interaction, f"/personality {name}".strip())

        @tree.command(name="retry", description="Retry your last message")
        async def slash_retry(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/retry", "Retrying~")

        @tree.command(name="undo", description="Remove the last exchange")
        async def slash_undo(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/undo")

        @tree.command(name="status", description="Show Hermes session status")
        async def slash_status(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/status", "Status sent~")

        @tree.command(name="sethome", description="Set this chat as the home channel")
        async def slash_sethome(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/sethome")

        @tree.command(name="stop", description="Stop the running Hermes agent")
        async def slash_stop(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/stop", "Stop requested~")

        @tree.command(name="steer", description="Inject a message after the next tool call (no interrupt)")
        @discord.app_commands.describe(prompt="Text to inject into the agent's next tool result")
        async def slash_steer(interaction: discord.Interaction, prompt: str):
            await self._run_simple_slash(interaction, f"/steer {prompt}".strip())

        @tree.command(name="compress", description="Compress conversation context")
        async def slash_compress(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/compress")

        @tree.command(name="title", description="Set or show the session title")
        @discord.app_commands.describe(name="Session title. Leave empty to show current.")
        async def slash_title(interaction: discord.Interaction, name: str = ""):
            await self._run_simple_slash(interaction, f"/title {name}".strip())

        @tree.command(name="resume", description="Resume a previously-named session")
        @discord.app_commands.describe(name="Session name to resume. Leave empty to list sessions.")
        async def slash_resume(interaction: discord.Interaction, name: str = ""):
            await self._run_simple_slash(interaction, f"/resume {name}".strip())

        @tree.command(name="usage", description="Show token usage for this session")
        async def slash_usage(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/usage")

        @tree.command(name="provider", description="Show available providers")
        async def slash_provider(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/provider")

        @tree.command(name="help", description="Show available commands")
        async def slash_help(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/help")

        @tree.command(name="insights", description="Show usage insights and analytics")
        @discord.app_commands.describe(days="Number of days to analyze (default: 7)")
        async def slash_insights(interaction: discord.Interaction, days: int = 7):
            await self._run_simple_slash(interaction, f"/insights {days}")

        @tree.command(name="reload-mcp", description="Reload MCP servers from config")
        async def slash_reload_mcp(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/reload-mcp")

        @tree.command(name="voice", description="Toggle voice reply mode")
        @discord.app_commands.describe(mode="Voice mode: on, off, tts, channel, leave, or status")
        @discord.app_commands.choices(mode=[
            discord.app_commands.Choice(name="channel — join your voice channel", value="channel"),
            discord.app_commands.Choice(name="leave — leave voice channel", value="leave"),
            discord.app_commands.Choice(name="on — voice reply to voice messages", value="on"),
            discord.app_commands.Choice(name="tts — voice reply to all messages", value="tts"),
            discord.app_commands.Choice(name="off — text only", value="off"),
            discord.app_commands.Choice(name="status — show current mode", value="status"),
        ])
        async def slash_voice(interaction: discord.Interaction, mode: str = ""):
            await self._run_simple_slash(interaction, f"/voice {mode}".strip())

        @tree.command(name="update", description="Update Hermes Agent to the latest version")
        async def slash_update(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/update", "Update initiated~")

        @tree.command(name="restart", description="Gracefully restart the Hermes gateway")
        async def slash_restart(interaction: discord.Interaction):
            await self._run_simple_slash(interaction, "/restart", "Restart requested~")

        @tree.command(name="approve", description="Approve a pending dangerous command")
        @discord.app_commands.describe(scope="Optional: 'all', 'session', 'always', 'all session', 'all always'")
        async def slash_approve(interaction: discord.Interaction, scope: str = ""):
            await self._run_simple_slash(interaction, f"/approve {scope}".strip())

        @tree.command(name="deny", description="Deny a pending dangerous command")
        @discord.app_commands.describe(scope="Optional: 'all' to deny all pending commands")
        async def slash_deny(interaction: discord.Interaction, scope: str = ""):
            await self._run_simple_slash(interaction, f"/deny {scope}".strip())

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

        @tree.command(name="queue", description="Queue a prompt for the next turn (doesn't interrupt)")
        @discord.app_commands.describe(prompt="The prompt to queue")
        async def slash_queue(interaction: discord.Interaction, prompt: str):
            await self._run_simple_slash(interaction, f"/queue {prompt}", "Queued for the next turn.")

        @tree.command(name="background", description="Run a prompt in the background")
        @discord.app_commands.describe(prompt="The prompt to run in the background")
        async def slash_background(interaction: discord.Interaction, prompt: str):
            await self._run_simple_slash(interaction, f"/background {prompt}", "Background task started~")

        @tree.command(name="btw", description="Ephemeral side question using session context")
        @discord.app_commands.describe(question="Your side question (no tools, not persisted)")
        async def slash_btw(interaction: discord.Interaction, question: str):
            await self._run_simple_slash(interaction, f"/btw {question}")

        # ── Auto-register any gateway-available commands not yet on the tree ──
        # This ensures new commands added to COMMAND_REGISTRY in
        # hermes_cli/commands.py automatically appear as Discord slash
        # commands without needing a manual entry here.
        try:
            from hermes_cli.commands import COMMAND_REGISTRY, _is_gateway_available, _resolve_config_gates

            already_registered = set()
            try:
                already_registered = {cmd.name for cmd in tree.get_commands()}
            except Exception:
                pass

            config_overrides = _resolve_config_gates()

            for cmd_def in COMMAND_REGISTRY:
                if not _is_gateway_available(cmd_def, config_overrides):
                    continue
                # Discord command names: lowercase, hyphens OK, max 32 chars.
                discord_name = cmd_def.name.lower()[:32]
                if discord_name in already_registered:
                    continue
                # Skip aliases that overlap with already-registered names
                # (aliases for explicitly registered commands are handled above).
                desc = (cmd_def.description or f"Run /{cmd_def.name}")[:100]
                has_args = bool(cmd_def.args_hint)

                if has_args:
                    # Command takes optional arguments — create handler with
                    # an optional ``args`` string parameter.
                    def _make_args_handler(_name: str, _hint: str):
                        @discord.app_commands.describe(args=f"Arguments: {_hint}"[:100])
                        async def _handler(interaction: discord.Interaction, args: str = ""):
                            await self._run_simple_slash(
                                interaction, f"/{_name} {args}".strip()
                            )
                        _handler.__name__ = f"auto_slash_{_name.replace('-', '_')}"
                        return _handler

                    handler = _make_args_handler(cmd_def.name, cmd_def.args_hint)
                else:
                    # Parameterless command.
                    def _make_simple_handler(_name: str):
                        async def _handler(interaction: discord.Interaction):
                            await self._run_simple_slash(interaction, f"/{_name}")
                        _handler.__name__ = f"auto_slash_{_name.replace('-', '_')}"
                        return _handler

                    handler = _make_simple_handler(cmd_def.name)

                auto_cmd = discord.app_commands.Command(
                    name=discord_name,
                    description=desc,
                    callback=handler,
                )
                try:
                    tree.add_command(auto_cmd)
                    already_registered.add(discord_name)
                except Exception:
                    # Silently skip commands that fail registration (e.g.
                    # name conflict with a subcommand group).
                    pass

            logger.debug(
                "Discord auto-registered %d commands from COMMAND_REGISTRY",
                len(already_registered),
            )
        except Exception as e:
            logger.warning("Discord auto-register from COMMAND_REGISTRY failed: %s", e)

        # Register skills under a single /skill command group with category
        # subcommand groups.  This uses 1 top-level slot instead of N,
        # supporting up to 25 categories × 25 skills = 625 skills.
        self._register_skill_group(tree)

    def _register_skill_group(self, tree) -> None:
        """Register a single ``/skill`` command with autocomplete on the name.

        Discord enforces an ~8000-byte per-command payload limit. The older
        nested layout (``/skill <category> <name>``) registered one giant
        command whose serialized payload grew linearly with the skill
        catalog — with the default ~75 skills the payload was ~14 KB and
        ``tree.sync()`` rejected the entire slash-command batch (issues
        #11321, #10259, #11385, #10261, #10214).

        Autocomplete options are fetched dynamically by Discord when the
        user types — they do NOT count against the per-command registration
        budget. So we register ONE flat ``/skill`` command with
        ``name: str`` (autocompleted) and ``args: str = ""``. This scales
        to thousands of skills with no size math, no splitting, and no
        hidden skills. The slash picker also becomes more discoverable —
        Discord live-filters by the user's typed prefix against both the
        skill name and its description.
        """
        try:
            from hermes_cli.commands import discord_skill_commands_by_category

            existing_names = set()
            try:
                existing_names = {cmd.name for cmd in tree.get_commands()}
            except Exception:
                pass

            # Reuse the existing collector for consistent filtering
            # (per-platform disabled, hub-excluded, name clamping), then
            # flatten — the category grouping was only useful for the
            # nested layout.
            categories, uncategorized, hidden = discord_skill_commands_by_category(
                reserved_names=existing_names,
            )
            entries: list[tuple[str, str, str]] = list(uncategorized)
            for cat_skills in categories.values():
                entries.extend(cat_skills)

            if not entries:
                return

            # Stable alphabetical order so the autocomplete suggestion
            # list is predictable across restarts.
            entries.sort(key=lambda t: t[0])

            # name -> (description, cmd_key) — used by both the autocomplete
            # callback and the handler for O(1) dispatch.
            skill_lookup: dict[str, tuple[str, str]] = {
                n: (d, k) for n, d, k in entries
            }

            async def _autocomplete_name(
                interaction: "discord.Interaction", current: str,
            ) -> list:
                """Filter skills by the user's typed prefix.

                Matches both the skill name and its description so
                "/skill pdf" surfaces skills whose description mentions
                PDFs even if the name doesn't. Discord caps this list at
                25 entries per query.
                """
                q = (current or "").strip().lower()
                choices: list = []
                for name, desc, _key in entries:
                    if not q or q in name.lower() or (desc and q in desc.lower()):
                        if desc:
                            label = f"{name} — {desc}"
                        else:
                            label = name
                        # Discord's Choice.name is capped at 100 chars.
                        if len(label) > 100:
                            label = label[:97] + "..."
                        choices.append(
                            discord.app_commands.Choice(name=label, value=name)
                        )
                        if len(choices) >= 25:
                            break
                return choices

            @discord.app_commands.describe(
                name="Which skill to run",
                args="Optional arguments for the skill",
            )
            @discord.app_commands.autocomplete(name=_autocomplete_name)
            async def _skill_handler(
                interaction: "discord.Interaction", name: str, args: str = "",
            ):
                entry = skill_lookup.get(name)
                if not entry:
                    await interaction.response.send_message(
                        f"Unknown skill: `{name}`. Start typing for "
                        f"autocomplete suggestions.",
                        ephemeral=True,
                    )
                    return
                _desc, cmd_key = entry
                await self._run_simple_slash(
                    interaction, f"{cmd_key} {args}".strip()
                )

            cmd = discord.app_commands.Command(
                name="skill",
                description="Run a Hermes skill",
                callback=_skill_handler,
            )
            tree.add_command(cmd)

            logger.info(
                "[%s] Registered /skill command with %d skill(s) via autocomplete",
                self.name, len(entries),
            )
            if hidden:
                logger.info(
                    "[%s] %d skill(s) filtered out of /skill (name clamp / reserved)",
                    self.name, hidden,
                )
        except Exception as exc:
            logger.warning("[%s] Failed to register /skill command: %s", self.name, exc)

    def _build_slash_event(self, interaction: discord.Interaction, text: str) -> MessageEvent:
        """Build a MessageEvent from a Discord slash command interaction."""
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        is_thread = isinstance(interaction.channel, discord.Thread)
        thread_id = None

        if is_dm:
            chat_type = "dm"
        elif is_thread:
            chat_type = "thread"
            thread_id = str(interaction.channel_id)
        else:
            chat_type = "group"

        chat_name = ""
        if not is_dm and hasattr(interaction.channel, "name"):
            chat_name = interaction.channel.name
            if hasattr(interaction.channel, "guild") and interaction.channel.guild:
                chat_name = f"{interaction.channel.guild.name} / #{chat_name}"

        # Get channel topic (if available).
        # For forum threads, inherit the parent forum's topic.
        chat_topic = self._get_effective_topic(interaction.channel, is_thread=is_thread)

        source = self.build_source(
            chat_id=str(interaction.channel_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(interaction.user.id),
            user_name=interaction.user.display_name,
            thread_id=thread_id,
            chat_topic=chat_topic,
        )

        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        channel_id = str(interaction.channel_id)
        parent_id = str(getattr(getattr(interaction, "channel", None), "parent_id", "") or "")
        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=interaction,
            channel_prompt=self._resolve_channel_prompt(channel_id, parent_id or None),
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

        # Track thread participation so follow-ups don't require @mention
        if thread_id:
            self._threads.mark(thread_id)

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

        # Inherit forum topic when the thread was created inside a forum channel.
        _chan = getattr(interaction, "channel", None)
        chat_topic = self._get_effective_topic(_chan, is_thread=True) if _chan else None

        source = self.build_source(
            chat_id=thread_id,
            chat_name=chat_name,
            chat_type="thread",
            user_id=str(interaction.user.id),
            user_name=interaction.user.display_name,
            thread_id=thread_id,
            chat_topic=chat_topic,
        )

        _parent_channel = self._thread_parent_channel(getattr(interaction, "channel", None))
        _parent_id = str(getattr(_parent_channel, "id", "") or "")
        _skills = self._resolve_channel_skills(thread_id, _parent_id or None)
        _channel_prompt = self._resolve_channel_prompt(thread_id, _parent_id or None)
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=interaction,
            auto_skill=_skills,
            channel_prompt=_channel_prompt,
        )
        await self.handle_message(event)

    def _resolve_channel_skills(self, channel_id: str, parent_id: str | None = None) -> list[str] | None:
        """Look up auto-skill bindings for a Discord channel/forum thread.

        Config format (in platform extra):
            channel_skill_bindings:
              - id: "123456"
                skills: ["skill-a", "skill-b"]
        Also checks parent_id so forum threads inherit the forum's bindings.
        """
        bindings = self.config.extra.get("channel_skill_bindings", [])
        if not bindings:
            return None
        ids_to_check = {channel_id}
        if parent_id:
            ids_to_check.add(parent_id)
        for entry in bindings:
            entry_id = str(entry.get("id", ""))
            if entry_id in ids_to_check:
                skills = entry.get("skills") or entry.get("skill")
                if isinstance(skills, str):
                    return [skills]
                if isinstance(skills, list) and skills:
                    return list(dict.fromkeys(skills))  # dedup, preserve order
        return None

    def _resolve_channel_prompt(self, channel_id: str, parent_id: str | None = None) -> str | None:
        """Resolve a Discord per-channel prompt, preferring the exact channel over its parent."""
        from gateway.platforms.base import resolve_channel_prompt
        return resolve_channel_prompt(self.config.extra, channel_id, parent_id)

    def _discord_require_mention(self) -> bool:
        """Return whether Discord channel messages require a bot mention."""
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in ("false", "0", "no", "off")
            return bool(configured)
        return os.getenv("DISCORD_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no", "off")

    def _discord_free_response_channels(self) -> set:
        """Return Discord channel IDs where no bot mention is required."""
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        if isinstance(raw, str) and raw.strip():
            return {part.strip() for part in raw.split(",") if part.strip()}
        return set()

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
        # Build a short thread name from the message. Strip Discord mention
        # syntax (users / roles / channels) so thread titles don't end up
        # showing raw <@id>, <@&id>, or <#id> markers — the ID isn't
        # meaningful to humans glancing at the thread list (#6336).
        content = (message.content or "").strip()
        # <@123>, <@!123>, <@&123>, <#123> — collapse to empty; normalize spaces.
        content = re.sub(r"<@[!&]?\d+>", "", content)
        content = re.sub(r"<#\d+>", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        thread_name = content[:80] if content else "Hermes"
        if len(content) > 80:
            thread_name = thread_name[:77] + "..."

        try:
            thread = await message.create_thread(name=thread_name, auto_archive_duration=1440)
            return thread
        except Exception as direct_error:
            display_name = getattr(getattr(message, "author", None), "display_name", None) or "unknown user"
            reason = f"Auto-threaded from mention by {display_name}"
            try:
                seed_msg = await message.channel.send(f"\U0001f9f5 Thread created by Hermes: **{thread_name}**")
                thread = await seed_msg.create_thread(
                    name=thread_name,
                    auto_archive_duration=1440,
                    reason=reason,
                )
                return thread
            except Exception as fallback_error:
                logger.warning(
                    "[%s] Auto-thread creation failed. Direct error: %s. Fallback error: %s",
                    self.name,
                    direct_error,
                    fallback_error,
                )
                return None

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str,
        description: str = "dangerous command",
        metadata: Optional[dict] = None,
    ) -> SendResult:
        """
        Send a button-based exec approval prompt for a dangerous command.

        The buttons call ``resolve_gateway_approval()`` to unblock the waiting
        agent thread — this replaces the text-based ``/approve`` flow on Discord.
        """
        if not self._client or not DISCORD_AVAILABLE:
            return SendResult(success=False, error="Not connected")

        try:
            # Resolve channel — use thread_id from metadata if present
            target_id = chat_id
            if metadata and metadata.get("thread_id"):
                target_id = metadata["thread_id"]

            channel = self._client.get_channel(int(target_id))
            if not channel:
                channel = await self._client.fetch_channel(int(target_id))

            # Discord embed description limit is 4096; show full command up to that
            max_desc = 4088
            cmd_display = command if len(command) <= max_desc else command[: max_desc - 3] + "..."
            embed = discord.Embed(
                title="⚠️ Command Approval Required",
                description=f"```\n{cmd_display}\n```",
                color=discord.Color.orange(),
            )
            embed.add_field(name="Reason", value=description, inline=False)

            view = ExecApprovalView(
                session_key=session_key,
                allowed_user_ids=self._allowed_user_ids,
            )

            msg = await channel.send(embed=embed, view=view)
            return SendResult(success=True, message_id=str(msg.id))

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "",
        session_key: str = "",
    ) -> SendResult:
        """Send an interactive button-based update prompt (Yes / No).

        Used by the gateway ``/update`` watcher when ``hermes update --gateway``
        needs user input (stash restore, config migration).
        """
        if not self._client or not DISCORD_AVAILABLE:
            return SendResult(success=False, error="Not connected")
        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))

            default_hint = f" (default: {default})" if default else ""
            embed = discord.Embed(
                title="⚕ Update Needs Your Input",
                description=f"{prompt}{default_hint}",
                color=discord.Color.gold(),
            )
            view = UpdatePromptView(
                session_key=session_key,
                allowed_user_ids=self._allowed_user_ids,
            )
            msg = await channel.send(embed=embed, view=view)
            return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an interactive select-menu model picker.

        Two-step drill-down: provider dropdown → model dropdown.
        Uses Discord embeds + Select menus via ``ModelPickerView``.
        """
        if not self._client or not DISCORD_AVAILABLE:
            return SendResult(success=False, error="Not connected")

        try:
            # Resolve target channel (use thread_id if present)
            target_id = chat_id
            if metadata and metadata.get("thread_id"):
                target_id = metadata["thread_id"]

            channel = self._client.get_channel(int(target_id))
            if not channel:
                channel = await self._client.fetch_channel(int(target_id))

            try:
                from hermes_cli.providers import get_label
                provider_label = get_label(current_provider)
            except Exception:
                provider_label = current_provider

            embed = discord.Embed(
                title="⚙ Model Configuration",
                description=(
                    f"Current model: `{current_model or 'unknown'}`\n"
                    f"Provider: {provider_label}\n\n"
                    f"Select a provider:"
                ),
                color=discord.Color.blue(),
            )

            view = ModelPickerView(
                providers=providers,
                current_model=current_model,
                current_provider=current_provider,
                session_key=session_key,
                on_model_selected=on_model_selected,
                allowed_user_ids=self._allowed_user_ids,
            )

            msg = await channel.send(embed=embed, view=view)
            return SendResult(success=True, message_id=str(msg.id))

        except Exception as e:
            logger.warning("[%s] send_model_picker failed: %s", self.name, e)
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

    def _get_effective_topic(self, channel: Any, is_thread: bool = False) -> Optional[str]:
        """Return the channel topic, falling back to the parent forum's topic for forum threads."""
        topic = getattr(channel, "topic", None)
        if not topic and is_thread:
            parent = getattr(channel, "parent", None)
            if parent and self._is_forum_parent(parent):
                topic = getattr(parent, "topic", None)
        return topic

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

    # ------------------------------------------------------------------
    # Attachment download helpers
    #
    # Discord attachments (images / audio / documents) are fetched via the
    # authenticated bot session whenever the Attachment object exposes
    # ``read()``. That sidesteps two classes of bug that hit the older
    # plain-HTTP path:
    #
    #   1. ``cdn.discordapp.com`` URLs increasingly require bot auth on
    #      download — unauthenticated httpx sees 403 Forbidden.
    #      (issue #8242)
    #   2. Some user environments (VPNs, corporate DNS, tunnels) resolve
    #      ``cdn.discordapp.com`` to private-looking IPs that our
    #      ``is_safe_url`` guard classifies as SSRF risks. Routing the
    #      fetch through discord.py's own HTTP client handles DNS
    #      internally so our guard isn't consulted for the attachment
    #      path. (issue #6587)
    #
    # If ``att.read()`` is unavailable (unexpected object shape / test
    # stub) or the bot session fetch fails, we fall back to the existing
    # SSRF-gated URL downloaders. The fallback keeps defense-in-depth
    # against any future Discord payload-schema drift that could slip a
    # non-CDN URL into the ``att.url`` field. (issue #11345)
    # ------------------------------------------------------------------

    async def _read_attachment_bytes(self, att) -> Optional[bytes]:
        """Read an attachment via discord.py's authenticated bot session.

        Returns the raw bytes on success, or ``None`` if ``att`` doesn't
        expose a callable ``read()`` or the read itself fails. Callers
        should treat ``None`` as a signal to fall back to the URL-based
        downloaders.
        """
        reader = getattr(att, "read", None)
        if reader is None or not callable(reader):
            return None
        try:
            return await reader()
        except Exception as e:
            logger.warning(
                "[Discord] Authenticated attachment read failed for %s: %s",
                getattr(att, "filename", None) or getattr(att, "url", "<unknown>"),
                e,
            )
            return None

    async def _cache_discord_image(self, att, ext: str) -> str:
        """Cache a Discord image attachment to local disk.

        Primary path: ``att.read()`` + ``cache_image_from_bytes``
        (authenticated, no SSRF gate).

        Fallback: ``cache_image_from_url`` (plain httpx, SSRF-gated).
        """
        raw_bytes = await self._read_attachment_bytes(att)
        if raw_bytes is not None:
            try:
                return cache_image_from_bytes(raw_bytes, ext=ext)
            except Exception as e:
                logger.debug(
                    "[Discord] cache_image_from_bytes rejected att.read() data; falling back to URL: %s",
                    e,
                )
        return await cache_image_from_url(att.url, ext=ext)

    async def _cache_discord_audio(self, att, ext: str) -> str:
        """Cache a Discord audio attachment to local disk.

        Primary path: ``att.read()`` + ``cache_audio_from_bytes``
        (authenticated, no SSRF gate).

        Fallback: ``cache_audio_from_url`` (plain httpx, SSRF-gated).
        """
        raw_bytes = await self._read_attachment_bytes(att)
        if raw_bytes is not None:
            try:
                return cache_audio_from_bytes(raw_bytes, ext=ext)
            except Exception as e:
                logger.debug(
                    "[Discord] cache_audio_from_bytes failed; falling back to URL: %s",
                    e,
                )
        return await cache_audio_from_url(att.url, ext=ext)

    async def _cache_discord_document(self, att, ext: str) -> bytes:
        """Download a Discord document attachment and return the raw bytes.

        Primary path: ``att.read()`` (authenticated, no SSRF gate).

        Fallback: SSRF-gated ``aiohttp`` download. This closes the gap
        where the old document path made raw ``aiohttp.ClientSession``
        requests with no safety check (#11345). The caller is responsible
        for passing the returned bytes to ``cache_document_from_bytes``
        (and, where applicable, for injecting text content).
        """
        raw_bytes = await self._read_attachment_bytes(att)
        if raw_bytes is not None:
            return raw_bytes

        # Fallback: SSRF-gated URL download.
        if not is_safe_url(att.url):
            raise ValueError(
                f"Blocked unsafe attachment URL (SSRF protection): {att.url}"
            )
        import aiohttp
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        async with aiohttp.ClientSession(**_sess_kw) as session:
            async with session.get(
                att.url,
                timeout=aiohttp.ClientTimeout(total=30),
                **_req_kw,
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                return await resp.read()

    async def _handle_message(self, message: DiscordMessage) -> None:
        """Handle incoming Discord messages."""
        # In server channels (not DMs), require the bot to be @mentioned
        # UNLESS the channel is in the free-response list or the message is
        # in a thread where the bot has already participated.
        #
        # Config (all settable via discord.* in config.yaml or DISCORD_* env vars):
        #   discord.require_mention: Require @mention in server channels (default: true)
        #   discord.free_response_channels: Channel IDs where bot responds without mention
        #   discord.ignored_channels: Channel IDs where bot NEVER responds (even when mentioned)
        #   discord.allowed_channels: If set, bot ONLY responds in these channels (whitelist)
        #   discord.no_thread_channels: Channel IDs where bot responds directly without creating thread
        #   discord.auto_thread: Auto-create thread on @mention in channels (default: true)

        thread_id = None
        parent_channel_id = None
        is_thread = isinstance(message.channel, discord.Thread)
        if is_thread:
            thread_id = str(message.channel.id)
            parent_channel_id = self._get_parent_channel_id(message.channel)

        is_voice_linked_channel = False
        if not isinstance(message.channel, discord.DMChannel):
            channel_ids = {str(message.channel.id)}
            if parent_channel_id:
                channel_ids.add(parent_channel_id)

            # Check allowed channels - if set, only respond in these channels
            allowed_channels_raw = os.getenv("DISCORD_ALLOWED_CHANNELS", "")
            if allowed_channels_raw:
                allowed_channels = {ch.strip() for ch in allowed_channels_raw.split(",") if ch.strip()}
                if not (channel_ids & allowed_channels):
                    logger.debug("[%s] Ignoring message in non-allowed channel: %s", self.name, channel_ids)
                    return

            # Check ignored channels - never respond even when mentioned
            ignored_channels_raw = os.getenv("DISCORD_IGNORED_CHANNELS", "")
            ignored_channels = {ch.strip() for ch in ignored_channels_raw.split(",") if ch.strip()}
            if channel_ids & ignored_channels:
                logger.debug("[%s] Ignoring message in ignored channel: %s", self.name, channel_ids)
                return

            free_channels = self._discord_free_response_channels()
            if parent_channel_id:
                channel_ids.add(parent_channel_id)

            require_mention = self._discord_require_mention()
            # Voice-linked text channels act as free-response while voice is active.
            # Only the exact bound channel gets the exemption, not sibling threads.
            voice_linked_ids = {str(ch_id) for ch_id in self._voice_text_channels.values()}
            current_channel_id = str(message.channel.id)
            is_voice_linked_channel = current_channel_id in voice_linked_ids
            is_free_channel = bool(channel_ids & free_channels) or is_voice_linked_channel

            # Skip the mention check if the message is in a thread where
            # the bot has previously participated (auto-created or replied in).
            in_bot_thread = is_thread and thread_id in self._threads

            if require_mention and not is_free_channel and not in_bot_thread:
                if self._client.user not in message.mentions:
                    return

            if self._client.user and self._client.user in message.mentions:
                message.content = message.content.replace(f"<@{self._client.user.id}>", "").strip()
                message.content = message.content.replace(f"<@!{self._client.user.id}>", "").strip()

        # Auto-thread: when enabled, automatically create a thread for every
        # @mention in a text channel so each conversation is isolated (like Slack).
        # Messages already inside threads or DMs are unaffected.
        # no_thread_channels: channels where bot responds directly without thread.
        auto_threaded_channel = None
        if not is_thread and not isinstance(message.channel, discord.DMChannel):
            no_thread_channels_raw = os.getenv("DISCORD_NO_THREAD_CHANNELS", "")
            no_thread_channels = {ch.strip() for ch in no_thread_channels_raw.split(",") if ch.strip()}
            skip_thread = bool(channel_ids & no_thread_channels) or is_free_channel
            auto_thread = os.getenv("DISCORD_AUTO_THREAD", "true").lower() in ("true", "1", "yes")
            is_reply_message = getattr(message, "type", None) == discord.MessageType.reply
            if auto_thread and not skip_thread and not is_voice_linked_channel and not is_reply_message:
                thread = await self._auto_create_thread(message)
                if thread:
                    is_thread = True
                    thread_id = str(thread.id)
                    auto_threaded_channel = thread
                    self._threads.mark(thread_id)

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
                        doc_ext = ""
                        if att.filename:
                            _, doc_ext = os.path.splitext(att.filename)
                            doc_ext = doc_ext.lower()
                        if doc_ext in SUPPORTED_DOCUMENT_TYPES:
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

        # Get channel topic (if available - TextChannels have topics, DMs/threads don't).
        # For threads whose parent is a forum channel, inherit the parent's topic
        # so forum descriptions (e.g. project instructions) appear in the session context.
        chat_topic = self._get_effective_topic(message.channel, is_thread=is_thread)

        # Build source
        source = self.build_source(
            chat_id=str(effective_channel.id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(message.author.id),
            user_name=message.author.display_name,
            thread_id=thread_id,
            chat_topic=chat_topic,
            is_bot=getattr(message.author, "bot", False),
        )

        # Build media URLs -- download image attachments to local cache so the
        # vision tool can access them reliably (Discord CDN URLs can expire).
        media_urls = []
        media_types = []
        pending_text_injection: Optional[str] = None
        for att in message.attachments:
            content_type = att.content_type or "unknown"
            if content_type.startswith("image/"):
                try:
                    # Determine extension from content type (image/png -> .png)
                    ext = "." + content_type.split("/")[-1].split(";")[0]
                    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                        ext = ".jpg"
                    cached_path = await self._cache_discord_image(att, ext)
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
                    cached_path = await self._cache_discord_audio(att, ext)
                    media_urls.append(cached_path)
                    media_types.append(content_type)
                    print(f"[Discord] Cached user audio: {cached_path}", flush=True)
                except Exception as e:
                    print(f"[Discord] Failed to cache audio attachment: {e}", flush=True)
                    media_urls.append(att.url)
                    media_types.append(content_type)
            else:
                # Document attachments: download, cache, and optionally inject text
                ext = ""
                if att.filename:
                    _, ext = os.path.splitext(att.filename)
                    ext = ext.lower()
                if not ext and content_type:
                    mime_to_ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
                    ext = mime_to_ext.get(content_type, "")
                if ext not in SUPPORTED_DOCUMENT_TYPES:
                    logger.warning(
                        "[Discord] Unsupported document type '%s' (%s), skipping",
                        ext or "unknown", content_type,
                    )
                else:
                    MAX_DOC_BYTES = 32 * 1024 * 1024
                    if att.size and att.size > MAX_DOC_BYTES:
                        logger.warning(
                            "[Discord] Document too large (%s bytes), skipping: %s",
                            att.size, att.filename,
                        )
                    else:
                        try:
                            raw_bytes = await self._cache_discord_document(att, ext)
                            cached_path = cache_document_from_bytes(
                                raw_bytes, att.filename or f"document{ext}"
                            )
                            doc_mime = SUPPORTED_DOCUMENT_TYPES[ext]
                            media_urls.append(cached_path)
                            media_types.append(doc_mime)
                            logger.info("[Discord] Cached user document: %s", cached_path)
                            # Inject text content for plain-text documents (capped at 100 KB)
                            MAX_TEXT_INJECT_BYTES = 100 * 1024
                            if ext in (".md", ".txt", ".log") and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                                try:
                                    text_content = raw_bytes.decode("utf-8")
                                    display_name = att.filename or f"document{ext}"
                                    display_name = re.sub(r'[^\w.\- ]', '_', display_name)
                                    injection = f"[Content of {display_name}]:\n{text_content}"
                                    if pending_text_injection:
                                        pending_text_injection = f"{pending_text_injection}\n\n{injection}"
                                    else:
                                        pending_text_injection = injection
                                except UnicodeDecodeError:
                                    pass
                        except Exception as e:
                            logger.warning(
                                "[Discord] Failed to cache document %s: %s",
                                att.filename, e, exc_info=True,
                            )

        event_text = message.content
        if pending_text_injection:
            event_text = f"{pending_text_injection}\n\n{event_text}" if event_text else pending_text_injection

        # Defense-in-depth: prevent empty user messages from entering session
        # (can happen when user sends @mention-only with no other text)
        if not event_text or not event_text.strip():
            event_text = "(The user sent a message with no text content)"

        _chan = message.channel
        _parent_id = str(getattr(_chan, "parent_id", "") or "")
        _chan_id = str(getattr(_chan, "id", ""))
        _skills = self._resolve_channel_skills(_chan_id, _parent_id or None)
        _channel_prompt = self._resolve_channel_prompt(_chan_id, _parent_id or None)

        reply_to_id = None
        reply_to_text = None
        if message.reference:
            reply_to_id = str(message.reference.message_id)
            if message.reference.resolved:
                reply_to_text = getattr(message.reference.resolved, "content", None) or None

        event = MessageEvent(
            text=event_text,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.id),
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            timestamp=message.created_at,
            auto_skill=_skills,
            channel_prompt=_channel_prompt,
        )

        # Track thread participation so the bot won't require @mention for
        # follow-up messages in threads it has already engaged in.
        if thread_id:
            self._threads.mark(thread_id)

        # Only batch plain text messages — commands, media, etc. dispatch
        # immediately since they won't be split by the Discord client.
        if msg_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(event)
        else:
            await self.handle_message(event)

    # ------------------------------------------------------------------
    # Text message aggregation (handles Discord client-side splits)
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        from gateway.session import build_session_key
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer.

        When Discord splits a long user message at 2000 chars, the chunks
        arrive within a few hundred milliseconds.  This merges them into
        a single event before dispatching.
        """
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)

        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text.

        Uses a longer delay when the latest chunk is near Discord's 2000-char
        split point, since a continuation chunk is almost certain.
        """
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info(
                "[Discord] Flushing text batch %s (%d chars)",
                key, len(event.text or ""),
            )
            # Shield the downstream dispatch so that a subsequent chunk
            # arriving while handle_message is mid-flight cannot cancel
            # the running agent turn.  _enqueue_text_event always cancels
            # the prior flush task when a new chunk lands; without this
            # shield, CancelledError would propagate from our task down
            # into handle_message → the agent's streaming request,
            # aborting the response the user was waiting on.  The new
            # chunk is handled by the fresh flush task regardless.
            await asyncio.shield(self.handle_message(event))
        except asyncio.CancelledError:
            # Only reached if cancel landed before the pop — the shielded
            # handle_message is unaffected either way.  Let the task exit
            # cleanly so the finally block cleans up.
            pass
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)


# ---------------------------------------------------------------------------
# Discord UI Components (outside the adapter class)
# ---------------------------------------------------------------------------

if DISCORD_AVAILABLE:

    class ExecApprovalView(discord.ui.View):
        """
        Interactive button view for exec approval of dangerous commands.

        Shows four buttons: Allow Once, Allow Session, Always Allow, Deny.
        Clicking a button calls ``resolve_gateway_approval()`` to unblock the
        waiting agent thread — the same mechanism as the text ``/approve`` flow.
        Only users in the allowed list can click.  Times out after 5 minutes.
        """

        def __init__(self, session_key: str, allowed_user_ids: set):
            super().__init__(timeout=300)  # 5-minute timeout
            self.session_key = session_key
            self.allowed_user_ids = allowed_user_ids
            self.resolved = False

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Verify the user clicking is authorized."""
            if not self.allowed_user_ids:
                return True  # No allowlist = anyone can approve
            return str(interaction.user.id) in self.allowed_user_ids

        async def _resolve(
            self, interaction: discord.Interaction, choice: str,
            color: discord.Color, label: str,
        ):
            """Resolve the approval via the gateway approval queue and update the embed."""
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
                embed.set_footer(text=f"{label} by {interaction.user.display_name}")

            # Disable all buttons
            for child in self.children:
                child.disabled = True

            await interaction.response.edit_message(embed=embed, view=self)

            # Unblock the waiting agent thread via the gateway approval queue
            try:
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(self.session_key, choice)
                logger.info(
                    "Discord button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, self.session_key, choice, interaction.user.display_name,
                )
            except Exception as exc:
                logger.error("Failed to resolve gateway approval from button: %s", exc)

        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "once", discord.Color.green(), "Approved once")

        @discord.ui.button(label="Allow Session", style=discord.ButtonStyle.grey)
        async def allow_session(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "session", discord.Color.blue(), "Approved for session")

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "always", discord.Color.purple(), "Approved permanently")

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "deny", discord.Color.red(), "Denied")

        async def on_timeout(self):
            """Handle view timeout -- disable buttons and mark as expired."""
            self.resolved = True
            for child in self.children:
                child.disabled = True

    class UpdatePromptView(discord.ui.View):
        """Interactive Yes/No buttons for ``hermes update`` prompts.

        Clicking a button writes the answer to ``.update_response`` so the
        detached update process can pick it up.  Only authorized users can
        click.  Times out after 5 minutes (the update process also has a
        5-minute timeout on its side).
        """

        def __init__(self, session_key: str, allowed_user_ids: set):
            super().__init__(timeout=300)
            self.session_key = session_key
            self.allowed_user_ids = allowed_user_ids
            self.resolved = False

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            if not self.allowed_user_ids:
                return True
            return str(interaction.user.id) in self.allowed_user_ids

        async def _respond(
            self, interaction: discord.Interaction, answer: str,
            color: discord.Color, label: str,
        ):
            if self.resolved:
                await interaction.response.send_message(
                    "Already answered~", ephemeral=True
                )
                return
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized~", ephemeral=True
                )
                return

            self.resolved = True

            # Update embed
            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed:
                embed.color = color
                embed.set_footer(text=f"{label} by {interaction.user.display_name}")

            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(embed=embed, view=self)

            # Write response file
            try:
                from hermes_constants import get_hermes_home
                home = get_hermes_home()
                response_path = home / ".update_response"
                tmp = response_path.with_suffix(".tmp")
                tmp.write_text(answer)
                tmp.replace(response_path)
                logger.info(
                    "Discord update prompt answered '%s' by %s",
                    answer, interaction.user.display_name,
                )
            except Exception as exc:
                logger.error("Failed to write update response: %s", exc)

        @discord.ui.button(label="Yes", style=discord.ButtonStyle.green, emoji="✓")
        async def yes_btn(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._respond(interaction, "y", discord.Color.green(), "Yes")

        @discord.ui.button(label="No", style=discord.ButtonStyle.red, emoji="✗")
        async def no_btn(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._respond(interaction, "n", discord.Color.red(), "No")

        async def on_timeout(self):
            self.resolved = True
            for child in self.children:
                child.disabled = True

    class ModelPickerView(discord.ui.View):
        """Interactive select-menu view for model switching.

        Two-step drill-down: provider dropdown → model dropdown.
        Edits the original message in-place as the user navigates.
        Times out after 2 minutes.
        """

        def __init__(
            self,
            providers: list,
            current_model: str,
            current_provider: str,
            session_key: str,
            on_model_selected,
            allowed_user_ids: set,
        ):
            super().__init__(timeout=120)
            self.providers = providers
            self.current_model = current_model
            self.current_provider = current_provider
            self.session_key = session_key
            self.on_model_selected = on_model_selected
            self.allowed_user_ids = allowed_user_ids
            self.resolved = False
            self._selected_provider: str = ""

            self._build_provider_select()

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            if not self.allowed_user_ids:
                return True
            return str(interaction.user.id) in self.allowed_user_ids

        def _build_provider_select(self):
            """Build the provider dropdown menu."""
            self.clear_items()
            options = []
            for p in self.providers:
                count = p.get("total_models", len(p.get("models", [])))
                label = f"{p['name']} ({count} models)"
                desc = "current" if p.get("is_current") else None
                options.append(
                    discord.SelectOption(
                        label=label[:100],
                        value=p["slug"],
                        description=desc,
                    )
                )
            if not options:
                return

            select = discord.ui.Select(
                placeholder="Choose a provider...",
                options=options[:25],
                custom_id="model_provider_select",
            )
            select.callback = self._on_provider_selected
            self.add_item(select)

            cancel_btn = discord.ui.Button(
                label="Cancel", style=discord.ButtonStyle.red, custom_id="model_cancel"
            )
            cancel_btn.callback = self._on_cancel
            self.add_item(cancel_btn)

        def _build_model_select(self, provider_slug: str):
            """Build the model dropdown for a specific provider."""
            self.clear_items()
            provider = next(
                (p for p in self.providers if p["slug"] == provider_slug), None
            )
            if not provider:
                return

            models = provider.get("models", [])
            options = []
            for model_id in models[:25]:
                short = model_id.split("/")[-1] if "/" in model_id else model_id
                options.append(
                    discord.SelectOption(
                        label=short[:100],
                        value=model_id[:100],
                    )
                )
            if not options:
                return

            select = discord.ui.Select(
                placeholder=f"Choose a model from {provider.get('name', provider_slug)}...",
                options=options,
                custom_id="model_model_select",
            )
            select.callback = self._on_model_selected
            self.add_item(select)

            back_btn = discord.ui.Button(
                label="◀ Back", style=discord.ButtonStyle.grey, custom_id="model_back"
            )
            back_btn.callback = self._on_back
            self.add_item(back_btn)

            cancel_btn = discord.ui.Button(
                label="Cancel", style=discord.ButtonStyle.red, custom_id="model_cancel2"
            )
            cancel_btn.callback = self._on_cancel
            self.add_item(cancel_btn)

        async def _on_provider_selected(self, interaction: discord.Interaction):
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized~", ephemeral=True
                )
                return

            provider_slug = interaction.data["values"][0]
            self._selected_provider = provider_slug
            provider = next(
                (p for p in self.providers if p["slug"] == provider_slug), None
            )
            pname = provider.get("name", provider_slug) if provider else provider_slug

            self._build_model_select(provider_slug)

            total = provider.get("total_models", 0) if provider else 0
            shown = min(len(provider.get("models", [])), 25) if provider else 0
            extra = f"\n*{total - shown} more available — type `/model <name>` directly*" if total > shown else ""

            await interaction.response.edit_message(
                embed=discord.Embed(
                    title="⚙ Model Configuration",
                    description=f"Provider: **{pname}**\nSelect a model:{extra}",
                    color=discord.Color.blue(),
                ),
                view=self,
            )

        async def _on_model_selected(self, interaction: discord.Interaction):
            if self.resolved:
                await interaction.response.send_message(
                    "Already resolved~", ephemeral=True
                )
                return
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized~", ephemeral=True
                )
                return

            self.resolved = True
            model_id = interaction.data["values"][0]

            try:
                result_text = await self.on_model_selected(
                    str(interaction.channel_id),
                    model_id,
                    self._selected_provider,
                )
            except Exception as exc:
                result_text = f"Error switching model: {exc}"

            self.clear_items()
            await interaction.response.edit_message(
                embed=discord.Embed(
                    title="⚙ Model Switched",
                    description=result_text,
                    color=discord.Color.green(),
                ),
                view=self,
            )

        async def _on_back(self, interaction: discord.Interaction):
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized~", ephemeral=True
                )
                return

            self._build_provider_select()

            try:
                from hermes_cli.providers import get_label
                provider_label = get_label(self.current_provider)
            except Exception:
                provider_label = self.current_provider

            await interaction.response.edit_message(
                embed=discord.Embed(
                    title="⚙ Model Configuration",
                    description=(
                        f"Current model: `{self.current_model or 'unknown'}`\n"
                        f"Provider: {provider_label}\n\n"
                        f"Select a provider:"
                    ),
                    color=discord.Color.blue(),
                ),
                view=self,
            )

        async def _on_cancel(self, interaction: discord.Interaction):
            self.resolved = True
            self.clear_items()
            await interaction.response.edit_message(
                embed=discord.Embed(
                    title="⚙ Model Configuration",
                    description="Model selection cancelled.",
                    color=discord.Color.greyple(),
                ),
                view=self,
            )

        async def on_timeout(self):
            self.resolved = True
            self.clear_items()

"""Discord adapter initialization and per-instance runtime state."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.platforms.helpers import MessageDeduplicator, ThreadParticipationTracker
from utils import env_float
from ..voice import VoiceReceiver
from .. import adapter as _adapter


def _new_nonconversational_tracker():
    return _adapter._DiscordNonConversationalMessageTracker()


class InitializationMixin:
    """Create the Discord client's state after the platform base is initialized."""


    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
        self._allowed_role_ids: set = set()  # For DISCORD_ALLOWED_ROLES filtering
        # Gate env snapshot captured in connect() inside the owning profile's scope; None until then.
        # None until then; accessors fall back to live scope-aware reads (issue #72348).
        self._gate_env_snapshot: Optional[Dict[str, str]] = None
        self.gateway_runner = None  # Set by gateway/run.py for cross-platform delivery
        self._voice_clients: Dict[int, Any] = {}  # guild_id -> VoiceClient
        self._voice_locks: Dict[int, asyncio.Lock] = {}  # guild_id -> serialize join/leave
        # Text batching: merge rapid successive messages (Telegram-style)
        self._text_batch_delay_seconds = env_float("HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS", 0.6)
        self._text_batch_split_delay_seconds = env_float("HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._voice_text_channels: Dict[int, int] = {}  # guild_id -> text_channel_id
        self._voice_sources: Dict[int, Dict[str, Any]] = {}  # guild_id -> linked text channel source metadata
        self._voice_timeout_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> timeout task
        self._voice_timeout_seconds = self._load_voice_timeout()
        self._playback_timeout_seconds = self._load_playback_timeout()
        self._voice_receivers: Dict[int, VoiceReceiver] = {}  # guild_id -> VoiceReceiver
        self._voice_listen_tasks: Dict[int, asyncio.Task] = {}  # guild_id -> listen loop
        self._voice_input_callback: Optional[Callable] = None  # set by run.py
        self._on_voice_disconnect: Optional[Callable] = None  # set by run.py
        # Voice-reply mode ("off"|"voice_only"|"all") per linked text-channel id (set by run.py) so
        # the inactivity timer keeps the bot in channel for /voice off, unlike /voice leave.
        self._voice_mode_getter: Optional[Callable] = None  # set by run.py
        # Continuous voice mixer per guild (ambient bed + ducked speech) so acks/TTS/thinking overlap.
        self._voice_mixers: Dict[int, Any] = {}  # guild_id -> VoiceMixer
        self._ambient_pcm_cache: Optional[bytes] = None  # decoded ambient bed
        self._voice_fx_cfg: Dict[str, Any] = self._load_voice_fx_config()
        # Threads the bot participated in (no @mention needed there); persisted across restarts.
        self._threads = ThreadParticipationTracker("discord")
        # Persistent typing loops per channel (DMs don't reliably show bot typing events).
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._bot_task: Optional[asyncio.Task] = None
        # Background task that runs post-connect housekeeping (command-menu registration + DM-topic setup)
        # off the connect path so a slow Bot API call (e.g. a set_my_commands stall for certain tokens)
        # cannot blow the gateway's connect timeout (#46298).
        self._post_connect_task: Optional[asyncio.Task] = None
        # WS liveness probe: REST 200 can't prove Gateway events still arrive, so sample WS
        # ready/open/ACK + heartbeat latency; consecutive failures -> retryable-fatal. 0 disables.
        self._liveness_interval_seconds = self._finite_positive_config_float(
            "websocket_liveness_interval_seconds", 15.0,
            env_key="HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        )
        self._liveness_failure_threshold = self._config_int(
            "websocket_liveness_failure_threshold", 2,
            env_key="HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
        )
        self._heartbeat_ack_max_age_seconds = self._finite_positive_config_float(
            "websocket_heartbeat_ack_max_age_seconds", 60.0,
        )
        self._max_latency_seconds = self._finite_positive_config_float(
            "websocket_max_latency_seconds", 30.0,
        )
        self._liveness_task: Optional[asyncio.Task] = None
        self._liveness_notification_task: Optional[asyncio.Task] = None
        # True while disconnect() intentionally closes discord.py (done callback: shutdown vs crash).
        self._disconnecting = False
        self._missed_message_backfill_task: Optional[asyncio.Task] = None
        from hermes_constants import get_hermes_home
        from plugins.platforms.discord.recovery import DiscordRecoveryStore
        self._discord_recovery_store = DiscordRecoveryStore(get_hermes_home())
        # Dedup cache: Discord RESUME replays events after reconnects.
        self._dedup = MessageDeduplicator()
        # Reply threading mode: "off", "first" (default; first chunk only), "all" (every chunk).
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'
        self._slash_commands: bool = self.config.extra.get("slash_commands", True)
        # Bot's last message ID per channel: lets history backfill skip the full channel.history() scan.
        self._last_self_message_id: Dict[str, str] = {}
        # Bot-authored lifecycle/status message IDs that must not bound history after restart.
        self._nonconversational_messages = _new_nonconversational_tracker()
        # Last truncated mid-stream preview per (chat_id, message_id): past the 2000 cap every edit
        # truncates to the SAME text, and re-sending only burns edit rate limit. Dropped on finalize.
        # Once an oversized streaming edit saturates at the 2000-char preview cap, every subsequent
        # progressive edit truncates to the SAME text; re-sending it is a no-op that still counts against
        # Discord's edit rate limit (~1 edit per stream tick for the rest of a long reply). Mirrors the
        # Telegram #58563 fix.
        self._last_overflow_preview: Dict[tuple, str] = {}
        self._warned_fail_closed_default = False

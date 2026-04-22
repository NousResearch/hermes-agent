"""
LiveKit voice platform adapter using WebRTC.

Connects to a LiveKit room, receives participant audio, transcribes via
hermes's STT pipeline, feeds into the agent loop, and publishes TTS
responses back as audio.

Requires:
    pip install 'hermes-agent[livekit]'
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET env vars

Configuration in config.yaml:
    platforms:
      livekit:
        enabled: true
        extra:
          url: "wss://your-project.livekit.cloud"
          api_key: "your-api-key"
          api_secret: "your-api-secret"
          room: "hermes"          # optional, default "hermes"
"""

import asyncio
import io
import logging
import math
import os
import struct
import subprocess
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    rtc = None  # type: ignore[assignment]

try:
    from livekit.api import AccessToken, VideoGrants, LiveKitAPI
    from livekit.protocol.room import ListParticipantsRequest
    LIVEKIT_API_AVAILABLE = True
except ImportError:
    LIVEKIT_API_AVAILABLE = False
    AccessToken = None  # type: ignore[assignment,misc]
    VideoGrants = None  # type: ignore[assignment,misc]
    LiveKitAPI = None  # type: ignore[assignment,misc]
    ListParticipantsRequest = None  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Voice detection
SILENCE_THRESHOLD_SECONDS = 1.5   # seconds of silence → end of utterance
MIN_SPEECH_DURATION = 0.5         # minimum seconds to process (skip noise)
RMS_SILENCE_FLOOR = 50            # PCM RMS below this is silence
POLL_INTERVAL = 0.2               # silence check interval when active
IDLE_POLL_INTERVAL = 2.0          # silence check interval when no remote participants

# LiveKit audio defaults
SAMPLE_RATE = 48000
NUM_CHANNELS = 1

# Reconnection
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
MAX_RECONNECT_ATTEMPTS = 10       # give up after this many consecutive failures

# Presence polling (when no humans in room, we stay out and poll).
# Defaults differ by deployment:
#   - LiveKit Cloud has real API rate limits and we pay per-minute, so
#     30s keeps headroom while still waking fast enough for normal UX.
#   - Self-hosted LiveKit has no limits and no cost pressure, so we poll
#     aggressively enough that the first speaker doesn't wait noticeably.
# Override with LIVEKIT_PRESENCE_POLL_INTERVAL (seconds) in the env if
# neither default fits.
PRESENCE_POLL_INTERVAL_CLOUD = 30.0
PRESENCE_POLL_INTERVAL_LOCAL = 5.0


def check_livekit_requirements() -> bool:
    """Check if LiveKit dependencies are available and configured."""
    if not LIVEKIT_AVAILABLE or not LIVEKIT_API_AVAILABLE:
        return False
    if not os.getenv("LIVEKIT_URL") or not os.getenv("LIVEKIT_API_KEY") or not os.getenv("LIVEKIT_API_SECRET"):
        return False
    return True


def _compute_rms(pcm_data: bytes) -> float:
    """Compute RMS energy of 16-bit PCM samples."""
    if len(pcm_data) < 2:
        return 0.0
    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / n_samples)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class LiveKitAdapter(BasePlatformAdapter):
    """LiveKit voice adapter using WebRTC.

    Joins a LiveKit room, captures participant audio, transcribes to text,
    and sends TTS replies back to the room.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LIVEKIT)

        extra = config.extra or {}
        self._url: str = extra.get("url") or os.getenv("LIVEKIT_URL", "")
        self._api_key: str = extra.get("api_key") or os.getenv("LIVEKIT_API_KEY", "")
        self._api_secret: str = extra.get("api_secret") or os.getenv("LIVEKIT_API_SECRET", "")
        self._room_name: str = extra.get("room") or os.getenv("LIVEKIT_ROOM", "hermes")
        self._agent_name: str = extra.get("agent_name") or os.getenv("LIVEKIT_AGENT_NAME", "Hermes")
        self._agent_avatar: str = extra.get("agent_avatar") or os.getenv("LIVEKIT_AGENT_AVATAR", "") or self._find_default_avatar()

        self._room: Optional["rtc.Room"] = None
        self._audio_source: Optional["rtc.AudioSource"] = None
        self._local_track: Optional["rtc.LocalAudioTrack"] = None
        self._silence_task: Optional[asyncio.Task] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._presence_task: Optional[asyncio.Task] = None
        self._graceful_leave: bool = False  # set while intentionally leaving

        # Per-participant audio buffers: identity -> (pcm bytearray, last_audio_time)
        self._audio_buffers: Dict[str, bytearray] = {}
        self._last_audio_time: Dict[str, float] = {}
        self._audio_streams: Dict[str, asyncio.Task] = {}

        # Pause audio capture during TTS playback
        self._paused = False

        # Per-participant speech state (for listening-start/stop events)
        self._speaking_participants: set[str] = set()

        self._presence_poll_interval: float = self._resolve_presence_poll_interval()

    def _resolve_presence_poll_interval(self) -> float:
        """Pick the presence-poll interval: env override > cloud/local default.

        LiveKit Cloud hosts on ``*.livekit.cloud``; anything else is treated
        as a self-hosted deployment and gets the faster default.
        """
        override = os.getenv("LIVEKIT_PRESENCE_POLL_INTERVAL", "").strip()
        if override:
            try:
                parsed = float(override)
                if parsed > 0:
                    logger.info("[%s] presence poll interval=%.1fs (LIVEKIT_PRESENCE_POLL_INTERVAL)", self.name, parsed)
                    return parsed
            except ValueError:
                logger.warning("[%s] LIVEKIT_PRESENCE_POLL_INTERVAL=%r is not a number; using default", self.name, override)

        is_cloud = ".livekit.cloud" in self._url.lower()
        interval = PRESENCE_POLL_INTERVAL_CLOUD if is_cloud else PRESENCE_POLL_INTERVAL_LOCAL
        logger.info("[%s] presence poll interval=%.1fs (%s default)", self.name, interval, "cloud" if is_cloud else "local")
        return interval

    @staticmethod
    def _find_default_avatar() -> str:
        """Look for a default avatar image in ~/.hermes/."""
        from pathlib import Path
        hermes_home = Path.home() / ".hermes"
        for name in ("agent.png", "agent.jpg"):
            path = hermes_home / name
            if path.is_file():
                return str(path)
        return ""

    def _resolve_avatar_url(self) -> str:
        """Convert avatar to a URL suitable for LiveKit metadata.

        If it's already a URL, use as-is. If it's a local file, encode
        as a data URI so it works without a web server.
        """
        avatar = self._agent_avatar
        if not avatar:
            return ""
        if avatar.startswith(("http://", "https://", "data:")):
            return avatar
        # Local file — base64 encode as data URI
        try:
            import base64
            from pathlib import Path
            path = Path(avatar).expanduser()
            if not path.is_file():
                return ""
            suffix = path.suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix, "image/png")
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"
        except Exception:
            return ""

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        """Start the LiveKit adapter.

        Presence-aware: if the room already has at least one remote
        participant, join immediately. Otherwise stay out and run a
        presence watcher that joins as soon as someone arrives. Either
        way the adapter is "connected" from the gateway's point of view.
        """
        if not LIVEKIT_AVAILABLE:
            logger.warning("[%s] livekit SDK not installed. Run: pip install 'hermes-agent[livekit]'", self.name)
            return False
        if not LIVEKIT_API_AVAILABLE:
            logger.warning("[%s] livekit-api not installed. Run: pip install 'hermes-agent[livekit]'", self.name)
            return False
        if not self._url or not self._api_key or not self._api_secret:
            logger.warning("[%s] LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET required", self.name)
            return False

        self._running = True

        # Check if anyone is in the room already. If not, don't consume a
        # participant slot — just watch.
        count = await self._count_remote_participants()
        if count > 0:
            logger.info("[%s] %d participant(s) already in '%s', joining", self.name, count, self._room_name)
            return await self._join_room()

        logger.info("[%s] Room '%s' empty, watching for participants (poll %.1fs)", self.name, self._room_name, self._presence_poll_interval)
        self._mark_connected()
        self._presence_task = asyncio.create_task(self._presence_watch_loop())
        return True

    async def _count_remote_participants(self) -> int:
        """Count non-local participants currently in the room via the Server API.

        Returns 0 on any error (room missing, network blip, etc.) — callers
        treat that as "nobody here, keep polling".
        """
        try:
            # Server API expects http(s):// scheme; convert from ws(s)://.
            http_url = self._url
            if http_url.startswith("wss://"):
                http_url = "https://" + http_url[6:]
            elif http_url.startswith("ws://"):
                http_url = "http://" + http_url[5:]
            http_url = http_url.rstrip("/")

            client = LiveKitAPI(url=http_url, api_key=self._api_key, api_secret=self._api_secret)
            try:
                resp = await client.room.list_participants(
                    ListParticipantsRequest(room=self._room_name)
                )
                return len(resp.participants)
            finally:
                await client.aclose()
        except Exception as e:
            logger.debug("[%s] presence check failed: %s", self.name, e)
            return 0

    async def _presence_watch_loop(self) -> None:
        """Poll the room; join as soon as a remote participant appears."""
        try:
            while self._running:
                await asyncio.sleep(self._presence_poll_interval)
                if not self._running:
                    return
                if self._room is not None:
                    # Something else joined us (manual reconnect?); stop polling.
                    return
                count = await self._count_remote_participants()
                if count > 0:
                    logger.info("[%s] Participant detected in '%s', joining", self.name, self._room_name)
                    if await self._join_room():
                        return  # joined — done polling
        except asyncio.CancelledError:
            return

    async def _join_room(self) -> bool:
        """Actually establish the LiveKit room connection and start audio I/O."""
        try:
            self._room = rtc.Room()

            # Register event handlers
            self._room.on("track_subscribed", self._on_track_subscribed)
            self._room.on("track_unsubscribed", self._on_track_unsubscribed)
            self._room.on("participant_disconnected", self._on_participant_disconnected)
            self._room.on("disconnected", self._on_disconnected)

            # Create access token
            import json as _json
            token = (
                AccessToken(api_key=self._api_key, api_secret=self._api_secret)
                .with_identity(f"hermes-{self._agent_name.lower()}")
                .with_name(self._agent_name)
                .with_grants(VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_update_own_metadata=True,
                ))
            )
            jwt_token = token.to_jwt()

            # Connect to room
            await self._room.connect(self._url, jwt_token)

            # Set metadata (including avatar) after connecting — avoids JWT size limits
            metadata = {}
            avatar_url = self._resolve_avatar_url()
            if avatar_url:
                metadata["avatar"] = avatar_url
            if metadata:
                await self._room.local_participant.set_metadata(_json.dumps(metadata))

            # Publish a local audio track for TTS playback
            self._audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
            self._local_track = rtc.LocalAudioTrack.create_audio_track(
                "hermes-voice", self._audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await self._room.local_participant.publish_track(self._local_track, options)

            # Start silence detection loop
            self._silence_task = asyncio.create_task(self._check_silence_loop())

            self._mark_connected()
            logger.info("[%s] Connected to room '%s' at %s", self.name, self._room_name, self._url)

            # If no explicit agent name was configured, ask the LLM and reconnect
            if not os.getenv("LIVEKIT_AGENT_NAME") and not (self.config.extra or {}).get("agent_name"):
                asyncio.create_task(self._resolve_agent_name())

            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from LiveKit room."""
        self._running = False
        self._mark_disconnected()

        if self._presence_task:
            self._presence_task.cancel()
            try:
                await self._presence_task
            except asyncio.CancelledError:
                pass
            self._presence_task = None

        if self._silence_task:
            self._silence_task.cancel()
            try:
                await self._silence_task
            except asyncio.CancelledError:
                pass
            self._silence_task = None

        # Cancel all audio stream tasks
        for task in self._audio_streams.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._audio_streams.clear()

        if self._room:
            self._graceful_leave = True
            try:
                await self._room.disconnect()
            finally:
                self._graceful_leave = False
            self._room = None

        self._audio_source = None
        self._local_track = None
        self._audio_buffers.clear()
        self._last_audio_time.clear()
        self._speaking_participants.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- LiveKit event handlers ---------------------------------------------

    def _on_track_subscribed(
        self,
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ):
        """Start capturing audio when a participant's audio track is subscribed."""
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        identity = participant.identity
        logger.info("[%s] Audio track subscribed: %s", self.name, identity)

        # Initialize buffer for this participant
        self._audio_buffers[identity] = bytearray()
        self._last_audio_time[identity] = time.monotonic()

        # Start receiving audio frames
        stream = rtc.AudioStream(track)
        task = asyncio.create_task(self._audio_receive_loop(stream, identity))
        self._audio_streams[identity] = task

    def _on_track_unsubscribed(
        self,
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ):
        """Clean up when a participant's audio track is unsubscribed."""
        identity = participant.identity
        logger.debug("[%s] Audio track unsubscribed: %s", self.name, identity)
        self._cleanup_participant(identity)

    def _on_participant_disconnected(self, participant: "rtc.RemoteParticipant"):
        """Clean up when a participant leaves the room.

        If we're now alone in the room, drop the connection and go back
        to presence polling — no need to consume a participant slot while
        nobody's here to talk to.
        """
        identity = participant.identity
        logger.info("[%s] Participant disconnected: %s", self.name, identity)
        self._cleanup_participant(identity)

        if self._room and not self._room.remote_participants:
            logger.info("[%s] Last participant left '%s', leaving room", self.name, self._room_name)
            asyncio.create_task(self._leave_and_watch())

    async def _leave_and_watch(self) -> None:
        """Tear down the room connection and resume presence polling."""
        # Stop silence detection and audio streams, but keep self._running
        # so the presence loop can resume us later.
        if self._silence_task:
            self._silence_task.cancel()
            try:
                await self._silence_task
            except asyncio.CancelledError:
                pass
            self._silence_task = None

        for task in self._audio_streams.values():
            task.cancel()
        self._audio_streams.clear()
        self._audio_buffers.clear()
        self._last_audio_time.clear()
        self._speaking_participants.clear()

        if self._room:
            self._graceful_leave = True
            try:
                await self._room.disconnect()
            except Exception as e:
                logger.debug("[%s] leave error: %s", self.name, e)
            finally:
                self._graceful_leave = False
            self._room = None
        self._audio_source = None
        self._local_track = None

        if self._running and (self._presence_task is None or self._presence_task.done()):
            self._presence_task = asyncio.create_task(self._presence_watch_loop())

    def _on_disconnected(self, reason: str = ""):
        """Handle unexpected room disconnection — schedule reconnection.

        Graceful leaves (empty room, full teardown) set ``_graceful_leave``
        so we don't fight with ``_leave_and_watch`` / ``disconnect``.
        """
        if not self._running or self._graceful_leave:
            return
        logger.warning("[%s] Disconnected from room: %s. Will reconnect.", self.name, reason)
        self._connect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Reconnect to LiveKit with exponential backoff.

        Caps at MAX_RECONNECT_ATTEMPTS consecutive failures — beyond that the
        adapter stays disconnected rather than spamming a misconfigured URL
        forever. The user can restart the gateway to retry.
        """
        backoff_idx = 0
        attempts = 0
        while self._running:
            if attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.error(
                    "[%s] Giving up after %d reconnect attempts. Restart the gateway to try again.",
                    self.name, attempts,
                )
                self._running = False
                self._mark_disconnected()
                return
            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Reconnecting in %ds (attempt %d/%d)...", self.name, delay, attempts + 1, MAX_RECONNECT_ATTEMPTS)
            await asyncio.sleep(delay)
            if not self._running:
                return
            try:
                if await self._join_room():
                    logger.info("[%s] Reconnected successfully", self.name)
                    return
            except Exception as e:
                logger.warning("[%s] Reconnect attempt failed: %s", self.name, e)
            backoff_idx += 1
            attempts += 1

    async def _resolve_agent_name(self):
        """Ask the LLM for the agent's name, then update the display name in-place."""
        try:
            from openai import AsyncOpenAI
            from hermes_cli.config import load_config

            config = load_config()
            model_config = config.get("model", {})
            provider = model_config.get("provider", "")
            model = model_config.get("default", "")

            # Use the runtime provider resolution to get the right client
            from hermes_cli.runtime_provider import resolve_requested_provider
            resolved = resolve_requested_provider(provider, model)
            if not resolved or not resolved.get("api_key"):
                return

            client = AsyncOpenAI(
                api_key=resolved["api_key"],
                base_url=resolved.get("base_url"),
            )
            resp = await client.chat.completions.create(
                model=resolved.get("model", model),
                messages=[{"role": "user", "content": "What is your name? Reply with ONLY your first name — no quotes, no punctuation, no explanation. It will be used as your on-screen display label in a video call."}],
                max_tokens=20,
            )
            name = resp.choices[0].message.content.strip().strip('"').strip("'").split()[0] if resp.choices else ""
            if not name or name.lower() == "hermes" or len(name) > 30:
                return

            logger.info("[%s] LLM says agent name is '%s', updating display name", self.name, name)
            self._agent_name = name
            await self._room.local_participant.set_name(name)
        except Exception as e:
            logger.debug("[%s] Could not resolve agent name from LLM: %s", self.name, e)

    def _cleanup_participant(self, identity: str):
        """Remove buffers and cancel audio stream for a participant."""
        task = self._audio_streams.pop(identity, None)
        if task:
            task.cancel()
        self._audio_buffers.pop(identity, None)
        self._last_audio_time.pop(identity, None)
        self._speaking_participants.discard(identity)

    # -- Audio capture and processing ---------------------------------------

    async def _audio_receive_loop(
        self,
        stream: "rtc.AudioStream",
        identity: str,
    ):
        """Receive audio frames from a participant and buffer them.

        This loop must drain the SDK's internal queue as fast as possible
        to avoid 'native audio stream queue overflow' warnings.  All
        heavy processing (RMS, silence detection) happens in
        _check_silence_loop instead.
        """
        try:
            async for event in stream:
                if self._paused:
                    continue
                if identity not in self._audio_buffers:
                    break

                self._audio_buffers[identity].extend(event.frame.data.tobytes())
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning("[%s] Audio receive error for %s: %s", self.name, identity, e)

    async def _check_silence_loop(self):
        """Periodically check for completed utterances (silence after speech).

        Each tick, we look at the tail of every participant's buffer to
        decide whether they are currently speaking or silent.  When
        silence exceeds the threshold, we extract the utterance and
        send it for transcription.

        Drops to a slower poll when no participants are buffered — saves
        CPU without delaying utterance detection (a joining participant
        will trigger ``_on_track_subscribed`` immediately, not on the
        next loop tick).
        """
        # bytes per poll interval (how much audio one tick represents)
        bytes_per_tick = int(SAMPLE_RATE * NUM_CHANNELS * 2 * POLL_INTERVAL)

        try:
            while self._running:
                # No one to listen to — sleep longer.
                if not self._audio_buffers:
                    await asyncio.sleep(IDLE_POLL_INTERVAL)
                    continue
                await asyncio.sleep(POLL_INTERVAL)

                for identity in list(self._audio_buffers.keys()):
                    buf = self._audio_buffers.get(identity)
                    if buf is None:
                        continue

                    buf_len = len(buf)
                    if buf_len == 0:
                        continue

                    # Check RMS of the most recent chunk to detect speech/silence
                    tail = bytes(buf[-bytes_per_tick:]) if buf_len >= bytes_per_tick else bytes(buf)
                    rms = _compute_rms(tail)

                    if rms > RMS_SILENCE_FLOOR:
                        # Active speech — update timestamp
                        self._last_audio_time[identity] = time.monotonic()
                        # Emit listening-start on first loud chunk of an utterance
                        if identity not in self._speaking_participants:
                            self._speaking_participants.add(identity)
                            asyncio.create_task(
                                self._publish_agent_event(
                                    "agent:listening-start", {"identity": identity}
                                )
                            )
                        continue

                    # Silent — check if silence has lasted long enough
                    last_time = self._last_audio_time.get(identity)
                    if last_time is None:
                        # Never spoke — discard accumulated noise
                        self._audio_buffers[identity] = bytearray()
                        continue

                    elapsed_silence = time.monotonic() - last_time
                    if elapsed_silence < SILENCE_THRESHOLD_SECONDS:
                        continue

                    # Trim trailing silence from the buffer (keep only up to
                    # SILENCE_THRESHOLD worth of trailing audio)
                    silence_bytes = int(SILENCE_THRESHOLD_SECONDS * SAMPLE_RATE * NUM_CHANNELS * 2)
                    speech_end = max(0, buf_len - silence_bytes)

                    duration = speech_end / (SAMPLE_RATE * NUM_CHANNELS * 2)
                    if duration < MIN_SPEECH_DURATION:
                        # Too short — discard as noise
                        self._audio_buffers[identity] = bytearray()
                        self._last_audio_time.pop(identity, None)
                        # False alarm — revert the listening-start we sent
                        if identity in self._speaking_participants:
                            self._speaking_participants.discard(identity)
                            asyncio.create_task(
                                self._publish_agent_event(
                                    "agent:listening-stop", {"identity": identity}
                                )
                            )
                        continue

                    # Extract the utterance (speech portion only) and reset
                    pcm_data = bytes(buf[:speech_end])
                    self._audio_buffers[identity] = bytearray()
                    self._last_audio_time.pop(identity, None)
                    self._speaking_participants.discard(identity)
                    asyncio.create_task(
                        self._publish_agent_event(
                            "agent:listening-stop", {"identity": identity}
                        )
                    )

                    logger.info("[%s] Utterance from %s: %.1fs audio", self.name, identity, duration)
                    asyncio.create_task(self._process_voice_input(identity, pcm_data))
        except asyncio.CancelledError:
            return

    async def _process_voice_input(self, identity: str, pcm_data: bytes):
        """Transcribe audio and feed into the agent loop."""
        try:
            # Write PCM to WAV temp file
            wav_data = _pcm_to_wav(pcm_data, SAMPLE_RATE, NUM_CHANNELS)
            tmp_dir = os.path.join(tempfile.gettempdir(), "hermes_livekit")
            os.makedirs(tmp_dir, exist_ok=True)
            wav_path = os.path.join(tmp_dir, f"utterance_{uuid.uuid4().hex[:8]}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_data)

            # Transcribe using hermes STT pipeline
            from tools.transcription_tools import transcribe_audio, get_stt_model_from_config
            stt_model = get_stt_model_from_config()
            result = await asyncio.to_thread(transcribe_audio, wav_path, model=stt_model)

            # Clean up temp file
            try:
                os.unlink(wav_path)
            except OSError:
                pass

            logger.info("[%s] STT result from %s: %s", self.name, identity, result)
            transcript = (result.get("transcript") or result.get("text") or "").strip() if isinstance(result, dict) else ""
            if not transcript:
                logger.info("[%s] Empty transcript from %s, skipping", self.name, identity)
                return

            logger.info("[%s] Transcript from %s: %s", self.name, identity, transcript[:80])

            # Publish the final user transcript so clients can update their UI.
            await self._publish_agent_event(
                "agent:user-transcript",
                {"transcript": transcript, "final": True, "identity": identity},
            )

            # Build message event
            source = self.build_source(
                chat_id=self._room_name,
                chat_name=self._room_name,
                chat_type="group",
                user_id=identity,
                user_name=identity,
            )

            event = MessageEvent(
                text=transcript,
                message_type=MessageType.VOICE,
                source=source,
                message_id=uuid.uuid4().hex[:12],
                media_urls=[],
                timestamp=datetime.now(tz=timezone.utc),
            )

            # Agent is about to invoke the LLM.
            await self._publish_agent_event("agent:thinking-start")
            await self.handle_message(event)
        except Exception as e:
            logger.error("[%s] Error processing voice from %s: %s", self.name, identity, e)

    # -- Outbound messaging -------------------------------------------------

    async def _publish_agent_event(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish an agent:* lifecycle event as JSON on the default data topic.

        Consumed by voice-agent.desktop (and any compatible client) to drive
        UI state — listening/thinking/speaking indicators and live transcript
        display. Topic is deliberately unset: the desktop client routes
        messages with no topic (or any topic other than "hermes-chat") to its
        JSON/event handler.
        """
        if not self._room:
            return
        try:
            import json as _json
            msg = {"type": event_type, "payload": payload or {}}
            await self._room.local_participant.publish_data(
                _json.dumps(msg).encode("utf-8"), reliable=True
            )
        except Exception as e:
            # Never let UI telemetry break the voice flow.
            logger.debug("[%s] agent event publish failed (%s): %s", self.name, event_type, e)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send text via data channel (best-effort for connected web clients)."""
        if not self._room:
            return SendResult(success=False, error="Not connected to room")

        try:
            data = content.encode("utf-8")
            await self._room.local_participant.publish_data(
                data, reliable=True, topic="hermes-chat"
            )
            # Mirror the content as an agent-transcript event so clients that
            # render a conversation log can add an assistant message.
            await self._publish_agent_event(
                "agent:agent-transcript", {"transcript": content, "final": True}
            )
            return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
        except Exception as e:
            logger.debug("[%s] Data channel send failed (non-critical): %s", self.name, e)
            # Not a failure — voice is the primary channel
            return SendResult(success=True, message_id=uuid.uuid4().hex[:12])

    async def play_tts(
        self,
        chat_id: str,
        audio_path: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Play TTS audio into the LiveKit room via the published audio track."""
        if not self._audio_source or not self._room:
            return SendResult(success=False, error="Not connected to room")

        try:
            # Pause capture to avoid echo
            self._paused = True

            # Decode audio file to raw PCM using ffmpeg
            pcm_data = await asyncio.to_thread(
                self._decode_audio_to_pcm, audio_path
            )
            if not pcm_data:
                self._paused = False
                return SendResult(success=False, error="Failed to decode audio")

            # Publish PCM frames to the audio source
            # LiveKit expects frames of a specific size
            samples_per_frame = SAMPLE_RATE // 50  # 20ms frames
            bytes_per_frame = samples_per_frame * NUM_CHANNELS * 2  # 16-bit

            await self._publish_agent_event("agent:speaking-start")

            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset:offset + bytes_per_frame]
                if len(chunk) < bytes_per_frame:
                    # Pad the last frame with silence
                    chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))

                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=samples_per_frame,
                )
                await self._audio_source.capture_frame(frame)
                offset += bytes_per_frame

            # Brief pause after playback before resuming capture
            await asyncio.sleep(0.3)
            self._paused = False
            await self._publish_agent_event("agent:speaking-stop")

            return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
        except Exception as e:
            self._paused = False
            await self._publish_agent_event("agent:speaking-stop")
            logger.error("[%s] TTS playback error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    @staticmethod
    def _decode_audio_to_pcm(audio_path: str) -> Optional[bytes]:
        """Decode an audio file to raw 16-bit PCM using ffmpeg."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", audio_path,
                    "-f", "s16le",        # raw 16-bit little-endian PCM
                    "-acodec", "pcm_s16le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", str(NUM_CHANNELS),
                    "-loglevel", "error",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("ffmpeg decode failed: %s", result.stderr.decode()[:200])
                return None
            return result.stdout
        except FileNotFoundError:
            logger.warning("ffmpeg not found — required for LiveKit TTS playback")
            return None
        except Exception as e:
            logger.warning("Audio decode error: %s", e)
            return None

    def prepare_tts_text(self, text: str) -> str:
        """Strip tool output, code blocks, URLs, and file paths for voice.

        The full response is already sent via data channel — TTS should
        only speak the conversational parts.
        """
        import re as _re

        # Remove fenced code blocks (```...```)
        text = _re.sub(r'```[\s\S]*?```', '', text)

        # Remove inline code (`...`)
        text = _re.sub(r'`[^`]+`', '', text)

        # Remove URLs
        text = _re.sub(r'https?://\S+', '', text)

        # Remove file paths (/foo/bar, ~/foo, C:\foo)
        text = _re.sub(r'(?:~|/|[A-Z]:\\)[\w./\\-]+', '', text)

        # Remove MEDIA: tags
        text = _re.sub(r'MEDIA:\S+', '', text)

        # Remove markdown formatting
        text = _re.sub(r'[*_`#\[\]()]', '', text)

        # Collapse whitespace
        text = _re.sub(r'\n{3,}', '\n\n', text)
        text = _re.sub(r'  +', ' ', text)

        return text[:4000].strip()

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No typing indicator for voice — no-op."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return info about the LiveKit room."""
        participants = []
        if self._room:
            for p in self._room.remote_participants.values():
                participants.append(p.identity)
        return {
            "name": self._room_name,
            "type": "group",
            "chat_id": chat_id,
            "participants": participants,
        }

"""Hermes Radio player orchestrator.

Manages the dual-mpv pattern (primary + voice), source switching,
crate-dig loop, and mic break scheduling.  Designed to run as a
background service within the Hermes CLI process.
"""

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from radio.mpv_client import MpvClient

logger = logging.getLogger(__name__)


def _station_name_from_url(url: str) -> str:
    """Extract a readable station name from a stream URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # Radio Garden: /listen/station-name/ID/channel.mp3
    # The station name slug is only present in long-form URLs.
    # Short URLs like /listen/43U_hUSz/channel.mp3 only have the ID.
    if "radio.garden" in hostname:
        parts = [p for p in parsed.path.split("/") if p and p != "channel.mp3"]
        # Find the slug after "listen" -- skip if it looks like an ID (short, alphanumeric)
        for i, p in enumerate(parts):
            if p == "listen" and i + 1 < len(parts):
                slug = parts[i + 1]
                # If slug has hyphens and is > 10 chars, it's a name not an ID
                if len(slug) > 10 or "-" in slug:
                    return slug.replace("-", " ").title()
        # Couldn't extract name -- return generic
        return "Radio Garden"

    # SomaFM: ice2.somafm.com/defcon-256-mp3 -> defcon
    if "somafm.com" in hostname:
        path = parsed.path.strip("/").split("-")[0]
        return f"SomaFM {path}" if path else "SomaFM"

    # Generic: use hostname or last path segment
    path = parsed.path.strip("/").split("/")[-1]
    if path and path not in ("stream", "channel.mp3", ""):
        return path.replace("-", " ").replace("_", " ").title()

    return hostname or "Radio"


class SourceMode(str, Enum):
    CRATE = "crate"       # Radiooooo track-by-track
    STREAM = "stream"     # Live radio (Radio Browser, SomaFM, Radio Garden, custom)
    LOCAL = "local"       # Local files


@dataclass
class NowPlaying:
    """Snapshot of the current playback state for display."""
    active: bool = False
    source_mode: str = ""
    title: str = ""
    artist: str = ""
    decade: int = 0
    country: str = ""
    mood: str = ""
    position: Optional[float] = None
    duration: Optional[float] = None
    volume: float = 55.0
    paused: bool = False
    station_name: str = ""
    # History for mic break context
    recent_tracks: List[Dict[str, str]] = field(default_factory=list)


class HermesRadio:
    """Main radio player.  Singleton within a Hermes CLI process."""

    _instance: Optional["HermesRadio"] = None

    def __init__(self):
        self._primary = MpvClient(label="main")
        self._voice = MpvClient(label="voice")
        self._source_mode: SourceMode = SourceMode.CRATE
        self._now = NowPlaying()
        self._crate_task: Optional[asyncio.Task] = None
        self._mic_break_active = False
        self._auto_mic_breaks = True
        self._mic_break_persona = "encyclopedic"
        self._duck_volume = 15
        self._duck_ramp_ms = 500
        self._tracks_since_break = 0
        self._break_every_n = 3
        self._running = False
        self._on_state_change: Optional[Callable] = None
        # Radiooooo client (lazy init)
        self._radiooooo = None
        # Crate dig config
        self._crate_decades: Optional[List[int]] = None
        self._crate_moods: Optional[List[str]] = None
        self._crate_country: Optional[str] = None
        self._crate_weighted = True
        # Weight overrides (None = use module defaults)
        self._mood_weights: Optional[Dict[str, float]] = None
        self._country_weights: Optional[Dict[str, float]] = None
        self._decade_weights: Optional[Dict[int, float]] = None
        # State poll task (created in start())
        self._state_poll_task = None
        self._muted = False
        self._pre_mute_volume = 55.0

    @classmethod
    def get(cls) -> "HermesRadio":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def active(cls) -> bool:
        """Check if the radio is currently active (without creating an instance)."""
        return cls._instance is not None and cls._instance._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the primary mpv instance."""
        if self._running:
            return
        await self._primary.start()

        # Listen for track-end events
        self._primary.on("end-file", self._on_track_end)
        self._primary.on("metadata-update", self._on_metadata_update)

        # Observe metadata changes
        await self._primary.observe_property(1, "media-title")
        await self._primary.observe_property(2, "metadata")

        self._running = True

        # Apply configured default volume
        await self._primary.set_volume(self._now.volume)

        self._state_poll_task = asyncio.create_task(self._poll_state())
        self._notify_state_change()
        try:
            from radio.log import info
            info("radio started")
        except Exception:
            pass
        logger.info("Hermes Radio started")

    async def stop(self) -> None:
        """Stop everything and clean up."""
        self._running = False

        for task in (self._crate_task, getattr(self, "_state_poll_task", None)):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self._primary.stop()

        if self._voice.running:
            await self._voice.stop()

        # Stop level meter
        try:
            from radio.level_meter import stop as stop_meter
            stop_meter()
        except Exception:
            pass

        if self._radiooooo:
            await self._radiooooo.close()
            self._radiooooo = None

        self._now = NowPlaying()
        self._notify_state_change()
        HermesRadio._instance = None
        logger.info("Hermes Radio stopped")

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    async def play_stream(self, url: str, station_name: str = "") -> str:
        """Play a live radio stream URL."""
        if not self._running:
            await self.start()
        self._cancel_crate()
        self._source_mode = SourceMode.STREAM
        self._now.source_mode = "stream"

        # Derive a station name if none provided
        if not station_name:
            station_name = _station_name_from_url(url)

        # Stop any active recording before switching (saves the file)
        if self.is_recording:
            result = await self.stop_recording()
            print(f"  {result}")

        self._now.station_name = station_name
        self._now.active = True
        await self._primary.loadfile(url)

        # Start level meter for reactive visualizer
        try:
            from radio.level_meter import start as start_meter
            start_meter(url)
        except Exception:
            pass

        # After a short delay, try to get a better name from mpv metadata
        async def _update_name():
            await asyncio.sleep(2)
            try:
                title = await self._primary.get_media_title()
                if title and title not in ("channel.mp3", url.split("/")[-1]):
                    if not self._now.station_name or self._now.station_name == _station_name_from_url(url):
                        self._now.station_name = title
                        self._notify_state_change()
                        # Update recent with the real name
                        from radio.config import add_recent_station
                        add_recent_station(title, url, source="stream")
            except Exception:
                pass
        asyncio.create_task(_update_name())

        # Log station to history
        try:
            from radio.history import log_station
            log_station(station_name=station_name, url=url, source="stream")
            from radio.config import add_recent_station
            add_recent_station(station_name, url, source="stream")
        except Exception:
            pass
        self._notify_state_change()
        return f"Tuned to {station_name}"

    async def play_crate(
        self,
        decades: Optional[List[int]] = None,
        moods: Optional[List[str]] = None,
        country: Optional[str] = None,
        weighted: bool = True,
    ) -> str:
        """Start crate-digging from Radiooooo."""
        if not self._running:
            await self.start()
        self._cancel_crate()
        self._source_mode = SourceMode.CRATE
        self._crate_decades = decades
        self._crate_moods = moods
        self._crate_country = country
        self._crate_weighted = weighted
        self._now.source_mode = "crate"
        self._now.station_name = "Crate Digger"
        self._now.active = True

        # Dig the first track
        track = await self._dig_track()
        if not track:
            return "No tracks found for those criteria"

        await self._play_track(track)
        # Start the crate loop
        self._crate_task = asyncio.create_task(self._crate_loop())
        self._notify_state_change()
        return f"Digging: {track.display}"

    async def play_local(self, path: str) -> str:
        """Play a local file or directory."""
        if not self._running:
            await self.start()
        self._cancel_crate()
        self._source_mode = SourceMode.LOCAL
        self._now.source_mode = "local"
        self._now.station_name = ""
        self._now.active = True

        p = Path(path).expanduser().resolve()
        if p.is_dir():
            # Queue all audio files
            exts = {".mp3", ".flac", ".ogg", ".wav", ".m4a", ".aac", ".opus", ".wma", ".webm"}
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
            if not files:
                return f"No audio files found in {p}"
            await self._primary.loadfile(str(files[0]))
            for f in files[1:]:
                await self._primary.loadfile(str(f), mode="append")
            self._notify_state_change()
            return f"Playing {len(files)} tracks from {p.name}"
        elif p.is_file():
            await self._primary.loadfile(str(p))
            self._notify_state_change()
            return f"Playing {p.name}"
        else:
            return f"Not found: {path}"

    async def skip(self) -> str:
        """Skip to the next track."""
        if self._mic_break_active:
            await self._abort_mic_break()
        if self._source_mode == SourceMode.CRATE:
            # Signal the crate loop to advance via the skip event
            if hasattr(self, '_skip_event') and self._skip_event:
                self._skip_event.set()
            return "Skipping..."
        else:
            try:
                await self._primary.playlist_next()
                return "Skipped"
            except Exception:
                return "Nothing to skip to"

    async def toggle_pause(self) -> str:
        await self._primary.toggle_pause()
        paused = await self._primary.is_paused()
        self._now.paused = paused
        self._notify_state_change()
        return "Paused" if paused else "Playing"

    async def set_volume(self, level: float) -> str:
        await self._primary.set_volume(level)
        self._now.volume = level
        self._muted = level <= 0
        if level > 0:
            self._pre_mute_volume = level
        self._notify_state_change()
        return f"Volume: {int(level)}%"

    async def start_recording(self, path: str = "") -> str:
        """Start recording the current stream to disk."""
        if not self._running:
            return "Radio is not playing"

        if not path:
            import os
            from datetime import datetime
            rec_dir = os.path.expanduser("~/.hermes/radio/recordings")
            os.makedirs(rec_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Build a meaningful name from station + track
            name_parts = []
            station = self._now.station_name or ""
            title = self._now.title or ""
            # Skip ID-like names (short alphanumeric, no spaces)
            if station and (len(station) > 10 or " " in station):
                name_parts.append(station)
            if title and title != station:
                name_parts.append(title)
            if not name_parts:
                name_parts.append(self._source_mode.value)

            raw_name = " - ".join(name_parts)
            # Sanitize: keep alphanumeric, spaces->underscores, strip non-ASCII
            safe = "".join(c for c in raw_name if c.isascii() and (c.isalnum() or c in " -_"))
            safe = safe.strip().replace(" ", "_")[:60] or "recording"
            path = os.path.join(rec_dir, f"{ts}_{safe}.mp3")

        try:
            await self._primary.set_property("stream-record", path)
            self._recording_path = path
            try:
                from radio.log import info
                info(f"recording started: {path}")
            except Exception:
                pass
            return f"Recording to {path}"
        except Exception as e:
            return f"Recording failed: {e}"

    async def stop_recording(self) -> str:
        """Stop recording the current stream."""
        try:
            await self._primary.set_property("stream-record", "")
            path = getattr(self, '_recording_path', '')
            self._recording_path = ""
            try:
                from radio.log import info
                info(f"recording stopped: {path}")
            except Exception:
                pass
            return f"Recording saved: {path}" if path else "Recording stopped"
        except Exception as e:
            return f"Stop recording failed: {e}"

    @property
    def is_recording(self) -> bool:
        return bool(getattr(self, '_recording_path', ''))

    async def adjust_volume(self, delta: float) -> str:
        current = await self._primary.get_volume()
        new_vol = max(0, min(100, current + delta))
        return await self.set_volume(new_vol)

    async def toggle_mute(self) -> str:
        current = await self._primary.get_volume()
        if self._muted or current <= 0:
            restore = self._pre_mute_volume if self._pre_mute_volume > 0 else 50.0
            await self._primary.set_volume(restore)
            self._now.volume = restore
            self._muted = False
            self._notify_state_change()
            return "Unmuted"

        self._pre_mute_volume = current if current > 0 else (self._pre_mute_volume or 50.0)
        await self._primary.set_volume(0)
        self._now.volume = 0
        self._muted = True
        self._notify_state_change()
        return "Muted"

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    async def status(self) -> Dict[str, Any]:
        """Return current playback state as a dict."""
        if not self._running:
            return {"active": False}
        mpv_status = await self._primary.status()
        return {
            "active": True,
            "source_mode": self._source_mode.value,
            "station_name": self._now.station_name,
            **mpv_status,
        }

    def now_playing(self) -> NowPlaying:
        """Return the current NowPlaying snapshot (sync, for UI)."""
        return self._now

    def set_state_callback(self, cb: Callable) -> None:
        """Register a callback for state changes (for UI refresh)."""
        self._on_state_change = cb

    def _notify_state_change(self) -> None:
        if self._on_state_change:
            try:
                self._on_state_change()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Crate-dig internals
    # ------------------------------------------------------------------

    async def _get_radiooooo(self):
        if self._radiooooo is None:
            from radio.radiooooo import RadioooooClient
            self._radiooooo = RadioooooClient()
        return self._radiooooo

    async def _dig_track(self):
        client = await self._get_radiooooo()
        return await client.dig(
            decades=self._crate_decades,
            moods=self._crate_moods,
            country=self._crate_country,
            weighted=self._crate_weighted,
            mood_weights=self._mood_weights,
            country_weights=self._country_weights,
            decade_weights=self._decade_weights,
        )

    async def _play_track(self, track) -> None:
        """Play a Radiooooo track and update now-playing state."""
        url = track.audio_url or track.audio_url_ogg
        if not url:
            logger.warning("Track has no audio URL: %s", track.id)
            return
        await self._primary.loadfile(url)
        # Start level meter for reactive visualizer
        try:
            from radio.level_meter import start as start_meter
            start_meter(url)
        except Exception:
            pass
        # Log to radio.log
        try:
            from radio.log import play as log_play
            log_play(track.artist, track.title, track.decade, track.country, track.mood)
        except Exception:
            pass
        self._now.title = track.title
        self._now.artist = track.artist
        self._now.decade = track.decade
        self._now.country = track.country
        self._now.mood = track.mood
        self._now.paused = False

        # Add to in-memory history (for LLM context)
        self._now.recent_tracks.append({
            "title": track.title,
            "artist": track.artist,
            "decade": str(track.decade),
            "country": track.country,
            "mood": track.mood,
        })
        if len(self._now.recent_tracks) > 10:
            self._now.recent_tracks = self._now.recent_tracks[-10:]

        # Persist: history log, track download, honcho sync (all optional)
        try:
            from radio.history import log_track, save_track, sync_to_honcho
            log_track(
                artist=track.artist, title=track.title, source="crate",
                decade=track.decade, country=track.country, mood=track.mood,
                duration=track.length, url=url,
            )
            save_track(url, track.artist, track.title, track.decade, track.country, track.mood)
            sync_to_honcho({
                "artist": track.artist, "title": track.title,
                "decade": track.decade, "country": track.country, "mood": track.mood,
            })
        except Exception:
            pass

        self._notify_state_change()

    async def _crate_loop(self) -> None:
        """Background loop that keeps digging tracks."""
        try:
            while self._running and self._source_mode == SourceMode.CRATE:
                # Pre-fetch the next track while current plays
                prefetch_task = asyncio.create_task(self._dig_track())

                # Wait for track to end naturally (eof) or user skip.
                # Two signals: end-file from mpv, or _skip_event from skip().
                end_event = asyncio.Event()
                self._skip_event = asyncio.Event()
                load_time = time.monotonic()

                def on_end(data):
                    reason = data.get("reason", "")
                    if reason not in ("eof", "error"):
                        return  # ignore "stop" -- only natural endings
                    # Debounce: ignore events within 2s of load (loadfile cascade)
                    if time.monotonic() - load_time < 2.0:
                        return
                    end_event.set()

                self._primary.on("end-file", on_end)

                # Wait for either natural end or skip
                wait_end = asyncio.create_task(end_event.wait())
                wait_skip = asyncio.create_task(self._skip_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        [wait_end, wait_skip],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                finally:
                    if on_end in self._primary._event_callbacks.get("end-file", []):
                        self._primary._event_callbacks["end-file"].remove(on_end)
                    self._skip_event = None

                if not self._running or self._source_mode != SourceMode.CRATE:
                    prefetch_task.cancel()
                    break

                # Get the prefetched track
                try:
                    next_track = await prefetch_task
                except asyncio.CancelledError:
                    break

                if not next_track:
                    # Retry with fresh params
                    next_track = await self._dig_track()
                    if not next_track:
                        logger.warning("Crate dig exhausted, stopping")
                        break

                # Play the next track first, then mic break over it
                await self._play_track(next_track)

                # Short delay to let playback start before mic break
                self._tracks_since_break += 1
                if (
                    self._auto_mic_breaks
                    and self._tracks_since_break >= self._break_every_n
                ):
                    await asyncio.sleep(2.0)  # let the music establish
                    await self._do_mic_break(next_track)
                    self._tracks_since_break = 0

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Crate loop error")

    def _cancel_crate(self) -> None:
        if self._crate_task and not self._crate_task.done():
            self._crate_task.cancel()
            self._crate_task = None

    # ------------------------------------------------------------------
    # Mic breaks
    # ------------------------------------------------------------------

    async def mic_break(self, text: Optional[str] = None) -> str:
        """Trigger a mic break.  If text is given, use it.  Otherwise auto-generate."""
        if not self._running:
            return "Radio is not playing"
        if self._mic_break_active:
            return "Mic break already in progress"

        if text:
            await self._speak(text)
            return "Mic break done"
        else:
            await self._do_mic_break()
            return "Mic break done"

    async def _do_mic_break(self, upcoming_track=None) -> None:
        """Generate and play a mic break with volume ducking."""
        if self._mic_break_active:
            return

        self._mic_break_active = True
        try:
            # Generate commentary
            commentary = await self._generate_commentary(upcoming_track)
            if not commentary:
                return

            audio_path = await self._render_tts(commentary)
            if audio_path:
                await self._speak_audio(audio_path)

            # Save mic break (optional, gated on config)
            try:
                from radio.history import log_mic_break, save_mic_break
                log_mic_break(
                    commentary=commentary,
                    audio_path=audio_path,
                    track_artist=self._now.artist,
                    track_title=self._now.title,
                )
                save_mic_break(commentary, audio_path)
            except Exception:
                pass
        except Exception:
            logger.exception("Mic break error")
        finally:
            self._mic_break_active = False

    async def _speak(self, text: str) -> None:
        """TTS render + volume duck + play on voice instance."""
        audio_path = await self._render_tts(text)
        if not audio_path:
            return
        await self._speak_audio(audio_path)

    async def _speak_audio(self, audio_path: str) -> None:
        """Play a pre-rendered audio file with volume ducking."""
        # Start voice mpv if needed
        if not self._voice.running:
            await self._voice.start()

        # Duck primary volume
        await self._primary.ramp_volume(
            self._duck_volume,
            duration_ms=self._duck_ramp_ms,
        )

        # Play the TTS clip at reduced volume (DJ shouldn't overpower music)
        await self._voice.set_volume(65)
        await self._voice.loadfile(audio_path)

        # Wait for the voice clip to finish
        voice_end = asyncio.Event()

        def on_voice_end(data):
            voice_end.set()

        self._voice.on("end-file", on_voice_end)
        try:
            await asyncio.wait_for(voice_end.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass
        finally:
            if on_voice_end in self._voice._event_callbacks.get("end-file", []):
                self._voice._event_callbacks["end-file"].remove(on_voice_end)

        # Restore primary volume
        await self._primary.ramp_volume(
            self._now.volume,
            duration_ms=self._duck_ramp_ms,
        )

    async def _abort_mic_break(self) -> None:
        """Immediately stop a mic break and restore volume."""
        if self._voice.running:
            await self._voice.stop()
        await self._primary.set_volume(self._now.volume)
        self._mic_break_active = False

    async def _render_tts(self, text: str) -> Optional[str]:
        """Render text to an audio file using Hermes' TTS tool."""
        try:
            from tools.tts_tool import text_to_speech_tool
            result_json = text_to_speech_tool(text=text)
            result = json.loads(result_json)
            if result.get("success"):
                return result["file_path"]
            else:
                logger.warning("TTS failed: %s", result.get("error"))
                return None
        except ImportError:
            logger.warning("TTS tool not available")
            return None
        except Exception:
            logger.exception("TTS render error")
            return None

    async def _generate_commentary(self, upcoming_track=None) -> Optional[str]:
        """Generate DJ commentary via the configured LLM."""
        now = self._now
        hour = time.localtime().tm_hour
        if hour < 6:
            time_vibe = "late night"
        elif hour < 12:
            time_vibe = "morning"
        elif hour < 18:
            time_vibe = "afternoon"
        else:
            time_vibe = "evening"

        # Build context
        current = f"{now.artist} - {now.title}" if now.title else "unknown"
        history = "; ".join(
            f"{t['artist']} - {t['title']} ({t['decade']}s, {t['country']})"
            for t in now.recent_tracks[-5:]
        )

        upcoming_info = ""
        if upcoming_track:
            upcoming_info = f"\nComing up next: {upcoming_track.artist} - {upcoming_track.title} ({upcoming_track.decade}s, {upcoming_track.country}, {upcoming_track.mood})"

        persona_prompts = {
            "encyclopedic": "You're a deeply knowledgeable music historian DJ. Share fascinating context about the music -- provenance, cultural significance, recording history, the label, the scene.",
            "deadpan": "You're a dry, sardonic late-night DJ. Minimal words, maximum effect. Understated observations.",
            "enthusiastic": "You're an infectiously excited college radio DJ discovering music for the first time. Genuinely thrilled.",
            "conspiratorial": "You're a paranoid late-night DJ who sees hidden connections between every track. Everything is linked.",
        }
        persona = persona_prompts.get(self._mic_break_persona, persona_prompts["encyclopedic"])

        prompt = f"""{persona}

You're doing a mic break on Hermes Radio. It's {time_vibe}. Keep it to 1-3 sentences. Conversational, not scripted. Never mention being an AI. No hashtags.

Just played: {current}
Recent history: {history}{upcoming_info}
Source: {now.source_mode}"""

        # Use Hermes' auxiliary LLM client
        try:
            from agent.auxiliary_client import call_llm
            messages = [{"role": "user", "content": prompt}]
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: call_llm(
                    task="radio",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.9,
                    timeout=10.0,
                ),
            )
            text = response.choices[0].message.content
            return text.strip() if text else None
        except ImportError:
            pass
        except Exception:
            logger.debug("LLM commentary generation failed, using template")

        # Fallback: simple template
        if upcoming_track:
            return f"That was {now.artist}. Coming up, {upcoming_track.artist} from the {upcoming_track.decade}s."
        return f"You're listening to Hermes Radio. That was {now.artist} -- {now.title}."

    # ------------------------------------------------------------------
    # mpv event handlers
    # ------------------------------------------------------------------

    async def _poll_state(self) -> None:
        """Background task that updates NowPlaying from mpv every ~300ms."""
        try:
            while self._running:
                try:
                    if self._primary.running and not await self._primary.is_idle():
                        self._now.position = await self._primary.get_position()
                        self._now.duration = await self._primary.get_duration()
                        self._now.volume = await self._primary.get_volume()
                        self._muted = self._now.volume <= 0

                        # Poll ICY metadata for streams (catches title changes)
                        if self._source_mode == SourceMode.STREAM:
                            try:
                                title = await self._primary.get_media_title()
                                last_raw = getattr(self, '_last_icy_raw', '')
                                if title and title != last_raw and title not in ("channel.mp3",):
                                    self._last_icy_raw = title
                                    # For streams: store full ICY as title, no split
                                    self._now.title = title
                                    self._now.artist = ""
                            except Exception:
                                pass
                except Exception:
                    pass
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass

    def _on_track_end(self, data: dict) -> None:
        """Handle track-end events from primary mpv."""
        pass  # The crate loop handles this via its own listener

    def _on_metadata_update(self, data: dict) -> None:
        """Handle metadata changes (ICY updates from live streams)."""
        if self._source_mode == SourceMode.STREAM:
            asyncio.create_task(self._refresh_stream_metadata())

    async def _refresh_stream_metadata(self) -> None:
        """Pull fresh metadata from mpv after an ICY update."""
        try:
            title = await self._primary.get_media_title()
            if title and title != self._now.title:
                self._now.title = title
                # Try to split "Artist - Title" format
                if " - " in title:
                    parts = title.split(" - ", 1)
                    self._now.artist = parts[0].strip()
                    self._now.title = parts[1].strip()
                else:
                    self._now.artist = ""
                self._notify_state_change()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def configure(self, config: dict) -> None:
        """Apply radio config from ~/.hermes/config.yaml."""
        radio_cfg = config.get("radio", {})
        mic_cfg = radio_cfg.get("mic_breaks", {})
        crate_cfg = radio_cfg.get("crate", {})

        self._now.volume = radio_cfg.get("default_volume", 55)
        self._auto_mic_breaks = mic_cfg.get("enabled", True)
        self._mic_break_persona = mic_cfg.get("persona", "encyclopedic")
        self._duck_volume = mic_cfg.get("duck_volume", 20)
        self._duck_ramp_ms = mic_cfg.get("duck_ramp_ms", 800)

        freq = mic_cfg.get("frequency", "every_track")
        if freq == "every_track":
            self._break_every_n = 1
        elif freq == "manual":
            self._auto_mic_breaks = False
        else:
            self._break_every_n = mic_cfg.get("interval", 3)

        self._crate_decades = crate_cfg.get("decades")
        self._crate_moods = crate_cfg.get("moods")
        self._crate_country = None
        self._crate_weighted = crate_cfg.get("weighted", True)

        # Weight overrides from config
        mood_w = crate_cfg.get("mood_weights")
        if mood_w and isinstance(mood_w, dict):
            self._mood_weights = {str(k): float(v) for k, v in mood_w.items()}

        country_w = crate_cfg.get("country_weights")
        if country_w and isinstance(country_w, dict):
            self._country_weights = {str(k).upper(): float(v) for k, v in country_w.items()}

        decade_w = crate_cfg.get("decade_weights")
        if decade_w and isinstance(decade_w, dict):
            self._decade_weights = {int(k): float(v) for k, v in decade_w.items()}


def check_radio_available() -> bool:
    """Check if mpv is installed."""
    return shutil.which("mpv") is not None

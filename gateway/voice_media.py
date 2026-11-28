"""Voice-channel media playback — the shared seam that routes ``MEDIA:`` audio into a live
voice channel instead of a text-channel attachment.

Symptom this fixes: while the bot is connected in a Discord voice channel (``/voice join``),
a reply carrying ``MEDIA:/path/to/audio.mp3`` still lands as a file attachment in the linked
text channel — only auto-TTS speech ever plays in the VC. Users ask for exactly the opposite
("play me a song in the voice channel"), and every audio MEDIA tag during a live session is
delivered where nobody is listening.

Design:
* The delivery core (``gateway/platforms/base.py``) calls :func:`route_media_attachments` with
  the turn's adapter, event and extracted media list BEFORE attachments are dispatched.
* An adapter opts in by implementing ``resolve_voice_playback_guild(event) -> Optional[int]``
  (membership proof is platform-specific; Discord keys it on the guild whose bound text channel
  IS this turn's chat — same proof as ``play_tts``). Adapters without the hook are untouched.
* Audio-extension local files for an opted-in, VC-connected guild are queued per guild and
  played sequentially through the adapter's ``play_voice_media(guild_id, path)``; everything
  else keeps the upstream attachment path unchanged. A failed/absent playback seam falls back
  to the file attachment rather than dropping the media.
* Config gate ``discord.voice_media_playback`` (default on): when off, audio keeps upstream
  behavior (file attachment with inline player).

The per-guild queue is module-level on purpose: MEDIA tags arrive from multiple delivery
paths (non-streaming chain, post-stream notification pass, session ingress) and all of them
must serialize onto one voice stream, not interleave packets.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Iterable, Optional

logger = logging.getLogger("gateway.voice_media")

# Audio extensions playable in a VC mixer (mirrors base._AUDIO_EXTS). A MEDIA: URL tag is never
# local — it stays on the upstream attachment path and is never handed to a mixer.
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".m2a", ".wav", ".m4a", ".flac"})

# guild_id -> [(adapter, path), ...] and one sequential worker task per guild.
_QUEUES: dict[int, list[tuple[Any, str]]] = {}
_QUEUE_WORKERS: dict[int, asyncio.Task] = {}
_LAST_ADAPTER: dict[int, Any] = {}  # guild -> adapter that last played (so stop() can cut the clip)
_BG_TASKS: set = set()  # strong refs to fire-and-forget tasks (asyncio GC pitfall)


def voice_media_playback_enabled(platform: str = "discord") -> bool:
    """Config gate ``<platform>.voice_media_playback`` (default True). Read per turn so a config
    change (or ``/voice media on|off``) applies without a gateway restart."""
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config() or {}
        raw = (cfg.get(platform) or {}).get("voice_media_playback", True)
        if isinstance(raw, str):
            return raw.strip().lower() not in ("false", "0", "no", "off")
        return bool(raw)
    except Exception:
        return True


def _is_local_audio(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return (
        ext in _AUDIO_EXTS
        and not path.startswith(("http://", "https://"))
        and os.path.isfile(path)
    )


def route_media_attachments(adapter: Any, event: Any, media_files: list) -> list:
    """Split ``media_files`` into what still goes as attachments vs what plays in the VC.

    Returns the kept (attachment) list; audio routed to the voice channel is removed from it.
    Never raises: on any internal problem the full upstream attachment path stays intact.
    """
    try:
        if not media_files:
            return media_files
        resolve = getattr(adapter, "resolve_voice_playback_guild", None)
        play = getattr(adapter, "play_voice_media", None)
        if not (callable(resolve) and callable(play)):
            return media_files
        guild_id_raw = resolve(event)
        if guild_id_raw is None:
            return media_files
        try:
            guild_id = int(guild_id_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return media_files
        platform = getattr(getattr(event, "source", None), "platform", None) or getattr(
            adapter, "name", "discord"
        )
        if guild_id is None or not voice_media_playback_enabled(str(platform)):
            # Not playing in a VC: force file-attachment routing for audio entries so they ride
            # the platform's regular upload path (native voice-message senders often demand
            # Opus and fail on mp3 — an attachment with inline player always works).
            return [(p, False) if _is_local_audio(p) else (p, v) for p, v in media_files]
        keep, played = [], []
        for entry in media_files:
            path = entry[0]
            if _is_local_audio(path):
                played.append(path)
            else:
                keep.append(entry)
        if not played:
            return media_files
        enqueue_voice_media(guild_id, adapter, played)
        return keep
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("voice_media: routing error, upstream attachment path intact: %s", exc)
        return media_files


def enqueue_voice_media(guild_id: int, adapter: Any, paths: Iterable[str]) -> None:
    """Append to this guild's sequential queue; spawn the drain worker if idle."""
    _QUEUES.setdefault(guild_id, []).extend((adapter, p) for p in paths)
    worker = _QUEUE_WORKERS.get(guild_id)
    if worker is None or worker.done():
        task = asyncio.ensure_future(_queue_worker(guild_id))
        _QUEUE_WORKERS[guild_id] = task
        _BG_TASKS.add(task)
        task.add_done_callback(_BG_TASKS.discard)


async def _queue_worker(guild_id: int) -> None:
    """Drain this guild's queue sequentially; a stop() clears the rest."""
    try:
        while _QUEUES.get(guild_id):
            adapter, path = _QUEUES[guild_id].pop(0)
            _LAST_ADAPTER[guild_id] = adapter
            try:
                ok = await adapter.play_voice_media(guild_id, path)
                logger.info(
                    "voice_media: played MEDIA audio in voice channel (guild=%s, %s, ok=%s)",
                    guild_id, path, ok,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("voice_media: voice playback failed for %s (%s)", path, exc)
    finally:
        _QUEUE_WORKERS.pop(guild_id, None)


def stop_voice_media(guild_id: int) -> int:
    """Cut the in-flight clip and drop queued files for this guild. Returns dropped count."""
    dropped = len(_QUEUES.pop(guild_id, []))
    worker = _QUEUE_WORKERS.get(guild_id)
    if worker is not None and not worker.done():
        worker.cancel()  # the blocked play_voice_media returns; the drain loop stops
    try:
        adapter = _LAST_ADAPTER.get(guild_id)
        stopper = getattr(adapter, "stop_voice_playback", None) if adapter is not None else None
        if callable(stopper):
            stopper(guild_id)
    except Exception as exc:
        logger.debug("voice_media: direct stop unavailable: %s", exc)
    return dropped


def voice_media_status(guild_id: Optional[int]) -> str:
    """One-line status for ``/voice media`` / ``/voice status``."""
    state = "on" if voice_media_playback_enabled() else "off"
    queued = len(_QUEUES.get(int(guild_id), [])) if guild_id is not None else 0
    playing = int(guild_id) in _QUEUE_WORKERS if guild_id is not None else False
    line = f"Voice media playback: {state}"
    if playing:
        line += " (playing now)"
    if queued:
        line += f" ({queued} queued)"
    return line

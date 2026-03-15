"""Radio listening history and track/mic break archival.

All features are optional and gated on config flags:
  ~/.hermes/radio/config.yaml:
    history: true        # log tracks to history.jsonl
    save_tracks: true    # download MP3s to tracks/
    save_mic_breaks: true # save commentary text + audio to mic_breaks/
    honcho_sync: true    # push listening sessions to Honcho

Directory structure:
  ~/.hermes/radio/
    config.yaml
    history.jsonl          # one JSON object per line
    tracks/                # downloaded MP3s with metadata
    mic_breaks/            # commentary text + audio files
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RADIO_DIR = Path(os.path.expanduser("~/.hermes/radio"))
HISTORY_FILE = RADIO_DIR / "history.jsonl"
TRACKS_DIR = RADIO_DIR / "tracks"
MIC_BREAKS_DIR = RADIO_DIR / "mic_breaks"


def _ensure_dirs():
    RADIO_DIR.mkdir(parents=True, exist_ok=True)


def _config_flag(key: str, default: bool = False) -> bool:
    """Read a boolean flag from radio config."""
    try:
        from radio.config import load
        return load().get(key, default)
    except Exception:
        return default


# -- History ----------------------------------------------------------------

def log_track(
    artist: str,
    title: str,
    source: str = "",
    decade: int = 0,
    country: str = "",
    mood: str = "",
    duration: float = 0,
    station_name: str = "",
    url: str = "",
) -> None:
    """Append a track entry to history.jsonl."""
    if not _config_flag("history"):
        return

    _ensure_dirs()

    entry = {
        "type": "track",
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
        "artist": artist,
        "title": title,
        "source": source,
        "decade": decade,
        "country": country,
        "mood": mood,
        "duration": duration,
        "station_name": station_name,
        "url": url,
    }

    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to write history", exc_info=True)


def log_mic_break(
    commentary: str,
    audio_path: Optional[str] = None,
    track_artist: str = "",
    track_title: str = "",
) -> None:
    """Log a mic break to history."""
    if not _config_flag("history"):
        return

    _ensure_dirs()

    entry = {
        "type": "mic_break",
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
        "commentary": commentary,
        "audio_path": audio_path,
        "track_artist": track_artist,
        "track_title": track_title,
    }

    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to write mic break history", exc_info=True)


def log_station(
    station_name: str,
    url: str,
    source: str = "",
) -> None:
    """Log a station tune-in to history."""
    if not _config_flag("history"):
        return

    _ensure_dirs()

    entry = {
        "type": "station",
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
        "station_name": station_name,
        "url": url,
        "source": source,
    }

    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to write station history", exc_info=True)


def get_history(limit: int = 50) -> list:
    """Read the last N history entries."""
    if not HISTORY_FILE.exists():
        return []
    try:
        lines = HISTORY_FILE.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines if line.strip()]
        return entries[-limit:]
    except Exception:
        return []


# -- Track saving -----------------------------------------------------------

def save_track(url: str, artist: str, title: str, decade: int = 0, country: str = "", mood: str = "") -> Optional[str]:
    """Download and save a track MP3 to ~/.hermes/radio/tracks/.

    Returns the saved file path, or None if saving is disabled/failed.
    """
    if not _config_flag("save_tracks"):
        return None

    _ensure_dirs()
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)

    # Build filename
    safe_artist = _safe_filename(artist)
    safe_title = _safe_filename(title)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{date_str}_{safe_artist}_{safe_title}.mp3"
    filepath = TRACKS_DIR / filename

    if filepath.exists():
        return str(filepath)

    try:
        import httpx
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)

        logger.info("Saved track: %s", filepath)

        # Try to tag with ffmpeg (non-blocking, best-effort)
        _tag_track(filepath, artist, title, decade, country, mood)

        return str(filepath)
    except Exception:
        logger.debug("Failed to save track", exc_info=True)
        return None


def _tag_track(filepath: Path, artist: str, title: str, decade: int, country: str, mood: str) -> None:
    """Add ID3 metadata to an MP3 file via ffmpeg."""
    if not shutil.which("ffmpeg"):
        return

    import subprocess
    tagged = filepath.with_suffix(".tagged.mp3")
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "quiet",
        "-i", str(filepath),
        "-metadata", f"artist={artist}",
        "-metadata", f"title={title}",
        "-metadata", f"album=Radiooooo {country} {decade}s",
        "-metadata", f"genre={mood}",
        "-metadata", f"comment=Hermes Radio crate dig",
        "-codec", "copy",
        str(tagged),
    ]
    try:
        subprocess.run(args, timeout=15, check=True)
        tagged.replace(filepath)
    except Exception:
        if tagged.exists():
            tagged.unlink()


# -- Mic break saving -------------------------------------------------------

def save_mic_break(commentary: str, audio_path: Optional[str] = None) -> Optional[str]:
    """Save mic break commentary and audio to ~/.hermes/radio/mic_breaks/.

    Returns the saved text file path, or None.
    """
    if not _config_flag("save_mic_breaks"):
        return None

    _ensure_dirs()
    MIC_BREAKS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = MIC_BREAKS_DIR / f"{ts}.txt"

    try:
        text_path.write_text(commentary, encoding="utf-8")

        # Copy audio file alongside the text
        if audio_path and os.path.exists(audio_path):
            ext = Path(audio_path).suffix or ".mp3"
            audio_dest = MIC_BREAKS_DIR / f"{ts}{ext}"
            shutil.copy2(audio_path, audio_dest)

        return str(text_path)
    except Exception:
        logger.debug("Failed to save mic break", exc_info=True)
        return None


# -- Honcho sync ------------------------------------------------------------

def sync_to_honcho(track_info: Dict[str, Any]) -> None:
    """Push a track play event to Honcho for cross-session recall."""
    if not _config_flag("honcho_sync"):
        return

    try:
        from tools.honcho_tools import honcho_conclude_tool
        summary = (
            f"Listened to {track_info.get('artist', '?')} - {track_info.get('title', '?')} "
            f"({track_info.get('decade', '')}s, {track_info.get('country', '')}, {track_info.get('mood', '')})"
        )
        honcho_conclude_tool({"conclusion": summary})
    except Exception:
        logger.debug("Honcho sync failed", exc_info=True)


# -- Helpers ----------------------------------------------------------------

def _safe_filename(s: str) -> str:
    """Make a string safe for use as a filename."""
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-")
    s = "".join(c for c in s if c.isalnum() or c in "-_ ")
    return s.strip()[:60] or "unknown"

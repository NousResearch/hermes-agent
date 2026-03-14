"""Audio level meter via ffmpeg sidecar.

Runs a lightweight ffmpeg process that reads the same audio source as mpv
and extracts per-frame RMS levels using the astats+ametadata filter chain.
Writes levels to a shared list that the mini player reads.

Uses no audio output (-f null) so it doesn't interfere with mpv playback.
The bandwidth cost is minimal for most streams (MP3/OGG are small).
"""

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)

# Shared state: ring buffer of recent RMS levels (dB, negative values)
# Read by mini_player.py, written by the meter thread.
_levels: Deque[float] = deque(maxlen=64)
_lock = threading.Lock()
_process: Optional[subprocess.Popen] = None
_thread: Optional[threading.Thread] = None
_running = False
_current_url = ""

# Regex to extract RMS level from ffmpeg ametadata output
_RMS_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=([-\d.]+)")
# Also grab per-channel for stereo spread
_CH1_RE = re.compile(r"lavfi\.astats\.1\.RMS_level=([-\d.]+)")
_CH2_RE = re.compile(r"lavfi\.astats\.2\.RMS_level=([-\d.]+)")


def start(url: str) -> None:
    """Start the level meter for the given audio URL/path."""
    global _process, _thread, _running, _current_url

    if not shutil.which("ffmpeg"):
        logger.debug("ffmpeg not found, level meter disabled")
        return

    # Don't restart if already metering the same URL
    if _running and _current_url == url:
        return

    stop()

    _current_url = url
    _running = True
    _thread = threading.Thread(target=_meter_loop, args=(url,), daemon=True, name="radio-level-meter")
    _thread.start()


def stop() -> None:
    """Stop the level meter."""
    global _process, _thread, _running, _current_url

    _running = False
    _current_url = ""

    if _process and _process.poll() is None:
        try:
            _process.kill()
            _process.wait(timeout=2)
        except Exception:
            pass
        _process = None

    with _lock:
        _levels.clear()


def get_levels(n: int = 12) -> List[float]:
    """Get the last N RMS levels as normalized values in [0.0, 1.0].

    Returns fewer than N values if not enough data yet.
    """
    with _lock:
        raw = list(_levels)[-n:] if _levels else []

    # Convert dB (typically -60 to 0) to 0.0-1.0 range
    result = []
    for db in raw:
        # Clamp to [-60, 0] range then normalize
        db = max(-60.0, min(0.0, db))
        # Map -60..0 to 0..1 with slight curve for better visual dynamics
        normalized = (db + 60.0) / 60.0
        normalized = normalized ** 0.7  # slight compression for livelier bars
        result.append(normalized)

    return result


def is_active() -> bool:
    return _running and _process is not None and _process.poll() is None


def _meter_loop(url: str) -> None:
    """Background thread: run ffmpeg and parse RMS levels."""
    global _process

    try:
        _process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "quiet",
                "-i", url,
                "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=/dev/stdout",
                "-f", "null",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line buffered
        )

        logger.info("Level meter started for %s (pid %d)", url[:60], _process.pid)

        while _running and _process.poll() is None:
            line = _process.stdout.readline()
            if not line:
                break

            m = _RMS_RE.search(line)
            if m:
                try:
                    level = float(m.group(1))
                    with _lock:
                        _levels.append(level)
                except ValueError:
                    pass

    except Exception:
        logger.debug("Level meter error", exc_info=True)
    finally:
        if _process and _process.poll() is None:
            try:
                _process.kill()
                _process.wait(timeout=2)
            except Exception:
                pass
        logger.debug("Level meter stopped")

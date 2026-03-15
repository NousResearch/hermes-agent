"""Audio level meter via ffmpeg sidecar.

Runs a lightweight ffmpeg process that reads the same audio source as mpv
and extracts per-frame RMS levels using the astats+ametadata filter chain.
Writes levels to a shared list that the mini player reads.

Uses no audio output (-f null) so it doesn't interfere with mpv playback.
The bandwidth cost is minimal for most streams (MP3/OGG are small).
"""

from dataclasses import dataclass
import logging
import os
import re
import shutil
import subprocess
import threading
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


@dataclass
class VisualizerFeatures:
    levels: List[float]
    energy: float
    peak: float
    transient: float
    motion: float
    decay: float
    active: bool


def _normalize_db(db: float) -> float:
    """Convert RMS dB (roughly -60..0) to [0,1] with a mild lift."""
    db = max(-60.0, min(0.0, db))
    normalized = (db + 60.0) / 60.0
    return normalized ** 0.7


def _resample(values: List[float], width: int) -> List[float]:
    """Resample values to width using simple linear interpolation."""
    width = max(1, width)
    if not values:
        return [0.0] * width
    if len(values) == 1:
        return [values[0]] * width
    if len(values) == width:
        return list(values)

    src_last = len(values) - 1
    out: List[float] = []
    for i in range(width):
        pos = i * src_last / max(1, width - 1)
        left = int(pos)
        right = min(src_last, left + 1)
        frac = pos - left
        sample = values[left] * (1.0 - frac) + values[right] * frac
        out.append(sample)
    return out


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

    result = []
    for db in raw:
        result.append(_normalize_db(db))

    return result


def get_feature_snapshot(width: int, *, smoothing: float = 0.0) -> VisualizerFeatures:
    """Return a compact feature snapshot for terminal visualizer rendering."""
    width = max(1, width)

    if not is_active():
        return VisualizerFeatures(
            levels=[0.0] * width,
            energy=0.0,
            peak=0.0,
            transient=0.0,
            motion=0.0,
            decay=0.0,
            active=False,
        )

    with _lock:
        raw_db = list(_levels)

    if not raw_db:
        return VisualizerFeatures(
            levels=[0.0] * width,
            energy=0.0,
            peak=0.0,
            transient=0.0,
            motion=0.0,
            decay=0.0,
            active=False,
        )

    normalized = [_normalize_db(db) for db in raw_db]
    levels = _resample(normalized, width)

    if smoothing > 0.0:
        smoothed: List[float] = []
        prev = levels[0]
        alpha = max(0.0, min(1.0, smoothing))
        for value in levels:
            prev = prev * alpha + value * (1.0 - alpha)
            smoothed.append(prev)
        levels = smoothed

    recent = normalized[-min(4, len(normalized)):]
    previous = recent[:-1]
    last = recent[-1]
    previous_mean = sum(previous) / len(previous) if previous else last
    diffs = [abs(b - a) for a, b in zip(levels, levels[1:])]

    return VisualizerFeatures(
        levels=levels,
        energy=sum(recent) / len(recent),
        peak=max(recent),
        transient=max(0.0, last - previous_mean),
        motion=(sum(diffs) / len(diffs)) if diffs else 0.0,
        decay=max(0.0, previous_mean - last),
        active=True,
    )


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

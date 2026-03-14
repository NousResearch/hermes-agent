"""Mini player widget for the Hermes CLI.

Provides a prompt_toolkit FormattedTextControl that renders a compact
now-playing bar below the input area.  Only visible when the radio is active.

The visualizer uses playback-position-seeded noise with smooth attack/decay
and peak hold, giving the appearance of audio-reactive bars.  Different
tracks produce different patterns because the seed incorporates the track
title.
"""

import hashlib
import math
import time
from typing import List, Tuple

# Block elements: 8 height levels
_BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_NUM_BARS = 12

# State for smooth animation (persists between renders)
_bar_levels = [0.0] * _NUM_BARS   # current smoothed level (0.0-1.0)
_peak_levels = [0.0] * _NUM_BARS  # peak hold level
_peak_decay = [0.0] * _NUM_BARS   # time since peak was set
_last_title = ""
_last_render = 0.0


def _noise(seed: int, idx: int) -> float:
    """Fast deterministic noise in [0, 1) from seed + index."""
    h = hashlib.md5(f"{seed}:{idx}".encode()).digest()
    return ((h[0] << 8) | h[1]) / 65536.0


def _generate_bars(position: float, title: str, paused: bool) -> str:
    """Generate animated bar characters.

    Uses real audio RMS levels from the ffmpeg sidecar when available,
    falling back to position-seeded noise.
    """
    global _bar_levels, _peak_levels, _peak_decay, _last_title, _last_render

    if paused:
        for i in range(_NUM_BARS):
            _bar_levels[i] *= 0.85
            _peak_levels[i] *= 0.9
        bars = []
        for i in range(_NUM_BARS):
            level = int(_bar_levels[i] * 7)
            bars.append(_BLOCKS[max(0, min(8, level))])
        return "".join(bars)

    # Try real audio levels first
    try:
        from radio.level_meter import get_levels, is_active
        if is_active():
            real_levels = get_levels(_NUM_BARS)
            if len(real_levels) >= 3:
                return _render_real_levels(real_levels)
    except ImportError:
        pass

    # Reset state when track changes
    if title != _last_title:
        _last_title = title
        _bar_levels = [0.0] * _NUM_BARS
        _peak_levels = [0.0] * _NUM_BARS
        _peak_decay = [0.0] * _NUM_BARS

    now = time.time()
    dt = min(now - _last_render, 0.5) if _last_render > 0 else 0.3
    _last_render = now

    # Derive a track-specific seed from the title
    title_seed = int(hashlib.md5((title or "x").encode()).hexdigest()[:8], 16)

    # Use position to create time-varying noise at different rates per bar
    # Lower bars = lower freq (slower variation), higher bars = higher freq
    pos = position if position and position > 0 else now

    for i in range(_NUM_BARS):
        # Each bar samples noise at a different rate
        # Lower indices = bass (slow), higher = treble (fast)
        freq = 1.5 + i * 0.8  # frequency multiplier
        phase = title_seed + i * 137  # phase offset per bar

        # Multi-octave noise for organic feel
        val = 0.0
        val += _noise(int(pos * freq) + phase, i) * 0.5
        val += _noise(int(pos * freq * 2.7) + phase + 1000, i) * 0.3
        val += _noise(int(pos * freq * 6.1) + phase + 2000, i) * 0.2

        # Shape: center-heavy (mids louder, like real music)
        center_boost = 0.6 + 0.4 * (1.0 - abs(i - _NUM_BARS / 2) / (_NUM_BARS / 2))
        val *= center_boost

        # Overall energy envelope -- slow sine wave so bars breathe
        energy = 0.6 + 0.4 * math.sin(pos * 0.4 + title_seed * 0.001)
        val *= energy

        # Boost everything up -- we want lively bars
        val = val * 1.6 + 0.15

        # Smooth attack/decay
        attack = 8.0   # fast attack
        decay = 3.0     # slower decay
        if val > _bar_levels[i]:
            _bar_levels[i] += (val - _bar_levels[i]) * min(1.0, attack * dt)
        else:
            _bar_levels[i] += (val - _bar_levels[i]) * min(1.0, decay * dt)

        # Peak hold with decay
        if _bar_levels[i] > _peak_levels[i]:
            _peak_levels[i] = _bar_levels[i]
            _peak_decay[i] = 0.0
        else:
            _peak_decay[i] += dt
            if _peak_decay[i] > 0.8:  # hold for 0.8s then decay
                _peak_levels[i] *= 0.92

    # Render bars
    bars = []
    for i in range(_NUM_BARS):
        level = int(_bar_levels[i] * 8)
        level = max(1, min(8, level))  # at least 1 when playing
        bars.append(_BLOCKS[level])

    return "".join(bars)


def _render_real_levels(levels: List[float]) -> str:
    """Render bars from real normalized audio levels [0.0-1.0].

    Each bar reads from a different time offset in the level history,
    creating a scrolling waveform effect. The most recent level drives
    the rightmost bar; older levels scroll left.
    """
    global _bar_levels, _last_render

    now = time.time()
    dt = min(now - _last_render, 0.5) if _last_render > 0 else 0.3
    _last_render = now

    n = len(levels)

    for i in range(_NUM_BARS):
        # Each bar reads from a staggered time position
        # Rightmost bar = most recent, leftmost = oldest
        idx = max(0, n - _NUM_BARS + i)
        if idx < n:
            val = levels[idx]
        else:
            val = levels[-1] if levels else 0.0

        # Add slight per-bar variation using hash of position
        # This creates visual spread even when all levels are similar
        jitter = _noise(int(now * 3), i) * 0.15
        val = max(0.0, min(1.0, val + jitter - 0.07))

        # Center-boost
        center = 1.0 - abs(i - _NUM_BARS / 2) / (_NUM_BARS / 2) * 0.25
        val *= center

        # Smooth attack/decay
        attack = 15.0
        decay = 5.0
        if val > _bar_levels[i]:
            _bar_levels[i] += (val - _bar_levels[i]) * min(1.0, attack * dt)
        else:
            _bar_levels[i] += (val - _bar_levels[i]) * min(1.0, decay * dt)

    bars = []
    for i in range(_NUM_BARS):
        level = int(_bar_levels[i] * 8)
        level = max(1, min(8, level))
        bars.append(_BLOCKS[level])

    return "".join(bars)


def get_mini_player_text() -> List[Tuple[str, str]]:
    """Return styled text fragments for the mini player bar.

    Returns an empty list when the radio is inactive (widget collapses to 0 height).
    """
    try:
        from radio.player import HermesRadio
        if not HermesRadio.active():
            return []

        now = HermesRadio.get().now_playing()
        if not now.active:
            return []
    except Exception:
        return []

    fragments: List[Tuple[str, str]] = []

    # Animated visualizer bars (driven by playback position)
    bars = _generate_bars(
        position=now.position or 0.0,
        title=f"{now.artist}-{now.title}",
        paused=now.paused,
    )
    fragments.append(("class:radio-bars", f"  {bars} "))

    # Artist - Title
    display = ""
    if now.artist and now.artist != "Unknown":
        display = f"{now.artist} \u2014 {now.title}" if now.title else now.artist
    elif now.title:
        display = now.title
    else:
        display = now.station_name or "..."

    # Truncate if too long
    if len(display) > 38:
        display = display[:35] + "..."

    fragments.append(("class:radio-title", display))

    # Tags (crate mode)
    if now.source_mode == "crate" and (now.decade or now.country):
        tags = []
        if now.decade:
            tags.append(f"{now.decade}s")
        if now.country:
            tags.append(now.country)
        if now.mood:
            tags.append(now.mood)
        fragments.append(("class:radio-tags", f"  [{' '.join(tags)}]"))

    # Station name (stream mode)
    elif now.source_mode == "stream" and now.station_name:
        fragments.append(("class:radio-station", f"  [{now.station_name}]"))

    # Position / Duration (for finite tracks)
    if now.duration and now.duration > 0 and now.position is not None:
        pos_fmt = _format_time(now.position)
        dur_fmt = _format_time(now.duration)
        fragments.append(("class:radio-time", f"  {pos_fmt}/{dur_fmt}"))

    # Volume
    fragments.append(("class:radio-vol", f"  vol {int(now.volume)}"))

    return fragments


def get_mini_player_height() -> int:
    """Return 1 when radio is active, 0 otherwise."""
    try:
        from radio.player import HermesRadio
        return 1 if HermesRadio.active() else 0
    except Exception:
        return 0


def _format_time(seconds: float) -> str:
    """Format seconds as m:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# Style tokens for the mini player
MINI_PLAYER_STYLES = {
    "radio-bars": "#7eb8f6",
    "radio-title": "#e6edf3 bold",
    "radio-tags": "#6e7681",
    "radio-station": "#7ee6a8",
    "radio-time": "#6e7681",
    "radio-vol": "#484f58",
}

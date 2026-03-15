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


def _get_mini_text() -> List[Tuple[str, str]]:
    """Return styled text fragments for the compact mini player bar.

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

    # Progress bar (second line)
    fragments.append(("", "\n"))
    if now.duration and now.duration > 0 and now.position is not None:
        bar_width = 52  # characters for progress bar
        progress = max(0.0, min(1.0, now.position / now.duration))
        filled = int(progress * bar_width)
        remaining = bar_width - filled
        fragments.append(("", "  "))  # left margin to match bars
        fragments.append(("class:radio-progress", "\u2501" * filled + "\u2578"))
        fragments.append(("class:radio-progress-bg", "\u2500" * max(0, remaining - 1)))
    else:
        # Streaming / unknown duration -- show a subtle line
        fragments.append(("", "  "))
        fragments.append(("class:radio-progress-bg", "\u2500" * 52))

    return fragments


# -- Expanded display mode --------------------------------------------------

_expanded = False  # toggled by 'v' key binding in cli.py

_BARS_EXPANDED = 32  # wider visualizer in expanded mode
_bar_levels_exp = [0.0] * _BARS_EXPANDED


def toggle_expanded() -> bool:
    """Toggle between mini and expanded display. Returns new state."""
    global _expanded
    _expanded = not _expanded
    return _expanded


def get_expanded_player_text() -> List[Tuple[str, str]]:
    """Return styled fragments for the expanded now-playing display."""
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
    W = 58  # display width

    # Top border
    fragments.append(("class:radio-border", f"  \u256d{'\u2500' * (W - 2)}\u256e\n"))

    # Row 1: HERMES RADIO + volume
    vol_str = f"vol {int(now.volume)}"
    pad = W - 4 - 12 - len(vol_str)
    fragments.append(("class:radio-border", "  \u2502 "))
    fragments.append(("class:radio-label", "HERMES RADIO"))
    fragments.append(("", " " * max(1, pad)))
    fragments.append(("class:radio-vol", vol_str))
    fragments.append(("class:radio-border", " \u2502\n"))

    # Row 2: separator
    fragments.append(("class:radio-border", f"  \u251c{'\u2500' * (W - 2)}\u2524\n"))

    # Row 3: empty line for breathing room
    fragments.append(("class:radio-border", "  \u2502"))
    fragments.append(("", " " * (W - 2)))
    fragments.append(("class:radio-border", "\u2502\n"))

    # Row 4-6: large visualizer (3 rows of bars)
    bars_str = _generate_bars_expanded(
        position=now.position or 0.0,
        title=f"{now.artist}-{now.title}",
        paused=now.paused,
    )
    for row in bars_str:
        pad = W - 4 - len(row)
        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.append(("class:radio-bars", row))
        fragments.append(("", " " * max(0, pad + 1)))
        fragments.append(("class:radio-border", "\u2502\n"))

    # Row 7: empty
    fragments.append(("class:radio-border", "  \u2502"))
    fragments.append(("", " " * (W - 2)))
    fragments.append(("class:radio-border", "\u2502\n"))

    # Row 8: Artist
    artist = now.artist if now.artist and now.artist != "Unknown" else ""
    if artist:
        aline = artist[:W - 4]
        pad = W - 4 - len(aline)
        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.append(("class:radio-title", aline))
        fragments.append(("", " " * max(0, pad + 1)))
        fragments.append(("class:radio-border", "\u2502\n"))

    # Row 9: Title
    title = now.title or now.station_name or "..."
    tline = title[:W - 4]
    pad = W - 4 - len(tline)
    fragments.append(("class:radio-border", "  \u2502 "))
    fragments.append(("class:radio-title-dim", tline))
    fragments.append(("", " " * max(0, pad + 1)))
    fragments.append(("class:radio-border", "\u2502\n"))

    # Row 10: Tags / station info
    tags = ""
    if now.source_mode == "crate":
        parts = []
        if now.decade:
            parts.append(f"{now.decade}s")
        if now.country:
            parts.append(now.country)
        if now.mood:
            parts.append(now.mood)
        tags = " \u00b7 ".join(parts)
    elif now.source_mode == "stream" and now.station_name:
        tags = now.station_name

    if tags:
        tline = tags[:W - 4]
        pad = W - 4 - len(tline)
        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.append(("class:radio-tags", tline))
        fragments.append(("", " " * max(0, pad + 1)))
        fragments.append(("class:radio-border", "\u2502\n"))

    # Row 11: empty
    fragments.append(("class:radio-border", "  \u2502"))
    fragments.append(("", " " * (W - 2)))
    fragments.append(("class:radio-border", "\u2502\n"))

    # Row 12: Progress bar + time
    bar_w = W - 18  # leave room for time display
    if now.duration and now.duration > 0 and now.position is not None:
        progress = max(0.0, min(1.0, now.position / now.duration))
        filled = int(progress * bar_w)
        remaining = bar_w - filled
        pos_fmt = _format_time(now.position)
        dur_fmt = _format_time(now.duration)
        time_str = f" {pos_fmt} / {dur_fmt}"

        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.append(("class:radio-progress", "\u2501" * filled + "\u2578"))
        fragments.append(("class:radio-progress-bg", "\u2500" * max(0, remaining - 1)))
        fragments.append(("class:radio-time", time_str))
        pad = W - 4 - bar_w - len(time_str)
        fragments.append(("", " " * max(0, pad + 1)))
        fragments.append(("class:radio-border", "\u2502\n"))
    else:
        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.append(("class:radio-progress-bg", "\u2500" * bar_w))
        fragments.append(("class:radio-time", "  \u221e "))
        pad = W - 4 - bar_w - 4
        fragments.append(("", " " * max(0, pad + 1)))
        fragments.append(("class:radio-border", "\u2502\n"))

    # Bottom border
    fragments.append(("class:radio-border", f"  \u2570{'\u2500' * (W - 2)}\u256f\n"))

    return fragments


def _generate_bars_expanded(position: float, title: str, paused: bool) -> List[str]:
    """Generate 3 rows of wide visualizer bars for expanded mode."""
    global _bar_levels_exp

    n = _BARS_EXPANDED

    # Get real or synthetic levels
    levels = [0.0] * n
    try:
        from radio.level_meter import get_levels, is_active
        if is_active():
            raw = get_levels(n)
            if len(raw) >= 3:
                for i in range(n):
                    idx = min(i, len(raw) - 1)
                    levels[i] = raw[idx]
                    # Add jitter for visual spread
                    j = _noise(int(time.time() * 3), i) * 0.12
                    levels[i] = max(0.0, min(1.0, levels[i] + j - 0.06))
    except ImportError:
        pass

    # If no real levels, use position-seeded noise
    if all(l == 0.0 for l in levels) and not paused:
        pos = position if position and position > 0 else time.time()
        title_seed = int(hashlib.md5((title or "x").encode()).hexdigest()[:8], 16)
        for i in range(n):
            freq = 1.2 + i * 0.5
            phase = title_seed + i * 137
            val = _noise(int(pos * freq) + phase, i) * 0.5
            val += _noise(int(pos * freq * 2.7) + phase + 1000, i) * 0.3
            val += _noise(int(pos * freq * 6.1) + phase + 2000, i) * 0.2
            center = 0.6 + 0.4 * (1.0 - abs(i - n / 2) / (n / 2))
            energy = 0.6 + 0.4 * math.sin(pos * 0.4 + title_seed * 0.001)
            levels[i] = val * center * energy * 1.6 + 0.15

    if paused:
        for i in range(n):
            _bar_levels_exp[i] *= 0.85
    else:
        dt = 0.3
        for i in range(n):
            val = levels[i]
            if val > _bar_levels_exp[i]:
                _bar_levels_exp[i] += (val - _bar_levels_exp[i]) * min(1.0, 12.0 * dt)
            else:
                _bar_levels_exp[i] += (val - _bar_levels_exp[i]) * min(1.0, 4.0 * dt)

    # Render 3 rows (top, mid, bottom) using block characters
    # Each bar has height 0-24 (3 rows x 8 levels)
    rows = ["", "", ""]
    for i in range(n):
        h = int(_bar_levels_exp[i] * 24)
        h = max(0, min(24, h))

        # Bottom row: 0-8
        bot = min(8, h)
        # Mid row: 8-16
        mid = min(8, max(0, h - 8))
        # Top row: 16-24
        top = min(8, max(0, h - 16))

        rows[0] += _BLOCKS[top]
        rows[1] += _BLOCKS[mid]
        rows[2] += _BLOCKS[bot]

    return rows  # top, mid, bottom


def get_mini_player_height() -> int:
    """Return display height: 0 (inactive), 2 (mini), or 15 (expanded)."""
    try:
        from radio.player import HermesRadio
        if not HermesRadio.active():
            return 0
        return 15 if _expanded else 2
    except Exception:
        return 0


def get_mini_player_text() -> List[Tuple[str, str]]:
    """Dispatch to mini or expanded renderer based on mode."""
    if _expanded:
        return get_expanded_player_text()
    return _get_mini_text()


def _format_time(seconds: float) -> str:
    """Format seconds as m:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"

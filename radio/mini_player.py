"""Mini player widget for the Hermes CLI.

Provides a prompt_toolkit FormattedTextControl that renders a compact
now-playing bar below the input area.  Only visible when the radio is active.

Visualizer uses Unicode braille characters (U+2800-U+28FF) for 2x4 dot
resolution per character cell.  Each frequency band is one column rendered
with braille dots, giving smooth sub-character animation.
"""

import hashlib
import math
import time
from typing import List, Tuple

# -- Braille rendering engine -----------------------------------------------
# Each braille character is a 2x4 grid of dots.  We use both columns
# (full width) for each bar, filling from bottom up.

# Both columns, bottom-to-top row order
_BRAILLE_ROWS = [
    0x40 | 0x80,  # row 3 (bottom): dots 7+8
    0x04 | 0x20,  # row 2: dots 3+6
    0x02 | 0x10,  # row 1: dots 2+5
    0x01 | 0x08,  # row 0 (top): dots 1+4
]

_BRAILLE_BASE = 0x2800

# Block elements fallback
_BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

_NUM_BARS = 16  # more bars for braille (each is 2px wide = 1 char)

# State for smooth animation (persists between renders)
_bar_levels = [0.0] * _NUM_BARS
_peak_levels = [0.0] * _NUM_BARS
_peak_decay = [0.0] * _NUM_BARS
_last_title = ""
_last_render = 0.0


def _braille_bar(height: float) -> str:
    """Convert a 0.0-1.0 height to a single braille character.

    Uses 4 vertical levels within one character (bottom-up fill).
    """
    filled = round(height * 4)
    filled = max(0, min(4, filled))
    code = 0
    for i in range(filled):
        code |= _BRAILLE_ROWS[i]
    return chr(_BRAILLE_BASE + code)


def _braille_bar_2row(height: float) -> List[str]:
    """Convert a 0.0-1.0 height to two braille characters (top, bottom).

    8 vertical levels across 2 characters stacked.
    """
    return _braille_bar_stack(height, rows=2)


def _braille_bar_stack(height: float, rows: int = 3) -> List[str]:
    """Convert a normalized height to N stacked braille characters."""
    total_levels = max(1, rows * 4)
    filled = round(height * total_levels)
    filled = max(0, min(total_levels, filled))

    out = []
    remaining = filled
    for _ in range(rows):
        row_fill = min(4, remaining)
        code = 0
        for i in range(row_fill):
            code |= _BRAILLE_ROWS[i]
        out.append(chr(_BRAILLE_BASE + code))
        remaining = max(0, remaining - 4)
    return list(reversed(out))


def _gradient_fragments(row: str, steps: int = 6) -> List[Tuple[str, str]]:
    """Split a row into themed gradient fragments for prompt_toolkit styling."""
    if not row:
        return []
    steps = max(1, min(steps, len(row)))
    width = len(row)
    fragments: List[Tuple[str, str]] = []
    start = 0
    for idx in range(steps):
        end = round((idx + 1) * width / steps)
        chunk = row[start:end]
        if chunk:
            fragments.append((f"class:radio-bars-grad-{idx}", chunk))
        start = end
    return fragments


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
        return "".join(_braille_bar(_bar_levels[i]) for i in range(_NUM_BARS))

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

    # Render as braille
    return "".join(_braille_bar(_bar_levels[i]) for i in range(_NUM_BARS))


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

    return "".join(_braille_bar(_bar_levels[i]) for i in range(_NUM_BARS))


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

    # Position / Duration (only for finite tracks, not streams)
    is_stream = now.source_mode == "stream"
    if not is_stream and now.duration and now.duration > 0 and now.position is not None:
        pos_fmt = _format_time(now.position)
        dur_fmt = _format_time(now.duration)
        fragments.append(("class:radio-time", f"  {pos_fmt}/{dur_fmt}"))

    # Volume dial -- starts from bottom-left, fills clockwise
    vol = int(now.volume)
    _DIAL = "\u25cb\u25d8\u25d3\u25d1\u25d2\u25cf"  # 6 positions: empty -> bl -> half-l -> half -> half-r -> full
    dial_idx = min(5, vol * 6 // 101)
    dial = _DIAL[dial_idx]
    fragments.append(("class:radio-vol", f"  {dial} {vol}"))

    # Check if radio control mode is active
    control_mode = False
    try:
        import radio.mini_player as _self_mod
        control_mode = getattr(_self_mod, '_control_mode_active', False)
    except Exception:
        pass

    # Second line: control hints, progress bar (finite tracks only), or nothing (streams)
    fragments.append(("", "\n"))
    if control_mode:
        fragments.append(("class:radio-control", "  Spc pause  n skip  m mute  -/+ vol  Tab expand  Ctrl+O/q exit"))
    elif not is_stream and now.duration and now.duration > 0 and now.position is not None:
        bar_width = 52
        progress = max(0.0, min(1.0, now.position / now.duration))
        filled = int(progress * bar_width)
        remaining = bar_width - filled
        fragments.append(("", "  "))
        fragments.append(("class:radio-progress", "\u2501" * filled + "\u2578"))
        fragments.append(("class:radio-progress-bg", "\u2500" * max(0, remaining - 1)))
    elif is_stream:
        # Stream mode: show LIVE indicator instead of progress bar
        fragments.append(("class:radio-station", "  LIVE"))
    else:
        fragments.append(("", "  "))
        fragments.append(("class:radio-progress-bg", "\u2500" * 52))

    return fragments

# Module-level flag set by cli.py when control mode is active
_control_mode_active = False


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

    # Row 1: HERMES RADIO -- TRANSMISSIONS ONLY + volume dial
    vol = int(now.volume)
    _DIAL = "\u25cb\u25d8\u25d3\u25d1\u25d2\u25cf"
    dial_idx = min(5, vol * 6 // 101)
    vol_str = f"{_DIAL[dial_idx]} {vol}"
    title_full = "HERMES RADIO \u2014 TRANSMISSIONS ONLY"
    pad = W - 4 - len(title_full) - len(vol_str)
    fragments.append(("class:radio-border", "  \u2502 "))
    fragments.append(("class:radio-label", "HERMES RADI"))
    fragments.append(("class:radio-control", "O"))
    fragments.append(("class:radio-tags", " \u2014 TRANSMISSIONS ONLY"))
    fragments.append(("", " " * max(1, pad)))
    fragments.append(("class:radio-vol", vol_str))
    fragments.append(("class:radio-border", " \u2502\n"))

    # Row 2: separator
    fragments.append(("class:radio-border", f"  \u251c{'\u2500' * (W - 2)}\u2524\n"))

    # Row 3: empty line for breathing room
    fragments.append(("class:radio-border", "  \u2502"))
    fragments.append(("", " " * (W - 2)))
    fragments.append(("class:radio-border", "\u2502\n"))

    # Row 4-6: large visualizer (3 rows of braille bars with theme gradient)
    bars_str = _generate_bars_expanded(
        position=now.position or 0.0,
        title=f"{now.artist}-{now.title}",
        paused=now.paused,
    )
    for row in bars_str:
        pad = W - 4 - len(row)
        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.extend(_gradient_fragments(row, steps=6))
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

    # Row 11: keyboard controls
    controls = "space pause  n skip  m mute  -/+ vol  V size"
    cline = controls[:W - 4]
    pad = W - 4 - len(cline)
    fragments.append(("class:radio-border", "  \u2502 "))
    fragments.append(("class:radio-tags", cline))
    fragments.append(("", " " * max(0, pad + 1)))
    fragments.append(("class:radio-border", "\u2502\n"))

    # Row 12: Progress bar + time (crate dig only) or LIVE (streams)
    is_stream = now.source_mode == "stream"
    bar_w = W - 18
    if not is_stream and now.duration and now.duration > 0 and now.position is not None:
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
    elif is_stream:
        live_text = "LIVE"
        pad = W - 4 - len(live_text)
        fragments.append(("class:radio-border", "  \u2502 "))
        fragments.append(("class:radio-station", live_text))
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
    """Generate 3 rows of wide braille visualizer bars for expanded mode."""
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

    # Load visualizer preset for attack/decay/rows/mirror
    try:
        from radio.visualizers import load_preset
        preset = load_preset()
    except Exception:
        preset = {"attack": 12.0, "decay": 4.0, "rows": 3, "mirror": False,
                  "center_boost": 0.25, "chars": "braille"}

    atk = preset.get("attack", 12.0)
    dec = preset.get("decay", 4.0)
    num_rows = preset.get("rows", 3)
    do_mirror = preset.get("mirror", False)
    char_set = preset.get("chars", "braille")
    cb = preset.get("center_boost", 0.25)

    if paused:
        for i in range(n):
            _bar_levels_exp[i] *= 0.85
    else:
        dt = 0.3
        for i in range(n):
            val = levels[i]
            # Apply center boost from preset
            if cb > 0:
                center = 1.0 - abs(i - n / 2) / (n / 2) * cb
                val *= center
            if val > _bar_levels_exp[i]:
                _bar_levels_exp[i] += (val - _bar_levels_exp[i]) * min(1.0, atk * dt)
            else:
                _bar_levels_exp[i] += (val - _bar_levels_exp[i]) * min(1.0, dec * dt)

    # Mirror mode: duplicate bars symmetrically
    bar_data = list(_bar_levels_exp[:n])
    if do_mirror:
        half = bar_data[:n // 2]
        bar_data = list(reversed(half)) + half

    # Render stacked rows
    num_rows = max(1, min(6, num_rows))
    rows = [""] * num_rows
    for i in range(len(bar_data)):
        level = max(0.0, min(1.0, bar_data[i]))
        if char_set == "blocks":
            # Use block characters (single row)
            idx = max(0, min(8, int(level * 8)))
            ch_stack = [_BLOCKS[idx]] * num_rows
        else:
            # Use braille (default)
            ch_stack = _braille_bar_stack(level, rows=num_rows)
        for row_idx, ch in enumerate(ch_stack):
            rows[row_idx] += ch

    return rows


def get_mini_player_height() -> int:
    """Return display height: 0 (inactive), 2 (mini), or 14 (expanded)."""
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

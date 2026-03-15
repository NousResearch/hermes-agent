"""Terminal-native visualizer engine for Hermes Radio.

Adapts the broad structure of the ascii-video pipeline to prompt_toolkit-safe
terminal rendering:

    feature snapshot -> scene selection -> field generation -> character mapping

This module intentionally stays cheap enough for continuous UI updates.
"""

from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Dict, List, Tuple

from radio.level_meter import VisualizerFeatures, get_feature_snapshot
from radio.visualizers import load_preset

_BRAILLE_ROWS = [
    0x40 | 0x80,
    0x04 | 0x20,
    0x02 | 0x10,
    0x01 | 0x08,
]
_BRAILLE_BASE = 0x2800
_BLOCKS = " ▁▂▃▄▅▆▇█"
_DOTS = " .·•◉"
_ASCII = " .:-=#@"


@dataclass
class TerminalGrid:
    cols: int
    rows: int


@dataclass
class VisualizerState:
    levels: List[float] = field(default_factory=list)
    seed_key: str = ""
    last_render: float = 0.0


_STATE: Dict[Tuple[str, int, int, str], VisualizerState] = {}


def _noise(seed: str, idx: int) -> float:
    digest = hashlib.md5(f"{seed}:{idx}".encode()).digest()
    return ((digest[0] << 8) | digest[1]) / 65535.0


def _synthetic_snapshot(width: int, position: float, title_seed: str) -> VisualizerFeatures:
    values: List[float] = []
    for i in range(width):
        base = 0.5 + 0.25 * math.sin(position * 1.7 + i * 0.41)
        sparkle = (_noise(title_seed, i) - 0.5) * 0.28
        contour = 0.15 * math.sin(position * (0.8 + i * 0.03) + i * 0.17)
        values.append(max(0.0, min(1.0, base + sparkle + contour)))
    diffs = [abs(b - a) for a, b in zip(values, values[1:])]
    energy = sum(values) / len(values) if values else 0.0
    peak = max(values) if values else 0.0
    transient = max(0.0, values[-1] - (sum(values[:-1]) / max(1, len(values) - 1))) if values else 0.0
    motion = sum(diffs) / len(diffs) if diffs else 0.0
    return VisualizerFeatures(
        levels=values,
        energy=energy,
        peak=peak,
        transient=transient,
        motion=motion,
        decay=0.0,
        active=False,
    )


def _resolve_features(width: int, position: float, title_seed: str) -> VisualizerFeatures:
    snapshot = get_feature_snapshot(width)
    if snapshot.active and any(snapshot.levels):
        return snapshot
    return _synthetic_snapshot(width, position, title_seed)


def _smooth_levels(levels: List[float], state: VisualizerState, attack: float, decay: float, paused: bool) -> List[float]:
    if not state.levels or len(state.levels) != len(levels):
        state.levels = [0.0] * len(levels)
    now = time.time()
    dt = min(now - state.last_render, 0.5) if state.last_render else 0.25
    state.last_render = now

    if paused:
        state.levels = [value * 0.85 for value in state.levels]
        return list(state.levels)

    attack_factor = min(1.0, max(0.0, attack) * dt)
    decay_factor = min(1.0, max(0.0, decay) * dt)
    for i, value in enumerate(levels):
        current = state.levels[i]
        if value > current:
            state.levels[i] = current + (value - current) * attack_factor
        else:
            state.levels[i] = current + (value - current) * decay_factor
    return list(state.levels)


def _apply_center_boost(levels: List[float], amount: float) -> List[float]:
    if amount <= 0.0 or not levels:
        return levels
    n = len(levels)
    out: List[float] = []
    center = max(1.0, (n - 1) / 2)
    for i, value in enumerate(levels):
        weight = 1.0 - abs(i - center) / center * amount
        out.append(max(0.0, min(1.0, value * weight)))
    return out


def _braille_stack(level: float, rows: int) -> List[str]:
    total_levels = max(1, rows * 4)
    filled = max(0, min(total_levels, round(level * total_levels)))
    out: List[str] = []
    remaining = filled
    for _ in range(rows):
        row_fill = min(4, remaining)
        code = 0
        for i in range(row_fill):
            code |= _BRAILLE_ROWS[i]
        out.append(chr(_BRAILLE_BASE + code))
        remaining = max(0, remaining - 4)
    return list(reversed(out))


def _scalar_stack(level: float, rows: int, charset: str) -> List[str]:
    idx = min(len(charset) - 1, max(0, round(level * (len(charset) - 1))))
    return [charset[idx]] * rows


def _render_bars(levels: List[float], rows: int, chars: str) -> List[str]:
    output = [""] * rows
    for level in levels:
        if chars == 'braille':
            stack = _braille_stack(level, rows)
        elif chars == 'blocks':
            stack = _scalar_stack(level, rows, _BLOCKS)
        elif chars == 'dots':
            stack = _scalar_stack(level, rows, _DOTS)
        else:
            stack = _scalar_stack(level, rows, _ASCII)
        for row_idx, char in enumerate(stack):
            output[row_idx] += char
    return output


def render_rows(*, preset_name: str | None, width: int, rows: int, paused: bool, position: float, title_seed: str) -> List[str]:
    """Render terminal visualizer rows for the active or requested preset."""
    width = max(1, width)
    rows = max(1, rows)
    preset = load_preset(preset_name)

    key = (preset.get('name', preset_name or 'default'), width, rows, title_seed)
    state = _STATE.setdefault(key, VisualizerState(seed_key=title_seed))

    features = _resolve_features(width, position, title_seed)
    levels = list(features.levels[:width])
    if len(levels) < width:
        levels.extend([0.0] * (width - len(levels)))

    levels = _apply_center_boost(levels, float(preset.get('center_boost', 0.0)))
    levels = _smooth_levels(levels, state, float(preset.get('attack', 12.0)), float(preset.get('decay', 4.0)), paused)

    if preset.get('mirror') and levels:
        half = levels[: max(1, width // 2)]
        reflected = list(reversed(half)) + half
        levels = reflected[:width]
        if len(levels) < width:
            levels.extend([levels[-1]] * (width - len(levels)))

    return _render_bars(levels, rows, str(preset.get('chars', 'braille')))

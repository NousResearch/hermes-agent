from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image
from . import converter


D30_CHARSET = ".,'`^\",:;Il!i><~+_-?][}{1)|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
DENSE_REF_CHARSET = "@$#MHAGXS532;:,. "
STATE_ORDER = ["idle", "thinking", "speaking"]


def parse_grid(spec: str) -> tuple[int, int]:
    width, height = spec.lower().split('x', 1)
    return int(width), int(height)


def normalize_lines(lines: Iterable[str], width: int, height: int) -> list[str]:
    rendered = [str(line).rstrip('\n') for line in lines]
    if len(rendered) < height:
        rendered.extend([''] * (height - len(rendered)))
    rendered = rendered[:height]
    return [(line + ' ' * width)[:width] for line in rendered]


def collapse_lines(lines: list[str], target_width: int, target_height: int, *, method: str = "majority") -> list[str]:
    source_height = len(lines)
    source_width = max((len(line) for line in lines), default=0)
    if source_width == 0 or source_height == 0:
        return [' ' * target_width for _ in range(target_height)]
    if source_width % target_width or source_height % target_height:
        raise ValueError(
            f'Cannot exactly collapse {source_width}x{source_height} to '
            f'{target_width}x{target_height}'
        )
    factor_x = source_width // target_width
    factor_y = source_height // target_height
    if factor_x != factor_y:
        raise ValueError(f"Non-square block factors not supported: fx={factor_x}, fy={factor_y}")
    normalized = normalize_lines(lines, source_width, source_height)

    if method == "edge-aware":
        from . import downsampling
        return downsampling.edge_aware_downsample(normalized, factor_y)
    elif method == "clahe":
        from . import downsampling
        return downsampling.clahe_downsample(normalized, factor_y)
    else:  # majority
        out: list[str] = []
        for y in range(target_height):
            row: list[str] = []
            for x in range(target_width):
                block: list[str] = []
                for dy in range(factor_y):
                    for dx in range(factor_x):
                        block.append(normalized[y * factor_y + dy][x * factor_x + dx])
                row.append(Counter(block).most_common(1)[0][0])
            out.append(''.join(row))
        return out


def charset_by_name(name: str) -> str:
    lookup = {
        'd30': D30_CHARSET,
        'dense-ref': DENSE_REF_CHARSET,
    }
    if name in lookup:
        return lookup[name]
    return name


def _frame_index(path: Path) -> int:
    match = re.search(r'(\d+)(?=\.[^.]+$)', path.name)
    return int(match.group(1)) if match else 0


def collect_frame_paths(frames_dir: str | Path) -> dict[str, list[Path]]:
    root = Path(frames_dir)
    grouped: dict[str, list[Path]] = {state: [] for state in STATE_ORDER}
    for state in STATE_ORDER:
        grouped[state] = sorted(root.glob(f'{state}_frame_*.png'), key=_frame_index)
    return grouped


def render_png_to_ascii_lines(
    png_path: str | Path,
    *,
    grid_width: int,
    grid_height: int,
    charset: str,
) -> list[str]:
    """Render a single PNG frame to ASCII lines using the pure-Python converter."""
    img = Image.open(png_path)
    lines = converter.convert_to_ascii(img, grid_width, grid_height, charset)
    return normalize_lines(lines, grid_width, grid_height)


def write_eikon(
    out_path: str | Path,
    state_frames: dict[str, list[list[str]]],
    *,
    grid_width: int,
    grid_height: int,
    eikon_id: str,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(frames) for frames in state_frames.values())
    ranges = []
    cursor = 0
    ordered_frames: list[tuple[str, list[str]]] = []
    for state in STATE_ORDER:
        frames = state_frames.get(state, [])
        if not frames:
            continue
        ranges.append({'s': state, 'b': cursor, 'c': len(frames)})
        for frame in frames:
            ordered_frames.append((state, normalize_lines(frame, grid_width, grid_height)))
        cursor += len(frames)
    with out.open('w', encoding='utf-8') as fh:
        fh.write(json.dumps({'t': 'eikon', 'v': 1, 'id': eikon_id, 'grid': {'w': grid_width, 'h': grid_height}}, ensure_ascii=False) + '\n')
        fh.write(json.dumps({'t': 'states', 'states': [state for state in STATE_ORDER if state_frames.get(state)], 'loopFrom': 0}, ensure_ascii=False) + '\n')
        fh.write(json.dumps({'t': 'frames', 'total': total, 'ranges': ranges}, ensure_ascii=False) + '\n')
        for index, (_, lines) in enumerate(ordered_frames):
            fh.write(json.dumps({'t': 'frame', 'i': index, 'g': lines}, ensure_ascii=False) + '\n')
    return out

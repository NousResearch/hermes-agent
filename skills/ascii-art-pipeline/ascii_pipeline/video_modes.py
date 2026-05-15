from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .eikon import (
    STATE_ORDER,
    charset_by_name,
    collapse_lines,
    collect_frame_paths,
    parse_grid,
    render_png_to_ascii_lines,
    write_eikon,
)


def build_eikon_from_frames(
    frames_dir: str | Path,
    out_path: str | Path,
    *,
    grid: str = '192x96',
    charset: str = 'dense-ref',
    collapse_to: str | None = None,
    eikon_id: str | None = None,
) -> dict[str, object]:
    frames_dir = Path(frames_dir)
    eikon_id = eikon_id or Path(out_path).stem
    grid_width, grid_height = parse_grid(grid)
    charmap = charset_by_name(charset)
    grouped = collect_frame_paths(frames_dir)

    rendered: dict[str, list[list[str]]] = {state: [] for state in STATE_ORDER}
    for state in STATE_ORDER:
        for png in grouped.get(state, []):
            rendered[state].append(
                render_png_to_ascii_lines(
                    png,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    charset=charmap,
                )
            )

    full_path = write_eikon(
        out_path,
        rendered,
        grid_width=grid_width,
        grid_height=grid_height,
        eikon_id=eikon_id,
    )

    result: dict[str, object] = {
        'full_size_eikon': str(full_path),
        'grid': [grid_width, grid_height],
        'state_counts': {state: len(rendered[state]) for state in STATE_ORDER},
        'collapse_eikon': None,
    }

    if collapse_to:
        target_width, target_height = parse_grid(collapse_to)
        collapsed = {
            state: [collapse_lines(lines, target_width, target_height) for lines in frames]
            for state, frames in rendered.items()
        }
        collapse_path = Path(out_path).with_name(Path(out_path).stem + f'-{target_width}x{target_height}.eikon')
        write_eikon(
            collapse_path,
            collapsed,
            grid_width=target_width,
            grid_height=target_height,
            eikon_id=collapse_path.stem,
        )
        result['collapse_eikon'] = str(collapse_path)

    return result


def _extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
) -> list[Path]:
    """Extract all frames from video to PNG sequence. Returns sorted list of frame paths."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vsync', '0',
        '-compression_level', '0',
    ]
    if fps is not None:
        ffmpeg_cmd += ['-vf', f'fps={fps}']
    ffmpeg_cmd.append(str(output_dir / 'frame_%04d.png'))

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frames = sorted(output_dir.glob('frame_*.png'))
    if not frames:
        raise RuntimeError("No frames extracted from video")
    return frames


def _compute_frame_deltas(frames: list[Path]) -> list[float]:
    """Compute mean absolute per-pixel difference between consecutive grayscale frames."""
    deltas = []
    prev = None
    for frame_path in frames:
        curr = np.array(Image.open(frame_path).convert('L'))
        if prev is not None:
            delta = np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16)))
            deltas.append(delta)
        prev = curr
    return deltas


def _assign_states_by_motion(
    deltas: list[float],
    states: list[str] | None = None,
) -> list[str]:
    """
    Cluster motion deltas into 3 motion states using percentile thresholds.
    Returns state assignment for each *delta* (len = frames-1).
    The first frame is assigned 'idle' by default.
    """
    states = states or STATE_ORDER
    if len(states) != 3:
        raise ValueError("Exactly 3 states required (idle, thinking, speaking)")

    low, mid = np.percentile(deltas, [33, 66])
    state_list = []
    for d in deltas:
        if d < low:
            state_list.append(states[0])   # idle
        elif d < mid:
            state_list.append(states[1])   # thinking
        else:
            state_list.append(states[2])   # speaking
    # Prepend idle for frame 0
    return [states[0]] + state_list


def build_eikon_from_video(
    video_path: str | Path,
    out_path: str | Path,
    *,
    grid: str = '192x96',
    charset: str = 'dense-ref',
    fps: float | None = None,
    states: list[str] | None = None,
    motion_phases: bool = True,
    eikon_id: str | None = None,
) -> dict[str, object]:
    """
    Build an animated eikon directly from a video file.

    Args:
        video_path: Source video (MP4, MOV, etc.)
        out_path: Destination .eikon file
        grid: Output ASCII grid 'WxH' (default 192x96)
        charset: Character set name ('dense-ref' or 'd30')
        fps: Extract frames at this rate (None = native fps)
        states: Ordered list of 3 state names (default: ['idle','thinking','speaking'])
        motion_phases: If True, auto-assign states by per-frame motion magnitude.
                        If False, all frames become 'idle'.
        eikon_id: Optional explicit eikon identifier

    Returns:
        Dict with eikon path, frame counts per state, and grid info.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    eikon_id = eikon_id or out_path.stem
    grid_width, grid_height = parse_grid(grid)
    charmap = charset_by_name(charset)

    # 1 — Extract frames to a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frames = _extract_frames_from_video(video_path, tmp_dir, fps=fps)
        print(f"Extracted {len(frames)} frames from {video_path.name}", file=sys.stderr)

        # 2 — Compute ASCII for every frame
        ascii_frames: list[list[str]] = []
        for i, frame in enumerate(frames):
            lines = render_png_to_ascii_lines(
                frame,
                grid_width=grid_width,
                grid_height=grid_height,
                charset=charmap,
            )
            ascii_frames.append(lines)
            if (i + 1) % 24 == 0:
                print(f"  Rendered {i+1}/{len(frames)} frames", file=sys.stderr)

        # 3 — Assign states
        if motion_phases and len(frames) > 1:
            deltas = _compute_frame_deltas(frames)
            state_assignments = _assign_states_by_motion(deltas, states)
        else:
            state_assignments = [states[0] if states else 'idle'] * len(frames)

        # 4 — Group by state
        grouped: dict[str, list[list[str]]] = {s: [] for s in STATE_ORDER}
        for lines, st in zip(ascii_frames, state_assignments):
            if st in grouped:
                grouped[st].append(lines)
            else:
                # Fallback: put in first known state
                grouped[STATE_ORDER[0]].append(lines)

        # 5 — Write eikon
        full_path = write_eikon(
            out_path,
            grouped,
            grid_width=grid_width,
            grid_height=grid_height,
            eikon_id=eikon_id,
        )

    result = {
        'full_size_eikon': str(full_path),
        'grid': [grid_width, grid_height],
        'state_counts': {s: len(grouped[s]) for s in STATE_ORDER},
        'total_frames': len(frames),
        'collapse_eikon': None,
    }
    return result

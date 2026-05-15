from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

HEAVY_GLYPHS = set("@#$%&*_~MW8B")
LIGHT_GLYPHS = set("Ili;:,. '\"")
BRAILLE_START = 0x2800
BRAILLE_END = 0x28FF


@dataclass(frozen=True)
class Frame:
    index: int
    lines: list[str]
    state: str | None = None
    fps: int | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class FrameMetrics:
    index: int
    state: str | None
    width: int
    height: int
    nonspace_chars: int
    unique_glyphs: int
    fill_ratio: float
    heavy_ratio: float
    light_ratio: float
    braille_ratio: float
    edge_signature: list[str]
    dimensions_ok: bool | None


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _normalize_lines(lines: list[str]) -> list[str]:
    if not lines:
        return []
    width = max(len(line) for line in lines)
    return [line.ljust(width) for line in lines]


def _char_diff(lines_a: list[str], lines_b: list[str]) -> float:
    norm_a = _normalize_lines(lines_a)
    norm_b = _normalize_lines(lines_b)
    height = max(len(norm_a), len(norm_b))
    width = max(
        max((len(line) for line in norm_a), default=0),
        max((len(line) for line in norm_b), default=0),
    )
    if height == 0 or width == 0:
        return 0.0
    diff = 0
    for row in range(height):
        left = norm_a[row] if row < len(norm_a) else " " * width
        right = norm_b[row] if row < len(norm_b) else " " * width
        left = left.ljust(width)
        right = right.ljust(width)
        diff += sum(a != b for a, b in zip(left, right))
    return diff / (height * width)


def analyze_lines(
    lines: list[str],
    *,
    frame_index: int = 0,
    state: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> FrameMetrics:
    width = max((len(line) for line in lines), default=0)
    height = len(lines)
    normalized = [line.ljust(width) for line in lines]
    chars = [ch for line in normalized for ch in line]
    nonspace = [ch for ch in chars if ch != " "]
    total_cells = max(1, width * height)
    nonspace_count = len(nonspace)
    total_nonspace = max(1, nonspace_count)
    unique_glyphs = len(set(nonspace))
    heavy_ratio = sum(ch in HEAVY_GLYPHS for ch in nonspace) / total_nonspace
    light_ratio = sum(ch in LIGHT_GLYPHS for ch in nonspace) / total_nonspace
    braille_ratio = sum(BRAILLE_START <= ord(ch) <= BRAILLE_END for ch in nonspace) / total_nonspace
    dimensions_ok = None
    if expected_width is not None and expected_height is not None:
        dimensions_ok = width == expected_width and height == expected_height
    edge_signature = [line[:12] for line in normalized[:3]]
    return FrameMetrics(
        index=frame_index,
        state=state,
        width=width,
        height=height,
        nonspace_chars=nonspace_count,
        unique_glyphs=unique_glyphs,
        fill_ratio=_round(nonspace_count / total_cells),
        heavy_ratio=_round(heavy_ratio),
        light_ratio=_round(light_ratio),
        braille_ratio=_round(braille_ratio),
        edge_signature=edge_signature,
        dimensions_ok=dimensions_ok,
    )


def _infer_state(frame_index: int, ranges: list[dict[str, Any]] | None) -> str | None:
    if not ranges:
        return None
    for item in ranges:
        begin = int(item.get("b", -1))
        count = int(item.get("c", 0))
        if begin <= frame_index < begin + count:
            return item.get("s")
    return None


def parse_eikon_text(text: str) -> list[Frame]:
    frames: list[Frame] = []
    current_state: str | None = None
    current_fps: int | None = None
    ranges: list[dict[str, Any]] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("t") == "frames":
            ranges = obj.get("ranges") or []
            continue
        if obj.get("t") in {"eikon", "states"}:
            continue
        if "state" in obj:
            current_state = obj.get("state")
            current_fps = obj.get("fps")
            continue
        if "data" in obj:
            payload = str(obj["data"])
            if "\\n" in payload and "\n" not in payload:
                payload = payload.replace("\\n", "\n")
            frames.append(
                Frame(
                    index=len(frames),
                    state=current_state,
                    fps=current_fps,
                    lines=payload.splitlines(),
                )
            )
            continue
        if obj.get("t") == "frame":
            index = int(obj.get("i", len(frames)))
            lines = [str(item) for item in obj.get("g", [])]
            frames.append(
                Frame(
                    index=index,
                    state=_infer_state(index, ranges),
                    fps=None,
                    lines=lines,
                )
            )
    return frames


def load_frames_from_path(path: str | Path) -> list[Frame]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".eikon":
        return parse_eikon_text(text)
    lines = text.splitlines()
    return [Frame(index=0, lines=lines, state=None, fps=None)]


def summarize_frames(
    frames: list[Frame],
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("No frames found")

    frame_metrics = [
        analyze_lines(
            frame.lines,
            frame_index=frame.index,
            state=frame.state,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        for frame in frames
    ]
    widths = sorted({metric.width for metric in frame_metrics})
    heights = sorted({metric.height for metric in frame_metrics})
    motion = [
        _char_diff(left.lines, right.lines)
        for left, right in zip(frames, frames[1:])
    ]
    state_counts: dict[str, int] = {}
    for frame in frames:
        key = frame.state or "unknown"
        state_counts[key] = state_counts.get(key, 0) + 1

    mean_heavy = fmean(metric.heavy_ratio for metric in frame_metrics)
    mean_light = fmean(metric.light_ratio for metric in frame_metrics)
    mean_braille = fmean(metric.braille_ratio for metric in frame_metrics)
    verdict = "mixed"
    if mean_braille > 0.5:
        verdict = "braille-dominant"
    elif mean_heavy >= 0.30 and mean_light <= 0.20:
        verdict = "high-contrast"
    elif mean_light >= 0.30:
        verdict = "low-contrast-garble-risk"

    return {
        "frames": len(frames),
        "states": state_counts,
        "dimensions": {
            "widths": widths,
            "heights": heights,
            "consistent": len(widths) == 1 and len(heights) == 1,
            "expected": [expected_width, expected_height] if expected_width is not None and expected_height is not None else None,
            "all_match_expected": None if expected_width is None or expected_height is None else all(metric.dimensions_ok for metric in frame_metrics),
        },
        "aggregate": {
            "unique_glyphs_mean": _round(fmean(metric.unique_glyphs for metric in frame_metrics), 2),
            "unique_glyphs_min": min(metric.unique_glyphs for metric in frame_metrics),
            "unique_glyphs_max": max(metric.unique_glyphs for metric in frame_metrics),
            "fill_ratio_mean": _round(fmean(metric.fill_ratio for metric in frame_metrics)),
            "heavy_ratio_mean": _round(mean_heavy),
            "light_ratio_mean": _round(mean_light),
            "braille_ratio_mean": _round(mean_braille),
            "motion_char_diff_mean": _round(fmean(motion) if motion else 0.0),
            "motion_char_diff_max": _round(max(motion) if motion else 0.0),
        },
        "sample_frame": asdict(frame_metrics[0]),
        "verdict": verdict,
    }


def diagnose_path(
    path: str | Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> dict[str, Any]:
    path = Path(path)
    frames = load_frames_from_path(path)
    summary = summarize_frames(
        frames,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    summary["path"] = str(path)
    summary["kind"] = "eikon" if path.suffix == ".eikon" else "text"
    return summary

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat


DEFAULT_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
)
PANEL_FILE_RE = re.compile(r"^p\d+\.png$")
DEFAULT_PANEL_PADDING = 24
DEFAULT_BOX_GAP = 18
DEFAULT_FORBIDDEN_OVERLAP_THRESHOLD = 0.08
SCROLL_SPACING = {
    "tight": 30,
    "medium": 70,
    "tall_drop": 180,
    "end_cliff": 260,
}


@dataclass(frozen=True)
class SpeakerStyle:
    speaker: str
    fill: tuple[int, int, int, int]
    outline: tuple[int, int, int, int]
    text_fill: tuple[int, int, int, int]
    bubble_kind: str
    font_size: int
    min_font_size: int
    radius: int
    text_padding: tuple[int, int, int, int]
    max_width_ratio: float
    max_height_ratio: float
    preferred_zone_kinds: tuple[str, ...]


STYLE_MAP: dict[str, SpeakerStyle] = {
    "mother": SpeakerStyle(
        speaker="mother",
        fill=(255, 255, 255, 248),
        outline=(28, 28, 28, 255),
        text_fill=(24, 24, 24, 255),
        bubble_kind="speech",
        font_size=32,
        min_font_size=24,
        radius=36,
        text_padding=(30, 24, 30, 24),
        max_width_ratio=0.5,
        max_height_ratio=0.3,
        preferred_zone_kinds=("speech", "caption"),
    ),
    "teacher_chat": SpeakerStyle(
        speaker="teacher_chat",
        fill=(226, 240, 255, 240),
        outline=(63, 109, 170, 255),
        text_fill=(22, 34, 59, 255),
        bubble_kind="chat_ui",
        font_size=25,
        min_font_size=19,
        radius=22,
        text_padding=(24, 20, 24, 20),
        max_width_ratio=0.5,
        max_height_ratio=0.2,
        preferred_zone_kinds=("chat_ui", "screen_note"),
    ),
    "friend_chat": SpeakerStyle(
        speaker="friend_chat",
        fill=(248, 248, 248, 240),
        outline=(96, 96, 96, 255),
        text_fill=(25, 25, 25, 255),
        bubble_kind="chat_ui",
        font_size=25,
        min_font_size=19,
        radius=22,
        text_padding=(24, 20, 24, 20),
        max_width_ratio=0.5,
        max_height_ratio=0.2,
        preferred_zone_kinds=("chat_ui", "speech"),
    ),
    "internal_note": SpeakerStyle(
        speaker="internal_note",
        fill=(255, 249, 208, 236),
        outline=(132, 109, 35, 255),
        text_fill=(50, 42, 20, 255),
        bubble_kind="screen_note",
        font_size=30,
        min_font_size=22,
        radius=18,
        text_padding=(24, 18, 24, 18),
        max_width_ratio=0.5,
        max_height_ratio=0.2,
        preferred_zone_kinds=("screen_note", "caption"),
    ),
    "caption": SpeakerStyle(
        speaker="caption",
        fill=(22, 26, 34, 224),
        outline=(255, 255, 255, 42),
        text_fill=(255, 255, 255, 255),
        bubble_kind="caption",
        font_size=28,
        min_font_size=21,
        radius=18,
        text_padding=(24, 18, 24, 18),
        max_width_ratio=0.48,
        max_height_ratio=0.2,
        preferred_zone_kinds=("caption", "screen_note"),
    ),
}


def resolve_font_path(cli_value: str | None) -> str | None:
    if cli_value:
        candidate = Path(cli_value)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Font not found: {cli_value}")
    for path in DEFAULT_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def placement_priority(item: dict[str, Any]) -> tuple[int, int]:
    template = template_for_item(item)
    zone_rank = 0 if template == "caption" else 1
    return (zone_rank, -len(item.get("text", "")))


def normalize_speaker(raw: str | None, item: dict[str, Any]) -> str:
    speaker = (raw or "").strip().lower()
    if item.get("kind") == "caption":
        return "caption"
    if speaker in {"text_focus", "internal_note"}:
        return "internal_note"
    if speaker in STYLE_MAP:
        return speaker
    tone = str(item.get("tone", "")).lower()
    if "internal" in tone:
        return "internal_note"
    if speaker == "teacher":
        return "teacher_chat"
    return speaker or "caption"


def template_for_item(item: dict[str, Any]) -> str:
    speaker = normalize_speaker(item.get("speaker"), item)
    if item.get("kind") == "caption" or speaker == "caption":
        return "caption"
    if speaker in {"teacher_chat", "friend_chat"}:
        return "chat_ui"
    if speaker == "internal_note":
        return "screen_note"
    return "speech"


def load_lettering(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    per_panel: dict[str, list[dict[str, Any]]] = {}
    for balloon in data.get("balloons", []):
        entry = dict(balloon)
        entry["kind"] = "balloon"
        per_panel.setdefault(balloon["panel_id"], []).append(entry)
    for index, caption in enumerate(data.get("captions", []), start=1):
        entry = dict(caption)
        entry["kind"] = "caption"
        entry["speaker"] = "caption"
        entry["id"] = entry.get("id") or f"{caption['panel_id']}_caption_{index:02d}"
        per_panel.setdefault(caption["panel_id"], []).append(entry)
    for panel_id, items in per_panel.items():
        per_panel[panel_id] = sorted(items, key=placement_priority)
    return per_panel, data


def load_scroll_plan(path: Path | None) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    if not path or not path.exists():
        return {}, {}, {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    spacing_map = {
        block["block_id"]: block.get("spacing", "medium")
        for block in data.get("blocks", [])
        if "block_id" in block
    }
    panel_to_block: dict[str, str] = {}
    block_map: dict[str, Any] = {}
    for index, block in enumerate(data.get("blocks", []), start=1):
        block_id = block.get("block_id", "")
        panel_to_block[f"p{index:02d}"] = block_id
        if block_id:
            block_map[block_id] = block
    return spacing_map, panel_to_block, block_map


def iter_panel_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and PANEL_FILE_RE.match(path.name))


def tokenize_text(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    tokens = re.findall(r"\S+\s*", normalized)
    return tokens or list(normalized)


def split_oversized_token(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    fragments: list[str] = []
    current = ""
    for char in token:
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > max_width:
            fragments.append(current.rstrip())
            current = char
            continue
        current = candidate
    if current:
        fragments.append(current.rstrip())
    return fragments


def wrap_korean_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    tokens = tokenize_text(text)
    if not tokens:
        return [""]
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{token}"
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current.rstrip())
            current = token.lstrip()
            if draw.textlength(current, font=font) > max_width:
                lines.extend(split_oversized_token(draw, current, font, max_width))
                current = ""
            continue
        if not current and draw.textlength(token, font=font) > max_width:
            lines.extend(split_oversized_token(draw, token, font, max_width))
            current = ""
            continue
        current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines or [text]


def rebalance_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if len(lines) < 2:
        return lines
    best = list(lines)
    best_score = wrapped_lines_balance_score(draw, best, font)
    current = list(lines)
    for _ in range(8):
        changed = False
        for idx in range(len(current) - 1):
            left = current[idx].rstrip()
            right = current[idx + 1].lstrip()
            if not left or not right:
                continue
            left_parts = left.split(" ")
            right_parts = right.split(" ")
            candidate_pairs = []
            if len(left_parts) > 1:
                moved = left_parts[-1]
                new_left = " ".join(left_parts[:-1]).strip()
                new_right = f"{moved} {right}".strip()
                candidate_pairs.append((new_left, new_right))
            if len(right_parts) > 1:
                moved = right_parts[0]
                new_left = f"{left} {moved}".strip()
                new_right = " ".join(right_parts[1:]).strip()
                candidate_pairs.append((new_left, new_right))
            for new_left, new_right in candidate_pairs:
                if not new_left or not new_right:
                    continue
                if draw.textlength(new_left, font=font) > max_width:
                    continue
                if draw.textlength(new_right, font=font) > max_width:
                    continue
                trial = list(current)
                trial[idx] = new_left
                trial[idx + 1] = new_right
                score = wrapped_lines_balance_score(draw, trial, font)
                if score < best_score:
                    best = trial
                    best_score = score
                    current = trial
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return best


def wrapped_lines_balance_score(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
) -> float:
    widths = [draw.textlength(line, font=font) for line in lines if line]
    if not widths:
        return 1e9
    max_width = max(widths)
    min_width = min(widths)
    score = (max_width - min_width) * 0.4
    if len(widths) >= 2:
        last_width = widths[-1]
        prev_width = widths[-2]
        if last_width < prev_width * 0.45:
            score += (prev_width * 0.45 - last_width) * 2.2
    if widths[0] < max_width * 0.45:
        score += (max_width * 0.45 - widths[0]) * 1.4
    return score + len(widths) * 2.5


def _speech_safe_insets(width: int, height: int) -> tuple[int, int]:
    inset_x = max(16, int(width * 0.11))
    inset_y = max(10, int(height * 0.08))
    return inset_x, inset_y


def speech_text_safe_dimensions(inner_width: int, inner_height: int) -> tuple[int, int]:
    inset_x, inset_y = _speech_safe_insets(inner_width, inner_height)
    safe_width = max(1, inner_width - inset_x * 2)
    safe_height = max(1, inner_height - inset_y * 2)
    return safe_width, safe_height


def _min_speech_inner_extent_for_safe_extent(safe_extent: int, *, min_inset: int, ratio: float) -> int:
    lower = max(1, safe_extent)
    upper = max(lower, safe_extent + min_inset * 2)

    def safe_size(extent: int) -> int:
        return max(1, extent - max(min_inset, int(extent * ratio)) * 2)

    while safe_size(upper) < safe_extent:
        upper *= 2

    while lower < upper:
        mid = (lower + upper) // 2
        if safe_size(mid) >= safe_extent:
            upper = mid
        else:
            lower = mid + 1
    return lower


def speech_text_fit_limits(max_text_width: int, max_text_height: int) -> tuple[int, int]:
    return max(50, max_text_width), max(30, max_text_height)


def speech_text_safe_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    inset_x, inset_y = _speech_safe_insets(width, height)
    return (x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y)


def speech_inner_size_for_text_bbox(text_width: int, text_height: int) -> tuple[int, int]:
    inner_width = _min_speech_inner_extent_for_safe_extent(text_width, min_inset=16, ratio=0.11)
    inner_height = _min_speech_inner_extent_for_safe_extent(text_height, min_inset=10, ratio=0.08)
    return inner_width, inner_height


def fit_text_block(
    text: str,
    style: SpeakerStyle,
    font_path: str | None,
    max_text_width: int,
    max_text_height: int,
) -> tuple[list[str], ImageFont.ImageFont, tuple[int, int]] | None:
    scratch = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(scratch)
    fit_width = max_text_width
    fit_height = max_text_height
    if style.bubble_kind == "speech":
        fit_width, fit_height = speech_text_fit_limits(max_text_width, max_text_height)
    best_fit: tuple[list[str], ImageFont.ImageFont, tuple[int, int], float] | None = None
    for size in range(style.font_size, style.min_font_size - 1, -1):
        font = load_font(font_path, size)
        lines = wrap_korean_text(draw, text, font, fit_width)
        lines = rebalance_wrapped_lines(draw, lines, font, fit_width)
        spacing = max(7, int(size * 0.34))
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing, align="center")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        fits = width <= fit_width and height <= fit_height
        if fits:
            score = wrapped_lines_balance_score(draw, lines, font) - size * 2.0
            candidate = (lines, font, (width, height), score)
            if best_fit is None or candidate[3] < best_fit[3]:
                best_fit = candidate
    if best_fit is None:
        return None
    return best_fit[0], best_fit[1], best_fit[2]


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def normalized_rect_to_pixels(
    rect: dict[str, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    x1 = int(round(rect["x"] * width))
    y1 = int(round(rect["y"] * height))
    x2 = int(round((rect["x"] + rect["w"]) * width))
    y2 = int(round((rect["y"] + rect["h"]) * height))
    return clamp_box((x1, y1, x2, y2), size)


def normalized_point_to_pixels(point: dict[str, float], size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    x = int(round(point["x"] * width))
    y = int(round(point["y"] * height))
    x = min(max(0, x), width)
    y = min(max(0, y), height)
    return x, y


def pixels_to_normalized_rect(box: tuple[int, int, int, int], size: tuple[int, int]) -> dict[str, float]:
    width, height = size
    return {
        "x": round(box[0] / width, 4),
        "y": round(box[1] / height, 4),
        "w": round((box[2] - box[0]) / width, 4),
        "h": round((box[3] - box[1]) / height, 4),
    }


def clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    panel_w, panel_h = size
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    x1 = min(max(0, x1), max(0, panel_w - width))
    y1 = min(max(0, y1), max(0, panel_h - height))
    return (x1, y1, x1 + width, y1 + height)


def expand_box(box: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    return (box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount)


def area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    base_area = area(a)
    if base_area <= 0:
        return 0.0
    return intersection_area(a, b) / base_area


def candidate_background_score(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[float, dict[str, float]]:
    crop = image.crop(box)
    gray = ImageOps.grayscale(crop)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(gray)
    edge_stat = ImageStat.Stat(edges)
    mean = stat.mean[0]
    stddev = stat.stddev[0]
    edge_mean = edge_stat.mean[0]
    busy_penalty = stddev * 0.35 + edge_mean * 0.55 + abs(mean - 168) * 0.05
    return busy_penalty, {
        "mean_luma": round(mean, 2),
        "stddev_luma": round(stddev, 2),
        "edge_mean": round(edge_mean, 2),
    }


def compose_longscroll(
    panel_outputs: list[dict[str, Any]],
    output_dir: Path,
    episode_id: str,
    spacing_map: dict[str, str],
    panel_to_block: dict[str, str],
    longscroll_name: str,
) -> Path | None:
    if not panel_outputs:
        return None
    ordered = sorted(panel_outputs, key=lambda item: item["panel_id"])
    images: list[tuple[Image.Image, int]] = []
    total_h = 0
    max_w = 0
    for index, panel in enumerate(ordered):
        image = Image.open(panel["output"]).convert("RGB")
        panel_id = panel["panel_id"]
        block_id = panel_to_block.get(panel_id, "")
        gap = SCROLL_SPACING.get(spacing_map.get(block_id, "medium"), 70)
        images.append((image, gap))
        max_w = max(max_w, image.width)
        total_h += image.height
        if index < len(ordered) - 1:
            total_h += gap

    canvas = Image.new("RGB", (max_w, total_h), (244, 244, 244))
    cursor_y = 0
    for index, (image, gap) in enumerate(images):
        canvas.paste(image, (0, cursor_y))
        cursor_y += image.height
        if index < len(images) - 1:
            cursor_y += gap

    longscroll_path = output_dir / longscroll_name
    canvas.save(longscroll_path)
    return longscroll_path

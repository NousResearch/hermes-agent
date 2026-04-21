from __future__ import annotations

import argparse
import json
import os
import re
import shlex
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
DEFAULT_OUTER_PADDING = 24
DEFAULT_BOX_GAP = 18


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
    preferred_zones: tuple[str, ...]
    tail: bool
    tail_bias: float


@dataclass
class LayoutResult:
    item: dict[str, Any]
    speaker: str
    box: tuple[int, int, int, int]
    inner_box: tuple[int, int, int, int]
    zone: str
    score: float
    lines: list[str]
    font_size: int
    bubble_kind: str
    tail_points: list[tuple[int, int]] | None
    background_metrics: dict[str, float]


STYLE_MAP: dict[str, SpeakerStyle] = {
    "mother": SpeakerStyle(
        speaker="mother",
        fill=(255, 255, 255, 248),
        outline=(28, 28, 28, 255),
        text_fill=(24, 24, 24, 255),
        bubble_kind="speech",
        font_size=31,
        min_font_size=22,
        radius=32,
        text_padding=(24, 22, 24, 22),
        max_width_ratio=0.46,
        max_height_ratio=0.27,
        preferred_zones=("top-right", "upper-mid", "top-left", "mid-right"),
        tail=True,
        tail_bias=0.70,
    ),
    "teacher_chat": SpeakerStyle(
        speaker="teacher_chat",
        fill=(250, 252, 255, 246),
        outline=(40, 66, 102, 255),
        text_fill=(22, 36, 60, 255),
        bubble_kind="chat",
        font_size=29,
        min_font_size=21,
        radius=28,
        text_padding=(24, 22, 24, 22),
        max_width_ratio=0.45,
        max_height_ratio=0.24,
        preferred_zones=("top-left", "upper-mid", "top-right", "mid-left"),
        tail=True,
        tail_bias=0.30,
    ),
    "friend_chat": SpeakerStyle(
        speaker="friend_chat",
        fill=(252, 252, 252, 244),
        outline=(72, 72, 72, 255),
        text_fill=(18, 18, 18, 255),
        bubble_kind="chat",
        font_size=29,
        min_font_size=21,
        radius=28,
        text_padding=(24, 22, 24, 22),
        max_width_ratio=0.45,
        max_height_ratio=0.24,
        preferred_zones=("top-left", "top-right", "upper-mid", "mid-left"),
        tail=True,
        tail_bias=0.25,
    ),
    "internal_note": SpeakerStyle(
        speaker="internal_note",
        fill=(255, 250, 214, 240),
        outline=(110, 92, 30, 255),
        text_fill=(44, 38, 20, 255),
        bubble_kind="note",
        font_size=29,
        min_font_size=21,
        radius=24,
        text_padding=(22, 18, 22, 18),
        max_width_ratio=0.48,
        max_height_ratio=0.22,
        preferred_zones=("upper-mid", "top-left", "top-right", "mid-left"),
        tail=False,
        tail_bias=0.50,
    ),
    "caption": SpeakerStyle(
        speaker="caption",
        fill=(22, 26, 34, 224),
        outline=(255, 255, 255, 42),
        text_fill=(255, 255, 255, 255),
        bubble_kind="caption",
        font_size=27,
        min_font_size=20,
        radius=22,
        text_padding=(20, 18, 20, 18),
        max_width_ratio=0.44,
        max_height_ratio=0.22,
        preferred_zones=("top-left", "top-right", "bottom-left", "upper-mid"),
        tail=False,
        tail_bias=0.50,
    ),
}

FALLBACK_ZONES = ("top-left", "top-right", "upper-mid", "mid-left", "mid-right", "bottom-left", "bottom-right")
ZONE_OFFSETS: dict[str, list[tuple[float, float]]] = {
    "top-left": [(0.0, 0.0), (0.02, 0.015), (0.04, 0.0), (0.01, 0.04)],
    "top-right": [(0.0, 0.0), (0.02, 0.015), (0.04, 0.0), (0.01, 0.04)],
    "upper-mid": [(0.0, 0.0), (-0.03, 0.015), (0.03, 0.015), (0.0, 0.04)],
    "mid-left": [(0.0, 0.0), (0.02, -0.02), (0.03, 0.03)],
    "mid-right": [(0.0, 0.0), (0.02, -0.02), (0.03, 0.03)],
    "bottom-left": [(0.0, 0.0), (0.02, -0.02), (0.04, 0.0)],
    "bottom-right": [(0.0, 0.0), (0.02, -0.02), (0.04, 0.0)],
}
MODE_WIDTH_SCALE = {
    "balanced": 1.0,
    "compact": 0.92,
    "airy": 1.10,
}
SCROLL_SPACING = {
    "tight": 30,
    "medium": 70,
    "tall_drop": 180,
    "end_cliff": 260,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heuristic Korean webtoon UI overlay renderer for panel directories."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing panel images such as p01.png.")
    parser.add_argument("--output-dir", required=True, help="Directory where rendered UI panels are written.")
    parser.add_argument("--lettering", required=True, help="Path to lettering_script.yaml.")
    parser.add_argument("--scroll-plan", help="Optional scroll_plan.yaml for longscroll assembly.")
    parser.add_argument(
        "--font-path",
        help="Font path for Korean text. Defaults to a detected Noto Sans CJK font when available.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_WIDTH_SCALE),
        default="balanced",
        help="Placement density profile. 'compact' uses tighter boxes, 'airy' allows wider boxes.",
    )
    parser.add_argument(
        "--mask-strength",
        type=float,
        default=0.45,
        help="0.0 disables likely-text masking. Higher values apply stronger soft masks.",
    )
    parser.add_argument(
        "--compose-longscroll",
        action="store_true",
        help="Compose output panel images into a longscroll when --scroll-plan is available.",
    )
    parser.add_argument(
        "--manifest-name",
        default="placement_manifest.json",
        help="Output manifest/debug JSON filename written under --output-dir.",
    )
    return parser.parse_args()


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


def load_lettering(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    per_panel: dict[str, list[dict[str, Any]]] = {}
    for balloon in data.get("balloons", []):
        entry = dict(balloon)
        entry["kind"] = "balloon"
        per_panel.setdefault(balloon["panel_id"], []).append(entry)
    for caption in data.get("captions", []):
        entry = dict(caption)
        entry["kind"] = "caption"
        entry["speaker"] = "caption"
        entry["id"] = entry.get("id") or f"{caption['panel_id']}_caption"
        per_panel.setdefault(caption["panel_id"], []).append(entry)
    for panel_id, items in per_panel.items():
        per_panel[panel_id] = sorted(items, key=placement_priority)
    return per_panel, data


def load_scroll_plan(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if not path or not path.exists():
        return {}, {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    spacing_map = {
        block["block_id"]: block.get("spacing", "medium")
        for block in data.get("blocks", [])
        if "block_id" in block
    }
    panel_to_block: dict[str, str] = {}
    for index, block in enumerate(data.get("blocks", []), start=1):
        panel_to_block[f"p{index:02d}"] = block.get("block_id", "")
    return spacing_map, panel_to_block


def placement_priority(item: dict[str, Any]) -> tuple[int, int]:
    speaker = normalize_speaker(item.get("speaker", "caption"), item)
    zone_rank = 0 if speaker == "caption" else 1
    text_len = len(item.get("text", ""))
    return (zone_rank, -text_len)


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
    return "mother" if speaker == "teacher" else speaker or "caption"


def iter_panel_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and PANEL_FILE_RE.match(path.name))


def tokenize_text(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    tokens = re.findall(r"\S+\s*", normalized)
    return tokens or list(normalized)


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


def fit_text_block(
    text: str,
    style: SpeakerStyle,
    font_path: str | None,
    max_text_width: int,
    max_text_height: int,
) -> tuple[list[str], ImageFont.ImageFont, tuple[int, int]] | None:
    scratch = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(scratch)
    for size in range(style.font_size, style.min_font_size - 1, -1):
        font = load_font(font_path, size)
        lines = wrap_korean_text(draw, text, font, max_text_width)
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=max(6, size // 4), align="center")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_text_width and height <= max_text_height:
            return lines, font, (width, height)
    return None


def compute_box_position(
    panel_size: tuple[int, int],
    zone: str,
    box_size: tuple[int, int],
    offset: tuple[float, float],
) -> tuple[int, int]:
    panel_w, panel_h = panel_size
    box_w, box_h = box_size
    ox, oy = offset
    if zone in {"top-left", "mid-left", "bottom-left"}:
        x = DEFAULT_OUTER_PADDING + int(panel_w * ox)
    elif zone in {"top-right", "mid-right", "bottom-right"}:
        x = panel_w - DEFAULT_OUTER_PADDING - box_w - int(panel_w * ox)
    else:
        x = int((panel_w - box_w) / 2 + panel_w * ox)

    if zone.startswith("top") or zone == "upper-mid":
        y = DEFAULT_OUTER_PADDING + int(panel_h * oy)
    elif zone.startswith("mid"):
        y = int((panel_h - box_h) / 2 + panel_h * oy)
    else:
        y = panel_h - DEFAULT_OUTER_PADDING - box_h - int(panel_h * abs(oy))
    return x, y


def clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    panel_w, panel_h = size
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    x1 = min(max(DEFAULT_OUTER_PADDING, x1), panel_w - DEFAULT_OUTER_PADDING - width)
    y1 = min(max(DEFAULT_OUTER_PADDING, y1), panel_h - DEFAULT_OUTER_PADDING - height)
    return (x1, y1, x1 + width, y1 + height)


def expand_box(box: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    return (box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount)


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


def detect_likely_text_regions(image: Image.Image, strength: float) -> list[dict[str, Any]]:
    if strength <= 0:
        return []
    panel_w, panel_h = image.size
    hsv = image.convert("HSV")
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    candidates: list[dict[str, Any]] = []
    size_options = ((0.72, 0.24), (0.62, 0.20), (0.55, 0.17))
    x_positions = (0.08, 0.16, 0.24)
    y_positions = (0.18, 0.26, 0.36, 0.46)

    for width_ratio, height_ratio in size_options:
        box_w = int(panel_w * width_ratio)
        box_h = int(panel_h * height_ratio)
        for xr in x_positions:
            for yr in y_positions:
                x1 = int(panel_w * xr)
                y1 = int(panel_h * yr)
                box = clamp_box((x1, y1, x1 + box_w, y1 + box_h), image.size)
                gray_stat = ImageStat.Stat(gray.crop(box))
                hsv_stat = ImageStat.Stat(hsv.crop(box))
                edge_stat = ImageStat.Stat(edges.crop(box))
                mean_luma = gray_stat.mean[0]
                std_luma = gray_stat.stddev[0]
                sat_mean = hsv_stat.mean[1]
                edge_mean = edge_stat.mean[0]
                cx = (box[0] + box[2]) / 2 / panel_w
                cy = (box[1] + box[3]) / 2 / panel_h
                center_bias = max(0.0, 22 - abs(cx - 0.5) * 38 - abs(cy - 0.42) * 46)
                score = edge_mean * 0.95 + std_luma * 0.45 + max(0.0, 90 - sat_mean) * 0.16 + center_bias
                if mean_luma < 35 or mean_luma > 240 or edge_mean < 10 or std_luma < 14:
                    continue
                candidates.append(
                    {
                        "box": box,
                        "score": round(score, 2),
                        "mean_luma": round(mean_luma, 2),
                        "std_luma": round(std_luma, 2),
                        "sat_mean": round(sat_mean, 2),
                        "edge_mean": round(edge_mean, 2),
                    }
                )

    selected: list[dict[str, Any]] = []
    threshold = 36 + strength * 18
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if candidate["score"] < threshold:
            continue
        if any(intersection_area(candidate["box"], item["box"]) > 0.35 * area(candidate["box"]) for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= (2 if strength >= 0.7 else 1):
            break
    return selected


def area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def build_tail_points(layout: LayoutResult, style: SpeakerStyle, panel_size: tuple[int, int]) -> list[tuple[int, int]] | None:
    if not style.tail:
        return None
    x1, y1, x2, y2 = layout.box
    panel_w, panel_h = panel_size
    base_y = y2 - 6
    available = max(26, x2 - x1 - 64)
    center_x = x1 + 32 + int(available * style.tail_bias)
    tip_x = int(center_x + (0.5 - style.tail_bias) * 42)
    tip_x = max(DEFAULT_OUTER_PADDING + 16, min(panel_w - DEFAULT_OUTER_PADDING - 16, tip_x))
    tip_y = min(panel_h - DEFAULT_OUTER_PADDING, y2 + int(panel_h * 0.08))
    base_half = 22
    return [(center_x - base_half, base_y), (center_x + base_half, base_y), (tip_x, tip_y)]


def select_layout_for_item(
    image: Image.Image,
    item: dict[str, Any],
    placed_boxes: list[tuple[int, int, int, int]],
    font_path: str | None,
    mode: str,
) -> LayoutResult:
    panel_w, panel_h = image.size
    speaker = normalize_speaker(item.get("speaker"), item)
    style = STYLE_MAP.get(speaker, STYLE_MAP["caption"])
    width_scale = MODE_WIDTH_SCALE[mode]
    preferred_zones = list(style.preferred_zones) + [zone for zone in FALLBACK_ZONES if zone not in style.preferred_zones]
    best: tuple[float, LayoutResult] | None = None

    for zone_index, zone in enumerate(preferred_zones):
        for offset_index, offset in enumerate(ZONE_OFFSETS.get(zone, [(0.0, 0.0)])):
            max_text_width = int(panel_w * style.max_width_ratio * width_scale) - style.text_padding[0] - style.text_padding[2]
            max_text_height = int(panel_h * style.max_height_ratio) - style.text_padding[1] - style.text_padding[3]
            fit = fit_text_block(item["text"], style, font_path, max_text_width, max_text_height)
            if not fit:
                continue
            lines, font, (text_w, text_h) = fit
            spacing = max(6, getattr(font, "size", style.font_size) // 4)
            outer_box = (
                0,
                0,
                text_w + style.text_padding[0] + style.text_padding[2],
                text_h + style.text_padding[1] + style.text_padding[3],
            )
            box_w = outer_box[2] - outer_box[0]
            box_h = outer_box[3] - outer_box[1]
            pos_x, pos_y = compute_box_position(image.size, zone, (box_w, box_h), offset)
            box = clamp_box((pos_x, pos_y, pos_x + box_w, pos_y + box_h), image.size)
            expanded = expand_box(box, DEFAULT_BOX_GAP)
            if any(boxes_intersect(expanded, expand_box(other, DEFAULT_BOX_GAP)) for other in placed_boxes):
                continue
            penalty, background = candidate_background_score(image, box)
            pref_bonus = 110 - zone_index * 11 - offset_index * 4
            vertical_bonus = max(0.0, 26 - (box[1] / panel_h) * 36) if zone.startswith("top") or zone == "upper-mid" else 10.0
            side_bonus = 8.0 if "left" in zone or "right" in zone else 4.0
            score = pref_bonus + vertical_bonus + side_bonus - penalty
            inner_box = (
                box[0] + style.text_padding[0],
                box[1] + style.text_padding[1],
                box[2] - style.text_padding[2],
                box[3] - style.text_padding[3],
            )
            result = LayoutResult(
                item=item,
                speaker=speaker,
                box=box,
                inner_box=inner_box,
                zone=zone,
                score=round(score, 2),
                lines=lines,
                font_size=getattr(font, "size", style.font_size),
                bubble_kind=style.bubble_kind,
                tail_points=None,
                background_metrics=background,
            )
            result.tail_points = build_tail_points(result, style, image.size)
            if best is None or score > best[0]:
                best = (score, result)

    if best is not None:
        return best[1]

    fallback_style = STYLE_MAP["caption"] if item.get("kind") == "caption" else style
    fallback_width = int(panel_w * fallback_style.max_width_ratio * width_scale)
    max_text_width = fallback_width - fallback_style.text_padding[0] - fallback_style.text_padding[2]
    max_text_height = int(panel_h * fallback_style.max_height_ratio) - fallback_style.text_padding[1] - fallback_style.text_padding[3]
    fit = fit_text_block(item["text"], fallback_style, font_path, max_text_width, max_text_height)
    if fit is None:
        lines = [item["text"]]
        font = load_font(font_path, fallback_style.min_font_size)
        scratch = Image.new("L", (8, 8), 0)
        draw = ImageDraw.Draw(scratch)
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=max(6, fallback_style.min_font_size // 4))
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    else:
        lines, font, (text_w, text_h) = fit
    box_w = text_w + fallback_style.text_padding[0] + fallback_style.text_padding[2]
    box_h = text_h + fallback_style.text_padding[1] + fallback_style.text_padding[3]
    fallback_zone = fallback_style.preferred_zones[0]
    pos_x, pos_y = compute_box_position(image.size, fallback_zone, (box_w, box_h), (0.0, 0.0))
    box = clamp_box((pos_x, pos_y, pos_x + box_w, pos_y + box_h), image.size)
    result = LayoutResult(
        item=item,
        speaker=speaker,
        box=box,
        inner_box=(
            box[0] + fallback_style.text_padding[0],
            box[1] + fallback_style.text_padding[1],
            box[2] - fallback_style.text_padding[2],
            box[3] - fallback_style.text_padding[3],
        ),
        zone=fallback_zone,
        score=-999.0,
        lines=lines,
        font_size=getattr(font, "size", fallback_style.min_font_size),
        bubble_kind=fallback_style.bubble_kind,
        tail_points=None,
        background_metrics={},
    )
    result.tail_points = build_tail_points(result, fallback_style, image.size)
    return result


def draw_soft_mask(
    base: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    radius: int,
    blur_radius: int,
) -> None:
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    softened = mask_layer.filter(ImageFilter.GaussianBlur(radius=max(0, blur_radius)))
    base.alpha_composite(softened)


def draw_ui_shape(base: Image.Image, layout: LayoutResult, font_path: str | None) -> None:
    style = STYLE_MAP.get(layout.speaker, STYLE_MAP["caption"])
    draw = ImageDraw.Draw(base)
    halo_box = expand_box(layout.box, 10)
    draw_soft_mask(base, halo_box, (245, 245, 245, 84), radius=style.radius + 10, blur_radius=5)
    draw.rounded_rectangle(layout.box, radius=style.radius, fill=style.fill, outline=style.outline, width=3)
    if layout.tail_points:
        draw.polygon(layout.tail_points, fill=style.fill, outline=style.outline)
        draw.line([layout.tail_points[0], layout.tail_points[2], layout.tail_points[1]], fill=style.outline, width=3)
    font = load_font(font_path, layout.font_size)
    text = "\n".join(layout.lines)
    spacing = max(6, layout.font_size // 4)
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    tx = layout.inner_box[0] + (layout.inner_box[2] - layout.inner_box[0] - text_w) / 2
    ty = layout.inner_box[1] + (layout.inner_box[3] - layout.inner_box[1] - text_h) / 2 - 1
    draw.multiline_text((tx, ty), text, font=font, fill=style.text_fill, spacing=spacing, align="center")


def render_panel(
    source_path: Path,
    output_path: Path,
    items: list[dict[str, Any]],
    font_path: str | None,
    mode: str,
    mask_strength: float,
) -> dict[str, Any]:
    base = Image.open(source_path).convert("RGBA")
    placements: list[LayoutResult] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for item in items:
        layout = select_layout_for_item(base, item, placed_boxes, font_path, mode)
        placements.append(layout)
        placed_boxes.append(layout.box)

    working = base.copy()
    auto_masks = detect_likely_text_regions(base, mask_strength)
    for mask in auto_masks:
        alpha = int(108 + mask_strength * 96)
        draw_soft_mask(working, tuple(mask["box"]), (244, 244, 244, alpha), radius=26, blur_radius=10)
    for layout in placements:
        draw_ui_shape(working, layout, font_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    working.convert("RGB").save(output_path)
    return {
        "panel_id": source_path.stem,
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "size": list(base.size),
        "masks": [
            {
                "box": list(mask["box"]),
                "score": mask["score"],
                "mean_luma": mask["mean_luma"],
                "std_luma": mask["std_luma"],
                "edge_mean": mask["edge_mean"],
                "sat_mean": mask["sat_mean"],
            }
            for mask in auto_masks
        ],
        "placements": [
            {
                "id": layout.item.get("id"),
                "kind": layout.item.get("kind"),
                "speaker": layout.speaker,
                "text": layout.item["text"],
                "zone": layout.zone,
                "score": layout.score,
                "font_size": layout.font_size,
                "bubble_kind": layout.bubble_kind,
                "box": list(layout.box),
                "inner_box": list(layout.inner_box),
                "tail_points": [list(point) for point in layout.tail_points] if layout.tail_points else None,
                "background_metrics": layout.background_metrics,
            }
            for layout in placements
        ],
    }


def compose_longscroll(
    panel_outputs: list[dict[str, Any]],
    output_dir: Path,
    episode_id: str,
    spacing_map: dict[str, str],
    panel_to_block: dict[str, str],
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

    longscroll_path = output_dir / f"{episode_id}_ui_longscroll.png"
    canvas.save(longscroll_path)
    return longscroll_path


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    lettering_path = Path(args.lettering)
    scroll_plan_path = Path(args.scroll_plan) if args.scroll_plan else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not lettering_path.exists():
        raise FileNotFoundError(f"Lettering file not found: {lettering_path}")

    font_path = resolve_font_path(args.font_path)
    panel_items, lettering_data = load_lettering(lettering_path)
    spacing_map, panel_to_block = load_scroll_plan(scroll_plan_path)
    panel_paths = iter_panel_paths(input_dir)
    if not panel_paths:
        raise RuntimeError(f"No panel images like p01.png found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_outputs: list[dict[str, Any]] = []
    created_files: list[Path] = []

    for panel_path in panel_paths:
        output_path = output_dir / panel_path.name
        panel_debug = render_panel(
            source_path=panel_path,
            output_path=output_path,
            items=panel_items.get(panel_path.stem, []),
            font_path=font_path,
            mode=args.mode,
            mask_strength=args.mask_strength,
        )
        panel_outputs.append(panel_debug)
        created_files.append(output_path)

    longscroll_path = None
    if args.compose_longscroll and scroll_plan_path:
        longscroll_path = compose_longscroll(
            panel_outputs=panel_outputs,
            output_dir=output_dir,
            episode_id=str(lettering_data.get("episode", "episode")),
            spacing_map=spacing_map,
            panel_to_block=panel_to_block,
        )
        if longscroll_path:
            created_files.append(longscroll_path)

    command = " ".join(shlex.quote(part) for part in [os.path.basename(__file__), *os.sys.argv[1:]])
    manifest = {
        "episode": lettering_data.get("episode"),
        "language": lettering_data.get("language", "ko"),
        "mode": args.mode,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "lettering": str(lettering_path.resolve()),
        "scroll_plan": str(scroll_plan_path.resolve()) if scroll_plan_path else None,
        "font_path": font_path,
        "mask_strength": args.mask_strength,
        "command": command,
        "panels": panel_outputs,
        "longscroll": str(longscroll_path.resolve()) if longscroll_path else None,
    }
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    created_files.append(manifest_path)

    print("Created files:")
    for path in created_files:
        print(f"- {relative_or_absolute(path)}")
    print("Example command:")
    print(command)


if __name__ == "__main__":
    main()

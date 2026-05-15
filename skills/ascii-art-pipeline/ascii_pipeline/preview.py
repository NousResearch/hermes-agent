from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .diagnostics import load_frames_from_path

DEFAULT_BG = (16, 18, 24)
DEFAULT_FG = (235, 225, 205)
FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/MesloLGS NF Regular.ttf",
    "/Library/Fonts/JetBrainsMono-Regular.ttf",
]


def _load_font(font_size: int, font_path: str | None = None) -> ImageFont.ImageFont:
    candidates = [font_path] if font_path else []
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, font_size)
    return ImageFont.load_default()


def _normalize_lines(lines: Iterable[str]) -> list[str]:
    rendered = [str(line) for line in lines]
    if not rendered:
        return [""]
    width = max(len(line) for line in rendered)
    return [line.ljust(width) for line in rendered]


def render_lines_to_image(
    lines: Iterable[str],
    output_path: str | Path,
    *,
    font_size: int = 18,
    margin: int = 24,
    line_spacing: int = 2,
    fg: tuple[int, int, int] = DEFAULT_FG,
    bg: tuple[int, int, int] = DEFAULT_BG,
    font_path: str | None = None,
) -> Path:
    normalized = _normalize_lines(lines)
    font = _load_font(font_size, font_path=font_path)
    bbox = font.getbbox("M")
    char_width = max(1, bbox[2] - bbox[0])
    try:
        ascent, descent = font.getmetrics()
        char_height = max(1, ascent + descent)
    except AttributeError:
        char_height = max(1, bbox[3] - bbox[1])

    image_width = margin * 2 + char_width * max(len(line) for line in normalized)
    image_height = margin * 2 + len(normalized) * (char_height + line_spacing)

    canvas = Image.new("RGB", (image_width, image_height), bg)
    draw = ImageDraw.Draw(canvas)
    for row, line in enumerate(normalized):
        y = margin + row * (char_height + line_spacing)
        draw.text((margin, y), line, fill=fg, font=font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def render_preview(
    input_path: str | Path,
    output_path: str | Path,
    *,
    frame_index: int = 0,
    font_size: int = 18,
    font_path: str | None = None,
) -> Path:
    frames = load_frames_from_path(input_path)
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"frame_index {frame_index} out of range for {len(frames)} frame(s)")
    return render_lines_to_image(
        frames[frame_index].lines,
        output_path,
        font_size=font_size,
        font_path=font_path,
    )

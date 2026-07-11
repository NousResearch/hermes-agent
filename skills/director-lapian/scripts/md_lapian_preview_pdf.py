#!/usr/bin/env python3
"""Render a Markdown lapian report into a readable PDF preview with images.

This is a fallback when Word / LibreOffice PDF export stalls on very large
image-heavy reports. It aims for audit-friendly preview output, not perfect
Markdown typography.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

import fitz


IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def clean_image_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    else:
        quote_idx = value.find(' "')
        if quote_idx != -1:
            value = value[:quote_idx]
    return unquote(value.strip())


def resolve_image(raw: str, md_dir: Path) -> Path | None:
    target = clean_image_target(raw)
    if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
        return None
    path = Path(target)
    if not path.is_absolute():
        path = md_dir / path
    return path if path.exists() else None


def choose_font() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def strip_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("|", "  ")
    text = re.sub(r"^\s*[-:]{3,}(\s+[-:]{3,})+\s*$", "", text)
    return text.strip()


def wrap_text(text: str, limit: int = 64) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for token in re.split(r"(\s+)", text):
        if len(current) + len(token) > limit and current:
            lines.append(current.rstrip())
            current = token.lstrip()
        else:
            current += token
    if current.strip():
        lines.append(current.rstrip())
    result: list[str] = []
    for line in lines:
        if len(line) <= limit:
            result.append(line)
            continue
        for start in range(0, len(line), limit):
            result.append(line[start : start + limit])
    return result


class PdfWriter:
    def __init__(self, out: Path, fontfile: str | None, image_width: float) -> None:
        self.out = out
        self.doc = fitz.open()
        self.fontfile = fontfile
        self.image_width = image_width
        self.page = None
        self.y = 0.0
        self.width = 595.0
        self.height = 842.0
        self.margin = 42.0
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=self.width, height=self.height)
        self.y = self.margin

    def ensure_space(self, need: float) -> None:
        if self.y + need > self.height - self.margin:
            self.new_page()

    def write_text(self, text: str, size: float = 9.0, gap: float = 2.0) -> None:
        lines = wrap_text(text, limit=72 if size <= 9 else 48)
        if not lines:
            self.y += 4
            return
        line_h = size * 1.45
        for line in lines:
            self.ensure_space(line_h + gap)
            rect = fitz.Rect(self.margin, self.y, self.width - self.margin, self.y + line_h + 2)
            self.page.insert_textbox(rect, line, fontsize=size, fontfile=self.fontfile, fontname="lapianfont")
            self.y += line_h + gap

    def write_image(self, path: Path, caption: str) -> None:
        try:
            pix = fitz.Pixmap(str(path))
            w, h = pix.width, pix.height
        except Exception as exc:
            self.write_text(f"[图片无法读取] {caption}: {path} ({exc})", size=8)
            return
        max_w = min(self.image_width, self.width - 2 * self.margin)
        scale = max_w / max(w, 1)
        draw_w = max_w
        draw_h = h * scale
        if draw_h > 230:
            draw_h = 230
            draw_w = w * (draw_h / max(h, 1))
        self.ensure_space(draw_h + 22)
        self.write_text(f"[图] {caption}", size=8, gap=0)
        rect = fitz.Rect(self.margin, self.y, self.margin + draw_w, self.y + draw_h)
        try:
            self.page.insert_image(rect, filename=str(path))
        except Exception as exc:
            self.write_text(f"[图片插入失败] {path} ({exc})", size=8)
            return
        self.y += draw_h + 8

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(self.out), garbage=4, deflate=True)
        self.doc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Markdown lapian report into PDF preview.")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-width", type=float, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    md = args.markdown.resolve()
    text = md.read_text(encoding="utf-8-sig")
    writer = PdfWriter(args.out.resolve(), choose_font(), args.image_width)
    image_count = 0
    missing_images: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        images = list(IMAGE_RE.finditer(line))
        line_without_images = IMAGE_RE.sub("", line)
        clean = strip_markdown(line_without_images)
        if clean:
            size = 14 if raw_line.startswith("# ") else 11 if raw_line.startswith("## ") else 9
            writer.write_text(clean, size=size)
        for match in images:
            image_count += 1
            image = resolve_image(match.group(2), md.parent)
            if not image:
                missing_images.append(clean_image_target(match.group(2)))
                writer.write_text(f"[缺图] {match.group(1)} {clean_image_target(match.group(2))}", size=8)
            else:
                writer.write_image(image, match.group(1) or image.name)

    writer.save()
    print(
        {
            "out": str(args.out.resolve()),
            "markdown": str(md),
            "image_refs": image_count,
            "missing_images": len(missing_images),
        }
    )
    if missing_images:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

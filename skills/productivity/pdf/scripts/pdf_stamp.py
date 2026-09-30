#!/usr/bin/env python3
"""Stamp text or an image at coordinates onto selected PDF pages.

Builds an in-memory single-page overlay with reportlab, then merges it
onto each selected page with pypdf. Coordinates are PDF points, origin
bottom-left. Covers 'sign here' arrows, diagonal DRAFT banners, and
page-corner labels.

Examples:
  pdf_stamp.py in.pdf -o out.pdf --text "DRAFT" --x 200 --y 400 \
      --font-size 60 --rotation 45 --opacity 0.3 --color "#cc0000"
  pdf_stamp.py in.pdf -o out.pdf --image sig.png --x 400 --y 60 \
      --width 120 --pages 3
"""
from __future__ import annotations

import argparse
import io
import json
import sys


def parse_pages(spec: str, page_count: int) -> list[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            start = int(start_s) if start_s else 1
            end = int(end_s) if end_s else page_count
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    bad = [p for p in pages if not 1 <= p <= page_count]
    if bad:
        raise ValueError(f"pages out of range 1-{page_count}: {sorted(bad)}")
    return sorted(pages)


def build_overlay(args, page_width: float, page_height: float) -> bytes:
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.saveState()
    try:
        c.setFillAlpha(float(args.opacity))
        c.setStrokeAlpha(float(args.opacity))
    except Exception:
        pass  # very old reportlab: no alpha support
    c.translate(float(args.x), float(args.y))
    if args.rotation:
        c.rotate(float(args.rotation))
    if args.text:
        c.setFont(args.font, float(args.font_size))
        c.setFillColor(HexColor(args.color))
        c.drawString(0, 0, args.text)
    else:
        kwargs = {}
        if args.width:
            kwargs["width"] = float(args.width)
        if args.height:
            kwargs["height"] = float(args.height)
        if "width" in kwargs and "height" not in kwargs:
            from PIL import Image as PILImage
            with PILImage.open(args.image) as im:
                kwargs["height"] = kwargs["width"] * im.height / im.width
        c.drawImage(args.image, 0, 0, mask="auto", **kwargs)
    c.restoreState()
    c.save()
    return buf.getvalue()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Stamp text or an image onto PDF pages.")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--text", help="Text to stamp")
    what.add_argument("--image", help="Image file to stamp (PNG/JPEG)")
    parser.add_argument("--x", type=float, required=True, help="X in points (origin bottom-left)")
    parser.add_argument("--y", type=float, required=True, help="Y in points")
    parser.add_argument("--pages", default="1-", help="1-based ranges, e.g. '1-3,5' (default: all)")
    parser.add_argument("--font", default="Helvetica", help="Font name (default Helvetica)")
    parser.add_argument("--font-size", type=float, default=24, help="Font size in points")
    parser.add_argument("--color", default="#000000", help="Text color as #RRGGBB")
    parser.add_argument("--rotation", type=float, default=0, help="Degrees counterclockwise")
    parser.add_argument("--opacity", type=float, default=1.0, help="0.0-1.0 (default 1.0)")
    parser.add_argument("--width", type=float, help="Image width in points")
    parser.add_argument("--height", type=float, help="Image height in points")
    parser.add_argument("--under", action="store_true",
                        help="Place the stamp under existing content instead of over it")
    parser.add_argument("--password", help="Password if the input is encrypted")
    args = parser.parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install pypdf reportlab'",
              file=sys.stderr)
        return 2

    reader = PdfReader(args.pdf)
    if reader.is_encrypted:
        if args.password is None or not reader.decrypt(args.password):
            print("Error: input is encrypted; pass --password", file=sys.stderr)
            return 3
    try:
        pages = set(parse_pages(args.pages, len(reader.pages)))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 4

    writer = PdfWriter()
    overlay_cache: dict[tuple[float, float], object] = {}
    for idx, page in enumerate(reader.pages, start=1):
        if idx in pages:
            size = (float(page.mediabox.width), float(page.mediabox.height))
            if size not in overlay_cache:
                overlay_pdf = PdfReader(io.BytesIO(build_overlay(args, *size)))
                overlay_cache[size] = overlay_pdf.pages[0]
            stamp = overlay_cache[size]
            if args.under:
                page.merge_page(stamp, over=False)
            else:
                page.merge_page(stamp)
        writer.add_page(page)
    with open(args.output, "wb") as fh:
        writer.write(fh)
    print(json.dumps({"output": args.output, "stamped_pages": sorted(pages),
                      "kind": "text" if args.text else "image"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

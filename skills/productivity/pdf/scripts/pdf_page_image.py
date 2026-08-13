#!/usr/bin/env python3
"""Export PDF pages as PNG images at a chosen DPI.

Rasterizer fallback chain: pypdfium2 (pip) -> pdftoppm (poppler-utils).
When neither is available, exits 0 with {"rendered": false, "missing": [...]}
so callers can branch instead of crashing.

Typical uses: visual verification with a vision model, and exporting
image-only (scanned) pages for hand-off to the ocr-and-documents skill.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_pages(spec: str, page_count: int) -> list[int]:
    """'1-3,5,9-' (1-based, inclusive) -> sorted page list."""
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


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Export PDF pages as PNG images.")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("--pages", default="1-", help="1-based ranges, e.g. '1-3,5' (default: all)")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default 150)")
    parser.add_argument("--out-dir", required=True, help="Directory for PNG files")
    parser.add_argument("--prefix", default="page", help="Output filename prefix (default 'page')")
    parser.add_argument("--password", help="Password for encrypted PDFs")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _raster

    if not _raster.available_backends():
        json.dump({"rendered": False, "missing": _raster.missing_hints()}, sys.stdout)
        print()
        return 0

    try:
        from pypdf import PdfReader
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install pypdf'", file=sys.stderr)
        return 2
    reader = PdfReader(args.pdf)
    if reader.is_encrypted:
        if args.password is None or not reader.decrypt(args.password):
            print("File is encrypted; pass --password.", file=sys.stderr)
            return 3
    page_count = len(reader.pages)

    try:
        pages = parse_pages(args.pages, page_count)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 4

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for pageno in pages:
        img = _raster.rasterize_page(args.pdf, pageno, dpi=args.dpi, password=args.password)
        if img is None:
            json.dump({"rendered": False, "missing": _raster.missing_hints()}, sys.stdout)
            print()
            return 0
        out_path = out_dir / f"{args.prefix}{pageno:03d}.png"
        img.save(out_path)
        files.append(str(out_path))
    json.dump({"rendered": True, "dpi": args.dpi, "page_count": page_count,
               "files": files}, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Extract page ranges from a PDF, optionally rotating and/or compressing pages."""
from __future__ import annotations

import argparse
import json
import sys


def parse_pages(spec: str, page_count: int) -> list[int]:
    """Parse a 1-based page spec like '1-3,5,9-' into 0-based indices."""
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            start = int(start_s) if start_s else 1
            end = int(end_s) if end_s else page_count
        else:
            start = end = int(part)
        if start < 1 or end > page_count or start > end:
            raise ValueError(f"Page range {part!r} out of bounds (1-{page_count})")
        indices.extend(range(start - 1, end))
    return indices


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Split/extract pages from a PDF (pypdf). Pages are 1-based: '1-3,5,9-'.")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("--pages", required=True, help="1-based page spec, e.g. '1-3,5,9-'")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    parser.add_argument("--rotate", type=int, default=0,
                        help="Rotate extracted pages clockwise (multiple of 90)")
    parser.add_argument("--compress", action="store_true",
                        help="Deflate content streams (modest savings; does not recompress images)")
    parser.add_argument("--password", help="Password if the input is encrypted")
    args = parser.parse_args()

    if args.rotate % 90 != 0:
        print("Error: --rotate must be a multiple of 90", file=sys.stderr)
        return 2

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install pypdf'", file=sys.stderr)
        return 2

    reader = PdfReader(args.pdf)
    if reader.is_encrypted:
        if args.password is None or not reader.decrypt(args.password):
            print("Error: input is encrypted; pass --password", file=sys.stderr)
            return 3
    try:
        indices = parse_pages(args.pages, len(reader.pages))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    writer = PdfWriter()
    for idx in indices:
        page = reader.pages[idx]
        if args.rotate:
            page.rotate(args.rotate)
        writer.add_page(page)
    if args.compress:
        for page in writer.pages:
            page.compress_content_streams()
    with open(args.output, "wb") as fh:
        writer.write(fh)
    print(json.dumps({"output": args.output, "page_count": len(indices), "rotated": args.rotate}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

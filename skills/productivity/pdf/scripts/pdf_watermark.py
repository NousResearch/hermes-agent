#!/usr/bin/env python3
"""Stamp/watermark every page of a PDF with page 1 of another PDF."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Overlay (stamp) or underlay (watermark) a one-page PDF onto every page.")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("--stamp", required=True, help="One-page PDF to apply (page 1 is used)")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    parser.add_argument("--under", action="store_true",
                        help="Place stamp under the page content (background watermark)")
    parser.add_argument("--password", help="Password if the input is encrypted")
    args = parser.parse_args()

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
    stamp_page = PdfReader(args.stamp).pages[0]

    writer = PdfWriter()
    writer.append(reader)
    for page in writer.pages:
        page.merge_page(stamp_page, over=not args.under)
    with open(args.output, "wb") as fh:
        writer.write(fh)
    print(json.dumps({"output": args.output, "page_count": len(writer.pages),
                      "mode": "under" if args.under else "over"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

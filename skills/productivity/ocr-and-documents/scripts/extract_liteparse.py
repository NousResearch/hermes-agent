#!/usr/bin/env python3
"""Extract fast plain text from local text PDFs using liteparse.

liteparse is an optional speed-first fallback for local PDFs when plain text and
reading order matter more than markdown/table fidelity. Keep pymupdf4llm as the
default for agent-ready markdown; use marker-pdf for scanned/OCR-heavy docs.

Install:
    uv pip install liteparse
    # or: pip install liteparse

Usage:
    python extract_liteparse.py document.pdf
    python extract_liteparse.py document.pdf --pages 1-3
    python extract_liteparse.py document.pdf --max-pages 5
    python extract_liteparse.py document.pdf --ocr       # slower; for light OCR attempts only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast plain-text PDF extraction with optional liteparse."
    )
    parser.add_argument("pdf", type=Path, help="Local PDF to extract")
    parser.add_argument(
        "--pages",
        help=(
            "liteparse target pages, e.g. '1', '1-3', or '1,3'. "
            "Uses liteparse's 1-based page selector."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Stop after this many pages, useful for quick reading-order checks.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable liteparse OCR. Default is disabled so text PDFs stay fast.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show liteparse timing output on stderr.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pdf.exists():
        print(f"Missing PDF: {args.pdf}", file=sys.stderr)
        return 2

    try:
        from liteparse import LiteParse
    except ImportError:
        print(
            "liteparse is not installed. Install it with: uv pip install liteparse",
            file=sys.stderr,
        )
        return 1

    parser = LiteParse(
        ocr_enabled=args.ocr,
        target_pages=args.pages,
        max_pages=args.max_pages,
        output_format="markdown",
        quiet=not args.verbose,
    )
    result = parser.parse(args.pdf)
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

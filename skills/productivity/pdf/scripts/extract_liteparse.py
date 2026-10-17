#!/usr/bin/env python3
"""Extract fast text from local PDFs using liteparse.

liteparse is an optional speed-first fallback for local PDFs when text and
reading order matter more than rich markdown/table fidelity. Keep pymupdf4llm
as the default for agent-ready markdown; use marker-pdf for scanned/OCR-heavy
docs.

Install:
    uv add 'liteparse==2.10.1'                         # existing uv project
    uv venv && uv pip install 'liteparse==2.10.1'      # standalone virtual environment

Usage:
    python extract_liteparse.py document.pdf
    python extract_liteparse.py document.pdf --pages 1-3
    python extract_liteparse.py document.pdf --max-pages 5
    python extract_liteparse.py document.pdf --ocr       # enable OCR (slower)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    """Return a positive integer for argparse options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast PDF text extraction with optional liteparse."
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
        type=positive_int,
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pdf.exists():
        print(f"Missing PDF: {args.pdf}", file=sys.stderr)
        return 2

    try:
        from liteparse import LiteParse
    except ImportError:
        print(
            "liteparse is not installed. In a uv project, run: "
            "uv add 'liteparse==2.10.1'\n"
            "For a standalone environment, run: "
            "uv venv && uv pip install 'liteparse==2.10.1'",
            file=sys.stderr,
        )
        return 1

    try:
        parser = LiteParse(
            ocr_enabled=args.ocr,
            target_pages=args.pages,
            max_pages=args.max_pages,
            output_format="markdown",
            quiet=not args.verbose,
        )
        result = parser.parse(args.pdf)
    except Exception as exc:
        print(f"liteparse failed: {exc}", file=sys.stderr)
        return 1

    text = getattr(result, "text", None)
    if not isinstance(text, str) or not text.strip():
        print("liteparse returned no text", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

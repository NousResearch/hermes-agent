#!/usr/bin/env python3
"""Check that runtime context files stay within a managed character budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUNTIME_CONTEXT_EXPLANATION = (
    "AGENTS.md is injected into every coding-agent session"
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("budget values must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check a context file's character count against warn and max budgets."
        )
    )
    parser.add_argument("file", type=Path, help="context file to check, e.g. AGENTS.md")
    parser.add_argument(
        "--warn-chars",
        type=_positive_int,
        default=30_000,
        help="warning budget in decoded characters (default: 30000)",
    )
    parser.add_argument(
        "--max-chars",
        type=_positive_int,
        default=38_000,
        help="maximum budget in decoded characters (default: 38000)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="text encoding used to decode the file (default: utf-8)",
    )
    parser.add_argument(
        "--strict-warning",
        action="store_true",
        help="exit nonzero when the file exceeds --warn-chars",
    )
    return parser


def _read_context_file(path: Path, encoding: str) -> tuple[str, int]:
    raw = path.read_bytes()
    return raw.decode(encoding), len(raw)


def check_file(
    path: Path,
    *,
    warn_chars: int,
    max_chars: int,
    encoding: str,
    strict_warning: bool = False,
) -> int:
    if warn_chars > max_chars:
        print(
            f"error: --warn-chars ({warn_chars}) must be <= --max-chars ({max_chars})",
            file=sys.stderr,
        )
        return 2

    if not path.exists():
        print(f"error: context file not found: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"error: context path is not a file: {path}", file=sys.stderr)
        return 2

    try:
        text, byte_count = _read_context_file(path, encoding)
    except UnicodeDecodeError as exc:
        print(
            f"error: could not decode {path} with encoding {encoding!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    char_count = len(text)
    label = str(path)
    print(
        f"{label}: chars={char_count} bytes={byte_count} "
        f"warn_chars={warn_chars} max_chars={max_chars}"
    )

    if char_count > max_chars:
        print(
            (
                f"error: {label} exceeds the context-file max budget "
                f"({char_count} > {max_chars} chars). "
                f"{RUNTIME_CONTEXT_EXPLANATION}; oversized runtime context "
                "increases latency/cost and can be truncated, hiding critical "
                "instructions. Keep invariants inline; move explanatory "
                "reference material to docs/reference and keep only short "
                "runtime-critical summaries plus links inline."
            ),
            file=sys.stderr,
        )
        return 1

    if char_count > warn_chars:
        print(
            (
                f"warning: {label} exceeds the warning budget "
                f"({char_count} > {warn_chars} chars). "
                "Consider moving long rationale, examples, and subsystem "
                "reference material to docs/reference before it reaches the "
                "hard max."
            ),
            file=sys.stderr,
        )
        return 1 if strict_warning else 0

    print(f"ok: {label} is within the context-file warning budget.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return check_file(
        args.file,
        warn_chars=args.warn_chars,
        max_chars=args.max_chars,
        encoding=args.encoding,
        strict_warning=args.strict_warning,
    )


if __name__ == "__main__":
    raise SystemExit(main())

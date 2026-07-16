#!/usr/bin/env python3
"""CLI wrapper for scripts/ci/verification_bundle.py.

This command is read-only: it writes rendered output only to stdout. Redirect
stdout at the shell if a report file is needed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
CI_DIR = SCRIPT_DIR / "ci"
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

from verification_bundle import format_json, format_markdown, suggest_bundle  # noqa: E402


def _validate_base_ref(base: str) -> str:
    value = str(base).strip()
    if not value:
        raise RuntimeError("base ref must not be empty")
    if value.startswith("-"):
        raise RuntimeError("base ref must not start with '-' or look like a git option")
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise RuntimeError("base ref must not contain whitespace or control characters")
    return value


def _paths_from_git(base: str) -> list[str]:
    safe_base = _validate_base_ref(base)
    try:
        proc = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--name-only", "--end-of-options", f"{safe_base}...HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("git diff timed out while collecting changed paths") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "git diff failed"
        raise RuntimeError(detail)
    return proc.stdout.splitlines()


def _collect_paths(args: argparse.Namespace) -> list[str]:
    if args.paths is not None:
        return list(args.paths)
    if args.read_stdin:
        return sys.stdin.read().splitlines()
    if args.from_git:
        return _paths_from_git(args.base)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest Hermes verification commands from changed paths. Writes only to stdout."
    )
    parser.add_argument("--paths", nargs="*", help="Changed paths to classify")
    parser.add_argument("--stdin", dest="read_stdin", action="store_true", help="Read changed paths from stdin")
    parser.add_argument("--from-git", action="store_true", help="Use git diff --name-only BASE...HEAD")
    parser.add_argument("--base", default="origin/main", help="Base ref for --from-git (default: origin/main)")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = _collect_paths(args)
        bundle = suggest_bundle(paths)
        rendered = format_json(bundle) if args.format == "json" else format_markdown(bundle)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

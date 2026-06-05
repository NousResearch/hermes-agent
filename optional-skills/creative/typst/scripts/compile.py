"""Wrapper around the `typst` CLI for the typst skill.

Used by the skill's verification step. Wraps `typst compile` so the agent
can call a single Python command rather than building the right `typst`
incantation each time. Also exposes --self-test, which invokes `typst
--version` and reports success — used by SKILL.md's verification section
to confirm the binary is on PATH.

Stdlib only. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_typst() -> str | None:
    """Return the path to the typst binary, or None if not found."""
    return shutil.which("typst")


def self_test() -> int:
    """Invoke `typst --version` and report success/failure."""
    binary = find_typst()
    if binary is None:
        print("FAIL: typst binary not found on PATH", file=sys.stderr)
        print("  install: curl -fsSL https://typst.community/typst-install/install.sh | bash", file=sys.stderr)
        return 1
    result = subprocess.run(
        [binary, "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        print(f"FAIL: typst --version exited {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(result.stdout.strip())
    print("OK: typst binary invoked successfully")
    return 0


def compile(input_path: str, output_path: str, fmt: str = "pdf") -> int:
    """Compile input.typ to output.{pdf,html,png}."""
    binary = find_typst()
    if binary is None:
        print("FAIL: typst binary not found on PATH", file=sys.stderr)
        return 1
    in_path = Path(input_path)
    if not in_path.is_file():
        print(f"FAIL: input not found: {in_path}", file=sys.stderr)
        return 1
    result = subprocess.run(
        [binary, "compile", "--format", fmt, str(in_path), output_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"FAIL: typst compile exited {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(f"wrote {output_path}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true", help="Verify typst is installed.")
    parser.add_argument("--input", help="Input .typ file.")
    parser.add_argument("--output", help="Output file path.")
    parser.add_argument(
        "--format",
        choices=("pdf", "html", "png", "svg"),
        default="pdf",
        help="Output format (default: pdf).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return self_test()
    if not args.input or not args.output:
        print("FAIL: --input and --output are required (or pass --self-test)", file=sys.stderr)
        return 2
    return compile(args.input, args.output, args.format)


if __name__ == "__main__":
    raise SystemExit(main())

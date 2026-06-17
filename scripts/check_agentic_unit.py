#!/usr/bin/env python3
"""Validate Hermes agentic-unit Markdown records.

This intentionally avoids depending on a user-local absolute path. The docs
checked here are part of this repository, so the validation contract needs to
run wherever the checkout is cloned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_FIELDS = (
    "unit_id",
    "surface",
    "goal",
    "current_state",
    "authority_boundary",
    "verification_criteria",
    "log_location",
    "completion_condition",
    "contract_category",
    "status",
)


def _iter_markdown(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".md":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.md") if p.is_file()))
    return files


def _has_agentic_header(text: str) -> bool:
    return "unit_id:" in text or "contract_category:" in text


def _missing_fields(text: str) -> list[str]:
    return [field for field in REQUIRED_FIELDS if f"{field}:" not in text]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--discover", action="append", default=[], type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])
    targets = [*args.paths, *args.discover]
    if not targets:
        targets = [Path("docs/plans"), Path("docs/playbooks")]

    files = _iter_markdown(targets)
    failures: list[str] = []
    checked = 0

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _has_agentic_header(text):
            continue
        checked += 1
        missing = _missing_fields(text)
        if missing:
            failures.append(f"{path}: missing {', '.join(missing)}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(f"PASS: agentic_unit_files={checked} markdown_files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

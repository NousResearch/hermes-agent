#!/usr/bin/env python3
"""RTL guard: apps/desktop must use logical direction Tailwind classes only.

The desktop ships a full Persian (RTL) locale. Two class families silently
break mirrored layouts while every LTR screenshot stays pixel-identical:

  1. ``rtl:`` / ``ltr:`` Tailwind variants — they hardcode one direction's
     overrides and fight the logical-class system (the web dashboard had a
     real bug of exactly this shape: ``rtl:pr-*`` overriding responsive
     ``pe-*`` classes depending on source order).
  2. Physical padding/margin classes — ``pl-/pr-/ml-/mr-`` (and negatives,
     arbitrary values, ``auto``). Under RTL they pad the wrong edge. The
     logical equivalents ``ps-/pe-/ms-/me-`` behave correctly in both
     directions and are identical to the physical classes in LTR.

This checker greps ``apps/desktop`` TypeScript for both families and fails
when either appears, so the migration stays migrated. Symmetric classes
(``px-/py-/mx-/my-``) and absolute-position ``left-/right-`` are out of
scope — they are direction-neutral / legitimately needed.

Dependency-free (stdlib only, like scripts/check_guide_walkthrough_sync.py)
so a CI job can run it before any setup step.

Usage:
    python scripts/check_desktop_rtl_classes.py            # gate mode
    python scripts/check_desktop_rtl_classes.py --all      # explicit full scan
    python scripts/check_desktop_rtl_classes.py path/...   # scan subset

Exit status:
    0 — no physical-direction or rtl:/ltr: variants found (or suppressed)
    1 — at least one unsuppressed match

Suppress an intentional use with a trailing comment marker:
    className="pl-2"  // desktop-rtl: ok — optically required, not mirrored
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = [REPO_ROOT / "apps" / "desktop" / "src", REPO_ROOT / "apps" / "desktop" / "electron"]
SCAN_SUFFIXES = {".ts", ".tsx"}

SUPPRESS_MARKER = re.compile(r"desktop-rtl\s*:\s*ok\b", re.IGNORECASE)

# Left boundary keeps us inside class-string tokens only: start of line,
# whitespace, quotes, backtick, template-literal brace, or a Tailwind
# variant prefix ("md:pl-2", "hover:-ml-1"). This avoids matching prose,
# identifiers (paddingLeft), and CSS files.
_BOUNDARY = r"(?<![\w-])"

# rtl:/ltr: Tailwind variants. Must NOT match `dir="rtl"` (no colon) or
# object keys like `{ rtl: true }` are excluded by requiring the variant to
# be followed by a class-ish token (letter, digit, -, or [).
VARIANT_RE = re.compile(_BOUNDARY + r"(rtl|ltr):(?=[\w\-\[])")

# Physical direction classes with a value: auto | number | arbitrary [..].
# Covers pl/pr/ml/mr, negatives, and variant-prefixed forms (md:pr-4).
PHYSICAL_RE = re.compile(
    _BOUNDARY + r"(-)?(p|m)([lr])-(auto|\d+(?:\.\d+)?|\[[^\]]+\])"
)

LETTER_MAP = {"l": "s", "r": "e"}


def logical_suggestion(match: re.Match) -> str:
    sign, prop, side, value = match.groups()
    return f"{'-' if sign else ''}{prop}{LETTER_MAP[side]}-{value}"


def iter_files(paths: list[Path]) -> list[Path]:
    roots = paths if paths else SCAN_ROOTS
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix in SCAN_SUFFIXES:
            files.append(root)
        elif root.is_dir():
            files.extend(
                p for p in sorted(root.rglob("*"))
                if p.suffix in SCAN_SUFFIXES and ".test." not in p.name
            )
    return files


def check_file(path: Path) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return findings
    for lineno, line in enumerate(lines, start=1):
        if SUPPRESS_MARKER.search(line):
            continue
        for match in VARIANT_RE.finditer(line):
            findings.append((lineno, f"rtl/ltr variant '{match.group(0).strip()}'", match.group(0).strip()))
        for match in PHYSICAL_RE.finditer(line):
            token = match.group(0).strip()
            findings.append((lineno, f"physical class '{token}' -> '{logical_suggestion(match)}'", token))
    return findings


def main() -> int:
    # Windows consoles default to a legacy codepage (cp1252); the checker's
    # output is ASCII anyway, but a stray Unicode char in a path must never
    # crash the gate — degrade instead of dying.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="full scan (default without paths)")
    parser.add_argument("paths", nargs="*", type=Path, help="files or dirs to scan")
    args = parser.parse_args()

    files = iter_files([Path(p) for p in args.paths])
    total = 0
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        for lineno, message, _token in check_file(path):
            if total == 0:
                print(f"{rel.name} — physical/rtl classes found (use logical ps/pe/ms/me):")
            print(f"  {rel}:{lineno}: {message}")
            total += 1

    if total:
        print(f"\n{total} desktop RTL class violation(s). "
              "Use logical classes (ps/pe/ms/me), drop rtl:/ltr: variants, "
              "or suppress with '// desktop-rtl: ok'.")
        return 1
    print(f"desktop RTL classes: clean ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

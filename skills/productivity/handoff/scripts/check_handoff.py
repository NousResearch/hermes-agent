#!/usr/bin/env python3
"""Validate a HANDOFF.md against the handoff skill's rules.

Mechanical checks only: the six required sections, no section left as an
empty placeholder, no obvious secret literals, and the closing line that
tells the next session what to do. Judgment (is the root cause really a
root cause, is the next step really testable) stays with the reader.

    python check_handoff.py HANDOFF.md
    python check_handoff.py HANDOFF.md --json

Exit codes: 0 clean, 1 problems found, 2 file missing or unreadable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED = (
    "Current Task",
    "Done",
    "Blockers",
    "Next Steps",
    "Landmines",
    "Suggested Skills",
)

CLOSING = "Next session: read HANDOFF.md first"

HEADING = re.compile(r"^##\s+(?P<name>[^#].*?)\s*$")
FENCE = re.compile(r"^\s*(?:```|~~~)")

PLACEHOLDER = re.compile(
    r"^(?:[-*]\s*)?(?:<[^>]*>|TODO|TBD|FIXME|N/?A|\.\.\.+|…)$",
    re.IGNORECASE,
)
# An explicitly empty section is a real answer: "no blockers" is information.
EXPLICIT_EMPTY = re.compile(
    r"^(?:[-*]\s*)?(?:none|nothing|no blockers?|n/?a)\b[\s.!]*$",
    re.IGNORECASE,
)

# Value-shaped credential patterns. The matched text is never echoed — a
# validator that prints the secret into a terminal or a CI log makes the leak
# worse than the one it reports.
SECRET = re.compile(
    r"""(?x)
      \bsk-[A-Za-z0-9_-]{16,}
    | \bgh[pousr]_[A-Za-z0-9]{20,}
    | \bgithub_pat_[A-Za-z0-9_]{20,}
    | \bxox[baprs]-[A-Za-z0-9-]{10,}
    | \bAKIA[0-9A-Z]{16}\b
    | \beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}
    | (?i:\b(?:api[_-]?key|access[_-]?token|secret|password|passwd)\b\s*[:=]\s*["']?[A-Za-z0-9_\-]{12,})
    """
)


def _strip_fences(lines: list[str]) -> list[tuple[int, str]]:
    """Drop fenced code blocks — a template's placeholders are not content."""
    out: list[tuple[int, str]] = []
    fenced = False
    for lineno, line in enumerate(lines, 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            out.append((lineno, line))
    return out


def parse_sections(text: str) -> dict[str, tuple[int, list[str]]]:
    """Map `## Name` → (heading line number, body lines), first occurrence wins."""
    sections: dict[str, tuple[int, list[str]]] = {}
    current: str | None = None
    body: list[str] = []
    for lineno, line in _strip_fences(text.splitlines()):
        match = HEADING.match(line)
        if match:
            if current is not None:
                sections.setdefault(current, (start, body))
            current = match.group("name").strip()
            start = lineno
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        sections.setdefault(current, (start, body))
    return sections


def _is_placeholder(body: list[str]) -> bool:
    content = [line.strip() for line in body if line.strip()]
    if not content:
        return True
    if any(EXPLICIT_EMPTY.match(line) for line in content):
        return False
    return all(PLACEHOLDER.match(line) for line in content)


def check(text: str) -> tuple[list[str], list[str]]:
    """Return (problems, warnings)."""
    problems: list[str] = []
    warnings: list[str] = []
    sections = parse_sections(text)

    for name in REQUIRED:
        if name not in sections:
            problems.append(f"missing section: ## {name}")
            continue
        lineno, body = sections[name]
        if _is_placeholder(body):
            problems.append(
                f"section ## {name} (line {lineno}) is empty or still a placeholder"
            )

    present = [name for name in sections if name in REQUIRED]
    expected = [name for name in REQUIRED if name in sections]
    if present != expected:
        warnings.append(
            "sections out of order: " + ", ".join(present)
        )

    for lineno, line in enumerate(text.splitlines(), 1):
        if SECRET.search(line):
            problems.append(
                f"line {lineno}: looks like a credential — reference where it lives instead"
            )

    if CLOSING not in text:
        problems.append(f"missing closing line: ▶ {CLOSING}")

    return problems, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a HANDOFF.md file.")
    parser.add_argument("path", type=Path, help="path to HANDOFF.md")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable results"
    )
    args = parser.parse_args(argv)

    try:
        text = args.path.read_text(encoding="utf-8")
    except OSError as exc:
        payload = {"ok": False, "path": str(args.path), "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"cannot read {args.path}: {exc}", file=sys.stderr)
        return 2

    problems, warnings = check(text)
    found = sum(1 for name in REQUIRED if name in parse_sections(text))

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not problems,
                    "path": str(args.path),
                    "sections_found": found,
                    "sections_required": len(REQUIRED),
                    "problems": problems,
                    "warnings": warnings,
                },
                indent=2,
            )
        )
    elif problems:
        print(f"{args.path}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        for warning in warnings:
            print(f"  ~ {warning}")
    else:
        print(
            f"✓ {args.path} ok — {found}/{len(REQUIRED)} sections, closing line present"
        )
        for warning in warnings:
            print(f"  ~ {warning}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""
Waking — load the agent's sanctum in one pass, or route to First Breath.

Run on activation. Determines the mode from the filesystem (and the --pulse
flag) and, when the sanctum exists, prints the full identity in a single read
(INDEX, PERSONA, CREED, BOND, MEMORY, CAPABILITIES) so the agent becomes itself
in one shot instead of six. In --pulse mode it also appends PULSE.md. When no
sanctum exists, it prints a directive to run First Breath.

This loads runtime memory only. It never reads or writes config or customize.toml.

Usage:
    uv run wake.py <project-root> [--pulse]

    project-root: The root of the project (where _bmad/ lives)
"""

import sys
from pathlib import Path

SKILL_NAME = "{skillName}"

# Load order — the "become yourself" set.
IDENTITY_FILES = [
    "INDEX.md",
    "PERSONA.md",
    "CREED.md",
    "BOND.md",
    "MEMORY.md",
    "CAPABILITIES.md",
]


def emit(path: Path) -> None:
    print(f"\n===== {path.name} =====")
    try:
        print(path.read_text(encoding="utf-8").rstrip())
    except FileNotFoundError:
        print(f"(missing: {path.name})")


def main() -> int:
    args = sys.argv[1:]
    pulse = "--pulse" in args
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        print("Usage: wake.py <project-root> [--pulse]", file=sys.stderr)
        return 2

    project_root = Path(positional[0]).resolve()
    sanctum = project_root / "_bmad" / "memory" / SKILL_NAME

    core_ok = (
        sanctum.is_dir()
        and (sanctum / "CREED.md").is_file()
        and (sanctum / "MEMORY.md").is_file()
    )
    if not core_ok:
        print("MODE: FIRST_BREATH")
        print(f"NO SANCTUM at {sanctum}")
        print("This is your one birth. Load references/first-breath.md and follow it.")
        return 0

    print("MODE: PULSE" if pulse else "MODE: WAKING")
    print(f"Sanctum: {sanctum}")
    for name in IDENTITY_FILES:
        emit(sanctum / name)
    if pulse:
        emit(sanctum / "PULSE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

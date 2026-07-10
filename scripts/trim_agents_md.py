#!/usr/bin/env python3
"""Regenerate .hermes.md from AGENTS.md — keeps only essential sections.

Run manually or automatically via .git/hooks/post-checkout / post-merge.
"""
import subprocess
import sys
from pathlib import Path

AGENTS_MD = Path(__file__).resolve().parent.parent / "AGENTS.md"
HERMES_MD = AGENTS_MD.parent / ".hermes.md"

# Sections to keep (1-indexed line ranges from AGENTS.md)
KEEP_RANGES = [
    (1, 28),      # What Hermes Is
    (213, 223),   # Development Environment
    (224, 276),   # Project Structure
    (300, 313),   # File Dependency Chain
    (314, 373),   # AIAgent Class + Agent Loop
    (374, 427),   # CLI + Slash Commands
    (428, 478),   # TUI Architecture
    (509, 558),   # Adding New Tools
    (582, 645),   # Configuration
    (735, 760),   # Plugins
    (853, 880),   # Skills
    (964, 982),   # Toolsets
    (983, 1016),  # Delegation
    (1128, 1157), # Important Policies
    (1214, 1278), # Known Pitfalls
    # Testing: from line 1279 to end — detected dynamically
]

HEADER = """# Hermes Agent - Development Guide (trimmed)

This is a trimmed version of AGENTS.md — keeps essential sections only.
Full version: AGENTS.md in this directory (auto-loaded when .hermes.md absent).
Regenerate: python3 scripts/trim_agents_md.py

"""


def main():
    if not AGENTS_MD.exists():
        print(f"ERROR: {AGENTS_MD} not found", file=sys.stderr)
        sys.exit(1)

    lines = AGENTS_MD.read_text().splitlines()
    total = len(lines)

    # Add "Testing" section (1279 to end)
    ranges = KEEP_RANGES + [(1279, total)]

    sections = []
    for start, end in ranges:
        section = "\n".join(lines[start - 1 : end])
        sections.append(section)

    trimmed = HEADER + "\n\n".join(sections)
    HERMES_MD.write_text(trimmed)
    print(f"Wrote {HERMES_MD} — {len(trimmed)} chars ({len(trimmed) * 100 // len(AGENTS_MD.read_text())}% of original)")


if __name__ == "__main__":
    main()

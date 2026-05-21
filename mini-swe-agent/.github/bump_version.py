#!/usr/bin/env python3
"""Bump the mini-swe-agent version in __init__.py."""

import re
import subprocess
import sys
from pathlib import Path

INIT_FILE = Path(__file__).resolve().parents[1] / "src" / "minisweagent" / "__init__.py"
VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


def main() -> None:
    text = INIT_FILE.read_text()
    match = VERSION_RE.search(text)
    if not match:
        print("Error: could not find __version__ in", INIT_FILE)
        sys.exit(1)

    current = match.group(1)
    print(f"Current version: {current}")

    new = input("New version: ").strip()
    if not new:
        print("No version entered, aborting.")
        sys.exit(1)

    updated = text[: match.start(1)] + new + text[match.end(1) :]
    INIT_FILE.write_text(updated)
    print(f"Updated {INIT_FILE.relative_to(Path.cwd())}: {current} -> {new}")

    # Commit the changes
    try:
        subprocess.run(["git", "add", str(INIT_FILE)], check=True)
        subprocess.run(["git", "commit", "-m", "Bump version"], check=True)
        print("Changes committed with message 'Bump version'")
    except subprocess.CalledProcessError as e:
        print(f"Error committing changes: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

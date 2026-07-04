#!/usr/bin/env python3
"""Print the platform-aware Hermes home path to stdout.

Usage:
    python scripts/get_hermes_home.py          # prints the resolved path
    python scripts/get_hermes_home.py --env    # prints export HERMES_HOME=...

Respects:
    - HERMES_HOME env var (highest priority)
    - Windows: %LOCALAPPDATA%/hermes
    - macOS/Linux: ~/.hermes

This is the single source-of-truth for *prompt-level* path resolution.
Code-level callers should import `hermes_constants.get_hermes_home()`
or `get_default_hermes_root()` instead.
"""

import os
import sys
from pathlib import Path


def _resolve() -> Path:
    """Resolve the Hermes home path, same logic as hermes_constants."""
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"

    return Path.home() / ".hermes"


def main() -> None:
    path = _resolve()
    if len(sys.argv) > 1 and sys.argv[1] == "--env":
        # Shell-safe export line, suitable for eval or sourcing
        print(f'export HERMES_HOME="{path}"')
    else:
        print(path)


if __name__ == "__main__":
    main()

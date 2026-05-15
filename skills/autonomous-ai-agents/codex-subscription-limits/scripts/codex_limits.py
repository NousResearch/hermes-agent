#!/usr/bin/env python3
"""Compatibility wrapper for the bundled Codex limits helper."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "agent" / "codex_limits.py").is_file():
            return parent
    return current.parents[4]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.codex_limits import *  # noqa: F401,F403,E402
from agent.codex_limits import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

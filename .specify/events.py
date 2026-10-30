#!/usr/bin/env python3
"""Stable host entry point for the repository continuity dispatcher."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from continuity_event import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

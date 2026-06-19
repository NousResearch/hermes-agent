#!/usr/bin/env python3
"""Run budgeted websearch-only investment theme discovery."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.investment_assistant.websearch_discovery_cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

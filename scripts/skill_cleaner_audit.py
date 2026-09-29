#!/usr/bin/env python3
"""Run the report-only Hermes skill cleaner audit.

This thin wrapper keeps the operator-facing command discoverable under scripts/
while the implementation stays importable/testable in tools.skill_cleaner.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.skill_cleaner import main


if __name__ == "__main__":
    raise SystemExit(main())

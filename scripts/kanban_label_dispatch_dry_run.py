#!/usr/bin/env python3
"""Dry-run wrapper for GitHub label to Kanban actor packet preview."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli.kanban_label_dispatch import main


if __name__ == "__main__":
    raise SystemExit(main())

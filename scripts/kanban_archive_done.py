#!/usr/bin/env python3
"""Cron entrypoint for stale Done Kanban task archival.

Recommended Hermes cron schedule for the approved policy:
  30 3 * * *

First backfill dry-run:
  python scripts/kanban_archive_done.py --dry-run --done-age-days 7 --batch-size 25
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli.kanban_archive_done import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

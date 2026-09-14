#!/usr/bin/env python3
"""Notify when a pending memory/skill approval has sat unreviewed too long.

Read-only, count-and-notify only. Never modifies, approves, or discards a
pending record -- that stays with `/memory pending` and `/skills pending`.
Root cause of a real gap: 12 records sat 7-12 days unnoticed because nothing
surfaced them until someone thought to check.

Silent when nothing is stale (no_agent cron convention: empty stdout = no
notification), matching sync-fork.sh and backlog-rotate.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foundation_cron_common import age_hours, local_now

STALE_AGE_DAYS = 3.0
SUBSYSTEMS = ("memory", "skills", "upstream_fix")


def _stale_items(pending_dir: Path, now) -> list[tuple[str, str, float]]:
    stale: list[tuple[str, str, float]] = []
    for subsystem in SUBSYSTEMS:
        directory = pending_dir / subsystem
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if not path.is_file():
                continue
            age_days = age_hours(path, now) / 24
            if age_days >= STALE_AGE_DAYS:
                stale.append((subsystem, path.stem, age_days))
    return stale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pending-dir", type=Path, default=Path.home() / ".hermes/pending"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = local_now()
    stale = _stale_items(args.pending_dir, now)
    if not stale:
        return 0

    stale.sort(key=lambda item: -item[2])
    plural = "s" if len(stale) != 1 else ""
    print(
        f"{len(stale)} pending approval{plural} unreviewed for "
        f"{STALE_AGE_DAYS:.0f}+ days:\n"
    )
    for subsystem, item_id, age_days in stale:
        print(f"  [{subsystem}] {item_id} — {age_days:.1f} days old")
    print("\nReview with: /memory pending   or   /skills pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

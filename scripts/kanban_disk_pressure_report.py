#!/usr/bin/env python3
"""Emit a read-only Kanban disk pressure report for Hermes cron.

No deletion, cleanup apply, restart, or env mutation is performed here. The
script is intentionally suitable for ``cronjob(no_agent=True)`` delivery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_cli.kanban_workspace_janitor import (
    collect_disk_pressure_report,
    format_disk_pressure_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Kanban disk pressure report")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human-readable markdown")
    parser.add_argument("--min-workspace-mib", type=int, default=10)
    parser.add_argument("--min-artifact-mib", type=int, default=10)
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes" / "kanban.db")
    parser.add_argument("--workspaces", type=Path, default=Path.home() / ".hermes" / "kanban" / "workspaces")
    args = parser.parse_args()

    report = collect_disk_pressure_report(
        db_path=args.db,
        workspaces_root=args.workspaces,
        min_workspace_bytes=args.min_workspace_mib * 1024 * 1024,
        min_artifact_bytes=args.min_artifact_mib * 1024 * 1024,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_disk_pressure_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

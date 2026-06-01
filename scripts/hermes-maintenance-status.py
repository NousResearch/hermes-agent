#!/usr/bin/env python3
"""Render Hermes Maintenance Self/Lineage status notification lines.

Default registry: docs/hermes-maintenance-lineage.json
Example:
  python scripts/hermes-maintenance-status.py --profile coordinator_hermes_maintenance --board hermes-maintenance
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli.kanban_status_lines import main  # noqa: E402


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--registry" not in argv:
        argv.extend(["--registry", str(REPO_ROOT / "docs" / "hermes-maintenance-lineage.json")])
    raise SystemExit(main(argv))

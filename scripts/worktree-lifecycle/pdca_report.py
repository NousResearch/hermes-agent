#!/usr/bin/env python3
"""Script-only Hermes cron entrypoint for WTL PDCA reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli.worktree_lifecycle import command_report


def main() -> int:
    parser = argparse.ArgumentParser()
    default_mode = "cleanup" if "cleanup" in Path(sys.argv[0]).stem else "light"
    parser.add_argument("--mode", choices=("light", "cleanup"), default=default_mode)
    parser.add_argument("--registry")
    args = parser.parse_args()
    result = command_report(argparse.Namespace(
        registry=args.registry,
        record=True,
        cleanup_review=args.mode == "cleanup",
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build Hermes' local skill FTS index and optional Obsidian dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.local_skill_index import build_skill_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-dir",
        action="append",
        dest="skill_dirs",
        required=True,
        help="Skill root to index; repeat for external roots",
    )
    parser.add_argument("--db", required=True, help="Output SQLite database")
    parser.add_argument("--usage", help="Optional .usage.json sidecar")
    parser.add_argument("--obsidian-map", help="Optional human-authored routing map")
    parser.add_argument("--obsidian-output", help="Optional generated Obsidian dashboard")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_skill_index(
        [Path(path).expanduser() for path in args.skill_dirs],
        Path(args.db).expanduser(),
        usage_path=Path(args.usage).expanduser() if args.usage else None,
        obsidian_map_path=Path(args.obsidian_map).expanduser() if args.obsidian_map else None,
        obsidian_output_path=(
            Path(args.obsidian_output).expanduser() if args.obsidian_output else None
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dry-run report for legacy Kanban summaries that mention supersession."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli import kanban_db as kb  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report tasks whose run summaries contain 'Superseded by t_<id>'. "
            "This command never changes task state."
        )
    )
    parser.add_argument(
        "--board",
        default=None,
        help="Board slug (default: active board)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kb.init_db(board=args.board)
    with kb.connect_closing(board=args.board) as conn:
        candidates = kb.find_supersession_candidates(conn)

    board = args.board or kb.get_current_board()
    report = {
        "board": board,
        "dry_run": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Board: {board}")
    print("Dry run: no task state was changed.")
    if not candidates:
        print("No legacy supersession summaries found.")
        return 0
    for candidate in candidates:
        task_id = candidate["task_id"]
        replacement_id = candidate["superseded_by"]
        state = (
            "already structured"
            if candidate["already_structured"]
            else "candidate"
        )
        target = (
            "target exists"
            if candidate["replacement_exists"]
            else "TARGET MISSING"
        )
        print(f"- {task_id} -> {replacement_id} [{state}; {target}]")
        if candidate["replacement_exists"] and not candidate["already_structured"]:
            print(
                f"  suggested: hermes kanban supersede "
                f"{task_id} {replacement_id}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

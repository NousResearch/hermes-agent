#!/usr/bin/env python3
"""CLI: C-min sidecar judge for board_only dual-bound parents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.kanban_orch_cmin import CMinError, live_judge_parent_task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C-min board_only sidecar lifecycle judge")
    p.add_argument("--task-id", required=True, help="Native parent task id")
    args = p.parse_args(argv)
    try:
        res = live_judge_parent_task(args.task_id)
    except CMinError as exc:
        print(json.dumps({"ok": False, "error": exc.code}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"ok": True, "result": res.to_dict()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

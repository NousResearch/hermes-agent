#!/usr/bin/env python3
"""CLI: one-shot cmin tick — advance open board_only sidecar requests from native truth."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.kanban_orch_cmin import CMinError, live_tick_once


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C-min tick: native done/children → sidecar completed")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true", help="machine JSON only")
    args = p.parse_args(argv)
    try:
        res = live_tick_once(limit=args.limit)
    except CMinError as exc:
        print(json.dumps({"ok": False, "error": exc.code}, ensure_ascii=False, indent=2))
        return 2
    payload = {"ok": True, "tick": res.to_dict()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

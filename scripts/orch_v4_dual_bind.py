#!/usr/bin/env python3
"""CLI: bind an existing native parent task into live ORCH V4 sidecar."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.kanban_orch_dual_bind import dual_bind_parent_task, preflight_dual_bind


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dual-bind native parent task into orch_v4 sidecar")
    p.add_argument("--task-id", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--cfg", default=None, help="Path to orch_v4_writer.json")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)

    pf = preflight_dual_bind(args.cfg)
    print(json.dumps({"preflight": pf}, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return 0 if pf.get("ok") else 2
    if not pf.get("ok"):
        return 2
    res = dual_bind_parent_task(task_id=args.task_id, title=args.title, cfg_path=args.cfg)
    print(json.dumps({"bind": res.to_dict()}, ensure_ascii=False, indent=2))
    if res.error:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

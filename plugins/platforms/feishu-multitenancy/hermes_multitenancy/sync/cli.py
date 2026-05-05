"""``python -m hermes_multitenancy.sync apply <users.json>`` entry point.

Minimal CLI for ops-style sync runs. Real deployments will likely call
``apply_users`` from a webhook receiver instead — this exists to provide
a one-command tier (cron + curl pipeline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .feishu_hr import UserSpec, apply_users


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes_multitenancy.sync", description="multitenancy routing sync")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_apply = sub.add_parser("apply", help="Reconcile routing table with a JSON user list")
    p_apply.add_argument("users_json", type=Path, help="Path to JSON array of {user_id, profile_name, open_id, union_id?}")
    p_apply.add_argument("--db", type=Path, default=None, help="Override DB path (default: ~/.hermes/multitenancy.db)")

    args = parser.parse_args(argv)

    if args.cmd == "apply":
        from hermes_multitenancy.routing import RoutingTable

        if not args.users_json.exists():
            print(f"error: {args.users_json} not found", file=sys.stderr)
            return 1
        raw = json.loads(args.users_json.read_text())
        if not isinstance(raw, list):
            print("error: users.json must be a JSON array", file=sys.stderr)
            return 1
        users = [UserSpec(**entry) for entry in raw]

        table = RoutingTable(args.db)
        try:
            stats = apply_users(table, users)
        finally:
            table.close()

        print(json.dumps(stats, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ORCH V4 isolated schema initializer / migrate helper.

Default posture: refuse live fleet kanban.db paths.
This is NOT a live writer cutover tool.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ORCH V4 isolated schema migrate/init")
    parser.add_argument("--db", required=True, help="Target SQLite path (isolated only by default)")
    parser.add_argument("--mode", choices=["sidecar", "inplace"], default="sidecar")
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Required together with ORCH_V4_ALLOW_LIVE=1 to touch live paths",
    )
    parser.add_argument(
        "--i-understand-live",
        action="store_true",
        help="Second explicit live acknowledgement",
    )
    args = parser.parse_args(argv)

    # Local imports keep --help fast and avoid import side effects.
    from hermes_cli.kanban_orch_db import OrchDBError, assert_not_live_path, open_orch_db, close_orch_db

    allow_live = bool(args.allow_live and args.i_understand_live and os.environ.get("ORCH_V4_ALLOW_LIVE") == "1")
    try:
        path = assert_not_live_path(args.db, allow_live=allow_live)
    except OrchDBError as exc:
        print(f"REFUSED:{exc.code}", file=sys.stderr)
        return 2

    if os.path.exists(path):
        print(f"REFUSED:db_exists:{path}", file=sys.stderr)
        return 3

    conn = open_orch_db(path, allow_live=allow_live, create=True, test_open_capability=True)
    try:
        if args.mode == "sidecar":
            from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema

            apply_sidecar_schema(conn)
        else:
            from hermes_cli.kanban_orch_schema_v4 import apply_schema

            apply_schema(conn)
        print(f"OK:initialized:{args.mode}:{path}")
        return 0
    finally:
        close_orch_db(conn)


if __name__ == "__main__":
    raise SystemExit(main())

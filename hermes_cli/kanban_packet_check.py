"""Read-only exact-packet approval check for isolated consumers."""
from __future__ import annotations

import argparse
import json
import sys

from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_packet_approval import require_packet_approval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check one exact packet approval without mutating it")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--board", default="default")
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        packet_sha256 = payload.get("packet_sha256")
        execution_identity = payload.get("execution_identity")
        if not isinstance(packet_sha256, str) or not isinstance(execution_identity, dict):
            raise ValueError("request requires packet_sha256 and execution_identity")
        with connect_closing(board=args.board) as conn:
            receipt = require_packet_approval(
                conn,
                args.task_id,
                packet_sha256=packet_sha256,
                execution_identity=execution_identity,
            )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"packet approval refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

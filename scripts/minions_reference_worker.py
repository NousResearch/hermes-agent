#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from agent.minions_reference import process_next_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one queued reference-minions job.")
    parser.add_argument("--spool-dir", default=None, help="Override spool directory")
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit")
    args = parser.parse_args()

    completion = process_next_job(spool_dir=args.spool_dir)
    if completion is None:
        print(json.dumps({"status": "idle"}, ensure_ascii=False))
        return 0
    print(json.dumps(completion, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

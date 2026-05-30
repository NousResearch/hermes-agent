#!/usr/bin/env python3.11
"""Audit open write handles for Hermes Lab process isolation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.lab_environment import lab_paths_from_env  # noqa: E402
from gateway.dev_control.lab_process_isolation import audit_process_isolation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that lab processes only write under lab roots.")
    parser.add_argument("--pid", action="append", default=[], help="Process id to audit. Repeatable.")
    parser.add_argument("--allow-root", action="append", default=[], help="Additional allowed write root.")
    parser.add_argument("--forbid-root", action="append", default=[], help="Additional forbidden root.")
    args = parser.parse_args()

    pids = args.pid or [str(os.getpid())]
    paths = lab_paths_from_env()
    allowed_roots = [paths["lab_home"], *args.allow_root]
    result = audit_process_isolation(
        pids=pids,
        allowed_roots=allowed_roots,
        forbidden_roots=args.forbid_root or None,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

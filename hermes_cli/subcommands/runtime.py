"""
CLI Subcommand: hermes runtime
Inspects and manages execution runtimes (local, ssh, container, sandbox, remote).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from runtime.manager import RuntimeManager


def main(args=None):
    mgr = RuntimeManager()
    runtimes = mgr.list_runtimes()

    print("=== Hermes Runtimes ===")
    for r in runtimes:
        active_mark = "*" if r["runtime_id"] == mgr.active_runtime_id else " "
        print(f"[{active_mark}] {r['runtime_id']} ({r['name']}) - Type: {r['type']} - Connected: {r['connected']}")
    print("=======================")


if __name__ == "__main__":
    main()

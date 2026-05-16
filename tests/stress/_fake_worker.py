#!/usr/bin/env python3
"""Fake worker process that exercises the real subprocess contract.

Reads HERMES_KANBAN_TASK from env, heartbeats periodically, does short
work, completes via the CLI. Designed to be spawned by the dispatcher
exactly the way `hermes chat -q` would be, minus the LLM cost.
"""

import json
import os
import subprocess
import sys
import time


def main():
    tid = os.environ["HERMES_KANBAN_TASK"]
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
    board = os.environ.get("HERMES_KANBAN_BOARD", "")

    # Orient first: exercise the same kanban_show surface dispatcher-spawned
    # workers receive before mutating lifecycle state.
    show = subprocess.run(
        ["hermes", "kanban", "show", tid, "--json"],
        check=True, capture_output=True, text=True,
    )
    shown = json.loads(show.stdout)
    if shown.get("task", {}).get("id") != tid:
        raise RuntimeError(f"kanban show returned the wrong task: {shown!r}")

    # Announce via CLI (goes through real argparse + init_db + etc)
    subprocess.run(
        ["hermes", "kanban", "heartbeat", tid, "--note", "started"],
        check=True, capture_output=True,
    )

    # Simulate work with periodic heartbeats
    for i in range(3):
        time.sleep(0.3)
        subprocess.run(
            ["hermes", "kanban", "heartbeat", tid, "--note", f"progress {i+1}/3"],
            check=True, capture_output=True,
        )

    # Complete with structured handoff
    subprocess.run(
        [
            "hermes", "kanban", "complete", tid,
            "--summary", f"real-subprocess worker finished {tid}",
            "--metadata", json.dumps({
                "workspace": workspace,
                "board": board,
                "worker_pid": os.getpid(),
                "iterations": 3,
                "oriented_with_kanban_show": True,
            }),
        ],
        check=True, capture_output=True,
    )


if __name__ == "__main__":
    main()

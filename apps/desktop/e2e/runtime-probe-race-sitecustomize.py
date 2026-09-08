"""Sandbox-only gate: let the real interpreter and Hermes CLI run unchanged."""
import atexit
import json
import os
from pathlib import Path
import sys
import time


def record(directory, name, **extra):
    # Separate, atomically published records avoid partial reads/concurrent
    # appends when Python's Windows venv redirector starts several children.
    target = directory / name
    target.mkdir(exist_ok=True)
    pending = target / f"{os.getpid()}.tmp"
    pending.write_text(
        json.dumps({"pid": os.getpid(), "argv": sys.orig_argv, **extra}),
        encoding="utf-8",
    )
    pending.replace(target / f"{os.getpid()}.json")


directory = os.environ.get("HERMES_E2E_RUNTIME_RACE_DIR")
if directory:
    directory = Path(directory)
    # Production supplies this secret only to actual backend spawns. Do not
    # change or short-circuit their startup: the record observes a real child.
    if os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN"):
        record(directory, "spawned")
    record(directory, "all")
    if sys.orig_argv[1:] == ["-m", "hermes_cli.main", "serve", "--help"]:
        record(directory, "entered", parent_pid=os.getppid())
        deadline = time.monotonic() + 25
        while not (directory / "release").exists():
            if time.monotonic() >= deadline:
                record(directory, "timed-out")
                break
            time.sleep(0.02)
        record(directory, "released")
        atexit.register(record, directory, "exited")

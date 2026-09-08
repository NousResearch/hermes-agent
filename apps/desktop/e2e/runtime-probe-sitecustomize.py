"""Sandbox-only hook: require the real import probe's Electron parent to reply.

Loaded by Python itself via PYTHONPATH; never replaces an interpreter or imports.
A fatal exit is intentional: Python otherwise ignores sitecustomize exceptions.
"""
import json
import os
from pathlib import Path
import socket
import sys
import time


def _gate_runtime_probe():
    gate = os.environ.get("HERMES_E2E_RUNTIME_PROBE_DIR")
    argv = sys.orig_argv
    if not gate or "-c" not in argv:
        return
    code = argv[argv.index("-c") + 1]
    if "import yaml" not in code or "import hermes_cli.config" not in code:
        return
    root = Path(gate)
    record = {"pid": os.getpid(), "argv": argv, "home": os.environ.get("HERMES_HOME")}
    (root / "entered.json").write_text(json.dumps(record), encoding="utf-8")
    deadline = time.monotonic() + 25
    port_file = root / "port"
    while not port_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Electron parent never started its loopback service")
        time.sleep(0.01)
    with socket.create_connection(("127.0.0.1", int(port_file.read_text())), timeout=20) as connection:
        connection.sendall(str(os.getpid()).encode("ascii"))
        with connection.makefile("rb") as response:
            reply = response.read(128).decode("ascii")
        if not reply.startswith("parent:"):
            raise RuntimeError(f"Unexpected parent response: {reply!r}")
        record["parent_pid"] = int(reply.removeprefix("parent:"))
    pending = root / "replied.json.tmp"
    pending.write_text(json.dumps(record), encoding="utf-8")
    pending.replace(root / "replied.json")


try:
    _gate_runtime_probe()
except Exception as error:
    gate = os.environ.get("HERMES_E2E_RUNTIME_PROBE_DIR")
    if gate:
        (Path(gate) / "failed.txt").write_text(str(error), encoding="utf-8")
    os._exit(91)

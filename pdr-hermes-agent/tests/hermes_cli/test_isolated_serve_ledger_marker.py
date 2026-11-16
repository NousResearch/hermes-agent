"""An isolated serve says so in the spawn ledger; an ordinary serve does not.

Desktop's attach-first discovery reads ``spawn-ledger.json`` and attaches to any live loopback
serve. ``hermes serve --isolated`` (the backend another machine's Desktop spawns over SSH) opts out
of the host singleton on the CLI side, so the ledger row must carry that fact as a structured field
for Desktop to honour it too. Real processes against a temp ``HERMES_HOME``; no mocks.
"""

import json
import os
import socket
import subprocess
import sys
import time

import psutil
import pytest


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _await_ledger_row(ledger, proc, log, *, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        pids = {proc.pid, *(c.pid for c in psutil.Process(proc.pid).children(recursive=True))}
        try:
            rows = json.loads(ledger.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []
        for row in rows:
            if row.get("purpose") == "serve" and row.get("pid") in pids and row.get("port"):
                return row
        time.sleep(0.1)
    pytest.fail(f"serve never registered (exit={proc.poll()}): {log.read_text(encoding='utf-8', errors='replace')}")


def _stop(proc):
    if proc.poll() is None:
        parent = psutil.Process(proc.pid)
        family = [*parent.children(recursive=True), parent]
        for p in family:
            p.terminate()
        _, alive = psutil.wait_procs(family, timeout=5)
        for p in alive:
            p.kill()
    proc.wait(timeout=10)


def test_isolated_serve_ledger_row_is_marked_and_ordinary_serve_is_not(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_GATEWAY_LOCK_DIR": str(tmp_path / "locks")}
    for key in ("HERMES_DESKTOP", "HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER",
                "HERMES_DASHBOARD_SESSION_TOKEN", "HERMES_SPAWN"):
        env.pop(key, None)
    ledger = home / "spawn-ledger.json"

    # Ordinary first: started second it would attach to the isolated one and exit.
    launches = [("ordinary", []), ("isolated", ["--isolated"])]
    procs, rows = {}, {}
    try:
        for name, extra in launches:
            log = tmp_path / f"{name}.log"
            with log.open("w", encoding="utf-8") as out:
                procs[name] = subprocess.Popen(
                    [sys.executable, "-m", "hermes_cli.main", "serve", *extra,
                     "--host", "127.0.0.1", "--port", str(_free_port())],
                    env=env, stdout=out, stderr=subprocess.STDOUT,
                )
            rows[name] = _await_ledger_row(ledger, procs[name], log)
    finally:
        for proc in procs.values():
            _stop(proc)

    assert rows["isolated"].get("isolated") is True
    assert not rows["ordinary"].get("isolated")

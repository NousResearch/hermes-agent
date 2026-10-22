"""In-sandbox command agent for the hand-off suite (stdlib only, run as a plain script).

The hand-off cells need several long-lived processes (gateway, dashboard, cron ticker, kanban
dispatcher) and the real ``hermes update`` to share ONE PID namespace, exactly as they share one
machine for a user: the updater's process scans must see the gateway it restarts, and a relaunched
gateway must be findable by the next ``hermes gateway status``. ``_helpers.sandbox_argv`` gives every
command its own namespace, so this agent is started once inside the sandbox and runs every command
of a cell on request.

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout, each carrying the
request ``id``. Requests are served on their own threads, so a blocking ``run`` (the update) never
stops the test from inspecting the process table meanwhile.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

_OUT = threading.Lock()
_CHILDREN: dict[int, subprocess.Popen] = {}


def _reply(rid, **payload) -> None:
    line = json.dumps({"id": rid, **payload}) + "\n"
    with _OUT:
        sys.stdout.write(line)
        sys.stdout.flush()


def _stat(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            return fh.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def _read(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def _procs() -> list[dict]:
    rows = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        fields = _stat(pid)
        if fields is None:
            continue
        env = {}
        for item in _read(f"/proc/{pid}/environ").split(b"\0"):
            key, sep, value = item.partition(b"=")
            if sep and key in (b"HERMES_HOME", b"INVOCATION_ID", b"HERMES_E2E_UNIT"):
                env[key.decode()] = value.decode(errors="replace")
        rows.append({
            "pid": pid, "state": fields[0], "ppid": int(fields[1]), "start": int(fields[19]),
            "cmdline": [a.decode(errors="replace") for a in _read(f"/proc/{pid}/cmdline").split(b"\0") if a],
            "env": env,
        })
    return rows


def _reap() -> None:
    for pid, proc in list(_CHILDREN.items()):
        if proc.poll() is not None:
            _CHILDREN.pop(pid, None)


def _reaper_loop() -> None:
    """Reap spawned children the moment they exit, as the user's shell does: an unreaped zombie
    still answers ``kill(pid, 0)``, so a restart watcher waiting for the old gateway to exit would
    see it alive forever."""
    while True:
        _reap()
        time.sleep(0.1)


def _run(req: dict) -> dict:
    stdin = subprocess.PIPE if req.get("input") is not None else subprocess.DEVNULL
    proc = subprocess.Popen(req["argv"], env=req["env"], cwd=req["cwd"], text=True, stdin=stdin,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    started = time.monotonic()
    try:
        out, err = proc.communicate(input=req.get("input"), timeout=req.get("timeout") or 600)
        return {"rc": proc.returncode, "stdout": out, "stderr": err, "timed_out": False,
                "elapsed": time.monotonic() - started}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
        return {"rc": proc.returncode, "stdout": out, "stderr": err, "timed_out": True,
                "elapsed": time.monotonic() - started}


def _spawn(req: dict) -> dict:
    log = open(req["log"], "ab")
    try:
        proc = subprocess.Popen(req["argv"], env=req["env"], cwd=req["cwd"], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    _CHILDREN[proc.pid] = proc
    return {"pid": proc.pid}


def _kill(req: dict) -> dict:
    try:
        os.kill(int(req["pid"]), int(req.get("sig", signal.SIGTERM)))
        return {"ok": True}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def _handle(req: dict) -> None:
    rid = req.get("id")
    try:
        _reap()
        op = req["op"]
        if op == "run":
            _reply(rid, **_run(req))
        elif op == "spawn":
            _reply(rid, **_spawn(req))
        elif op == "procs":
            _reply(rid, procs=_procs())
        elif op == "kill":
            _reply(rid, **_kill(req))
        elif op == "ping":
            _reply(rid, pid=os.getpid())
        else:
            _reply(rid, error=f"unknown op {op!r}")
    except Exception as exc:  # the agent must answer every request, whatever it was
        _reply(rid, error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    threading.Thread(target=_reaper_loop, daemon=True).start()
    for raw in sys.stdin:
        raw = raw.strip()
        if raw:
            threading.Thread(target=_handle, args=(json.loads(raw),), daemon=True).start()


if __name__ == "__main__":
    main()

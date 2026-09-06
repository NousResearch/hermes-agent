"""Out-of-scope sentinel: stopping the scope also kills orphaned descendants."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    import runpy

    __package__ = runpy.run_path(str(Path(__file__).resolve().with_name("_binding.py")))["load_runtime"]().__name__

from .lifecycle import Registry, alive, atomic_json, stop_scope, remove_runtime


def cleanup(home, realm_id):
    registry = Registry(home)
    with registry.lock():
        if not registry.path(realm_id).exists():
            return
        record = registry.get(realm_id)
        stop_scope(record)
        remove_runtime(registry, record)


def run(home, realm_id):
    registry = Registry(home)
    record = registry.get(realm_id)
    runtime = Path(record["runtime_dir"])
    from .lifecycle import identity

    atomic_json(runtime / "supervisor.json", identity(os.getpid()))
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--unit=" + record["scope"],
        "--property=BindsTo=" + record["guardian_unit"],
        "--property=After=" + record["guardian_unit"],
        "--description=Hermes realm " + record["generation"],
        "--property=TimeoutStopSec=3s",
        "--",
        "systemd-inhibit",
        "--what=sleep",
        "--mode=block",
        "--who=Hermes Realms",
        "--why=Background realm is active",
        sys.executable,
        "-P",
        "-m",
        "realms.bootstrap",
        "worker",
        str(runtime),
    ]
    runner = None
    try:
        runner = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True)
        deadline = time.monotonic() + 25
        ready = runtime / "ready.json"
        while not ready.exists():
            if runner.poll() is not None:
                raise RuntimeError("realm scope exited during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("realm startup timed out")
            time.sleep(0.1)
        startup = json.loads(ready.read_text(encoding="utf-8"))
        while all(alive(process) for process in startup["processes"].values()):
            with registry.lock():
                current = registry.get(realm_id)
                abandoned_start = (
                    current["status"] == "starting"
                    and time.time() - current["created_at"] >= 35
                )
                if (
                    abandoned_start
                    or time.time() - current["last_activity"] >= current["idle_ttl"]
                ):
                    # Linearize expiry with env/reuse renewal. Cleanup may wait
                    # for the lock again, but no caller can renew this lease.
                    current["status"] = "stopping"
                    registry.put(current)
                    print("realm idle lease expired", flush=True)
                    break
            time.sleep(0.2)
        dead = {
            name: process
            for name, process in startup["processes"].items()
            if not alive(process)
        }
        if dead:
            print("realm components exited: " + json.dumps(dead), flush=True)
    except Exception as exc:
        if runtime.exists():
            atomic_json(runtime / "error.json", {"error": str(exc)})
        print(str(exc), file=sys.stderr, flush=True)
    finally:
        try:
            cleanup(home, realm_id)
        finally:
            if runner is not None:
                try:
                    runner.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


if __name__ == "__main__":
    if sys.argv[1] == "cleanup":
        cleanup(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])

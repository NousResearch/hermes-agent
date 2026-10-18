#!/usr/bin/env python3
"""Isolated Phase 1 execution-supervisor proof of concept.

This experiment deliberately is not a Hermes runtime integration. It uses the
existing mission ownership API and a disposable database, while a transient
systemd scope and Bubblewrap confine a controlled local worker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any

from hermes_cli import mission_ownership as missions


class SupervisorError(RuntimeError):
    """A supervisor boundary could not be established or verified."""


def _systemd_env() -> dict[str, str]:
    """Route the supervisor's manager calls through the host user-manager socket only."""
    env = dict(os.environ)
    runtime = Path(f"/run/user/{os.getuid()}")
    private = runtime / "systemd" / "private"
    if not private.exists():
        raise SupervisorError(f"user systemd manager socket is unavailable: {private}")
    env["XDG_RUNTIME_DIR"] = str(runtime)
    env.pop("DBUS_SESSION_BUS_ADDRESS", None)
    return env


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    missions.ensure_schema(conn)
    conn.execute("""CREATE TABLE IF NOT EXISTS poc_scope_ops (
        mission_id TEXT PRIMARY KEY, lease_id TEXT NOT NULL UNIQUE,
        unit_name TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
        cgroup_path TEXT, owner_id TEXT NOT NULL, generation INTEGER NOT NULL,
        worker_pid INTEGER, created_at INTEGER NOT NULL
    )""")
    return conn


def _unit_info(unit: str) -> tuple[str, str, str]:
    unit_ref = unit if unit.endswith(".scope") else unit + ".scope"
    result = subprocess.run(
        ["systemctl", "--user", "show", "--property=LoadState", "--property=ActiveState",
         "--property=ControlGroup", "--value", unit_ref], capture_output=True, text=True, timeout=5,
        check=False,
        env=_systemd_env(),
    )
    if result.returncode:
        raise SupervisorError(f"systemd query failed for {unit}: {result.stderr.strip()}")
    values = result.stdout.splitlines()
    if len(values) < 2:
        raise SupervisorError(f"systemd returned incomplete state for {unit}")
    return values[0], values[1], values[2] if len(values) > 2 else ""


def _populated(cgroup_path: str) -> bool:
    if not cgroup_path:
        return False
    events = Path("/sys/fs/cgroup", cgroup_path.lstrip("/"), "cgroup.events")
    try:
        values = dict(line.split() for line in events.read_text(encoding="ascii").splitlines())
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        raise SupervisorError(f"could not inspect cgroup quiescence: {exc}") from exc
    if "populated" not in values:
        raise SupervisorError(f"cgroup event state has no populated field: {events}")
    return values["populated"] == "1"


def _cgroup_pids(cgroup_path: str) -> set[int]:
    path = Path("/sys/fs/cgroup", cgroup_path.lstrip("/"), "cgroup.procs")
    try:
        return {int(value) for value in path.read_text(encoding="ascii").splitlines()}
    except (OSError, ValueError) as exc:
        raise SupervisorError(f"could not inspect scope process membership: {exc}") from exc


def _cgroup_memory(cgroup_path: str) -> dict[str, int | None]:
    base = Path("/sys/fs/cgroup", cgroup_path.lstrip("/"))
    result: dict[str, int | None] = {}
    for name in ("memory.current", "memory.peak"):
        try:
            result[name] = int((base / name).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            result[name] = None
    return result


def _unit_populated(unit: str) -> tuple[bool, str, str]:
    load, active, cgroup = _unit_info(unit)
    if load == "not-found":
        return False, "", active
    if cgroup:
        return _populated(cgroup), cgroup, active
    if active in {"inactive", "failed"}:
        return False, "", active
    raise SupervisorError(f"unit {unit} has no inspectable cgroup in state {active}")


def _wait_empty(unit: str, timeout: float = 8.0) -> str:
    deadline = time.monotonic() + timeout
    last_cgroup = ""
    while time.monotonic() < deadline:
        populated, cgroup, _active = _unit_populated(unit)
        last_cgroup = cgroup or last_cgroup
        if not populated:
            return last_cgroup
        time.sleep(0.025)
    raise SupervisorError(f"scope did not become quiescent before timeout: {unit}")


def _workspace(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("workspace", "input", "output", "protected", "mask")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    approved = paths["input"] / "approved.txt"
    protected = paths["protected"] / "mission.txt"
    if not approved.exists():
        approved.write_text("approved read-only input\n", encoding="utf-8")
    if not protected.exists():
        protected.write_text("protected mission state\n", encoding="utf-8")
    (paths["mask"] / "empty").mkdir(exist_ok=True)
    return paths


def _sandbox_command(paths: dict[str, Path], program: str, program_args: list[str] | None = None) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SupervisorError("Bubblewrap is required; refusing to run an unsandboxed worker")
    args = [bwrap, "--unshare-net", "--size", "67108864", "--tmpfs", "/"]
    for directory in ("usr", "etc", "lib64", "home", "root", "run", "mission"):
        args.extend(("--dir", f"/{directory}"))
    args.extend(("--ro-bind", "/usr", "/usr"))
    for name in ("ld.so.cache", "passwd", "group", "nsswitch.conf"):
        source = Path("/etc") / name
        if source.is_file():
            args.extend(("--ro-bind", str(source), f"/etc/{name}"))
    loader = Path("/lib64/ld-linux-x86-64.so.2")
    if loader.is_file():
        args.extend(("--ro-bind", str(loader), "/lib64/ld-linux-x86-64.so.2"))
    args.extend(("--dev", "/dev", "--ro-bind", str(paths["mask"] / "empty"), "/dev/shm"))
    for directory in ("home", "root", "run"):
        args.extend(("--ro-bind", str(paths["mask"] / "empty"), f"/{directory}"))
    args.extend(("--dir", "/mission/output"))
    args.extend(("--ro-bind", str(paths["input"] / "approved.txt"), "/mission/input.txt"))
    args.extend(("--ro-bind", str(paths["protected"] / "mission.txt"), "/mission/protected.txt"))
    args.extend(("--bind", str(paths["output"]), "/mission/output"))
    args.extend(("--remount-ro", "/"))
    args.extend(("--clearenv", "--setenv", "HOME", "/", "--setenv", "PATH", "/usr/bin:/bin",
                 "--setenv", "LANG", "C.UTF-8", "--", "/usr/bin/python3", "-c", program,
                 *(program_args or [])))
    return args


def _worker_program() -> str:
    parent = r'''import json, os, pathlib, subprocess, sys
out = pathlib.Path('/mission/output')
(out / 'input-read.txt').write_text(pathlib.Path('/mission/input.txt').read_text())
try:
    pathlib.Path('/mission/output/root-write-probe').write_text('allowed')
except OSError as exc:
    raise SystemExit(f'authorized output unexpectedly read-only: {exc}')
outside = sys.argv[3]
child_code = sys.argv[4]
child = subprocess.Popen([sys.executable, '-c', child_code, sys.argv[1], outside],
                         start_new_session=True, close_fds=True)
while not pathlib.Path('/mission/output/child-started.json').exists():
    pass
(out / 'parent-exited').write_text(json.dumps({'parent_pid': os.getpid(), 'child_pid': child.pid}))
if sys.argv[2] == 'crash':
    os._exit(71)
'''
    return f"exec({parent!r})"


class MissionSupervisorPoc:
    """Tiny supervisor tied to the existing mission ownership database."""

    def __init__(self, db_path: Path, root: Path):
        isolation = Path(os.environ.get("HERMES_ISOLATION_ROOT", "")).resolve()
        self.db_path = db_path.resolve()
        self.root = root.resolve()
        self._validate_paths(isolation)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = _workspace(self.root)
        self.conn = _connect(self.db_path)

    def _validate_paths(self, isolation: Path) -> None:
        if not isolation or not isolation.is_dir() or not self.root.is_relative_to(isolation):
            raise SupervisorError("experiment workspace must be inside the existing HERMES_ISOLATION_ROOT")
        if not self.db_path.is_relative_to(isolation):
            raise SupervisorError("supervisor board must be inside HERMES_ISOLATION_ROOT")
        state_keys = ("HOME", "HERMES_HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                      "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR")
        for key in state_keys:
            value = os.environ.get(key)
            if not value or not Path(value).expanduser().resolve().is_relative_to(isolation):
                raise SupervisorError(f"{key} must resolve inside HERMES_ISOLATION_ROOT")
        protected = [Path(__file__).resolve().parents[1]]
        for key in ("HERMES_PROTECTED_HERMES", "HERMES_PROTECTED_KANBAN_DB", "HERMES_PROTECTED_X"):
            value = os.environ.get(key)
            if value:
                protected.append(Path(value).expanduser().resolve())
        for path in (self.root, self.db_path):
            if any(path == target or path.is_relative_to(target) or target.is_relative_to(path)
                   for target in protected):
                raise SupervisorError(f"experiment path overlaps a protected path: {path}")

    def admit(self, key: str, owner: str = "fred") -> str:
        mission_id, _created = missions.admit(
            self.conn, ingress_key=key, request="Disposable supervisor proof of concept",
            contract={"outcome": "prove supervised local execution", "acceptance": ["quiescent handoff"]},
            owner_id=owner,
        )
        return mission_id

    def start(self, mission_id: str, *, owner_id: str = "fred", generation: int = 1,
              delay: float = 0.5, crash: bool = False, wait: float = 8.0,
              launch_gate: threading.Event | None = None,
              launch_waiting: threading.Event | None = None) -> dict[str, Any]:
        mission = missions.get(self.conn, mission_id)
        if not mission or (mission["owner_id"], mission["owner_generation"]) != (owner_id, generation):
            raise missions.StaleOwnerError("worker start does not match current mission owner")
        for name in ("child-started.json", "parent-exited", "child-finished", "write-boundary.json"):
            (self.paths["output"] / name).unlink(missing_ok=True)
        unit = "hermes-mission-poc-" + secrets.token_hex(6)
        lease = "poc_" + secrets.token_hex(10)
        with missions._write_txn(self.conn):
            current = self.conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
            missions._assert_owner(current, "session", owner_id, generation)
            existing = self.conn.execute("SELECT state FROM poc_scope_ops WHERE mission_id=?", (mission_id,)).fetchone()
            if existing and existing["state"] != "transferred":
                raise SupervisorError("mission already has a supervisor operation")
            now = int(time.time())
            self.conn.execute(
                "INSERT INTO poc_scope_ops VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(mission_id) DO UPDATE SET lease_id=excluded.lease_id,"
                "unit_name=excluded.unit_name,state=excluded.state,cgroup_path=NULL,"
                "owner_id=excluded.owner_id,generation=excluded.generation,worker_pid=NULL,created_at=excluded.created_at",
                (mission_id, lease, unit, "starting", None, owner_id, generation, None, now),
            )
            self.conn.execute(
                "INSERT INTO mission_execution_leases VALUES (?,?,?,?,?,?,?,?,?)",
                (lease, mission_id, "session", owner_id, generation, "systemd-scope", os.getpid(), None, now),
            )
        if launch_gate is not None:
            if launch_waiting is not None:
                launch_waiting.set()
            if not launch_gate.wait(timeout=5):
                raise SupervisorError("test launch barrier timed out; lease remains fail-closed")
        program = _worker_program()
        command = ["systemd-run", "--user", "--scope", "--unit=" + unit,
                   "--property=KillMode=control-group", "--"] + _sandbox_command(
                       self.paths, program,
                       [str(delay), "crash" if crash else "normal",
                        str(self.paths["protected"] / "mission.txt"),
                        r'''import json, os, pathlib, sys, time
out = pathlib.Path('/mission/output')
pid = os.getpid()
(out / 'child-started.json').write_text(json.dumps({'pid': pid, 'sid': os.getsid(0)}))
time.sleep(float(sys.argv[1]))
result = {'child_pid': pid, 'child_sid': os.getsid(0)}
for label, target in (('input', '/mission/input.txt'), ('protected', '/mission/protected.txt'),
                      ('outside', sys.argv[2]), ('unapproved_output', '/mission/unapproved.txt'),
                      ('cgroup_control', '/sys/fs/cgroup/cgroup.procs')):
    try:
        with open(target, 'a', encoding='utf-8') as handle:
            handle.write('forbidden\n')
        result[label] = 'UNEXPECTED_WRITE'
    except OSError as exc:
        result[label] = {'denied': True, 'errno': exc.errno, 'error': type(exc).__name__}
(out / 'write-boundary.json').write_text(json.dumps(result, sort_keys=True))
(out / 'child-finished').write_text('done')
'''])
        launch_log = (self.root / "systemd-run.log").open("w", encoding="utf-8")
        launcher = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=launch_log,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                                     env=_systemd_env())
        launch_started = time.monotonic()
        deadline = launch_started + wait
        cgroup = ""
        while time.monotonic() < deadline:
            populated, cgroup, active = _unit_populated(unit)
            if cgroup and populated:
                child_file = self.paths["output"] / "child-started.json"
                parent_file = self.paths["output"] / "parent-exited"
                if child_file.exists() and parent_file.exists():
                    child = json.loads(child_file.read_text(encoding="utf-8"))
                    parent = json.loads(parent_file.read_text(encoding="utf-8"))
                    pids = _cgroup_pids(cgroup)
                    if child["pid"] not in pids or parent["parent_pid"] in pids:
                        time.sleep(0.01)
                        continue
                    child_cgroup = ""
                    try:
                        for line in Path("/proc", str(child["pid"]), "cgroup").read_text(encoding="ascii").splitlines():
                            if line.startswith("0::"):
                                child_cgroup = line[3:]
                    except OSError:
                        pass
                    if child_cgroup != cgroup:
                        raise SupervisorError(f"detached worker escaped scope: {child_cgroup} != {cgroup}")
                    self.conn.execute(
                        "UPDATE poc_scope_ops SET state='running',cgroup_path=?,worker_pid=? WHERE mission_id=?",
                        (cgroup, child["pid"], mission_id),
                    )
                    result = {"unit": unit, "cgroup": cgroup, "cgroup_pids": sorted(pids),
                              "parent_pid": parent["parent_pid"], "child_pid": child["pid"],
                              "child_session": child["sid"], "active": active, "populated": True,
                              "launch_ms": round((time.monotonic() - launch_started) * 1000, 2),
                              "memory_bytes": _cgroup_memory(cgroup)}
                    if crash:
                        return result
                    return result
            if launcher.poll() not in (None, 0):
                launch_log.close()
                detail = (self.root / "systemd-run.log").read_text(encoding="utf-8", errors="replace")
                raise SupervisorError(f"systemd-run failed ({launcher.returncode}): {detail[-2000:]}")
            time.sleep(0.02)
        launch_log.close()
        raise SupervisorError(f"scope failed to become inspectably active: {unit}")

    def handoff(self, mission_id: str, *, owner_id: str = "fred", generation: int = 1,
                new_owner: str = "specialist") -> dict[str, Any]:
        op = self.conn.execute("SELECT * FROM poc_scope_ops WHERE mission_id=?", (mission_id,)).fetchone()
        if op is None:
            raise SupervisorError("no supervisor operation is recorded")
        if op["state"] in {"starting", "cancelling"}:
            try:
                missions.handoff(self.conn, mission_id=mission_id, expected_owner_kind="session",
                                 expected_owner_id=owner_id, expected_generation=generation,
                                 new_owner_kind="session", new_owner_id=new_owner, progress={})
            except missions.MissionError as exc:
                return {"transferred": False, "reason": str(exc), "state": op["state"],
                        "quiescence_verified": False}
            raise SupervisorError("ownership transferred during an unresolved scope transition")
        populated, cgroup, active = _unit_populated(op["unit_name"])
        if populated:
            self.conn.execute("UPDATE poc_scope_ops SET cgroup_path=? WHERE mission_id=?", (cgroup, mission_id))
            try:
                missions.handoff(self.conn, mission_id=mission_id, expected_owner_kind="session",
                                 expected_owner_id=owner_id, expected_generation=generation,
                                 new_owner_kind="session", new_owner_id=new_owner, progress={})
            except missions.MissionError as exc:
                return {"transferred": False, "reason": str(exc), "active": active, "populated": True}
            raise SupervisorError("ownership transferred while the scope remained populated")
        cgroup = cgroup or op["cgroup_path"] or self._wait_recorded_unit(op["unit_name"])
        with missions._write_txn(self.conn):
            self.conn.execute("DELETE FROM mission_execution_leases WHERE id=?", (op["lease_id"],))
            new_generation = missions.handoff(
                self.conn, mission_id=mission_id, expected_owner_kind="session", expected_owner_id=owner_id,
                expected_generation=generation, new_owner_kind="session", new_owner_id=new_owner,
                progress={"supervisor": "scope quiescent", "cgroup": cgroup, "populated": False},
            )
            self.conn.execute("UPDATE poc_scope_ops SET state='transferred',cgroup_path=? WHERE mission_id=?",
                              (cgroup, mission_id))
        return {"transferred": True, "generation": new_generation, "cgroup": cgroup, "populated": False}

    def _wait_recorded_unit(self, unit: str) -> str:
        load, active, cgroup = _unit_info(unit)
        if load == "not-found" or active in {"inactive", "failed"}:
            return cgroup
        raise SupervisorError(f"scope is not quiescent: {unit} ({active})")

    def wait_empty(self, mission_id: str, timeout: float = 8.0) -> str:
        op = self.conn.execute("SELECT unit_name FROM poc_scope_ops WHERE mission_id=?", (mission_id,)).fetchone()
        if not op:
            raise SupervisorError("no supervisor operation is recorded")
        cgroup = _wait_empty(op["unit_name"], timeout)
        self.conn.execute("UPDATE poc_scope_ops SET state='quiescent',cgroup_path=? WHERE mission_id=?",
                          (cgroup, mission_id))
        return cgroup

    def cancel(self, mission_id: str, timeout: float = 8.0,
               cancel_gate: threading.Event | None = None,
               cancel_waiting: threading.Event | None = None) -> str:
        op = self.conn.execute("SELECT unit_name FROM poc_scope_ops WHERE mission_id=?", (mission_id,)).fetchone()
        if not op:
            raise SupervisorError("no supervisor operation is recorded")
        self.conn.execute("UPDATE poc_scope_ops SET state='cancelling' WHERE mission_id=?", (mission_id,))
        if cancel_gate is not None:
            if cancel_waiting is not None:
                cancel_waiting.set()
            if not cancel_gate.wait(timeout=5):
                raise SupervisorError("test cancellation barrier timed out; lease remains fail-closed")
        populated, _cgroup, _active = _unit_populated(op["unit_name"])
        if populated:
            result = subprocess.run(["systemctl", "--user", "kill", "--kill-whom=all",
                                     "--signal=SIGKILL", op["unit_name"] + ".scope"], capture_output=True,
                                    text=True, timeout=5, env=_systemd_env(), check=False)
            if result.returncode:
                raise SupervisorError(f"scope cancellation failed: {result.stderr.strip()}")
        cgroup = _wait_empty(op["unit_name"], timeout)
        self.conn.execute("UPDATE poc_scope_ops SET state='quiescent',cgroup_path=? WHERE mission_id=?",
                          (cgroup, mission_id))
        return cgroup

    def recover(self, mission_id: str, *, cancel: bool = False) -> dict[str, Any]:
        op = self.conn.execute("SELECT * FROM poc_scope_ops WHERE mission_id=?", (mission_id,)).fetchone()
        if not op:
            raise SupervisorError("no persisted operation to recover")
        if op["state"] == "starting":
            return {"recovered": False, "state": "starting", "quiescence_verified": False}
        populated, cgroup, active = _unit_populated(op["unit_name"])
        if populated and not cancel:
            return {"recovered": False, "state": "running", "active": active,
                    "populated": True, "cgroup": cgroup}
        if populated:
            cgroup = self.cancel(mission_id)
        else:
            cgroup = cgroup or op["cgroup_path"]
        self.conn.execute("UPDATE poc_scope_ops SET state='quiescent',cgroup_path=? WHERE mission_id=?",
                          (cgroup, mission_id))
        return {"recovered": True, "state": "quiescent", "populated": False, "cgroup": cgroup}

    def escape_attempts(self) -> dict[str, Any]:
        checks = {
            "systemd_scope": ["/usr/bin/systemd-run", "--user", "--scope", "--unit=hermes-poc-escape",
                              "--", "/usr/bin/true"],
            "systemd_service": ["/usr/bin/systemctl", "--user", "start", "hermes-poc-escape.service"],
            "container": ["/usr/bin/docker", "info"],
            "systemd_timer": ["/usr/bin/systemd-run", "--user", "--on-active=1h",
                              "--unit=hermes-poc-escape-timer", "--", "/usr/bin/true"],
            "at_scheduler": ["/usr/bin/at", "now"],
            "crontab_scheduler": ["/usr/bin/crontab", "-l"],
        }
        code = r'''import json, subprocess, sys
checks = json.loads(sys.argv[1]); result = {}
for name, argv in checks.items():
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=3)
        result[name] = {'exit_code': run.returncode, 'output': (run.stdout + run.stderr)[-500:]}
    except OSError as exc:
        result[name] = {'unavailable': type(exc).__name__, 'error': str(exc)}
    except subprocess.TimeoutExpired:
        result[name] = {'blocked': False, 'error': 'timed out'}
open('/mission/output/escape-attempts.json', 'w').write(json.dumps(result, sort_keys=True))
'''
        embedded = f"exec({code!r})"
        command = ["systemd-run", "--user", "--scope", "--unit=hermes-mission-poc-escape-" + secrets.token_hex(4),
                   "--"] + _sandbox_command(self.paths, embedded, [json.dumps(checks)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=10,
                                env=_systemd_env(), check=False)
        if result.returncode:
            raise SupervisorError(f"escape-probe scope failed: {result.stderr.strip()}")
        output = self.paths["output"] / "escape-attempts.json"
        attempts = json.loads(output.read_text(encoding="utf-8"))
        units: dict[str, str] = {}
        for unit in ("hermes-poc-escape.scope", "hermes-poc-escape.service",
                     "hermes-poc-escape-timer.timer"):
            probe = subprocess.run(["systemctl", "--user", "show", "--property=LoadState", "--value", unit],
                                   capture_output=True, text=True, timeout=5, env=_systemd_env(), check=False)
            units[unit] = probe.stdout.strip() if probe.returncode == 0 else "query-failed"
        attempts["host_unit_load_states"] = units
        return attempts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mission-id", default="")
    parser.add_argument("--key", default="poc:default")
    parser.add_argument("--mode", choices=("start", "handoff", "recover", "cancel", "demo"), default="demo")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        supervisor = MissionSupervisorPoc(args.db, args.root)
        mission_id = args.mission_id or supervisor.admit(args.key)
        if args.mode == "start":
            print(json.dumps({"mission_id": mission_id, **supervisor.start(
                mission_id, delay=args.delay, crash=args.crash)}), flush=True)
            return 71 if args.crash else 0
        if args.mode == "handoff":
            print(json.dumps(supervisor.handoff(mission_id)), flush=True)
            return 0
        if args.mode == "cancel":
            print(json.dumps({"cgroup": supervisor.cancel(mission_id)}), flush=True)
            return 0
        if args.mode == "recover":
            result = supervisor.recover(mission_id, cancel=args.cancel)
            print(json.dumps(result), flush=True)
            return 0
        result = supervisor.start(mission_id, delay=args.delay)
        blocked = supervisor.handoff(mission_id)
        if blocked["transferred"]:
            raise SupervisorError("handoff succeeded while detached work was active")
        supervisor.wait_empty(mission_id)
        transfer = supervisor.handoff(mission_id)
        escape = supervisor.escape_attempts()
        samples = [result["launch_ms"]]
        for index in range(3):
            sample_id = supervisor.admit(f"{args.key}:latency:{index}")
            started = supervisor.start(sample_id, delay=0.1)
            supervisor.wait_empty(sample_id)
            supervisor.handoff(sample_id)
            samples.append(started["launch_ms"])
        samples.sort()
        metrics = {"launch_ms_samples": samples, "launch_ms_median": samples[len(samples) // 2],
                   "scope_memory_bytes": result["memory_bytes"]}
        print(json.dumps({"mission_id": mission_id, "scope": result, "blocked_handoff": blocked,
                          "transfer": transfer, "escapes": escape, "metrics": metrics},
                         sort_keys=True), flush=True)
        return 0
    except (SupervisorError, missions.MissionError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
